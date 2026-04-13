# Copyright (c) Meta Platforms, Inc. and affiliates.

import time
from itertools import chain
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from apps.aunet.data.regex_cutting import RegexArgs, RegexPool
from apps.aunet.hierarchical import (
    HierarchicalArgs,
    HierarchicalTransformer,
)
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    batch_prompts,
    pack_prompts,
    sample_tokens,
)
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.transformer import Attention, causal_mask, generate_doc_mask_mod, lengths_to_local_ids, lengths_to_start_ids


def causal_sliding_mask(sliding_window: int):
    def mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= sliding_window)
    return mask

def load_consolidated_model_and_tokenizer(consolidated_path, model_cls, model_args_cls):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    
    updated_state_dict = {}
    for key, value in st_dict["model"].items():
        if "transitions." in key and "parametrizations" in key:
            key = key.replace("parametrizations", "_orig_mod.parametrizations")
        updated_state_dict[key] = value
    st_dict["model"] = updated_state_dict

    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    regex_args = dataclass_from_dict(RegexArgs, config.data.regex, strict=False)
    regex_pool = RegexPool(regex_args)


    return model, tokenizer, regex_pool, config

class PackedHierarchicalCausalTransformerGeneratorArgs(PackedCausalTransformerGeneratorArgs):
    regex_buffer_size: Optional[int] = None
    use_regex_for_level_mask: bool = True

class PackedInternalCausalTransformerGenerator(PackedCausalTransformerGenerator):
    def generate_next_token(self, current_token, hierarchical_mask):
        doc_mask = self.current_doc_id[hierarchical_mask].unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id[hierarchical_mask].unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        if self.model.sliding_window is not None:
            local_mask =(self.current_tok_id[hierarchical_mask] - self.model.sliding_window).unsqueeze(1) <= self.padded_tok_id.unsqueeze(0)
        else:
            local_mask = torch.ones_like(caus_mask)
        mask = doc_mask & caus_mask & local_mask
        
        # mask the offset of the kv_cache
        _offsets = []
        for module in self.model.modules():
            if isinstance(module, Attention):
                _offset = module.kv_cache.offset
                module.kv_cache.offset = _offset[hierarchical_mask]
                _offsets.append(_offset)
        
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id[hierarchical_mask],  # n_seqs
            mask=mask,
            attn_impl="sdpa",
        )
        self.current_tok_id[hierarchical_mask] += 1

        # restaure original offset
        idx = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = _offsets[idx]
                idx += 1

        return out
    
    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        if self.model.sliding_window is None:
            doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        else:
            doc_mask_mod = generate_doc_mask_mod(causal_sliding_mask(self.model.sliding_window), lengths, padded_lengths)
        self.prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)
    
class PackedHierarchicalCausalTransformerGenerator(PackedCausalTransformerGenerator):
    def __init__(
        self,
        cfg: PackedHierarchicalCausalTransformerGeneratorArgs,
        model: HierarchicalTransformer,
        tokenizer: Tokenizer,
        regex_pool: RegexPool,
    ):
        self.regex_buffer_size = cfg.regex_buffer_size
        self.regex_pool = regex_pool
        self.use_regex_for_level_mask = cfg.use_regex_for_level_mask

        self.encoders = [
            PackedInternalCausalTransformerGenerator(cfg, enc, tokenizer)
            for enc in model.encoders
        ]

        self.trunk = PackedInternalCausalTransformerGenerator(cfg, model.trunk, tokenizer)
        self.decoders = [
            PackedInternalCausalTransformerGenerator(cfg, dec, tokenizer)
            for dec in model.decoders
        ]

        self.transitions = [
            trans for trans in model.transitions
        ]
        
        super().__init__(cfg, model, tokenizer)

    def get_level_mask(self, sampled_tokens, features):
        level_mask = None
        if self.model.lambda_level and self.use_regex_for_level_mask:
            level_mask = self.model._level_mask_logits(features, sampled_tokens).argmax(dim=-1).flatten()
        else:
            level_mask = self.regex_pool.get_levels_mask_gen(sampled_tokens.tolist()[0])
            level_mask = torch.tensor(level_mask, dtype=torch.long).cuda()

        assert level_mask is not None

        return level_mask

    def prefill(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        level_mask: torch.Tensor,
    ):    
        pool_masks, pool_mask_idcs, nb_toks = self.model.get_pool_mask(level_mask, max_seqlen=None, return_idcs=True, force_first=True)

        all_doc_lengths = [lengths]
        doc_lengths = lengths
        for pool_mask in pool_masks:
            assert pool_mask[0, doc_lengths.sum():].sum() == 0
            doc_lengths = torch.tensor(
                [m.sum() for m in pool_mask.squeeze()[:doc_lengths.sum()].split(doc_lengths.tolist(), dim=0)],
                dtype=torch.int,
                device=self.device,
            )
            all_doc_lengths.append(doc_lengths)
        
        self.setup_residual_cache(lengths.size(0))
        self.setup_decoder_cache(lengths.size(0))
        self.setup_local_position_cache(lengths.size(0))
        
        residuals = []
        positions = []
        repeat_idcs = []
        x = self.model.tok_embeddings(tokens)
        it = zip(self.encoders, self.transitions, pool_masks, pool_mask_idcs, all_doc_lengths[:-1])
        for i, (encoder, trans, mask, mask_id, doc_lengths) in enumerate(it):
            
            x = encoder.prefill(x, doc_lengths)
            residuals.append(x)
            
            # caching last token of each sequense
            last_tokens_per_seq = doc_lengths.cumsum(0) - 1
            self.residual_cache[i][:] = x[0, last_tokens_per_seq]

            # local positions
            seqlen = x.size(1)
            repeat_idx = mask[0, :seqlen].cumsum(0)
            repeat_idx = repeat_idx - repeat_idx[0].clone()
            position = torch.arange(seqlen, device=self.device)
            position = (position - mask_id.gather(0, repeat_idx)) % getattr(trans, 'max_pos', 16)
            positions.append(position)
            repeat_idcs.append(repeat_idx)

            # caching last local position of each sequence
            self.position_cache[i][:] = position[last_tokens_per_seq] + 1
            
            x = trans.down(x, mask, encoder.model.rope_embeddings.freqs_cis, mask_id.unsqueeze(0), position.unsqueeze(0))

        x = self.trunk.prefill(x, all_doc_lengths[-1])
        self.decoder_cache[-1][:] = x[0, all_doc_lengths[-1].cumsum(0) - 1]

        it = list(
            zip(
                self.decoders,
                self.transitions,
                residuals,
                self.model.adding_residuals,
                pool_masks,
                pool_mask_idcs,
                all_doc_lengths[:-1],
                positions,
                repeat_idcs,
                self.decoder_cache[:-1]
            )
        )
        it = it[::-1]
        for decoder, trans, res, add_res, mask, mask_id, doc_lengths, position, repeat_idx, dec_cache in it:
            x = trans.up(x, res, mask, decoder.model.rope_embeddings.freqs_cis, mask_id.unsqueeze(0), position.unsqueeze(0), repeat_idx.unsqueeze(0))
            if add_res:
                x = res + x
            x = decoder.prefill(x, doc_lengths)
            dec_cache[:] = x[0, doc_lengths.cumsum(0) - 1]
        
        features = None
        if self.model.lambda_level:
            features = x.clone()

        return self.model.vocab(x), features

    def generate_next_token(self, current_token, level_mask):
        pool_masks, pool_mask_idcs, nb_toks = self.model.get_pool_mask(level_mask, max_seqlen=None, return_idcs=True, force_first=False)
        mask_for_current_stage = [torch.arange(current_token.size(1), device=current_token.device)] + [m for m in pool_mask_idcs]

        nb_toks.insert(0, current_token.size(1))

        x = self.model.tok_embeddings(current_token)
        it = zip(
            nb_toks[:-1],
            self.encoders,
            self.transitions,
            pool_masks,
            pool_mask_idcs,
            mask_for_current_stage[:-1],
            self.position_cache
        )
        for  i, (nb, enc, trans, mask, mask_id, current_mask, position) in enumerate(it):
            if nb > 0: # There is token to pool from
                absolute_idx = level_mask[0] >= i
                x = enc.generate_next_token(x, absolute_idx)
                self.residual_cache[i][absolute_idx, :] = x[0, :]
                if mask_id.numel() > 0:
                    position[level_mask[0] > i] = 0
                x = trans.down(x, mask, enc.model.rope_embeddings.freqs_cis, mask_id.unsqueeze(0), position[absolute_idx].unsqueeze(0))
        
        if nb_toks[-1] > 0:
            absolute_idx = (level_mask[0] == len(nb_toks)-1)
            x = self.trunk.generate_next_token(x, absolute_idx)
            self.decoder_cache[-1][absolute_idx, :] = x[0, :]

        it = zip(
            nb_toks[:-1],
            self.decoders,
            self.transitions,
            self.residual_cache,
            self.model.adding_residuals,
            pool_masks,
            pool_mask_idcs,
            mask_for_current_stage[:-1],
            self.position_cache,
            self.decoder_cache[1:],
            self.decoder_cache[:-1],
        )
        it = list(it)[::-1]
        i = len(nb_toks) - 2
        for nb, decoder, trans, res, add_res, mask, mask_id, current_mask, position, dec_cache, next_dec_cache in it:
            if nb > 0:
                absolute_idx = level_mask[0] >= i
                x = trans.up(dec_cache[absolute_idx].unsqueeze(0), res[absolute_idx].unsqueeze(0), mask, decoder.model.rope_embeddings.freqs_cis, mask_id.unsqueeze(0), position[absolute_idx].unsqueeze(0))
                if add_res:
                    x = res[absolute_idx].unsqueeze(0) + x
                next_dec_cache[None, absolute_idx, :] = decoder.generate_next_token(x, absolute_idx)
                
                position[absolute_idx] += 1
                position[absolute_idx] %= getattr(trans, 'max_pos', 16)
            i -= 1

        x = self.decoder_cache[0].unsqueeze(0)

        features = None
        if self.model.lambda_level:
            features = x.clone()

        return self.model.vocab(x), features
    
    @torch.inference_mode()
    def generate(self, prompts):
        # Tokenize
        prompts = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]
        # Truncate
        max_prompt_len = self.max_prompt_len or min(
            self.model.encoders[0].max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []

        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()

            level_mask = self.regex_pool.get_levels_mask_prefill(batch, size=self.regex_buffer_size, force_first=True)
            level_mask = torch.tensor(level_mask, dtype=torch.long).unsqueeze(0)
            level_mask = level_mask.cuda()

            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits, features = self.prefill(packed_batch.unsqueeze(0), lengths, level_mask)
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]
            if features is not None:
                features = features[:, lengths.cumsum(0) - 1]
            
            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):
                level_mask = self.get_level_mask(current_token, features).unsqueeze(0)
                next_logits, features = self.generate_next_token(current_token, level_mask)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-self.max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in self.until]
                        )
                        is_done[seq_id] = (
                            contains_end_string or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                x = logit[:-1].cpu()
                y = torch.tensor(p[1:])
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none"))
                greedy.append(x.argmax(dim=-1) == y)

        return generation, loglikelihood, greedy
    
    @property
    def max_prompt_len(self):
        return self._max_prompt_len
    
    @max_prompt_len.setter
    def max_prompt_len(self, value: int):
        if value is not None and value > 1:
            value *= 2
        self._max_prompt_len = value
        for generator in chain(self.encoders, [self.trunk], self.decoders):
            generator.max_prompt_len = value
    
    @property
    def max_gen_len(self):
        return self._max_gen_len
    
    @max_gen_len.setter
    def max_gen_len(self, value: int):
        if value is not None and value > 1:
            value *= 2
        self._max_gen_len = value
        for generator in chain(self.encoders, [self.trunk], self.decoders):
            generator.max_gen_len = value
    
    @property
    def max_tokens(self):
        return self._max_tokens
    
    @max_tokens.setter
    def max_tokens(self, value: int):
        self._max_tokens = value
        for generator in chain(self.encoders, [self.trunk], self.decoders):
            generator.max_tokens = value
            self.update_rope(generator, value)
    
    def update_rope(self, generator: PackedCausalTransformerGenerator, max_seqlen: int):
        theta, head_dim = generator.model.rope_embeddings.theta, generator.model.rope_embeddings.head_dim
        generator.model.rope_embeddings = getattr(generator.model.rope_embeddings, '_orig_mod', generator.model.rope_embeddings).__class__(theta=theta, head_dim=head_dim, max_seqlen=max_seqlen)
        generator.model.rope_embeddings.reset_parameters()
        generator.model.rope_embeddings.to(generator.device)

    def setup_residual_cache(self, bsz):
        self.residual_cache = [torch.zeros((bsz, enc.dim), dtype=self.dtype, device=self.device) for enc in self.model.encoders]
    
    def setup_decoder_cache(self, bsz):
        self.decoder_cache = [torch.zeros((bsz, dec.dim), dtype=self.dtype, device=self.device) for dec in self.model.decoders]
        self.decoder_cache.append(torch.zeros(bsz, self.model.trunk.dim, dtype=self.dtype, device=self.device))

    def setup_local_position_cache(self, bsz):
        self.position_cache = [torch.zeros(bsz, dtype=int, device=self.device) for _ in range(len(self.model.encoders))]
    
    def set_hiearchical_mask(self, trans, hierarchical_mask):
        if hasattr(trans, "down_attn"):
            trans.down_attn.set_hierarchical_mask(hierarchical_mask)
        if hasattr(trans, "up_attn"):
            trans.up_attn.set_hierarchical_mask(hierarchical_mask)

def init_model_and_tokenizer(
    config_path,
):
    config = OmegaConf.load(config_path)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = config.model
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)

    model = HierarchicalTransformer(model_args)
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedHierarchicalCausalTransformerGeneratorArgs, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, regex_pool, _ = load_consolidated_model_and_tokenizer(
        cfg.ckpt, model_cls=HierarchicalTransformer, model_args_cls=HierarchicalArgs
    )

    gen_cfg.max_prompt_len = 2048#16000
    gen_cfg.max_tokens = 8192
    gen_cfg.max_gen_len = 1024
    gen_cfg.temperature = 0.0
    # gen_cfg.top_p = 0.95
    # gen_cfg.dtype = "fp32"
    gen_cfg.use_regex_for_level_mask = False
    gen_cfg.regex_buffer_size = None

    generator = PackedHierarchicalCausalTransformerGenerator(gen_cfg, model, tokenizer, regex_pool)

    # Allow multiple prompts
    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")
        print(f"Loglikelihood: {loglikelihood[i].mean().item()}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
