# Copyright (c) Meta Platforms, Inc. and affiliates.

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from apps.fastRNN.hawk.core_hawk import RGLRU
from apps.fastRNN.hawk.hawk import LMHawk, LMHawkArgs
from apps.fastRNN.minGRU.core_gru import GRU
from apps.fastRNN.minGRU.mingru import LMMinGRU, LMMinGRUArgs
from apps.fastRNN.minLSTM.core_lstm import LSTM
from apps.fastRNN.minLSTM.minlstm import LMMinLSTM, LMMinLSTMArgs
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
)
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer


def load_consolidated_model_and_tokenizer(consolidated_path):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    if config.model_type.lower() == "mingru":
        model_cls = LMMinGRU
        model_args_cls = LMMinGRUArgs
    elif config.model_type.lower() == "minlstm":
        model_cls = LMMinLSTM
        model_args_cls = LMMinLSTMArgs
    elif config.model_type.lower() == "hawk":
        model_cls = LMHawk
        model_args_cls = LMHawkArgs
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"], strict=False)
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, config


class StateCache(nn.Module):
    def __init__(self, bsz, n_heads, head_dim, conv_size, conv_dim, dtype, device):
        super().__init__()
        state_shape = (n_heads, head_dim, bsz)
        if conv_size is None:
            conv_shape = (0,)
        else:
            conv_shape = (bsz, conv_dim, conv_size)

        self.register_buffer(
            "conv_cache",
            torch.zeros(conv_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "state_cache",
            torch.zeros(state_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def reset(self):
        self.conv_cache.zero_()
        self.state_cache.zero_()


@dataclass
class PackedRNNGeneratorArgs(PackedCausalTransformerGeneratorArgs):
    pass


class PackedRNNGenerator(PackedCausalTransformerGenerator):
    def __init__(
        self,
        cfg: PackedRNNGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.cu_seqlens = None
        self.tok_idx = None

    def clear_cache(self, lengths: torch.Tensor):
        for module in self.model.modules():
            if isinstance(module, (GRU, LSTM, RGLRU)):
                module.cache = StateCache(
                    lengths.size(0),
                    module.n_heads,
                    module.head_dim,
                    module.conv_size,
                    module.conv_dim,
                    self.dtype,
                    self.device,
                )

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        self.clear_cache(lengths)

        self.cu_seqlens = lengths.cumsum(0)
        self.cu_seqlens = torch.cat(
            [torch.tensor([0], device=self.device), self.cu_seqlens]
        ).int()

        self.tok_idx = torch.repeat_interleave(lengths).int().unsqueeze(0).to(self.device)


    @torch.compiler.disable
    def setup_generation(self, lengths):
        pass

    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.tok_idx,
            cu_seqlens=self.cu_seqlens,
            impl="parallel",
        )

        return prefill_out

    def generate_next_token(self, current_token):
        out = self.model.forward(
            current_token,
            cu_seqlens=None,
            impl="sequential",
        )
        return out

    def generate(self, prompts):
        return super().generate(prompts)


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(PackedRNNGeneratorArgs, cfg, strict=False)
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedRNNGenerator(gen_cfg, model, tokenizer)

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

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
