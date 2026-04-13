# Copyright (c) Meta Platforms, Inc. and affiliates.

import time
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch import nn

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.mamba.core_mamba import SSM
from apps.mamba.mamba import LMMamba, LMMambaArgs, StateCache
from lingua.args import dataclass_from_dict
from lingua.tokenizer import Tokenizer


@dataclass
class PackedCausalMambaGeneratorArgs(PackedCausalTransformerGeneratorArgs):
    pass


class PackedCausalMambaGenerator(PackedCausalTransformerGenerator):
    def __init__(
        self,
        cfg: PackedCausalMambaGeneratorArgs,
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

        self.prefill_tok_id = None
        self.cu_seqlens = None

    def clear_cache(self, lengths: torch.Tensor):
        for module in self.model.modules():
            if isinstance(module, SSM):
                module.cache = StateCache(
                    lengths.size(0),
                    module.n_heads,
                    module.head_dim,
                    module.state_dim,
                    module.conv_size,
                    module.conv_dim,
                    self.dtype,
                    self.device,
                )

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        self.clear_cache(lengths)

        self.prefill_tok_id = torch.repeat_interleave(lengths).unsqueeze(0).int()
        self.cu_seqlens = lengths.cumsum(0)
        self.cu_seqlens = torch.cat(
            [torch.tensor([0], device=self.device), self.cu_seqlens]
        ).int()

    @torch.compiler.disable
    def setup_generation(self, lengths):
        pass

    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            cu_seqlens=self.cu_seqlens,
            ssm_impl="ssm",
        )

        return prefill_out

    def generate_next_token(self, current_token):
        out = self.model.forward(
            current_token,
            tok_idx=None,
            cu_seqlens=None,
            ssm_impl="ssm_update",
        )
        return out

    def generate(self, prompts):
        return super().generate(prompts)


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(PackedCausalMambaGeneratorArgs, cfg, strict=False)
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(
        cfg.ckpt, model_cls=LMMamba, model_args_cls=LMMambaArgs
    )

    generator = PackedCausalMambaGenerator(gen_cfg, model, tokenizer)

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
