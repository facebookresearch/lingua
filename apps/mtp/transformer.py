# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from xformers.ops import AttentionBias, fmha

from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    cross_entropy,
)


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMMTPArgs(BaseTransformerArgs):

    seed: int = 42
    n_future_head: int = 1

    vocab_size: int = -1

    attn_impl: str = "sdpa"
    mask: str = "causal"
    sliding_window: Optional[int] = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMMTPArgs):
        super().__init__(args)
        self.sliding_window = args.sliding_window
        self.mask = args.mask
        self.attn_impl = args.attn_impl

        self.n_future_head = args.n_future_head

        assert self.n_future_head >= 1
        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.heads = nn.ModuleList()
        for _ in range(self.n_future_head):
            self.heads.append(
                nn.Linear(
                    args.dim,
                    args.vocab_size,
                    bias=False,
                )
            )

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[List[torch.Tensor]] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, self.attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        norm_h = self.norm(h)
        if target is not None:
            if self.training:
                ce = []
                for i, head in enumerate(self.heads):
                    logits = torch.utils.checkpoint.checkpoint(
                        head,
                        norm_h,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    ce.append(cross_entropy(logits, target[..., i]))
            else:
                head = self.heads[0]
                logits = head(norm_h)
                ce = cross_entropy(logits, target)
            return ce
        else:
            return self.heads[0](norm_h)

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        for head in self.heads:
            nn.init.trunc_normal_(
                head.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self):
        super().init_weights()


def build_fsdp_grouping_plan(model_args: LMMTPArgs) -> List[Tuple[str, bool]]:
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    return group_plan
