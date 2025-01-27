# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
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
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

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
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
