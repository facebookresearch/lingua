# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from apps.fastRNN.component.rnn_common import conv1d, scan
from lingua.probe import log_stats
from lingua.transformer import FeedForward, InitStdFactor, RMSNorm


@dataclass
class BaseHawkArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 1

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    lru_dim_multiplier: Optional[float] = None

    conv_size: Optional[int] = None

    norm_eps: float = 1e-5

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


_MAX_SQRT_GRADIENT: float = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


def sqrt_bounded_derivative(x: torch.Tensor) -> torch.Tensor:
    return SqrtBoundDerivative.apply(x)


class RGLRU(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        conv_size: Optional[int] = None,
    ):
        super().__init__()

        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        assert (
            head_dim * n_heads == dim
        ), f"dim {dim} must be equal to n_heads {n_heads} * head_dim {head_dim}"

        self.c = 8.0

        self.conv_size = conv_size
        if conv_size is not None:
            assert (dim % 8 == 0) and (
                conv_size in [2, 3, 4]
            ), f"Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, got {dim} and {conv_size}"
            self.conv_dim = self.dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.register_parameter("a", nn.Parameter(torch.empty((head_dim))))

        self.input_gate = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.a_gate = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        if self.conv_size is not None:
            conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            x = conv1d(
                x=x.transpose(1, 2),
                conv_weight=conv1d_w,
                tok_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                impl=impl,
                cache=self.cache.conv_cache if hasattr(self, "cache") else None,
            ).transpose(1, 2)

        gate_x = F.sigmoid(self.input_gate(x.view_as(x)))
        gate_a = F.sigmoid(self.a_gate(x.view_as(x)))

        gate_x = gate_x.transpose(1, 2).reshape(
            bsz * self.n_heads, self.head_dim, seqlen
        )
        gate_a = gate_a.transpose(1, 2).reshape(
            bsz * self.n_heads, self.head_dim, seqlen
        )

        a = (
            F.softplus(self.a)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bsz * self.n_heads, self.head_dim, seqlen)
        )

        log_a = -self.c * gate_a * a
        a = log_a.exp()
        multiplier = sqrt_bounded_derivative(1.0 - (2.0 * log_a).exp())

        x = x.transpose(1, 2).reshape(bsz * self.n_heads, self.head_dim, seqlen)

        h = scan(
            a=a.contiguous(),
            b=(multiplier * gate_x * x).contiguous(),
            cu_seqlens=cu_seqlens,
            impl=impl,
            cache=self.cache.state_cache if hasattr(self, "cache") else None,
        )

        h = h.view(bsz, self.dim, seqlen).transpose(1, 2)
        h = log_stats(h, "hidden_state")

        return h

    def reset_parameters(self, init_std, factor):
        in_init_std = init_std or (self.dim ** (-0.5))
        in_init_std = in_init_std / factor

        for w in [self.input_gate, self.a_gate]:
            nn.init.trunc_normal_(
                w.weight, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std
            )

        min_rad, max_rad = 0.9, 0.999
        self.a.data.uniform_(min_rad**2 + 1e-8, max_rad**2 + 1e-8)
        self.a.data.log_().mul_(0.5)

        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(
                self.conv_weight, std=conv_std, a=-3 * conv_std, b=3 * conv_std
            )


class RGLRUBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        multiple_of: int,
        lru_dim_multiplier: Optional[float],
        conv_size: Optional[int] = None,
    ):
        super().__init__()

        if lru_dim_multiplier is not None:
            hidden_dim = int(lru_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            hidden_dim % n_heads == 0
        ), f"Hidden dim must be divisible by n_heads: {hidden_dim} % {n_heads} != 0"

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.wy = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wx = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.rglru = RGLRU(
            dim=hidden_dim,
            n_heads=n_heads,
            head_dim=hidden_dim // n_heads,
            conv_size=conv_size,
        )

        self.wo = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        h = self.rglru(self.wx(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        h = h * F.silu(self.wy(x))
        y = x + self.wo(h)

        return y

    def init_weights(self, init_std: Optional[float], factor: InitStdFactor):
        self.rglru.reset_parameters(init_std, factor)

        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor

        for w in [self.wy, self.wx]:
            nn.init.trunc_normal_(
                w.weight, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std
            )

        nn.init.trunc_normal_(
            self.wo.weight, std=out_init_std, a=-3 * out_init_std, b=3 * out_init_std
        )


class HawkBlock(nn.Module):
    def __init__(self, args: BaseHawkArgs):
        super().__init__()

        self.rlgru_block = RGLRUBlock(
            dim=args.dim,
            hidden_dim=int(4 / 3 * args.dim),
            n_heads=args.n_heads,
            conv_size=args.conv_size,
            multiple_of=args.multiple_of,
            lru_dim_multiplier=args.lru_dim_multiplier,
        )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.rlgru_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        x = x + self.rlgru_block(self.rlgru_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, init_std: Optional[float], factor: InitStdFactor):
        self.rlgru_block.init_weights(init_std, factor)
        self.rlgru_norm.reset_parameters()
        self.feed_forward.reset_parameters()
        self.ffn_norm.reset_parameters()


class BaseHawk(nn.Module):
    def __init__(self, args: BaseHawkArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(HawkBlock(args))

    def forward(
        self, h: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        return h

    def reset_parameters(self):
        pass

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
