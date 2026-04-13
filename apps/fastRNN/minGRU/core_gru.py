# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from apps.fastRNN.component.rnn_common import conv1d, scan
from lingua.probe import log_stats
from lingua.transformer import InitStdFactor, RMSNorm


@dataclass
class BaseMinGRUArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 1

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    conv_size: Optional[int] = None

    norm_eps: float = 1e-5

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


def sequential_step(
    states: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return a * states + b


class GRU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,  # h_t dim (state expansion)
        n_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        conv_size: Optional[int] = None,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            hidden_dim % n_heads == 0
        ), f"Hidden dim must be divisible by n_heads: {hidden_dim} % {n_heads} != 0"

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.conv_size = conv_size
        if conv_size is not None:
            assert ((self.hidden_dim) % 8 == 0) and (
                conv_size in [2, 3, 4]
            ), f"Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, got {self.hidden_dim} and {conv_size}"
            self.conv_dim = self.hidden_dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.w = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wz = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wh_tilde = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        w0 = self.w(x.view_as(x))

        z = self.wz(x.view_as(x))
        tilde_h = self.wh_tilde(x.view_as(x))

        z = z.transpose(1, 2)
        tilde_h = tilde_h.transpose(1, 2)

        if self.conv_size is not None:
            conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            z = conv1d(
                x=z,
                conv_weight=conv1d_w,
                tok_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                impl=impl,
                cache=self.cache.conv_cache if hasattr(self, "cache") else None,
            )

        z = F.sigmoid(z)

        z = z.reshape(bsz * self.n_heads, self.head_dim, seq_len)
        tilde_h = tilde_h.reshape(bsz * self.n_heads, self.head_dim, seq_len)

        h = scan(
            a=(1 - z),
            b=(z * tilde_h),
            cu_seqlens=cu_seqlens,
            impl=impl,
            cache=self.cache.state_cache if hasattr(self, "cache") else None,
        )

        h = h.view(bsz, self.hidden_dim, seq_len).transpose(1, 2)
        h = log_stats(h, "hidden_state")

        h = h * F.silu(w0)

        out = self.wo(h)

        return out

    def reset_parameters(self, init_std, factor):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor

        for w in [self.w, self.wz, self.wh_tilde]:
            nn.init.trunc_normal_(
                w.weight, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std
            )

        nn.init.trunc_normal_(
            self.wo.weight, std=out_init_std, a=-3 * in_init_std, b=3 * in_init_std
        )

        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(
                self.conv_weight, std=conv_std, a=-3 * conv_std, b=3 * conv_std
            )


class GRUBlock(nn.Module):
    def __init__(self, args: BaseMinGRUArgs):
        super().__init__()

        self.gru_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.gru = GRU(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            n_heads=args.n_heads,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            conv_size=args.conv_size,
        )

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        x = x + self.gru(self.gru_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        return x

    def init_weights(self, init_std: Optional[float], factor: InitStdFactor):
        self.gru.reset_parameters(init_std, factor)
        self.gru_norm.reset_parameters()


class BaseMinGRU(nn.Module):
    def __init__(self, args: BaseMinGRUArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(GRUBlock(args))

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
