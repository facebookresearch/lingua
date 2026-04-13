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
class BaseMinLSTMArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 1

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    conv_size: Optional[int] = None

    norm_eps: float = 1e-5

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


class LSTM(nn.Module):
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
            ), f"Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim % 8 == 0, got {self.hidden_dim} and {conv_size}"
            self.conv_dim = 2 * self.hidden_dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.w = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.wfi = nn.Linear(
            dim,
            2 * hidden_dim,
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

        fi = self.wfi(x.view_as(x)).transpose(1, 2)
        h_tilde = self.wh_tilde(x.view_as(x)).transpose(1, 2)

        if self.conv_size is not None:
            conv1d_w = log_stats(self.conv_weight, "conv1d.w")
            fi = conv1d(
                x=fi,
                conv_weight=conv1d_w,
                tok_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                impl=impl,
                cache=self.cache.conv_cache if hasattr(self, "cache") else None,
            )

        fi = fi.reshape(bsz * self.n_heads, 2 * self.head_dim, seq_len)
        h_tilde = h_tilde.reshape(bsz * self.n_heads, self.head_dim, seq_len)

        f, i = fi.chunk(2, dim=1)
        f, i = F.sigmoid(f), F.sigmoid(i)
        denom = 1 / (f + i + 1e-4)

        h = scan(
            a=(f * denom),
            b=(h_tilde * i * denom),
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

        for w in [self.w, self.wfi, self.wh_tilde]:
            nn.init.trunc_normal_(
                w.weight, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std
            )

        nn.init.trunc_normal_(
            self.wo.weight, std=out_init_std, a=-3 * in_init_std, b=3 * in_init_std
        )

        if self.conv_size is not None:
            conv_std = init_std or (self.conv_size ** (-0.5))
            nn.init.trunc_normal_(
                self.conv_weight,
                mean=0.0,
                std=conv_std,
                a=-3 * conv_std,
                b=3 * conv_std,
            )


class LSTMBlock(nn.Module):
    def __init__(self, args: BaseMinLSTMArgs):
        super().__init__()

        self.lstm_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lstm = LSTM(
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
        x = x + self.lstm(self.lstm_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        return x

    def init_weights(self, init_std: Optional[float], factor: InitStdFactor):
        self.lstm.reset_parameters(init_std, factor)
        self.lstm_norm.reset_parameters()


class BaseMinLSTM(nn.Module):
    def __init__(self, args: BaseMinLSTMArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(LSTMBlock(args))

    def forward(
        self, x: torch.Tensor, tok_idx: torch.Tensor, cu_seqlens: torch.Tensor, impl: str = "parallel"
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, tok_idx=tok_idx, cu_seqlens=cu_seqlens, impl=impl)
        return x

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
