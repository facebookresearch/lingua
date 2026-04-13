# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn
from torch.nn import functional as F

from apps.mamba.component.causal_conv1d_compilable import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from apps.mamba.component.ssm_compilable import mamba_chunk_scan_combined
from lingua.probe import log_stats
from lingua.transformer import InitStdFactor, RMSNorm


@dataclass
class InitArgs:
    dt_max: float = 0.1
    dt_min: float = 0.001

    dt_init_floor: float = 1e-4

    A_init_min: float = 1
    A_init_max: float = 16


@dataclass
class BaseMambaArgs:

    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8

    state_dim: int = 128
    n_groups: int = 1
    conv_size: Optional[int] = None

    dt_bias: bool = False
    D_has_head_dim: bool = False
    learnable_init_states: bool = False

    ssm_chunk_size: int = 256

    vocab_size: int = -1

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256
    """
    Enforces that the SwiGLU hidden layer size is a multiple
    of large power of 2.
    """

    norm_eps: float = 1e-5

    init_use_depth: bool = False
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    init_args: InitArgs = field(default_factory=InitArgs)
    seed: int = 42


class SSM(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        state_dim: int,
        n_heads: int,
        n_groups: int,
        conv_size: Optional[int],
        dt_bias: bool,
        D_has_head_dim: Optional[bool],
        learnable_init_states: bool,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        # Fused kernel and sharding options
        chunk_size=256,
    ):
        super().__init__()

        self.dim = dim

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            self.hidden_dim % n_heads == 0
        ), f"Hidden dim must be divisible by n_heads: {self.hidden_dim} % {n_heads} != 0"

        self.state_dim = state_dim
        self.head_dim = self.hidden_dim // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.dt_limit = dt_limit

        self.chunk_size = chunk_size

        # Order: [z, x, B, C, dt]
        d_in_proj = (
            2 * self.hidden_dim + 2 * self.n_groups * self.state_dim + self.n_heads
        )
        self.in_proj = nn.Linear(dim, d_in_proj, bias=False)

        self.conv_size = conv_size
        self.conv_dim = None
        if conv_size is not None:
            self.conv_dim = self.hidden_dim + 2 * self.n_groups * self.state_dim
            assert (self.conv_dim % 8 == 0) and (
                conv_size in [2, 3, 4]
            ), f"Causal conv1d only supports conv_size in [2, 3, 4] and hidden_dim/head_dim % 8 == 0, got {self.conv_dim} and {conv_size}"
            self.conv_dim = self.hidden_dim + 2 * self.n_groups * self.state_dim
            self.conv_weight = nn.Parameter(torch.empty((self.conv_dim, conv_size)))

        self.learnable_init_states = learnable_init_states
        if learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(n_heads, self.head_dim, state_dim)
            )

        self.dt_bias = None
        if dt_bias:
            self.dt_bias = nn.Parameter(torch.empty(n_heads))
        self.A_log = nn.Parameter(torch.empty(n_heads))

        if D_has_head_dim is None:
            self.D = None
        elif D_has_head_dim:
            self.D = nn.Parameter(torch.ones(n_heads, self.head_dim))
        else:
            self.D = nn.Parameter(torch.ones(n_heads))

        self.out_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)

        self.ssm_norm = RMSNorm(self.hidden_dim, eps=1e-5)

        self.dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        zxbcdt = self.in_proj(x)

        # Causal conv1d path
        if self.conv_size is not None:
            z, xBC, dt = torch.split(
                zxbcdt,
                [
                    self.hidden_dim,
                    self.hidden_dim + 2 * self.n_groups * self.state_dim,
                    self.n_heads,
                ],
                dim=-1,
            )

            conv1d = log_stats(self.conv_weight, "conv1d.w")
            xBC = log_stats(xBC, "conv1d.in")

            if ssm_impl == "ssm":  # For training
                if hasattr(self, "cache"):
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0),
                        cu_seqlens,
                        state_len=self.cache.conv_cache.shape[-1],
                    )
                    self.cache.conv_cache.copy_(conv_varlen_states)

                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=conv1d,
                    bias=None,
                    activation="silu",
                    seq_idx=tok_idx,
                ).transpose(1, 2)

            elif ssm_impl == "ssm_update":  # For generation only
                xBC = causal_conv1d_update(
                    x=xBC.squeeze(0),
                    conv_state=self.cache.conv_cache,
                    weight=self.conv_weight,
                    bias=None,
                    activation="silu",
                ).unsqueeze(0)

            else:
                raise NotImplementedError(
                    f"SSM implementation {ssm_impl} not supported"
                )

            xBC = log_stats(xBC, "conv1d.out")

            x, B, C = torch.split(
                xBC,
                [
                    self.hidden_dim,
                    self.n_groups * self.state_dim,
                    self.n_groups * self.state_dim,
                ],
                dim=-1,
            )
        else:
            z, x, B, C, dt = torch.split(
                zxbcdt,
                [
                    self.hidden_dim,
                    self.hidden_dim,
                    self.n_groups * self.state_dim,
                    self.n_groups * self.state_dim,
                    self.n_heads,
                ],
                dim=-1,
            )

        initial_states = None
        if self.learnable_init_states:
            initial_states = self.init_states.expand(bsz, -1, -1, -1)

        x = x.view(
            bsz, seq_len, self.n_heads, self.head_dim
        )  # (bsz, seq_len, n_heads, head_dim)

        A_log = log_stats(self.A_log, "A_log")
        A = -torch.exp(A_log.float())
        B = B.view(
            bsz, seq_len, self.n_groups, self.state_dim
        )  # (bsz, seq_len, ngroups, state_dim)
        C = C.view(
            bsz, seq_len, self.n_groups, self.state_dim
        )  # (bsz, seq_len, ngroups, state_dim)

        A, B, C = log_stats(A, "A"), log_stats(B, "B"), log_stats(C, "C")  # For probing

        if ssm_impl == "ssm":  # For training
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=tok_idx,
                cu_seqlens=cu_seqlens,
                initial_states=initial_states,
                **self.dt_limit_kwargs,
            )  # (bsz, seq_len, n_heads, head_dim)

            if hasattr(self, "cache"):
                y, varlen_states = y
                self.cache.state_cache.copy_(varlen_states)

        elif ssm_impl == "ssm_update":  # For generation only
            x = x.squeeze(0)
            A = A[..., None, None].expand(self.n_heads, self.head_dim, self.state_dim)
            dt = dt.permute(1, 2, 0).expand(seq_len, self.n_heads, self.head_dim)
            D = self.D
            if D is not None and D.dim() == 1:
                D = D.unsqueeze(1).expand(self.n_heads, self.head_dim)
            B, C = B.squeeze(0), C.squeeze(0)
            y = selective_state_update(
                self.cache.state_cache,
                x,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=(
                    torch.zeros(self.n_heads, self.head_dim).to(x)
                    if self.dt_bias is None
                    else self.dt_bias.unsqueeze(1).expand(self.n_heads, self.head_dim)
                ),
                dt_softplus=True,
            ).unsqueeze(0)

        else:
            raise NotImplementedError(f"SSM implementation {ssm_impl} not supported")

        y = y.view(bsz, seq_len, self.hidden_dim)

        # Could be different activation function, including None, Mamba people post_norm here also (sometime norm(z)*y or norm(z*y))
        y = log_stats(y, "ssm_out")
        # y = self.ssm_norm(y)
        y = self.ssm_norm(y * F.silu(z))

        out = self.out_proj(y)

        return out

    def reset_parameters(self, init_std, factor, init_args: InitArgs):
        # Linear layers
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = out_init_std / factor

        nn.init.trunc_normal_(
            self.in_proj.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

        nn.init.trunc_normal_(
            self.out_proj.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )

        # SSM
        if self.dt_bias is not None:
            self.dt_bias.uniform_(init_args.dt_min, init_args.dt_max)
            self.dt_bias.clamp_(min=init_args.dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            self.dt_bias.data = self.dt_bias.data + torch.log(
                -torch.expm1(-self.dt_bias.data)
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

        if self.learnable_init_states:
            self.init_states.zero_()

        # Initialize A
        self.A_log.uniform_(init_args.A_init_min, init_args.A_init_max)
        self.A_log.log_()

        self.D.data.fill_(1.0)
        self.ssm_norm.reset_parameters()


class MambaBlock(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()

        self.ssm_norm = RMSNorm(args.dim, args.norm_eps)
        self.ssm = SSM(
            dim=args.dim,
            hidden_dim=3 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            state_dim=args.state_dim,
            n_heads=args.n_heads,
            n_groups=args.n_groups,
            conv_size=args.conv_size,
            dt_bias=args.dt_bias,
            D_has_head_dim=args.D_has_head_dim,
            learnable_init_states=args.learnable_init_states,
            chunk_size=args.ssm_chunk_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        x = x + self.ssm(
            self.ssm_norm(x), tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl
        )
        return x

    def init_weights(self, init_std=None, factor=1.0, init_args: InitArgs = InitArgs()):
        self.ssm_norm.reset_parameters()
        self.ssm.reset_parameters(init_std, factor, init_args)


class BaseMamba(nn.Module):
    def __init__(self, args: BaseMambaArgs):
        super().__init__()
        self.model_dim = args.dim
        self.init_base_std = args.init_base_std

        self.init_args = args.init_args
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MambaBlock(args))

    def forward(
        self,
        h: torch.Tensor,
        tok_idx: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        ssm_impl: str = "ssm",
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, tok_idx=tok_idx, cu_seqlens=cu_seqlens, ssm_impl=ssm_impl)
        return h

    def reset_parameters(self):
        pass

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.model_dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
