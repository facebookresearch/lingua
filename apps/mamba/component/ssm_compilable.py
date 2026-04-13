# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Optional, Tuple

import torch
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd, _mamba_chunk_scan_combined_fwd


@torch.compile(fullgraph=True)
def _compiled_mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=None):
    return _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit)

@torch.compile(fullgraph=True)
def _compiled_mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False, dt_limit=None):
    return _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit)


@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_fwd(
    x: torch.Tensor, 
    dt: torch.Tensor, 
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor, 
    chunk_size: int, 
    D: Optional[torch.Tensor] = None, 
    z: Optional[torch.Tensor] = None,  
    dt_bias: Optional[torch.Tensor] = None, 
    initial_states: Optional[torch.Tensor] = None,  
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit)

    return out, out_x if out_x is not None else out.new_empty(0), rest[0] if cu_seqlens is not None else out.new_empty(0)

@ssm_chunk_scan_combined_fwd.register_fake
def _ssm_chunk_scan_combined_fwd_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
):
    _, _, n_heads, head_dim = x.shape
    return (
        torch.empty_like(x), 
        torch.empty_like(x) if z is not None else None, 
        x.new_empty((cu_seqlens.size(0)-1, n_heads, head_dim, B.size(0))) if cu_seqlens is not None else None,
    )

@torch.library.custom_op(
    "mamba_ssm::ssm_chunk_scan_combined_bwd", 
    mutates_args=(),
    device_types="cuda",
)
def ssm_chunk_scan_combined_bwd(
    dout: torch.Tensor,
    x: torch.Tensor, 
    dt: torch.Tensor, 
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int, 
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=None, seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit)
    return (
        dx,
        ddt,
        dA,
        dB,
        dC,
        dD if dD is not None else dx.new_empty(0),
        dz if dz is not None else dx.new_empty(0),
        ddt_bias if ddt_bias is not None else dx.new_empty(0),
        dinitial_states if dinitial_states is not None else dx.new_empty(0)
    )

@ssm_chunk_scan_combined_bwd.register_fake
def _ssm_chunk_scan_combined_bwd_fake(
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[List[float]] = None
):
    return (
        torch.empty_like(x),
        torch.empty_like(dt),
        torch.empty_like(A),
        torch.empty_like(B),
        torch.empty_like(C),
        torch.empty_like(D) if D is not None else None,
        torch.empty_like(z) if z is not None else None,
        torch.empty_like(dt_bias) if dt_bias is not None else None,
        torch.empty_like(initial_states) if initial_states is not None else None,
    )


def ssm_chunk_scan_combined_setup_context(ctx, inputs, output):
    x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit = inputs
    out, out_x, state_varlen = output

    ctx.save_for_backward(out if z is None else out_x, x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx)
    ctx.dt_softplus = dt_softplus
    ctx.chunk_size = chunk_size
    ctx.dt_limit = dt_limit

def ssm_chunk_scan_combined_bridge(ctx, dout, dout_x, dout_state_varlen):
    out, x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors

    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = ssm_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)

    return (
        dx,
        ddt,
        dA,
        dB,
        dC, 
        None,
        dD if D is not None else None, 
        dz if z is not None else None,
        ddt_bias if dt_bias is not None else None, 
        dinitial_states if initial_states is not None else None, 
        None,
        None,
        None,
        None,
    )

# Register custom autograd function
torch.library.register_autograd(
    "mamba_ssm::ssm_chunk_scan_combined_fwd",
    ssm_chunk_scan_combined_bridge,
    setup_context=ssm_chunk_scan_combined_setup_context,
)

def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    
    out, _, varlen_states  = ssm_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit)
    if cu_seqlens is not None:
        return out, varlen_states
    return out

if __name__ == "__main__":
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as mamba_chunk_scan_combined_ref

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn(2, 3, 4, 5).cuda()
    dt = torch.randn(2, 3, 4).cuda()
    A = torch.randn(4).cuda()
    B = torch.randn(2, 3, 4, 5).cuda()
    C = torch.randn(2, 3, 4, 5).cuda()
    chunk_size = 2
    D = torch.randn(4, 5).cuda()
    z = torch.randn(2, 3, 4, 5).cuda()
    dt_bias = torch.randn(4).cuda()

    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    compiled_mamba_chunk_scan_combined = torch.compile(mamba_chunk_scan_combined)
    out = compiled_mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out.min(), out.max(), out.mean(), out.std())

    out_ref = mamba_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias)

    print(out_ref.min(), out_ref.max(), out_ref.mean(), out_ref.std())
