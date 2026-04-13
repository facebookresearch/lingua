# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import causal_conv1d_cuda
import torch


# Causal Conv1D Forward Function
@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_fwd",
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    # Ensure activation is valid
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # Ensure x is contiguous
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()

    # Make bias and seq_idx contiguous if they exist
    bias = bias.contiguous() if bias is not None else None
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None

    # Translate activation to bool for custom CUDA kernel
    use_activation = activation in ["silu", "swish"]

    # Call custom CUDA kernel for forward pass
    out = causal_conv1d_cuda.causal_conv1d_fwd(
        x, weight, bias, seq_idx, None, None, use_activation
    )
    return out

# Register a fake forward pass for tracing
@causal_conv1d_fwd.register_fake
def _causal_conv1d_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    torch._check(x.shape[-2] == weight.shape[0])
    return torch.empty_like(x)

# Causal Conv1D Backward Function
@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_bwd", 
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    activation: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Ensure dout is contiguous
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()

    # Call custom CUDA kernel for backward pass
    dx, dweight, dbias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
        x, weight, bias, dout, seq_idx, None, None, None, False, activation
    )

    # Handle optional bias gradient
    dbias = dbias if bias is not None else torch.empty((0,), device=dout.device)
    
    return dx, dweight, dbias

# Register a fake backward pass for tracing
@causal_conv1d_bwd.register_fake
def _causal_conv1d_bwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    activation: bool,
):
    return (
        torch.empty_like(x),
        torch.empty_like(weight),
        torch.empty_like(bias) if bias is not None else None,
    )

# Setup context for autograd
def causal_conv1d_setup_context(ctx, inputs, output):
    x, weight, bias, seq_idx, activation = inputs
    ctx.activation = activation in ["silu", "swish"]
    ctx.save_for_backward(x, weight, bias, seq_idx)

# Bridge for backward pass in autograd
def causal_conv1d_bwd_bridge(ctx, dout):
    x, weight, bias, seq_idx = ctx.saved_tensors
    dx, dweight, dbias = causal_conv1d_bwd(x, weight, bias, dout, seq_idx, ctx.activation)
    
    # Handle None return values
    dbias = dbias if bias is not None else None
    return dx, dweight, dbias, None, None

# Register custom autograd function
torch.library.register_autograd(
    "mamba_causal_conv1d::causal_conv1d_fwd",
    causal_conv1d_bwd_bridge,
    setup_context=causal_conv1d_setup_context,
)

# Define a higher-level function to invoke the custom op
def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, activation=None):
    return causal_conv1d_fwd(x, weight, bias, seq_idx, activation)


@torch.library.custom_op(
    "mamba_causal_conv1d::causal_conv1d_update",
    mutates_args=(),
    device_types="cuda",
)
def causal_conv1d_update_fwd(
    x: torch.Tensor, 
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    out = causal_conv1d_cuda.causal_conv1d_update(
        x, conv_state, weight, bias, activation, cache_seqlens
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

@causal_conv1d_update_fwd.register_fake
def _causal_conv1d_update_fwd(
    x: torch.Tensor, 
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(x)

def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    return causal_conv1d_update_fwd(x, conv_state, weight, bias, activation, cache_seqlens)

# Test the implementation
if __name__ == "__main__":
    from causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_ref

    torch.manual_seed(0)

    x = torch.randn(8, 32, 16, device="cuda", requires_grad=True)
    weight = torch.randn(32, 3, device="cuda", requires_grad=True)
    bias = None#torch.randn(32, device="cuda", requires_grad=True)

    # Test the forward and backward pass
    print("Custom Implementation")
    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    out.sum().backward()

    print(out.min(), out.max(), out.mean(), out.std())
    print(x.grad.min(), x.grad.max(), x.grad.mean(), x.grad.std())
    print(weight.grad.min(), weight.grad.max(), weight.grad.mean(), weight.grad.std())

    # Try compiling the function using torch.compile
    x.grad.zero_(), weight.grad.zero_()
    compiled_conv1d = torch.compile(causal_conv1d_fn)
    print(compiled_conv1d)

    # Run the compiled function
    print("Compiled Implementation")
    out = compiled_conv1d(x, weight, bias, activation="silu")
    out.sum().backward()

    print(out.min(), out.max(), out.mean(), out.std())
    print(x.grad.min(), x.grad.max(), x.grad.mean(), x.grad.std())
    print(weight.grad.min(), weight.grad.max(), weight.grad.mean(), weight.grad.std())

    print("Reference Implementation")
    x.grad.zero_(), weight.grad.zero_()
    out = causal_conv1d_fn_ref(x, weight, bias, activation="silu")
    out.sum().backward()

    print(out.min(), out.max(), out.mean(), out.std())
    print(x.grad.min(), x.grad.max(), x.grad.mean(), x.grad.std())
    print(weight.grad.min(), weight.grad.max(), weight.grad.mean(), weight.grad.std())
