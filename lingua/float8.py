# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
import warnings
from typing import Callable

import torch

# avoid division by zero when calculating scale
EPS = 1e-12


def scale(t, amax_t):
    min_v = torch.finfo(torch.float8_e4m3fn).min
    max_v = torch.finfo(torch.float8_e4m3fn).max
    scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
    t_fp8 = (t / scale_t).clamp(min=min_v, max=max_v).to(torch.float8_e4m3fn)
    return t_fp8, scale_t


def matmul(first, amax_first, second_t, amax_second_t, bias, use_fast_accum):
    first_fp8, scale_first = scale(first, amax_first)
    second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
    # PyTorch's row-wise scaled matmul kernel is based on CUTLASS and is quite
    # slow when fast_accum is disabled. Hence we fall back to an "unscaled"
    # matmul, which uses cuBLAS, and apply the scale manually afterwards.
    output = torch._scaled_mm(
        first_fp8,
        second_t_fp8.t(),
        scale_a=scale_first.new_ones((1, 1)),
        scale_b=scale_second_t.t().new_ones((1, 1)),
        bias=None,
        out_dtype=torch.bfloat16,
        use_fast_accum=use_fast_accum,
    )
    output = (output * scale_first * scale_second_t.t()).to(torch.bfloat16)
    if bias is not None:
        output = output + bias
    return output


@torch.compiler.allow_in_graph
class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t, bias):
        amax_a = a.abs().amax(dim=-1, keepdim=True)
        amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
        out = matmul(a, amax_a, b_t, amax_b_t, bias, use_fast_accum=True)

        ctx.a_requires_grad = a.requires_grad
        ctx.b_requires_grad = b_t.requires_grad
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False

        ctx.save_for_backward(a, b_t, amax_b_t.max())

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b_t, amax_b = ctx.saved_tensors

        # Workaround for https://github.com/pytorch/pytorch/issues/141881.
        # The partitioner would pre-compute the transposed scaling of the weight
        # in the forward (as it's most efficient, but it actually uses too much
        # memory). We prevent that by making the scaling depend on the gradient
        # in a way that has no effect and will be optimized away later.
        b_t = b_t + grad_out[0, 0] * 0

        if ctx.a_requires_grad:
            b = b_t.t().contiguous()
            amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
            amax_b = amax_b.repeat(b.shape[0], 1)
            grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None, use_fast_accum=False)
        else:
            grad_a = None
        if ctx.b_requires_grad:
            grad_b = grad_out.t() @ a
        else:
            grad_b = None
        if ctx.bias_requires_grad:
            grad_bias = grad_out.sum(dim=0)
        else:
            grad_bias = None

        return grad_a, grad_b, grad_bias


class Fp8Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = Fp8LinearFn.apply(input.flatten(end_dim=-2), self.weight, self.bias)
        out = out.unflatten(0, input.shape[:-1])
        return out


def named_replace(fn: Callable[[torch.nn.Module, str], torch.nn.Module], module: torch.nn.Module, name="") -> torch.nn.Module:
    for child_name, child_module in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        new_child_module = named_replace(fn, child_module, full_name)
        setattr(module, child_name, new_child_module)
    module = fn(module, name)
    return module


def convert_linears_to_fp8(root_module: torch.nn.Module, recipe: str, filter: str) -> torch.nn.Module:
    if recipe not in ["rowwise"]:
        raise RuntimeError(f"Unknown float8 recipe {recipe!r}")

    if recipe == "rowwise" and torch.__version__ < "2.5":
        # We need https://github.com/pytorch/pytorch/pull/134781.
        warnings.warn("Float8 row-wise scaling is slow in PyTorch prior to v2.5.0")

    # Multi-kernel makes Inductor auto-tune between a regular "streaming"-based
    # reduction kernel and a "persistent" reduction kernel. Since fp8 has some
    # multi-pass steps (e.g., first get amax, then scale), persistent kernels
    # should perform better.
    torch._inductor.config.triton.multi_kernel = 1

    filter_re = re.compile(filter)
    def replace(module: torch.nn.Module, name: str) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Linear) or not filter_re.search(name):
            return module
        if type(module) == torch.nn.Linear:
            if recipe == "rowwise":
                new_module = Fp8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                    device=module.weight.device,
                )
                new_module.weight = module.weight
                new_module.bias = module.bias
            else:
                assert False, recipe
        else:
            assert False, str(type(module))
        return new_module
    out = named_replace(replace, root_module)

    # Force re-compile everything
    torch._dynamo.reset_code_caches()
    from torch._inductor.cudagraph_trees import reset_cudagraph_trees
    reset_cudagraph_trees()

    return out
