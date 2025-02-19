# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
import warnings
from typing import Callable

import torch
from torch.distributed.tensor import DTensor, Partial, Shard

# avoid division by zero when calculating scale
EPS = 1e-12


def get_splitk(t):
    # When tensor parallelism splits the operands along the reduction dim, it's
    # more natural (and efficient, and accurate) to do sub-row-wise scaling, so
    # that each rank can compute its own scales independently.
    if isinstance(t, DTensor) and t.placements == (Shard(dim=1),):
        return t.device_mesh.size()
    else:
        return 1


def mul_tiled(a, *bs):
    # If b is m x n, divide a into m x n chunks and multiply each by an element of b
    for b in bs:
        a = a.unflatten(0, (b.shape[0], -1)).unflatten(-1, (b.shape[-1], -1))
        a = a * b[:, None, :, None]
        a = a.flatten(end_dim=1).flatten(start_dim=-2)
    return a


def apply_to_partial(fn, t, *args, **kwargs):
    # With tensor parallelism, _scaled_mm returns a "partial" result, but we do
    # manual (post-)scaling which we want to apply to each partial term
    # separately, thus we do this hack to "unpack" the DTensors.
    if isinstance(t, DTensor) and t.placements == (Partial(),):
        return torch.distributed.tensor.experimental.local_map(fn, [*t.placements])(t, *args, **kwargs)
    else:
        return fn(t, *args, **kwargs)


def scale(t, amax_t):
    max_v = torch.finfo(torch.float8_e4m3fn).max
    scale_t = torch.clamp(amax_t.float(), min=EPS) / max_v
    t_fp8 = mul_tiled(t, scale_t.reciprocal()).to(torch.float8_e4m3fn)
    return t_fp8, scale_t


def matmul(first, amax_first, second_t, amax_second_t, bias, use_fast_accum):
    first_fp8, scale_first = scale(first, amax_first)
    second_t_fp8, scale_second_t = scale(second_t, amax_second_t)

    # PyTorch's row-wise scaled matmul kernel is based on CUTLASS and is quite
    # slow when fast_accum is disabled. Hence we fall back to an "unscaled"
    # matmul, which uses cuBLAS, and apply the scale manually afterwards.
    post_scales = []
    post_bias = None
    if not use_fast_accum:
        post_scales = [scale_first, scale_second_t.t()]
        scale_first = scale_first.new_ones((1, 1))
        scale_second_t = scale_second_t.t().new_ones((1, 1))
        post_bias, bias = bias, None

    res = torch._scaled_mm(
        first_fp8,
        second_t_fp8.t(),
        scale_a=scale_first,
        scale_b=scale_second_t.t(),
        bias=bias,
        out_dtype=torch.bfloat16,
        use_fast_accum=use_fast_accum,
    )

    res = apply_to_partial(mul_tiled, res, *post_scales).to(torch.bfloat16)
    if post_bias is not None:
        res += post_bias

    return res


@torch.compiler.allow_in_graph
class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t, bias):
        amax_a = a.abs().unflatten(-1, (get_splitk(a), -1)).amax(dim=-1)
        amax_b_t = b_t.abs().unflatten(-1, (get_splitk(b_t), -1)).amax(dim=-1)
        out = matmul(a, amax_a, b_t, amax_b_t, bias, use_fast_accum=True)

        ctx.a_requires_grad = a.requires_grad
        ctx.b_requires_grad = b_t.requires_grad
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False

        ctx.save_for_backward(a, b_t, amax_b_t)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b_t, amax_b_t = ctx.saved_tensors

        # Workaround for https://github.com/pytorch/pytorch/issues/141881.
        # The partitioner would pre-compute the transposed scaling of the weight
        # in the forward (as it's most efficient, but it actually uses too much
        # memory). We prevent that by making the scaling depend on the gradient
        # in a way that has no effect and will be optimized away later.
        # Care is needed to support tensor parallelism and circumvent bugs.
        b_t = b_t + grad_out[:1, :, None].squeeze(0) * 0

        if ctx.a_requires_grad:
            b = b_t.t().contiguous()
            amax_grad_out = (
                grad_out.abs().unflatten(-1, (get_splitk(grad_out), -1)).amax(dim=-1)
            )
            amax_b = amax_b_t.t().unflatten(-1, (get_splitk(b), -1)).amax(dim=-1)
            amax_b = amax_b.repeat_interleave(
                b.shape[0] // amax_b.shape[0], dim=0, output_size=b.shape[0]
            )
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


# We need some upstream PyTorch fixes which are only present in v2.7+ or in
# nightlies starting from January 7, 2025. For earlier versions, we copy-pasted
# the relevant pieces of code below.
if torch.__version__ < "2.7.0.dev20250107":
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import (
        OpSchema,
        OpStrategy,
        PlacementStrategy,
        RuntimeSchemaInfo,
    )
    from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
    from torch.distributed.tensor._ops._math_ops import (
        _infer_reduction_dims,
        common_reduction_strategy,
    )
    from torch.distributed.tensor._ops.utils import (
        generate_redistribute_costs,
        is_tensor_shardable,
        prod,
        register_op_strategy,
    )
    from torch.distributed.tensor.placement_types import Replicate

    # Cherry-pick of https://github.com/pytorch/pytorch/pull/143747

    LINEAR_REDUCTION_OP_MAP = {
        torch.ops.aten.amax.default: "max",
        torch.ops.aten.amin.default: "min",
    }

    @register_op_strategy(
        list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1)
    )
    def linear_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
        args_schema = op_schema.args_schema
        input_strategy = args_schema[0]
        assert isinstance(input_strategy, OpStrategy)
        dims = None
        if len(op_schema.args_schema) > 1:
            dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

        reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

        keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
        reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
        return common_reduction_strategy(
            mesh,
            input_strategy,
            reduce_dims,
            keep_dim=keep_dim,
            reduction_linear=True,
            reduction_op=reduction_op,
        )

    # Cherry-pick of https://github.com/pytorch/pytorch/pull/143760

    def _mm_like_strategy(
        mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
    ) -> OpStrategy:
        (
            self_strategy,
            mat2_strategy,
            scale_self_strategy,
            scale_mat2_strategy,
            bias_strategy,
            scale_result_strategy,
            *_,
        ) = op_schema.args_schema
        assert isinstance(self_strategy, OpStrategy)
        assert isinstance(mat2_strategy, OpStrategy)
        assert isinstance(scale_self_strategy, OpStrategy)
        assert isinstance(scale_mat2_strategy, OpStrategy)
        assert bias_strategy is None
        assert scale_result_strategy is None
        # generate all possible strategies for mm
        mm_strategy = gen_einsum_strategies(mm_equation, mesh)
        assert isinstance(mm_strategy, OpStrategy)
        # filter out invalid strategies and associate costs
        strategies = mm_strategy.strategies
        filtered_strategies = []
        for strtg in strategies:
            assert isinstance(strtg, PlacementStrategy)
            assert strtg.input_specs is not None
            self_spec = strtg.input_specs[0]
            mat2_spec = strtg.input_specs[1]
            assert isinstance(self_spec, DTensorSpec)
            assert isinstance(mat2_spec, DTensorSpec)
            scale_self_spec = (
                DTensorSpec(self_spec.mesh, (Replicate(),))
                if prod(scale_self_strategy.shape) == 1
                else self_spec
            )
            scale_mat2_spec = (
                DTensorSpec(mat2_spec.mesh, (Replicate(),))
                if prod(scale_mat2_strategy.shape) == 1
                else mat2_spec
            )
            strtg.input_specs.extend([scale_self_spec, scale_mat2_spec])
            if (
                is_tensor_shardable(self_strategy.shape, self_spec)
                and is_tensor_shardable(mat2_strategy.shape, mat2_spec)
                and is_tensor_shardable(scale_self_strategy.shape, scale_self_spec)
                and is_tensor_shardable(scale_mat2_strategy.shape, scale_mat2_spec)
            ):
                redistribute_cost = [
                    generate_redistribute_costs(self_strategy, self_spec),
                    generate_redistribute_costs(mat2_strategy, mat2_spec),
                    generate_redistribute_costs(scale_self_strategy, scale_self_spec),
                    generate_redistribute_costs(scale_mat2_strategy, scale_mat2_spec),
                ]
                strtg.redistribute_cost = redistribute_cost
                filtered_strategies.append(strtg)

        mm_strategy.strategies = filtered_strategies

        return mm_strategy

    @register_op_strategy(torch.ops.aten._scaled_mm.default)
    def mm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
        return _mm_like_strategy("mk,kn->mn", mesh, op_schema)
