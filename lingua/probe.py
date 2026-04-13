# Copyright (c) Meta Platforms, Inc. and affiliates.

# This file from the xFormers repo is just a example of how to implement
# probing of the activations of a model, without changing anything.
# By default, the linear inputs/outputs/gradients are logged, as well as
# the attention logits+entropy. It is possible to log an additional tensor, eg:
# x = log_stats(x, "name")
#
# Known limitations:
# * Only a subset of the attention biases is supported
# * Torch-compile is disabled automatically when this is enabled
# * Only tested with bf16/f16/f32 datatypes

import contextlib
import functools
import json
import math
import os
import uuid
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.fx.operator_schemas import normalize_function
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.module_tracker import ModuleTracker
from xformers.ops import fmha


@torch.library.custom_op("torchprobe::log", mutates_args=(), device_types=None)
def _log(x: torch.Tensor, name: str, uid: str) -> None:
    pass


@_log.register_fake
def _log_fake(x: torch.Tensor, name: str, uid: str) -> None:
    pass


class _LogStats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str):
        uid = str(uuid.uuid4())
        torch.ops.torchprobe.log(x, name, uid)
        ctx.name = name
        ctx.uid = uid
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        torch.ops.torchprobe.log(grad, f"{ctx.name}.g", ctx.uid)
        return grad, None


_PROBING_ENABLED = False


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:
    if not _PROBING_ENABLED:
        return x
    return _LogStats.apply(x, name)


QUANTILES = [
    0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01,
    0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95,
    0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999,
]


@functools.cache
def _get_quantiles(device: torch.device, dtype) -> torch.Tensor:
    return torch.tensor(QUANTILES, device=device, dtype=dtype)


def _get_stats(x_: torch.Tensor, remove_inf=False) -> Dict[str, Any]:
    if x_.dtype not in [torch.float, torch.double, torch.float16, torch.bfloat16]:
        return {}
    x = x_.flatten()
    if remove_inf:
        x = x[x.abs() < float("inf")]
    if x.dtype is not torch.double:
        x = x.float()
    xabs = x.abs()
    quantiles = _get_quantiles(x.device, x.dtype)
    mean = x.mean()
    std = x.std()
    return {
        "shape": tuple(x_.shape),
        "mean": mean,
        "std": std,
        "skew": (((x - mean) / std) ** 3).double().mean(),
        "kurtosis": (((x - mean) / std) ** 4).double().mean(),
        "abs.mean": xabs.mean(),
        "max": x.max(),
        "min": x.min(),
        # Note: `quantile` takes at most 2**24 elements, see
        # https://github.com/pytorch/pytorch/issues/64947
        "quantiles": torch.quantile(x[: 2**24], quantiles),
    }


def _mask_attn_causal_inplace(logits: torch.Tensor, q_idx, q_len, kv_len) -> None:
    assert logits.ndim == 4
    logits[:, :, :, q_idx + kv_len - q_len + 1 :] = -math.inf


def _mask_attn_logits(
    logits: torch.Tensor,
    q_idx: List[int],
    *,
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert logits.dtype is torch.float32
    # Handle BlockDiagonalMask
    if cu_seqlens_q is not None:
        assert cu_seqlens_k is not None
        # Expect BHMqMkv
        assert logits.ndim == 4, logits.shape
        qs = cu_seqlens_q.tolist()
        ks = cu_seqlens_k.tolist()
        q_batchid = []
        k_batchid = [-2] * logits.shape[-1]
        q_idx_i = 0
        for bid, (q0, q1, k0, k1) in enumerate(zip(qs, qs[1:], ks, ks[1:])):
            for k in range(k0, k1):
                k_batchid[k] = bid
            while q_idx_i < len(q_idx) and q_idx[q_idx_i] < q1:
                q_batchid.append(bid)
                if causal:
                    _mask_attn_causal_inplace(
                        logits[:, :, q_idx_i : q_idx_i + 1, k0:k1],
                        q_idx[q_idx_i] - q0,
                        q1 - q0,
                        k1 - k0,
                    )
                q_idx_i += 1
        mask_out = (
            torch.tensor(q_batchid, device=logits.device)[None, None, :, None]
            != torch.tensor(k_batchid, device=logits.device)[None, None, None, :]
        )
        logits[mask_out.expand_as(logits)] = -math.inf
        assert q_idx_i == len(q_idx)
    elif causal:
        for q_idx_i in range(len(q_idx)):
            _mask_attn_causal_inplace(
                logits[:, :, q_idx_i : q_idx_i + 1, :],
                q_idx[q_idx_i],
                logits.shape[2],
                logits.shape[3],
            )
    return logits


def _attn_queries_subset(num_queries: int) -> List[int]:
    return list(range(0, num_queries, max(1, num_queries // 128)))


@torch.no_grad()
def _compute_attn_stats_sdpa(
    probe,
    path: str,
    # supports arguments both cudnn + flash backends
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    compute_log_sumexp=True,
    return_debug_mask=False,
    **kwargs,
):
    if scale is None:
        scale = 1 / (query.shape[-1] ** 0.5)
    # Filter-out not supported cases
    if attn_mask is not None or attn_bias is not None or dropout_p != 0.0 or kwargs:
        probe.store[f"{path}::attn"] = {
            "query.shape": tuple(query.shape),
            "key.shape": tuple(key.shape),
            "value.shape": tuple(value.shape),
            "attn_mask": attn_mask.shape if attn_mask is not None else None,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "unk_kwargs": list(kwargs.keys()),
        }
        return
    # Take a subset of the queries and compute the logits
    query_s = _attn_queries_subset(query.shape[-2])
    logits = query[:, :, query_s] @ key.transpose(-1, -2) * scale
    logits = _mask_attn_logits(logits.float(), query_s, causal=is_causal)
    p = logits.float().softmax(-1)
    masked_logsoft = logits.log_softmax(-1).where((logits > -math.inf), torch.zeros_like(logits))
    entropy = -(p * masked_logsoft).sum(-1)
    probe.log_tensor(f"{path}::attn_entropy", entropy)
    probe.log_tensor(f"{path}::attn_logits", logits, remove_inf=True)


@torch.no_grad()
def _compute_attn_stats_flash(
    probe,
    path: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    p: float,
    softmax_scale: float,
    is_causal: bool,
    window_left: int,
    window_right: int,
    return_softmax: bool,
    block_tables: Optional[torch.Tensor],
    unpadded_lse: bool = False,
) -> None:
    # Filter-out not supported cases
    if seqused_k is not None or p != 0.0 or window_left >= 0 or window_right >= 0 or block_tables is not None:
        probe.store[f"{path}::attn"] = {
            "query.shape": tuple(query.shape),
            "key.shape": tuple(key.shape),
            "value.shape": tuple(value.shape),
            "op": "flash",
        }
        return

    if cu_seqlens_q is not None:
        assert query.ndim == 3, query.shape
        query, key, value = query[None], key[None], value[None]
    assert query.ndim == 4, query.shape

    # Take a subset of the queries and compute the logits
    query_s = _attn_queries_subset(query.shape[1])
    logits = query[:, query_s].transpose(1, 2) @ key.transpose(1, 2).transpose(-1, -2) * softmax_scale
    logits = _mask_attn_logits(
        logits.float(),
        query_s,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=is_causal,
    )
    p = logits.float().softmax(-1)
    masked_logsoft = logits.log_softmax(-1).where((logits > -math.inf), torch.zeros_like(logits))
    entropy = -(p * masked_logsoft).sum(-1)
    probe.log_tensor(f"{path}::attn_entropy", entropy)
    probe.log_tensor(f"{path}::attn_logits", logits, remove_inf=True)


def _tensors_to_python(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.tolist()


# class syntax
class LinearBwType(Enum):
    DW = 1
    DX = 2
    UNKNOWN = 3


class AutoProbeD(TorchDispatchMode):
    def __init__(self, module: nn.Module, write_file: Optional[str] = None) -> None:
        self.write_file = Path(write_file) if write_file is not None else None
        self.write_tensors_tmpdir: Optional[Path] = None
        self.compile_disabler = TorchCompileDisabler(module)
        self.mod_tracker = ModuleTracker()
        self.count_per_path: Dict[str, int] = defaultdict(int)
        self.store: Dict[str, Dict[str, Any]] = {}
        self.linear_data: Dict[str, Tuple[Any, Any, Any, Any, Any]] = {}
        self.uid_to_path: Dict[str, str] = {}
        self.metadata: Any = None
        self.enabled = False
        self.verbose = bool(int(os.environ.get("PROBE_VERBOSE", "0")))

    def __enter__(self):
        global _PROBING_ENABLED
        assert not self.enabled, "Entered probe twice"
        self.compile_disabler.__enter__()
        self.mod_tracker.__enter__()
        super().__enter__()
        self.enabled = True
        _PROBING_ENABLED = True
        # self._setup_tensors_logging()
        return self

    def __exit__(self, *args) -> None:
        global _PROBING_ENABLED
        assert self.enabled, "Exiting probe without entering it"
        super().__exit__(*args)
        self.mod_tracker.__exit__(*args)
        self.compile_disabler.__exit__(*args)
        self._flush_and_clear()
        _PROBING_ENABLED = False
        self.enabled = False

    def _setup_tensors_logging(self):
        if self.write_file is not None:
            self.write_file.parent.mkdir(exist_ok=True)
            self.write_tensors_tmpdir = self.write_file.parent / f"{self.write_file.name}-tmp-{str(uuid.uuid4())[:8]}"
            self.write_tensors_tmpdir.mkdir(exist_ok=True)

    def _flush_and_clear(self) -> None:
        if self.write_file is not None:
            dump_data = tree_map(_tensors_to_python, self.store)
            with self.write_file.open("a") as fd:
                json.dump(
                    {
                        "data": dump_data,
                        "meta": self.metadata,
                        "version": 2,
                        "quantiles": QUANTILES,
                    },
                    fd,
                )
                fd.write("\n")
        if self.write_tensors_tmpdir is not None:
            assert self.write_file is not None
            dump_dir = self.write_tensors_tmpdir.parent / f"{self.write_file.name}-dump"
            dump_dir.mkdir(exist_ok=True)
            dir_name = ""
            if "it" in self.metadata:
                dir_name = f"it{int(self.metadata['it']):010}"
            if dir_name == "" or (dump_dir / dir_name).exists():
                num_files = len(list(dump_dir.glob(f"{dir_name}v*")))
                dir_name = f"{dir_name}v{num_files}"
            dump_dir = dump_dir / dir_name
            assert not dump_dir.exists()
            self.write_tensors_tmpdir.rename(dump_dir)
            self.write_tensors_tmpdir = None
        self.store.clear()
        self.count_per_path.clear()
        self.uid_to_path.clear()

    def _find_bw_path_and_type(self, path: str, out: torch.Tensor, args) -> Tuple[str, LinearBwType]:
        """
        We are in the BW pass, and process a GEMM.
        Let's figure out:
        (1) The path for the FW pass (might differ in case of ModuleTracker bug)
        (2) The type of BW pass (eg `dw` or `dx`)
        """

        def _is_path_correct_dw(path: str) -> bool:
            # dW.t = dY.t @ X
            in_shape, w_shape, out_shape, input_sm, weight_sm = self.linear_data[path]
            return out.shape == (w_shape[1], w_shape[0]) and torch.allclose(input_sm, args[1][:4, :4])

        def _is_path_correct_dx(path: str) -> bool:
            # dX = dY @ W.t
            in_shape, w_shape, out_shape, input_sm, weight_sm = self.linear_data[path]
            return out.shape == in_shape and torch.allclose(weight_sm, args[1][:4, :4])

        if path in self.linear_data:
            if _is_path_correct_dw(path):
                return path, LinearBwType.DW
            if _is_path_correct_dx(path):
                return path, LinearBwType.DX
        for candidate_path in self.mod_tracker.parents:
            if candidate_path not in self.linear_data:
                continue
            if _is_path_correct_dw(candidate_path):
                return candidate_path, LinearBwType.DW
            if _is_path_correct_dx(candidate_path):
                return candidate_path, LinearBwType.DX
        return path, LinearBwType.UNKNOWN

    def log_tensor(self, name: str, x: torch.Tensor, **kwargs) -> None:
        self.store[name] = _get_stats(x, **kwargs)
        if self.write_tensors_tmpdir is not None:
            name_safe = name.replace("::", "__").replace("/", "")
            torch.save(x, self.write_tensors_tmpdir / f"{name_safe}.pkl")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        path = None
        # Find longest path
        for p in self.mod_tracker.parents:
            if p == "Global":
                continue
            if path is None or len(p) > len(path):
                path = p
        if path is None:
            path = "Global"
        path = path.replace("._checkpoint_wrapped_module", "")
        out = func(*args, **kwargs)

        # Handle linear layers
        if func._overloadpacket in [torch.ops.aten.addmm, torch.ops.aten.mm]:
            weight: torch.Tensor
            input: torch.Tensor
            if not self.mod_tracker.is_bw:
                # (technically, weight is transposed)
                if func._overloadpacket == torch.ops.aten.addmm:
                    _bias, input, weight = args[:3]
                else:
                    assert func._overloadpacket == torch.ops.aten.mm
                    input, weight = args[:2]
                self.log_tensor(f"{path}::in", input)
                self.log_tensor(f"{path}::w", weight)
                self.log_tensor(f"{path}::out", out)
                self.linear_data[path] = (
                    input.shape,
                    weight.shape,
                    out.shape,
                    input[:4, :4].clone(),
                    weight[:4, :4].T.clone(),
                )
            elif func._overloadpacket == torch.ops.aten.mm:
                # XXX: Try to find the actual path for the linear layer
                # This is messed with with Francisco's FSDP sometimes
                new_path, bwtype = self._find_bw_path_and_type(path, out, args)
                if new_path != path:
                    if self.verbose:
                        print(f"E: Fixing path `{path}` -> `{new_path}")
                    path = new_path

                if bwtype == LinearBwType.DW:
                    # dW.t = dY.t @ X
                    self.log_tensor(f"{path}::w.g", out)
                elif bwtype == LinearBwType.DX:
                    # dX = dY @ W.t
                    self.log_tensor(f"{path}::in.g", out)
                    self.log_tensor(f"{path}::out.g", args[0])
        elif func._overloadpacket in [
            torch.ops.aten._scaled_dot_product_flash_attention,
            torch.ops.aten._scaled_dot_product_cudnn_attention,
        ]:
            _, kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
            _compute_attn_stats_sdpa(self, path, **kwargs)
        elif func._overloadpacket == fmha.flash.FwOp.OPERATOR:
            _, kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
            _compute_attn_stats_flash(self, path, **kwargs)
        elif func._overloadpacket == torch.ops.torchprobe.log:
            uid = args[2]
            path = self.uid_to_path.setdefault(uid, path)
            self.log_tensor(f"{path}::{args[1]}", args[0])
        if self.verbose:
            print(f"{'[BW]' if self.mod_tracker.is_bw else '[FW]'} `{path}`: {func}")
        return out


def _find_all_submodules_compiled(out: List[nn.Module], module: nn.Module) -> None:
    if module._compiled_call_impl is not None:
        out.append(module)
    for c in module.children():
        _find_all_submodules_compiled(out, module=c)


class TorchCompileDisabler:
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.submodules_compiled: List[nn.Module] = []
        self.compiled_call_impl: List[Any] = []
        self.disable_compile = torch.compiler.disable()
        torch._dynamo.config.raise_on_ctx_manager_usage = False  # type: ignore

    def __enter__(self) -> None:
        # Remove all `_compiled_call_impl` attributes to effectively
        # "undo" compilation
        self.submodules_compiled.clear()
        _find_all_submodules_compiled(self.submodules_compiled, self.module)
        self.compiled_call_impl = [m._compiled_call_impl for m in self.submodules_compiled]
        for m in self.submodules_compiled:
            m._compiled_call_impl = None
        self.disable_compile.__enter__()  # type: ignore

    def __exit__(self, *args) -> None:
        self.disable_compile.__exit__(*args)  # type: ignore
        for m, c_impl in zip(self.submodules_compiled, self.compiled_call_impl):
            m._compiled_call_impl = c_impl
        self.compiled_call_impl = []


Probe = AutoProbeD

# EXAMPLE USAGE
d = 512
seqlen = 4
bs = 2


class Attention1(nn.Module):
    def forward(self, x):
        attn_bias = fmha.attn_bias.LowerTriangularFromBottomRightMask()
        return fmha.memory_efficient_attention(x, x, x, attn_bias=attn_bias).reshape([x.shape[0], seqlen, -1])


class Attention2(nn.Module):
    def forward(self, x):
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens([seqlen] * bs).make_causal()
        xr = x.reshape([1, 2 * seqlen, x.shape[2], x.shape[3]])
        return fmha.memory_efficient_attention(xr, xr, xr, attn_bias=attn_bias).reshape([x.shape[0], seqlen, -1])


class AttentionSDPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wo = nn.Linear(d, d)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.wo(F.scaled_dot_product_attention(x, x, x).transpose(1, 2).reshape([x.shape[0], seqlen, -1]))


class AttentionSDPAFlash(AttentionSDPA):
    def forward(self, x):
        x = x.transpose(1, 2)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return self.wo(F.scaled_dot_product_attention(x, x, x).transpose(1, 2).reshape([x.shape[0], seqlen, -1]))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(d, 16)
        self.trunk = nn.Sequential(
            nn.Linear(d, d),
            nn.Linear(d, d),
        )
        self.q_proj = nn.Linear(d, d, bias=False)
        self.trunk.compile()
        self.attn1 = Attention1()
        self.attn2 = Attention2()
        self.attnSDPA = AttentionSDPA()
        self.attnSDPAflash = AttentionSDPAFlash()

    def forward(self, x):
        B, nHeads, D = x.shape[0], d // 64, 64
        x = self.q_proj(x).reshape([B, seqlen, nHeads, D])
        x = self.attn1(x) + self.attn2(x) + self.attnSDPA(x) + self.attnSDPAflash(x)
        x = log_stats(x, "attns_out")
        return self.head(self.trunk(x))


def test_masking() -> None:
    q_seqlen = [1, 1, 14, 12]
    kv_seqlen = [2, 2, 14, 18]
    attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(q_seqlen, kv_seqlen).make_causal_from_bottomright()
    logits = torch.randn([1, 1, sum(q_seqlen), sum(kv_seqlen)], dtype=torch.float32, device="cuda")
    bias = attn_bias.materialize(logits.shape, dtype=logits.dtype, device=logits.device)
    logits_masked = logits.clone()
    _mask_attn_logits(
        logits_masked,
        list(range(logits.shape[2])),
        causal=True,
        cu_seqlens_q=attn_bias.q_seqinfo.seqstart,
        cu_seqlens_k=attn_bias.k_seqinfo.seqstart,
    )
    assert (logits + bias == logits_masked).all().item()


def test_toy_model() -> None:
    # Test masking
    kw = dict(device="cuda", dtype=torch.float16)
    x = torch.randn([bs, seqlen, d], **kw)
    m = Model()
    m.head = checkpoint_wrapper(m.head, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)
    m.to(**kw)
    m.compile()
    optim = torch.optim.SGD(m.parameters(), lr=0.0)
    probe = AutoProbeD(m, "./probe.json")

    for i in range(4):
        with contextlib.ExitStack() as stack:
            print(f"########### STEP {i}")
            if i % 4 == 1:
                stack.enter_context(probe)
                probe.metadata = {"it": i}
            y = m(x)
            g = torch.randn_like(y)
            y.backward(g)
            if i % 4 == 1:
                assert probe.enabled
                # Make sure we registered all linears
                print(list(probe.store.keys()))
                for key in [
                    "Model::attns_out",
                    "Model::attns_out.g",
                    "Model.attn1::attn_logits",
                    "Model.attn2::attn_logits",
                    "Model.attnSDPA::attn_logits",
                    "Model.attnSDPAflash::attn_logits",
                    "Model.head::w",
                    "Model.head::w.g",
                    "Model.head::in",
                    "Model.head::in.g",
                    "Model.head::out",
                    "Model.head::out.g",
                    "Model.trunk.0::in",
                    "Model.trunk.1::in",
                ]:
                    assert key in probe.store, f"Missing key: '{key}'"
                # .. and that the values are correct
                for key, tensor in [
                    ("Model.head::w", m.head.weight),
                    ("Model.head::w.g", m.head.weight.grad),
                    ("Model.q_proj::in", x),
                    ("Model.q_proj::w.g", m.q_proj.weight.grad),
                    ("Model.head::out", y),
                    ("Model.head::out.g", g),
                ]:
                    assert key in probe.store, f"Missing key: '{key}'"
                    assert torch.allclose(
                        probe.store[key]["abs.mean"], tensor.float().abs().mean()
                    ), f"'{key}' mismatches"
                # Check we don't have `nans`
                for key, value in probe.store.items():
                    if "abs.mean" in value:
                        assert math.isfinite(value["abs.mean"].item()), f"Inf/Nan for {key}"
            optim.step()
            optim.zero_grad()
