# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops.fmha import AttentionBias

from apps.aunet.index_matmul import IndexedMatMul
from apps.main.transformer import (
    create_causal_mask,
    cross_entropy,
)
from lingua.metrics import get_num_params
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
)


@dataclass
class CausalTransformerArgs(BaseTransformerArgs):
    sliding_window: Optional[int] = None

@dataclass
class HierarchicalArgs:
    dimensions: Optional[List[int]] = None
    head_dims: Optional[List[int]] = None
    layers: Optional[List[int]] = None
    sliding_windows: Optional[List[int]] = None
    max_seqlens: Optional[List[int]] = None
    residuals: Optional[List[bool]] = None

    block: BaseTransformerArgs = field(default_factory=BaseTransformerArgs)

    pooling_type: str = "simple_indexed_matmul"

    seed: int = 42
    vocab_size: int = -1
    lambda_level: float = 0.0

    norm_eps: float = 1e-5

    # Estimates used to avoid having to instantiate an actual model
    @property
    def estimated_non_embed_param_count(self) -> int:
        n_heads = [dim/head_dim for dim, head_dim in zip(self.dimensions, self.head_dims)]
        return estimate_non_embed_param_count(self.dimensions, self.layers, n_heads, self.block.n_kv_heads or n_heads, 4, self.block.ffn_dim_multiplier, self.block.multiple_of)

    @property
    def estimated_param_count(self) -> int:
        vocab_factor = 2
        return self.estimated_non_embed_param_count + vocab_factor*self.dimensions[0]*self.vocab_size

    def non_embed_flops_per_token(self, seq_len: int) -> int:
        max_seqlens = [seq_len] + self.max_seqlens[1:]
        seq_len_ratios = [max_seq_len / seq_len for max_seq_len in max_seqlens]
        n_heads = [dim/head_dim for dim, head_dim in zip(self.dimensions, self.head_dims)]
        
        linear_w = estimate_effective_param_count(self.dimensions, self.layers, seq_len_ratios, n_heads, self.block.n_kv_heads or n_heads, 4, self.block.ffn_dim_multiplier, self.block.multiple_of)

        return estimate_flops_per_token(self.dimensions, self.layers, max_seqlens, seq_len_ratios, self.sliding_windows, linear_w)

    def flops_per_token(self, seq_len: int) -> int:
        seqlens = [seq_len] + self.max_seqlens[1:]
        seq_len_ratios = [max_seq_len / seq_len for max_seq_len in seqlens]

        n_heads = [dim/head_dim for dim, head_dim in zip(self.dimensions, self.head_dims)]

        linear_w = estimate_effective_param_count(self.dimensions, self.layers, seq_len_ratios, n_heads, self.block.n_kv_heads or n_heads, 4, self.block.ffn_dim_multiplier, self.block.multiple_of)
        linear_w += self.dimensions[0]*self.vocab_size

        return estimate_flops_per_token(self.dimensions, self.layers, seqlens, seq_len_ratios, self.sliding_windows, linear_w)

    def tokens_per_second(self, seq_len: int, flops_per_second: int) -> int:
        return flops_per_second / self.flops_per_token(seq_len)

    def mem_usage(self, batch_size: int, seq_len: int) -> tuple[int, int]:
        return estimate_mem_usage(self.dimensions, self.layers, batch_size, seq_len, self.max_seqlens, 4, self.block.ffn_dim_multiplier, self.block.multiple_of, self.vocab_size)

    def max_batch_size(self, max_mem_budget: float = 80e9, seq_len: int = 1024) -> int:
        activation_mem, vocab_mem = self.mem_usage(1, seq_len)
        mem_per_batch = activation_mem + vocab_mem
        return math.floor(max_mem_budget / mem_per_batch)


def estimate_flops_per_token(dimensions: List[int], layers: List[int], seqlens: List[int], seqlens_ratio: List[int], sliding_windows: List[int], num_linear_params: int) -> int:
    """Calculate FLOPs per token for a given model configuration"""
    dimensions = dimensions.copy()
    layers = layers.copy()
    if isinstance(layers, list):
        layers = [l*2 if k < len(layers) - 1 else l for k, l in enumerate(layers)]
    else:
        layers[:, :-1] = layers[:, :-1]*2
    sliding_windows = sliding_windows.copy()
    
    attn_flops = lambda dim, seqlen, n_layers, sliding_window:  6 * dim * seqlen * n_layers * (sliding_window * (2 * seqlen - sliding_window))/(seqlen**2)
    # FLOPs per token is independent of batch size and grad accumulation
    if isinstance(dimensions, list):
        attn_flops = sum(attn_flops(dim, seqlen, n_layers, min(seqlen, sliding_window))*ratio for dim, seqlen, sliding_window, n_layers, ratio in zip(dimensions, seqlens, sliding_windows, layers, seqlens_ratio))
    else:
        sliding_windows = np.minimum(seqlens, sliding_windows)
        attn_flops = ((6 * dimensions * layers * (seqlens * (sliding_windows * (2 * seqlens - sliding_windows))/(seqlens**2)))*seqlens_ratio).sum(axis=1)
    total_flops = attn_flops + 6 * num_linear_params

    return total_flops

def estimate_non_embed_param_count(dimensions: List[int], layers: List[int], n_heads: List[int], n_kv_heads: int, ffn_exp: float, ffn_dim_multiplier: float, multiple_of: int) -> int:
    """Estimate number of parameters in the model (excluding embeddings)"""
    dimensions = dimensions.copy()
    layers = layers.copy()
    if isinstance(layers, list):
        layers = [l*2 if k < len(layers) - 1 else l for k, l in enumerate(layers)]
    else:
        layers[:, :-1] = layers[:, :-1]*2
    n_heads = n_heads.copy()
    n_kv_heads = n_heads.copy()
    ffn_dim_multiplier = ffn_dim_multiplier if ffn_dim_multiplier is not None else 1.0
    if isinstance(ffn_dim_multiplier, (float, int)):
        lffn_dim = [math.ceil(int(int(2*(dim * ffn_exp)/3) * ffn_dim_multiplier) / multiple_of) * multiple_of for dim in dimensions]
    else:
        lffn_dim = np.ceil(np.floor((np.floor(2*(dimensions * ffn_exp)/3)) * ffn_dim_multiplier[..., None]) / multiple_of[..., None]) * multiple_of[..., None]

    params_per_layer = lambda dim, n_heads, n_kv_heads, ffn_dim: (
        # QKV projections
        dim * dim +  # Q
        2 * dim * dim * (n_kv_heads / n_heads) +  # K,V with kv_heads
        # Output projection
        dim * dim +
        # FFN
        3 * dim * ffn_dim
    )

    if isinstance(dimensions, list):
        return sum(nlayers*params_per_layer(dim, n_heads, n_kv_heads, ffn_dim) for dim, n_heads, n_kv_heads, ffn_dim, nlayers in zip(dimensions, n_heads, n_kv_heads, lffn_dim, layers))
    else:
        return ((dimensions * dimensions + 2 * dimensions * dimensions * (n_kv_heads / n_heads)[..., None] + dimensions * dimensions + 3 * dimensions * lffn_dim) * layers).sum(axis=1)

def estimate_mem_usage(dimensions: List[int], layers: List[int], batch_size: int, seq_len: int, max_seqlens: List[int], ffn_exp: float, ffn_dim_multiplier: float, multiple_of: int, vocab_size: int) -> tuple[int, int]:
    """Estimate memory usage for model with given parameters (without requiring a TrainArgs object)"""
    nb_dim_tensors = 8
    nb_ffn_tensors = 3
    nb_bytes_per_activation = 2

    ffn_dim_multiplier = ffn_dim_multiplier if ffn_dim_multiplier is not None else 1.0
    
    n_layers = [l*2 if k < len(layers) - 1 else l for k, l in enumerate(layers)] 
    ffn_dims = [math.ceil(int(int(2*(d * ffn_exp)/3) * ffn_dim_multiplier) / multiple_of) * multiple_of for d in dimensions]
    seq_lens = [seq_len] + max_seqlens[1:]

    activation_mem = lambda dim, ffn_dim, n_layers, batch_size, seq_len: nb_bytes_per_activation*(nb_dim_tensors*dim+nb_ffn_tensors*ffn_dim)*batch_size*seq_len*n_layers

    # Model activations memory
    activation_mem = sum(activation_mem(dim, ffn_dim, n_layers, batch_size, seq_len) for dim, ffn_dim, n_layers, seq_len in zip(dimensions, ffn_dims, n_layers, seq_lens))*1.2 # 20% margin... better bound for later
    # Memory for handling vocab operations (embeddings, output layer)
    vocab_mem = sum([nb_bytes_per_activation*2*vocab_size*batch_size*seq_len for seq_len in seq_lens])
    
    return activation_mem, vocab_mem

def estimate_effective_param_count(dimensions: List[int], layers: List[int], seq_len_ratios: List[int], n_heads: List[int], n_kv_heads: int, ffn_exp: float, ffn_dim_multiplier: float, multiple_of: int) -> int:
    """Calculate number of parameters in the model (excluding embeddings)"""
    dimensions = dimensions.copy()
    layers = layers.copy()
    if isinstance(layers, list):
        layers = [l*2 if k < len(layers) - 1 else l for k, l in enumerate(layers)]
    else:
        layers[:, :-1] = layers[:, :-1]*2
    n_heads = n_heads.copy()
    n_kv_heads = n_heads.copy()
    ffn_dim_multiplier = ffn_dim_multiplier if ffn_dim_multiplier is not None else 1.0
    if isinstance(ffn_dim_multiplier, (float, int)):
        lffn_dim = [math.ceil(int(int(2*(dim * ffn_exp)/3) * ffn_dim_multiplier) / multiple_of) * multiple_of for dim in dimensions]
    else:
        lffn_dim = np.ceil(np.floor((np.floor(2*(dimensions * ffn_exp)/3)) * ffn_dim_multiplier[..., None]) / multiple_of[..., None]) * multiple_of[..., None]
    
    def params_per_layer(dim, n_heads, n_kv_heads, ffn_dim):
        return dim * dim +  2 * dim * dim * (n_kv_heads / n_heads) +  dim * dim + 3 * dim * ffn_dim

    if isinstance(dimensions, list):
        return sum(nlayers*seq_len_ratio*params_per_layer(dim, n_heads, n_kv_heads, ffn_dim) for dim, n_heads, n_kv_heads, ffn_dim, nlayers, seq_len_ratio in zip(dimensions, n_heads, n_kv_heads, lffn_dim, layers, seq_len_ratios))
    else:
        return ((dimensions * dimensions + 2 * dimensions * dimensions * (n_kv_heads / n_heads)[..., None] + dimensions * dimensions + 3 * dimensions * lffn_dim) * layers * seq_len_ratios).sum(axis=1)
    
def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )

class CausalTransformer(BaseTransformer):
    def __init__(self, args: CausalTransformerArgs):
        super().__init__(args)

        self.n_layers = args.n_layers
        self.sliding_window = args.sliding_window
        self.dim = args.dim
        self.max_seqlen = args.max_seqlen
        self.num_params = get_num_params(self)

    def forward(
        self,
        h: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "fmha",
    ):
        bsz, seqlen, dim = h.shape

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, "fmha", self.sliding_window)
        )

        return super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

    def flops_per_token(self):
        atten_s = min(self.max_seqlen, self.sliding_window or float("inf"))
        lin_flops = get_num_flop_per_token(
            self.num_params, self.n_layers, self.dim, atten_s
        )
        return lin_flops

def non_parametric_trans_down(x:  torch.Tensor, expand_idx: torch.Tensor, features_weight: torch.Tensor, features_ratio: int):
    features_weight = features_weight.to(dtype=x.dtype)
    return x[:, :, expand_idx] * features_weight[None, None, :] + x[:, :, expand_idx].roll(features_ratio, -1) * (1 - features_weight[None, None, :])

def non_parametic_trans_up(x: torch.Tensor, features_ratio: int):
    bsz, seqlen, dim = x.shape
    return x.view(bsz, seqlen, dim//features_ratio, features_ratio).mean(-1)

class MaxSumMask(nn.Module):
    def __init__(self, seq_len: int, numel: int):
        """
        Initialize the ConstantSumMask class with numel and arange as attributes.

        Args:
            numel (int): The maximum number of True values that should remain in each row.
            seq_len (int): The length of the sequence (second dimension of the mask tensor).
        """
        super().__init__()
        self.numel = numel
        self.seq_len = seq_len

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Truncate the number of True values in the mask to numel from left to right,
        then add remaining True values at the end to ensure exactly numel are True.

        Args:
            mask (torch.Tensor): A boolean mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: A boolean mask tensor with exactly numel True values per row.
        """
        idxs = (~mask).squeeze().int().argsort(stable=True)
        idxs = idxs[:, : self.numel]

        return idxs

class SimpleTransition(nn.Module):
    def __init__(
        self, seqlen_in, seqlen_out, dim_in, dim_out, head_dim, rope_theta, eps_norm, non_parametric: bool = False, indexed_matmul: bool = False, repeat: bool = False,
    ):
        super().__init__()
        self.seqlen_in = seqlen_in
        self.seqlen_out = seqlen_out
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.non_parametric = non_parametric
        self.repeat = repeat
        self.indexed_matmul = indexed_matmul
        self.max_sum_mask = MaxSumMask(seqlen_in, seqlen_out)

        if not self.non_parametric:
            self.down_norm = RMSNorm(dim_in, eps=eps_norm)
            self.trans_down = nn.Linear(dim_in, dim_out, bias=False)
            self.up_norm = RMSNorm(dim_out, eps=eps_norm)
            self.trans_up = nn.Linear(dim_out, dim_in, bias=False)
        else:
            assert dim_out % dim_in == 0, f"dim_out {dim_out} must be a multiple of dim_in {dim_in}"
            features_ratio = dim_out // dim_in
            self.features_ratio = features_ratio
            self.register_buffer("features_weight", (torch.arange(1, features_ratio+1)/features_ratio).repeat(dim_in), persistent=True)
            self.register_buffer("expand_idx", torch.arange(dim_in).repeat_interleave(features_ratio), persistent=True)
            if dim_out == dim_in:
                self.trans_down = lambda x, **kwargs: x
                self.trans_up = lambda x: x
            else:
                self.trans_down = non_parametric_trans_down
                self.trans_up = partial(non_parametic_trans_up, features_ratio=features_ratio)

        if self.indexed_matmul:
            self.max_pos = 16
            self.repeat = True
            self.indexed_linear = IndexedMatMul(self.max_pos, dim_out, dim_in)
            if self.non_parametric:
                self.indexed_norm = RMSNorm(dim_out, eps=eps_norm)

        if self.repeat:
            self.max_pos = 16
            self.register_buffer("position", torch.arange(seqlen_in), persistent=False)
            if not self.indexed_matmul:
                self.pos_encoding = torch.nn.Embedding(self.max_pos, dim_out)

    def down(self, x, mask, freq_cis, mask_idx=None, position=None):
        # For generation
        if mask_idx is not None:
            assert not self.training
            if mask_idx.numel() == 0:
                return x
            idx = mask_idx
        else:
            idx = self.idx = self.max_sum_mask(mask)

        x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.dim_in))

        if hasattr(self, "down_norm"):
            x = self.down_norm(x)

        self.trans_down_kwargs = {}
        if self.non_parametric:
            self.trans_down_kwargs = {'expand_idx':self.expand_idx, 'features_weight':self.features_weight, 'features_ratio':self.features_ratio}
        
        out = self.trans_down(x, **self.trans_down_kwargs)
        
        return out

    def up(self, x, res, mask, freq_cis, mask_idx=None, position=None, repeat_idx=None):
        if mask_idx is not None:
            assert not self.training
            if not self.repeat and mask_idx.numel() == 0:
                return torch.zeros_like(res)
            idx = mask_idx
        else:
            idx = self.idx

        if self.repeat:
            if position is not None:
                assert not self.training

                if hasattr(self, "up_norm"):
                    x = self.up_norm(x)

                if repeat_idx is not None:
                    x = x.gather(1, repeat_idx.unsqueeze(-1).expand(-1, -1, self.dim_out))

                if self.indexed_matmul:
                    return self.trans_up(x) + self.indexed_linear(self.indexed_norm(x) if hasattr(self, "indexed_norm") else x, position)
                
                return self.trans_up(x + self.pos_encoding(position))
            
            with torch.no_grad():
                repeat_idx = mask[:, :self.seqlen_in].cumsum(1)
                repeat_idx = repeat_idx - repeat_idx[:, 0].clone().unsqueeze(-1)
                position = (self.position - idx.gather(1, repeat_idx)) % self.max_pos
            
            if hasattr(self, "up_attn_norm"):
                n = self.up_attn_norm(x).gather(1, repeat_idx.unsqueeze(-1).expand(-1, -1, self.dim_out))

            if hasattr(self, "up_norm"):
                x = self.up_norm(x)
            
            x = x.gather(1, repeat_idx.unsqueeze(-1).expand(-1, -1, self.dim_out))

            if self.indexed_matmul:
                return self.trans_up(x) + self.indexed_linear(self.indexed_norm(x) if hasattr(self, "indexed_norm") else x, position)

            return self.trans_up(x + self.pos_encoding(position))
        
        if hasattr(self, "up_norm"):
            x = self.up_norm(x)
        
        if position is None:
            m = mask.gather(1, idx)
            x = m.to(dtype=x.dtype).unsqueeze(-1) * (self.trans_up(x))  # Removing values
        else:
            assert not self.training
            x = self.trans_up(x)
            if repeat_idx is None:
                x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.dim_in))

        h = torch.zeros_like(res)
        h.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, self.dim_in), x)
        
        return h

    def reset_parameters(self):
        in_init_std = (self.dim_in ** (-0.5))
        if hasattr(self, "down_norm"):
            self.down_norm.reset_parameters()
        if hasattr(self.trans_down, "weight"):
            nn.init.trunc_normal_(
                self.trans_down.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        out_init_std = (self.dim_out ** (-0.5))
        if hasattr(self, "up_norm"):
            self.up_norm.reset_parameters()
        if hasattr(self.trans_up, "weight"):
            nn.init.trunc_normal_(
                self.trans_up.weight,
                mean=0.0,
                std=out_init_std,
                a=-3 * out_init_std,
                b=3 * out_init_std,
            )

        if hasattr(self, "indexed_linear"):
            self.indexed_linear.reset_parameters()
        if hasattr(self, "indexed_norm"):
            self.indexed_norm.reset_parameters()

        if self.repeat:
            self.position[...] = torch.arange(self.seqlen_in)
            if hasattr(self, "pos_encoding"):
                self.pos_encoding.reset_parameters()

class HierarchicalTransformer(nn.Module):
    def __init__(self, args: HierarchicalArgs):
        super().__init__()
        self.seed = args.seed

        self.dim = args.dimensions[0]
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, self.dim)

        self.vocab_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.vocab = nn.Linear(self.dim, args.vocab_size, bias=False)

        self.lambda_level = args.lambda_level
        if self.lambda_level > 0:
            self.level_mask_norm = RMSNorm(self.dim * 2, eps=args.norm_eps)
            self.level_mask = nn.Linear(self.dim * 2, len(args.dimensions), bias=False)
            self.level_mask.forward = torch.compiler.disable(self.level_mask.forward)

        self.adding_residuals = args.residuals or [True] * (len(args.dimensions)-1)

        input_dims = args.dimensions[:-1]
        head_dims = args.head_dims[:-1]
        output_dims = args.dimensions[1:]
        input_seqlens = args.max_seqlens[:-1]
        output_seqlens = args.max_seqlens[1:]
        sliding_windows = args.sliding_windows or [None] * len(args.dimensions)
        self.enc_args = list(
            zip(
                input_dims,
                head_dims,
                output_dims,
                input_seqlens,
                output_seqlens,
                args.layers,
                sliding_windows,
            )
        )

        base_block = asdict(deepcopy(args.block))

        self.pooling_type = args.pooling_type

        transition_cls = None
        transition_kwargs = {}
        if args.pooling_type.startswith("simple"):
            transition_cls = SimpleTransition
            transition_kwargs["non_parametric"] = "non_param" in args.pooling_type.lower()
            transition_kwargs["repeat"] = "repeat" in args.pooling_type.lower()
            transition_kwargs["indexed_matmul"] = "indexed_matmul" in args.pooling_type.lower()
        else:
            raise ValueError(f"Pooling type {args.pooling_type} not supported")

        self.encoders: List[CausalTransformer] = nn.ModuleList()
        self.decoders: List[CausalTransformer] = nn.ModuleList()
        self.transitions: List[SimpleTransition] = nn.ModuleList()



        for d_i, h_d, d_o, s_i, s_o, l, w in self.enc_args:
            block = CausalTransformerArgs(**base_block)
            block.dim = d_i
            block.head_dim = h_d
            block.n_layers = l
            block.sliding_window = w
            block.max_seqlen = s_i
            self.encoders.append(CausalTransformer(block))
            self.transitions.append(
                transition_cls(
                    s_i,
                    s_o,
                    d_i,
                    d_o,
                    h_d,
                    block.rope_theta,
                    block.norm_eps,
                    **transition_kwargs,
                )
            )
            self.decoders.append(CausalTransformer(block))

        tru = CausalTransformerArgs(**base_block)
        tru.dim = args.dimensions[-1]
        tru.head_dim = args.head_dims[-1]
        tru.n_layers = args.layers[-1]
        tru.sliding_window = sliding_windows[-1]
        tru.max_seqlen = args.max_seqlens[-1]
        self.trunk = CausalTransformer(tru)
        # self.trunk.rope_embeddings = torch.compiler.disable(self.trunk.rope_embeddings) # When using multiple levels compile bug here

        self.num_params = [get_num_params(e) for e in self.encoders]
        self.num_params += [get_num_params(self.trunk)]

    def forward(
        self,
        token_values: torch.Tensor,  # bsz, seqlen
        level_mask: torch.Tensor,  # bsz, seqlen
        target: torch.Tensor,  # bsz, seqlen, n_future
        target_level_mask: torch.Tensor,  # bsz, seqlen
    ):
        masks, _, nb_toks = self.get_pool_mask(level_mask, [e.max_seqlen for e in self.encoders] + [self.trunk.max_seqlen], return_idcs=False, force_first=True)

        residuals = []
        x = self.tok_embeddings(token_values)

        it = zip(self.encoders, self.transitions, masks)
        for i, (encoder, trans, mask) in enumerate(it):
            x = encoder(x, attn_impl="fmha")
            residuals.append(x)
            x = trans.down(x, mask, encoder.rope_embeddings.freqs_cis)

        x = self.trunk(x, attn_impl="fmha")

        it = list(
            zip(
                self.decoders,
                self.transitions,
                residuals,
                self.adding_residuals,
                masks
            )
        )
        it = it[::-1]
        for decoder, trans, res, add_res, mask in it:
            x = trans.up(x, res, mask, decoder.rope_embeddings.freqs_cis)
            if add_res:
                x = res + x
            x = decoder(x, attn_impl="fmha")

        logits = self.vocab(self.vocab_norm(x))
        loss = cross_entropy(logits, target)
        
        mask_loss = None
        mask_logits = self._level_mask_logits(x, target)
        if mask_logits is not None:
            mask_loss = cross_entropy(mask_logits, target_level_mask)

        return loss, mask_loss, nb_toks

    def _level_mask_logits(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = None
        if self.lambda_level > 0:
            target_emb = torch.cat([features, self.tok_embeddings(target)], dim=-1)
            logits = self.level_mask(self.level_mask_norm(target_emb))

        return logits
    
    @torch.no_grad()
    def get_pool_mask(self, level_mask: torch.Tensor, max_seqlen: Optional[List[int]] = None, return_idcs: bool = True, force_first: bool = False) -> List[torch.Tensor]:
        nb_levels = len(self.encoders)
        pool_mask = []
        pool_mask_idcs = []
        nb_toks = []
        remaining = torch.ones_like(level_mask, dtype=torch.bool)
        if force_first:
            level_mask[:, 0] = nb_levels
        for i in range(nb_levels):
            mask = level_mask > i

            if max_seqlen is not None:
                mask.masked_fill_((mask.cumsum(1) > max_seqlen[i+1]), False)
            next_remaining = mask & remaining
            
            _fix_allocation_mask = torch.zeros_like(mask)
            beg_mask = (torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand(mask.size(0), -1) < remaining.sum(1).unsqueeze(1))
            _fix_allocation_mask[beg_mask] = mask[remaining]
            mask = _fix_allocation_mask

            nb_toks.append(mask.sum())
            pool_mask.append(mask)
            if return_idcs:
                pool_mask_idcs.append(mask.nonzero(as_tuple=True)[1])

            remaining = next_remaining
        return pool_mask, pool_mask_idcs, nb_toks

    def flops_per_token(self):
        flops = 0
        init_seqlen = self.encoders[0].max_seqlen
        for enc, dec in zip(self.encoders, self.decoders):
            flops += enc.max_seqlen / init_seqlen * enc.flops_per_token()
            flops += enc.max_seqlen / init_seqlen * dec.flops_per_token()
        flops += self.trunk.max_seqlen / init_seqlen * self.trunk.flops_per_token()
        return flops

    def reset_parameters(self, init_std=None):
        init_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        self.vocab_norm.reset_parameters()
        nn.init.trunc_normal_(
            self.vocab.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        for lin in self.transitions:
            lin.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for encoder in self.encoders:
            encoder.init_weights()
        for decoder in self.decoders:
            decoder.init_weights()
        self.trunk.init_weights()

def get_no_recompute_ops():
    return None

def build_fsdp_grouping_plan(model_args: HierarchicalArgs):
    group_plan: List[Tuple[int, bool]] = []

    for k, l in enumerate(model_args.layers[:-1]):
        for i in range(l):
            group_plan.append((f"encoders.{k}.layers.{i}", False))
            group_plan.append((f"decoders.{k}.layers.{i}", False))

    # Grouping by layers
    for i in range(model_args.layers[-1]):
        group_plan.append((f"trunk.layers.{i}", False))

    return group_plan