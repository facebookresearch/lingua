# Copyright (c) Meta Platforms, Inc. and affiliates.

import time

import torch
import torch.nn.functional as F


####################################
# This file implement MultiLinear Upsampling described in the paper
# Custom Autograd Function using Grouped GEMMs
# This could be better, there is a sync in the forward and the backword. That's why `torch.compile` is disabled this function.
####################################
class FastIndexedMatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, indices):
        """
        Forward:
          X:       [bs, seqlen, dim]
          W:       [n, dim, dim]   (n = number of distinct weight matrices)
          indices: [seqlen] (int32, values in 0,...,n-1)
        Computes:
          Y[b, s, :] = X[b, s, :] @ W[indices[s]]
        """
        bs, seqlen, dim = X.shape
        n, out_features, in_features = W.shape

        Y = X.new_empty((bs, seqlen, out_features))
        # Save for backward.
        ctx.save_for_backward(X, W, indices)
        ctx.bs, ctx.seqlen, ctx.dim = bs, seqlen, dim
        ctx.n, ctx.in_features, ctx.out_features = n, in_features, out_features

        # Process each weight index group.
        for n_idx in range(n):
            # Get sequence positions where indices == n_idx.
            bs_idx, s_idx = torch.where(indices == n_idx)
            if s_idx.numel() == 0:
                continue
            # X_sub has shape [bs, num_selected, dim]
            X_sub = X[bs_idx, s_idx, :]
            # Compute Y_sub = X_sub @ W[n_idx]
            Y[bs_idx, s_idx, :] = F.linear(X_sub, W[n_idx], bias=None)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        """
        Backward:
          grad_Y: [bs, seqlen, dim] (gradient from above)
        Computes:
          grad_X[b, s, :] = grad_Y[b, s, :] @ W[indices[s]].T
          grad_W[n, :, :] = sum_{b,s: indices[s]==n} (X[b,s,:].T @ grad_Y[b,s,:])
        """
        X, W, indices = ctx.saved_tensors
        bs, seqlen, dim = ctx.bs, ctx.seqlen, ctx.dim
        n = ctx.n

        grad_X = grad_Y.new_zeros((bs, seqlen, dim))  # initialize to zeros
        grad_W = torch.zeros_like(W)
        for n_idx in range(n):
            bs_idx, s_idx = torch.where(indices == n_idx)
            if s_idx.numel() == 0:
                continue
            grad_X[bs_idx, s_idx, :] = torch.matmul(grad_Y[bs_idx, s_idx, :], W[n_idx])
            grad_W[n_idx] = torch.einsum('sd,se->ed', X[bs_idx, s_idx, :], grad_Y[bs_idx, s_idx, :])
        return grad_X, grad_W, None

####################################
# PyTorch Module Wrapper
####################################
class IndexedMatMul(torch.nn.Module):
    def __init__(self, num_linear, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_linear = num_linear
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((num_linear, out_features, in_features), **factory_kwargs)
        )

    @torch.compiler.disable
    def forward(self, X, indices):
        return FastIndexedMatMulFunction.apply(X, self.weight, indices)
    
    def extra_repr(self):
        return f"num_linear={self.num_linear},in_features={self.in_features}, out_features={self.out_features}"
    
    def reset_parameters(self):
        std = self.in_features ** (-0.5)
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)


####################################
# Test: Check Gradient Correctness, Measure Speed, and Benchmark GPU Memory
####################################
def test_gradient_and_speed():
    torch.manual_seed(42)
    # Typical values:
    bs, seqlen, dim, n = 4, 128, 2048, 6
    hidden_dim = dim*2
    X = torch.randn(bs, seqlen, dim, device='cuda', requires_grad=True)
    W = torch.randn(n, hidden_dim, dim, device='cuda', requires_grad=True)
    indices = torch.randint(0, n, (bs, seqlen), device='cuda', dtype=torch.int32)

    # --- Custom Module Benchmark ---
    module = IndexedMatMul(n, dim, hidden_dim, device='cuda', dtype=torch.float32)
    module.weight.data = W  # assign weights for testing

    # Warm-up and forward pass timing for custom module.
    Y = module(X, indices)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    Y = module(X, indices)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    forward_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Custom module forward pass elapsed time: {elapsed * 1000:.2f} ms")
    print(f"Custom module forward pass peak memory usage: {forward_peak:.2f} MB")

    torch.cuda.reset_peak_memory_stats()
    loss = Y.sum()
    loss.backward()
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Custom module backward pass peak memory usage: {backward_peak:.2f} MB")

    # --- Reference Implementation Benchmark ---
    # Create new copies of X and W for the reference computation.
    X_ref = X.detach().clone().requires_grad_()
    W_ref = W.detach().clone().requires_grad_()

    # Gather corresponding weights based on indices.
    # indices is [bs, seqlen]. Flatten and view as [bs, seqlen, dim, dim]
    big_W = W_ref[indices.flatten()].view(bs, seqlen, hidden_dim, dim)

    torch.cuda.reset_peak_memory_stats()
    start_ref = time.time()
    # Each X_ref[b, s, :] @ big_W[b, s, :, :] --> Result is [bs, seqlen, dim]
    Y_ref = torch.einsum('bsd,bsde->bse', X_ref, big_W.transpose(-1, -2))
    torch.cuda.synchronize()
    elapsed_ref = time.time() - start_ref
    forward_peak_ref = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Reference forward pass elapsed time: {elapsed_ref * 1000:.2f} ms")
    print(f"Reference forward pass peak memory usage: {forward_peak_ref:.2f} MB")

    torch.cuda.reset_peak_memory_stats()
    loss_ref = Y_ref.sum()
    loss_ref.backward()
    torch.cuda.synchronize()
    backward_peak_ref = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Reference backward pass peak memory usage: {backward_peak_ref:.2f} MB")

    # --- Compare Results ---
    max_diff_loss = (Y - Y_ref).abs().max().item()
    max_diff_X = (X.grad - X_ref.grad).abs().max().item()
    max_diff_W = (module.weight.grad - W_ref.grad).abs().max().item()
    print("Max diff in loss:", max_diff_loss)
    print("Max diff in X grad:", max_diff_X)
    print("Max diff in W grad:", max_diff_W)
    assert max_diff_X < 5e-4, "Gradient for X differs from reference!"
    assert max_diff_W < 5e-4, "Gradient for W differs from reference!"
    print("Gradient test passed.")
    torch.nn.Linear

if __name__ == '__main__':
    test_gradient_and_speed()