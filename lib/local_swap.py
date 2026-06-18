"""
Residual-Guided Local Swap (RGLS) for RIA Pruning
==================================================

This module implements the RGLS algorithm which performs post-pruning optimization
by swapping weights near the threshold boundary, guided by residual signals.

Key concepts:
- keep_mask: True = keep weight, False = prune weight
- Maintains exact sparsity ratio after swap
- Only supports unstructured sparsity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional


def zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute z-score normalization."""
    mean = x.mean()
    std = x.std()
    if std < eps:
        std = eps
    return (x - mean) / std


def build_keep_mask_from_scores(
    scores: torch.Tensor,
    sparsity_ratio: float,
    per_outneuron: bool = False
) -> torch.Tensor:
    """
    Build keep_mask from importance scores.
    
    Args:
        scores: Importance scores, shape [out_features, in_features]
        sparsity_ratio: Fraction of weights to prune (0.5 = prune 50%)
        per_outneuron: If True, prune per output neuron
    
    Returns:
        keep_mask: Boolean tensor, True = keep, False = prune
    """
    if per_outneuron:
        # Per output neuron pruning
        sort_res = torch.sort(scores, dim=-1, stable=True)
        prune_count = int(scores.shape[1] * sparsity_ratio)
        indices = sort_res[1][:, :prune_count]
        prune_mask = torch.zeros_like(scores, dtype=torch.bool)
        prune_mask.scatter_(1, indices, True)
        keep_mask = ~prune_mask
    else:
        # Global threshold pruning
        total = scores.numel()
        prune_count = int(total * sparsity_ratio)
        thresh = torch.sort(scores.flatten())[0][prune_count]
        keep_mask = scores > thresh
    
    return keep_mask


def get_candidates_topk(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    largest: bool
) -> torch.Tensor:
    """
    Get top-k candidates from positions where mask is True.
    
    Args:
        scores: Score tensor (flattened)
        mask: Boolean mask indicating valid positions
        k: Number of candidates to select
        largest: If True, select largest scores; if False, select smallest
    
    Returns:
        flat_indices: Indices of selected candidates in flattened tensor
    """
    valid_indices = torch.where(mask.flatten())[0]
    valid_scores = scores.flatten()[valid_indices]
    
    k = min(k, len(valid_indices))
    if k == 0:
        return torch.tensor([], dtype=torch.long, device=scores.device)
    
    _, topk_local = torch.topk(valid_scores, k, largest=largest)
    flat_indices = valid_indices[topk_local]
    
    return flat_indices


def compute_benefit_chunked(
    W: torch.Tensor,
    keep_target: torch.Tensor,
    add_candidates_flat: torch.Tensor,
    X_batches: List[torch.Tensor],
    out_chunk: int = 512,
    clip_residual: float = 0.0,
    accum_dtype: torch.dtype = torch.float32,
    debug: bool = False
) -> torch.Tensor:
    """
    Compute residual-guided benefit scores for add_candidates only.
    
    Math:
        ΔW = W ⊙ (~keep_target)  # pruned weights
        R = X @ ΔW^T             # residual [tokens, out]
        C = X^T @ R              # [in, out]
        benefit[j,i] = |W[j,i] * C[i,j]|
    
    Args:
        W: Weight tensor [out_features, in_features]
        keep_target: Initial keep mask (bool), True = keep
        add_candidates_flat: Flat indices of add candidates
        X_batches: List of input activation tensors [tokens, in_features]
        out_chunk: Chunk size for out_features dimension
        clip_residual: If >0, clip residuals to [-v, v]
        accum_dtype: Dtype for accumulation
        debug: Print debug info
    
    Returns:
        benefit_add: Benefit scores for add_candidates [len(add_candidates)]
    """
    out_features, in_features = W.shape
    device = W.device
    
    # Pruned weights (will contribute to residual)
    prune_mask = ~keep_target
    delta_W = W * prune_mask.float()  # [out, in]
    
    # Prepare output tensor
    num_candidates = len(add_candidates_flat)
    benefit_add = torch.zeros(num_candidates, dtype=accum_dtype, device=device)
    
    if num_candidates == 0:
        return benefit_add
    
    # Precompute candidate positions
    out_idx_all = add_candidates_flat // in_features
    in_idx_all = add_candidates_flat % in_features
    
    # Process in chunks over out_features
    for o0 in range(0, out_features, out_chunk):
        o1 = min(o0 + out_chunk, out_features)
        
        delta_W_chunk = delta_W[o0:o1, :].to(accum_dtype)  # [chunk, in]
        W_chunk = W[o0:o1, :].to(accum_dtype)
        
        # Accumulate C_chunk over batches
        C_chunk = torch.zeros((in_features, o1 - o0), dtype=accum_dtype, device=device)
        
        for X_b in X_batches:
            X_b = X_b.to(accum_dtype)  # [tokens, in]
            
            # R_chunk = X_b @ delta_W_chunk^T  -> [tokens, chunk]
            R_chunk = X_b @ delta_W_chunk.t()
            
            # Clip residual if requested
            if clip_residual > 0:
                R_chunk = torch.clamp(R_chunk, -clip_residual, clip_residual)
            
            # C_chunk += X_b^T @ R_chunk  -> [in, chunk]
            C_chunk += X_b.t() @ R_chunk
        
        # Find candidates in this chunk
        chunk_mask = (out_idx_all >= o0) & (out_idx_all < o1)
        chunk_indices = torch.where(chunk_mask)[0]
        
        if len(chunk_indices) > 0:
            local_out = out_idx_all[chunk_indices] - o0
            local_in = in_idx_all[chunk_indices]
            
            # benefit = |W[j,i] * C[i,j]|
            w_vals = W_chunk[local_out, local_in]
            c_vals = C_chunk[local_in, local_out]
            benefit_vals = torch.abs(w_vals * c_vals)
            
            benefit_add[chunk_indices] = benefit_vals
        
        # Free memory
        del C_chunk, delta_W_chunk, W_chunk
    
    # Check for numerical issues
    if debug:
        nan_count = torch.isnan(benefit_add).sum().item()
        inf_count = torch.isinf(benefit_add).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"[RGLS Warning] benefit has {nan_count} NaN, {inf_count} Inf values")
        benefit_add = torch.nan_to_num(benefit_add, nan=0.0, posinf=0.0, neginf=0.0)
    
    return benefit_add


def local_swap_one_linear(
    W: torch.Tensor,
    keep_target: torch.Tensor,
    ria_scores: torch.Tensor,
    X_batches: List[torch.Tensor],
    swap_ratio: float,
    swap_candidate_bandwidth: float,
    swap_alpha: float,
    refill_out_chunk: int,
    refill_clip_residual: float,
    refill_dtype: str,
    swap_debug: bool,
    layer_name: str = ""
) -> Tuple[torch.Tensor, Dict]:
    """
    Perform RGLS local swap on a single linear layer.
    
    Args:
        W: Weight tensor [out_features, in_features]
        keep_target: Initial keep mask from RIA pruning (bool, True=keep)
        ria_scores: RIA importance scores [out_features, in_features]
        X_batches: List of input activation tensors
        swap_ratio: Fraction of total params to swap
        swap_candidate_bandwidth: Candidate pool size as fraction of total
        swap_alpha: Weight for benefit vs RIA score
        refill_out_chunk: Chunk size for out_features
        refill_clip_residual: Residual clipping value
        refill_dtype: Accumulation dtype
        swap_debug: Print debug info
        layer_name: Layer name for logging
    
    Returns:
        keep_final: Final keep mask after swap
        stats: Dictionary of statistics
    """
    device = W.device
    out_features, in_features = W.shape
    total = out_features * in_features
    
    # Determine accumulation dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    accum_dtype = dtype_map.get(refill_dtype, torch.float32)
    
    stats = {
        "layer_name": layer_name,
        "shape": (out_features, in_features),
        "total": total,
        "swap_k": 0,
        "cand_m": 0,
        "mask_diff": 0,
        "mask_diff_ratio": 0.0
    }
    
    # Step 1: Compute swap count
    kept_count = keep_target.sum().item()
    pruned_count = total - kept_count
    swap_k = round(total * swap_ratio)
    swap_k = min(swap_k, kept_count, pruned_count)
    swap_k = max(swap_k, 0)
    
    stats["kept_count"] = kept_count
    stats["pruned_count"] = pruned_count
    stats["swap_k"] = swap_k
    
    # If no swap needed, return original mask
    if swap_k == 0:
        print(f"[RGLS] {layer_name}: swap_k=0, skipping (baseline equivalent)")
        return keep_target.clone(), stats
    
    # Step 2: Build candidate sets
    cand_m = round(total * swap_candidate_bandwidth)
    cand_m = max(cand_m, swap_k)
    stats["cand_m"] = cand_m
    
    # Drop candidates: lowest RIA scores among kept weights
    drop_candidates = get_candidates_topk(ria_scores, keep_target, cand_m, largest=False)
    
    # Add candidates: highest RIA scores among pruned weights
    add_candidates = get_candidates_topk(ria_scores, ~keep_target, cand_m, largest=True)
    
    if len(drop_candidates) == 0 or len(add_candidates) == 0:
        print(f"[RGLS] {layer_name}: Empty candidate pool, skipping")
        return keep_target.clone(), stats
    
    # Step 3: Compute benefit for add_candidates
    benefit_add = compute_benefit_chunked(
        W, keep_target, add_candidates,
        X_batches, refill_out_chunk, refill_clip_residual,
        accum_dtype, swap_debug
    )
    
    # Step 4: Compute add_score
    ria_add = ria_scores.flatten()[add_candidates]
    
    z_benefit = zscore(benefit_add)
    z_ria = zscore(ria_add)
    add_score = swap_alpha * z_benefit + (1 - swap_alpha) * z_ria
    
    # Select top swap_k from add_candidates by add_score
    _, topk_add_local = torch.topk(add_score, min(swap_k, len(add_score)), largest=True)
    add_set = add_candidates[topk_add_local]
    
    # Select bottom swap_k from drop_candidates by ria_scores
    ria_drop = ria_scores.flatten()[drop_candidates]
    _, topk_drop_local = torch.topk(ria_drop, min(swap_k, len(ria_drop)), largest=False)
    drop_set = drop_candidates[topk_drop_local]
    
    # Ensure we swap equal numbers
    actual_swap = min(len(add_set), len(drop_set))
    add_set = add_set[:actual_swap]
    drop_set = drop_set[:actual_swap]
    
    # Step 5: Apply swap
    keep_final = keep_target.clone().flatten()
    keep_final[drop_set] = False
    keep_final[add_set] = True
    keep_final = keep_final.view(out_features, in_features)
    
    # Verify sparsity preserved
    mask_diff = (keep_final != keep_target).sum().item()
    final_kept = keep_final.sum().item()
    
    stats["mask_diff"] = mask_diff
    stats["mask_diff_ratio"] = mask_diff / total
    stats["final_kept"] = final_kept
    stats["actual_swap"] = actual_swap
    
    # Sanity check
    assert final_kept == kept_count, f"Sparsity changed! {kept_count} -> {final_kept}"
    
    # Print summary
    sparsity = 1.0 - kept_count / total
    print(f"[RGLS] {layer_name}: shape={W.shape}, sparsity={sparsity:.4f}, "
          f"swap_k={swap_k}, cand_m={cand_m}, mask_diff={mask_diff} ({stats['mask_diff_ratio']*100:.3f}%)")
    
    if swap_debug:
        print(f"  [Debug] drop_cand RIA: min={ria_scores.flatten()[drop_candidates].min():.6f}, "
              f"max={ria_scores.flatten()[drop_candidates].max():.6f}")
        print(f"  [Debug] add_cand RIA: min={ria_add.min():.6f}, max={ria_add.max():.6f}")
        print(f"  [Debug] benefit: mean={benefit_add.mean():.6f}, std={benefit_add.std():.6f}")
        print(f"  [Debug] add_score topk: {add_score[topk_add_local[:5]].tolist()}")
    
    return keep_final, stats


def run_rgls_tests():
    """
    Unit tests for RGLS functionality.
    Run this to verify correctness.
    """
    print("=" * 60)
    print("Running RGLS Unit Tests")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Create test data
    out_f, in_f = 64, 128
    W = torch.randn(out_f, in_f, device=device)
    ria_scores = torch.abs(W) + torch.randn_like(W) * 0.1
    sparsity = 0.5
    
    keep_target = build_keep_mask_from_scores(ria_scores, sparsity)
    
    # Create fake X_batches
    X_batches = [torch.randn(32, in_f, device=device) for _ in range(4)]
    
    # Test 1: Sparsity preservation
    print("\nTest 1: Sparsity preservation...")
    keep_final, stats = local_swap_one_linear(
        W, keep_target, ria_scores, X_batches,
        swap_ratio=0.01, swap_candidate_bandwidth=0.02,
        swap_alpha=0.5, refill_out_chunk=32,
        refill_clip_residual=0.0, refill_dtype="fp32",
        swap_debug=False, layer_name="test_layer"
    )
    assert keep_final.sum() == keep_target.sum(), "FAILED: Sparsity not preserved!"
    print("PASSED: keep_final.sum() == keep_target.sum()")
    
    # Test 2: Candidate pool size
    print("\nTest 2: Candidate pool size...")
    cand_m = stats["cand_m"]
    swap_k = stats["swap_k"]
    assert cand_m >= swap_k, f"FAILED: cand_m ({cand_m}) < swap_k ({swap_k})"
    print(f"PASSED: cand_m ({cand_m}) >= swap_k ({swap_k})")
    
    # Test 3: swap_ratio=0 equivalence
    print("\nTest 3: swap_ratio=0 equivalence...")
    keep_final_zero, _ = local_swap_one_linear(
        W, keep_target, ria_scores, X_batches,
        swap_ratio=0.0, swap_candidate_bandwidth=0.02,
        swap_alpha=0.5, refill_out_chunk=32,
        refill_clip_residual=0.0, refill_dtype="fp32",
        swap_debug=False, layer_name="test_layer_zero"
    )
    assert torch.all(keep_final_zero == keep_target), "FAILED: swap_ratio=0 should be identical!"
    print("PASSED: swap_ratio=0 produces identical mask")
    
    # Test 4: mask_diff upper bound
    print("\nTest 4: mask_diff upper bound...")
    mask_diff = stats["mask_diff"]
    assert mask_diff <= 2 * swap_k, f"FAILED: mask_diff ({mask_diff}) > 2*swap_k ({2*swap_k})"
    print(f"PASSED: mask_diff ({mask_diff}) <= 2*swap_k ({2*swap_k})")
    
    print("\n" + "=" * 60)
    print("All RGLS tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_rgls_tests()

