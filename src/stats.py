"""Statistical significance tests for head-level effects.

Two tests, one for each experiment type:

1. Max-statistic permutation test (CMA) — FWER control via sign-flip null
   distribution. From Yang et al. (2025), Appendix B.2.

2. Wilcoxon signed-rank + FDR correction (rescue patching) — per-head
   two-sided test on probability deltas, Benjamini-Hochberg corrected.
"""

import logging

import torch
import numpy as np

from src.config import N_PERMUTATIONS, P_VALUE_THRESHOLD

logger = logging.getLogger(__name__)


def run_permutation_test(
    scores: torch.Tensor,
    n_permutations: int = N_PERMUTATIONS,
    alpha: float = P_VALUE_THRESHOLD,
    seed: int = 42,
) -> dict:
    """Run max-statistic permutation test on CMA scores.

    Args:
        scores: Tensor of shape [n_prompts, n_layers, n_heads] — per-prompt CMA scores
        n_permutations: Number of random permutations
        alpha: Family-wise error rate threshold

    Returns:
        Dict with keys:
            threshold: significance threshold value
            significant_mask: boolean tensor [n_layers, n_heads]
            observed_mean: mean CMA scores [n_layers, n_heads]
            null_max_distribution: array of max values from each permutation
    """
    rng = np.random.default_rng(seed)
    n_prompts, n_layers, n_heads = scores.shape

    # Move to CPU for speed (this is pure numpy-style computation)
    scores_np = scores.cpu().numpy()

    # Observed mean CMA score per (layer, head)
    observed_mean = scores_np.mean(axis=0)  # [n_layers, n_heads]

    # Null distribution: for each permutation, randomly sign-flip each prompt's
    # scores with probability 0.5, compute mean, record the max
    null_max_values = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # Random sign flips: +1 or -1 for each prompt
        signs = rng.choice([-1, 1], size=(n_prompts, 1, 1))
        permuted_mean = (scores_np * signs).mean(axis=0)
        null_max_values[perm] = permuted_mean.max()

    # Threshold: (1 - alpha) quantile of null max distribution
    threshold = np.percentile(null_max_values, (1 - alpha) * 100)

    # Significant heads: observed mean exceeds threshold
    significant_mask = observed_mean > threshold

    n_significant = significant_mask.sum()
    logger.info(
        f"Permutation test: threshold={threshold:.4f}, "
        f"{n_significant} significant heads (alpha={alpha})"
    )

    return {
        "threshold": float(threshold),
        "significant_mask": torch.tensor(significant_mask),
        "observed_mean": torch.tensor(observed_mean),
        "null_max_distribution": null_max_values,
    }


def extract_significant_heads(
    test_result: dict,
) -> list[tuple[int, int, float]]:
    """Extract list of significant (layer, head, score) tuples.

    Returns list sorted by CMA score (descending).
    """
    mask = test_result["significant_mask"]
    mean_scores = test_result["observed_mean"]

    heads = []
    layers, head_idxs = torch.where(mask)
    for l, h in zip(layers.tolist(), head_idxs.tolist()):
        heads.append((l, h, mean_scores[l, h].item()))

    heads.sort(key=lambda x: x[2], reverse=True)
    return heads


# ---------------------------------------------------------------------------
# Rescue patching significance (Wilcoxon signed-rank + FDR)
# ---------------------------------------------------------------------------


def compute_rescue_significance(
    all_prob_deltas: np.ndarray | torch.Tensor,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> dict:
    """Compute statistical significance of rescue patching effects per head.

    Runs a two-sided Wilcoxon signed-rank test on each head's per-pair
    probability deltas, then applies FDR correction (Benjamini-Hochberg)
    across all heads.

    This is the rescue-patching analog of run_permutation_test() for CMA:
    it converts raw per-pair effect sizes into a significance mask.

    Args:
        all_prob_deltas: Array of shape [n_pairs, n_layers, n_heads].
        alpha: Significance level for FDR correction.
        method: Multiple testing correction method (default: 'fdr_bh').

    Returns:
        Dict with:
            significant_mask: boolean array [n_layers, n_heads]
            qvalues: array [n_layers, n_heads] — FDR-corrected q-values
            pvalues: array [n_layers, n_heads] — uncorrected p-values
            n_significant: int — total significant heads
            n_positive: int — significant with positive mean delta (rescuers)
            n_negative: int — significant with negative mean delta (harmful)
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    if isinstance(all_prob_deltas, torch.Tensor):
        all_prob_deltas = all_prob_deltas.cpu().float().numpy()

    n_pairs, n_layers, n_heads = all_prob_deltas.shape
    pvalues = np.ones((n_layers, n_heads))

    for L in range(n_layers):
        for H in range(n_heads):
            x = all_prob_deltas[:, L, H]
            if np.any(x != 0):
                try:
                    _, p = wilcoxon(x, alternative="two-sided")
                    pvalues[L, H] = p
                except ValueError:
                    pass

    # FDR correction across all heads
    flat_pvals = pvalues.ravel()
    reject, qvals_flat, _, _ = multipletests(flat_pvals, alpha=alpha, method=method)

    significant_mask = reject.reshape(n_layers, n_heads)
    qvalues = qvals_flat.reshape(n_layers, n_heads)

    mean_delta = all_prob_deltas.mean(axis=0)
    n_significant = int(significant_mask.sum())
    n_positive = int((significant_mask & (mean_delta > 0)).sum())
    n_negative = int((significant_mask & (mean_delta < 0)).sum())

    logger.info(
        f"Rescue significance: {n_significant} / {n_layers * n_heads} heads "
        f"(q < {alpha}) — {n_positive} positive, {n_negative} negative"
    )

    return {
        "significant_mask": significant_mask,
        "qvalues": qvalues,
        "pvalues": pvalues,
        "n_significant": n_significant,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
