"""Rescue patching: patch activations from correct prompts into wrong prompts.

For each head, takes the activation from a prompt the model gets RIGHT and
patches it into a prompt the model gets WRONG (same rule, same structure,
different tokens). If the wrong prediction flips to correct, that head
carries the critical information that makes the difference between success
and failure.

This complements standard CMA by directly identifying the bottleneck heads — the ones that intermittently fail at
low shot counts.

Usage:
    python scripts/03_rescue_patching.py                              # defaults (2-shot ABA)
    python scripts/03_rescue_patching.py --shots 2 4                  # multiple shot counts
    python scripts/03_rescue_patching.py --shots 2 --pairs 50 --rule ABA
    python scripts/03_rescue_patching.py --shots 2 --positions all    # patch all positions
    python scripts/03_rescue_patching.py --shots 2 --positions cma    # patch CMA positions only

Estimated runtime: ~2-3 hours per shot count with 100 pairs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
import time

import torch
import numpy as np

from src.config import (
    DEVICE, MODEL_NAME, N_LAYERS, N_HEADS,
    get_patch_positions, get_shot_results_dir,
)
from src.model_utils import load_model, print_memory_usage
from src.prompt_generation import load_vocab, validate_vocab, generate_rescue_pairs
from src.cma import run_rescue_experiment
from src.stats import compute_rescue_significance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_rescue_for_shot_count(model, vocab, n_shot, n_pairs, rule, position_mode, has_bos):
    """Run rescue patching at one shot count."""
    logger.info(f"\n{'='*70}")
    logger.info(f"RESCUE PATCHING: {n_shot}-shot {rule}")
    logger.info(f"{'='*70}")

    shot_dir = get_shot_results_dir(n_shot)
    rescue_dir = shot_dir / "rescue"
    rescue_dir.mkdir(parents=True, exist_ok=True)

    # Determine patch positions
    if position_mode == "all":
        patch_positions = None  # patches all positions
        pos_label = "all"
    else:
        # Use the CMA positions: example C tokens + final position
        patch_pos = get_patch_positions(has_bos, n_shot)
        # Union of all CMA position types
        all_cma_pos = set()
        for positions in patch_pos.values():
            all_cma_pos.update(positions)
        patch_positions = sorted(all_cma_pos)
        pos_label = f"cma_{patch_positions}"

    logger.info(f"  Patch positions: {pos_label}")

    pairs = generate_rescue_pairs(
        vocab, model, n_pairs, rule,
        n_shot=n_shot,
        seed=n_shot * 100 + (0 if rule == "ABA" else 1),
    )

    if not pairs:
        logger.warning(f"No rescue pairs generated! Skipping {n_shot}-shot {rule}.")
        return None

    # Run rescue experiment
    start = time.time()
    result = run_rescue_experiment(
        model, pairs, patch_positions, DEVICE,
    )
    elapsed = time.time() - start

    logger.info(f"\n  Results ({elapsed:.0f}s):")
    logger.info(f"    Pairs processed: {result['n_pairs']}")
    logger.info(f"    Mean baseline P(correct) on wrong prompts: "
                f"{np.mean(result['baseline_probs']):.4f}")

    # Find best heads by flip rate
    flip_rate = result["flip_rate"].cpu().numpy()
    prob_delta = result["mean_prob_delta"].cpu().numpy()

    top_flip_idx = np.argsort(flip_rate.ravel())[::-1][:10]
    logger.info(f"\n  Top 10 heads by flip rate:")
    for idx in top_flip_idx:
        l, h = idx // N_HEADS, idx % N_HEADS
        logger.info(
            f"    L{l:2d}H{h}: flip_rate={flip_rate[l,h]:.1%}, "
            f"prob_delta={prob_delta[l,h]:+.4f}"
        )

    # Save results
    save_data = {
        "n_shot": n_shot,
        "rule": rule,
        "n_pairs": result["n_pairs"],
        "position_mode": position_mode,
        "patch_positions": patch_positions,
        "mean_baseline_prob": float(np.mean(result["baseline_probs"])),
        "top_rescue_heads": [
            {
                "layer": int(idx // N_HEADS),
                "head": int(idx % N_HEADS),
                "flip_rate": float(flip_rate[idx // N_HEADS, idx % N_HEADS]),
                "mean_prob_delta": float(prob_delta[idx // N_HEADS, idx % N_HEADS]),
            }
            for idx in top_flip_idx[:20]
        ],
    }

    json_file = rescue_dir / f"rescue_{rule.lower()}_{pos_label.split('_')[0]}.json"
    with open(json_file, "w") as f:
        json.dump(save_data, f, indent=2)

    # Save tensors for notebook analysis
    torch.save(result["mean_prob_delta"], rescue_dir / f"prob_delta_{rule.lower()}.pt")
    torch.save(result["flip_rate"], rescue_dir / f"flip_rate_{rule.lower()}.pt")
    if "all_prob_deltas" in result:
        torch.save(result["all_prob_deltas"], rescue_dir / f"all_prob_deltas_{rule.lower()}.pt")

    # Compute and save rescue significance (Wilcoxon + FDR)
    if "all_prob_deltas" in result:
        sig_result = compute_rescue_significance(result["all_prob_deltas"])
        torch.save(
            torch.tensor(sig_result["significant_mask"]),
            rescue_dir / f"rescue_sig_{rule.lower()}.pt",
        )
        save_data["rescue_significance"] = {
            "n_significant": sig_result["n_significant"],
            "n_positive": sig_result["n_positive"],
            "n_negative": sig_result["n_negative"],
        }
        # Re-save JSON with significance stats
        with open(json_file, "w") as f:
            json.dump(save_data, f, indent=2)

    logger.info(f"  Saved to {rescue_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Rescue patching: correct → wrong prompts")
    parser.add_argument("--shots", type=int, nargs="+", default=[2],
                        help="Shot counts to test (default: 2)")
    parser.add_argument("--pairs", type=int, default=100,
                        help="Number of rescue pairs per shot count (default: 100)")
    parser.add_argument("--rule", type=str, default="ABA", choices=["ABA", "ABB"],
                        help="Rule to test (default: ABA — where the model struggles)")
    parser.add_argument("--positions", type=str, default="all", choices=["all", "cma"],
                        help="Patch all positions or just CMA-relevant positions (default: all)")
    args = parser.parse_args()

    logger.info(f"Rescue Patching for {MODEL_NAME}")
    logger.info(f"Shot counts: {args.shots}")
    logger.info(f"Rule: {args.rule}")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Position mode: {args.positions}")

    logger.info("\nLoading model...")
    model = load_model()
    has_bos = model.cfg.default_prepend_bos
    print_memory_usage()

    logger.info("Loading and validating vocabulary...")
    vocab = load_vocab()
    vocab = validate_vocab(vocab, model)

    start_time = time.time()

    for n_shot in sorted(args.shots):
        run_rescue_for_shot_count(
            model, vocab, n_shot, args.pairs, args.rule, args.positions, has_bos,
        )

    elapsed = time.time() - start_time
    logger.info(f"\nAll rescue patching complete ({elapsed:.0f}s = {elapsed/3600:.1f}h)")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
