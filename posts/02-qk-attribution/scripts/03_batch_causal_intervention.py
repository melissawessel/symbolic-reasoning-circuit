"""Batch causal feature intervention: suppression on correct + rescue on wrong prompts.

Runs feature interventions across many prompts and computes aggregate statistics:
- Correct prompts: does suppressing the feature break the prediction?
- Wrong prompts: does amplifying the feature rescue failures?

Prerequisite: generate eval prompts for the target rule (script 01 defaults to ABA only):
    python posts/02-qk-attribution/scripts/01_generate_eval_prompts.py --shots 2 --rules ABA ABB

Usage:
    python posts/02-qk-attribution/scripts/03_batch_causal_intervention.py \
        --shots 2 --layer 14 --head 0 --feature 4958 --rule ABA --width 65k \
        --n-correct 100 --n-wrong 100 --min-activation 0.01

    python posts/02-qk-attribution/scripts/03_batch_causal_intervention.py \
        --shots 2 --layer 14 --head 0 --feature 16986 --rule ABB --width 65k \
        --n-correct 100 --n-wrong 100 --min-activation 0.01
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DEVICE, PROMPTS_DIR, RESULTS_DIR, get_token_positions
from src.model_utils import load_model, print_memory_usage
from src.qk_ov_attribution import load_residual_sae, neuronpedia_url, RES_SAE_WIDTH
from src.causal_feature_intervention import run_batch_intervention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts(n_shot: int, rule: str):
    """Load pre-evaluated prompts, split into correct and wrong."""
    prompt_file = PROMPTS_DIR / f"eval_{rule.lower()}_{n_shot}shot_prompts.json"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"No prompts found at {prompt_file}. "
            f"Run: python posts/02-qk-attribution/scripts/01_generate_eval_prompts.py --shots {n_shot} --rules {rule}"
        )

    with open(prompt_file) as f:
        all_prompts = json.load(f)

    correct = [p for p in all_prompts if p.get("predicted_correct", True)]
    wrong = [p for p in all_prompts if not p.get("predicted_correct", True)]
    logger.info(f"Loaded {len(all_prompts)} prompts: {len(correct)} correct, {len(wrong)} wrong")
    return correct, wrong


def compute_flip_stats(batch_result, scales, direction):
    """Compute per-prompt flip statistics.

    Args:
        batch_result: BatchInterventionResult
        scales: list of scale values
        direction: "suppress" (check scales < 0 break correct) or
                   "rescue" (check scales > 1 rescue wrong)
    """
    base_idx = scales.index(1.0)
    n_flipped = 0
    deltas = []

    for pp in batch_result.per_prompt:
        pc = np.array(pp["correct_ans_prob"])
        pw = np.array(pp["wrong_ans_prob"])

        if direction == "rescue":
            baseline_wrong = pc[base_idx] < pw[base_idx]
            if not baseline_wrong:
                continue
            for si, s in enumerate(scales):
                if s > 1.0 and pc[si] > pw[si]:
                    n_flipped += 1
                    deltas.append(pc[si] - pc[base_idx])
                    break
        elif direction == "suppress":
            baseline_correct = pc[base_idx] > pw[base_idx]
            if not baseline_correct:
                continue
            for si, s in enumerate(scales):
                if s < 0 and pc[si] < pw[si]:
                    n_flipped += 1
                    deltas.append(pc[base_idx] - pc[si])
                    break

    # Count eligible prompts
    if direction == "rescue":
        n_eligible = sum(
            1 for pp in batch_result.per_prompt
            if np.array(pp["correct_ans_prob"])[base_idx] < np.array(pp["wrong_ans_prob"])[base_idx]
        )
    else:
        n_eligible = sum(
            1 for pp in batch_result.per_prompt
            if np.array(pp["correct_ans_prob"])[base_idx] > np.array(pp["wrong_ans_prob"])[base_idx]
        )

    # Significance tests
    from scipy.stats import wilcoxon, binomtest

    # 1. Wilcoxon signed-rank: paired test on P(correct) at baseline vs intervention
    #    For suppress: compare scale=1 vs most negative scale
    #    For rescue: compare scale=1 vs largest scale
    baseline_probs = []
    intervention_probs = []
    if direction == "rescue":
        best_scale_idx = max(range(len(scales)), key=lambda i: scales[i])
    else:
        best_scale_idx = min(range(len(scales)), key=lambda i: scales[i])

    for pp in batch_result.per_prompt:
        pc = np.array(pp["correct_ans_prob"])
        baseline_probs.append(pc[base_idx])
        intervention_probs.append(pc[best_scale_idx])

    baseline_probs = np.array(baseline_probs)
    intervention_probs = np.array(intervention_probs)

    if len(baseline_probs) >= 5:
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                intervention_probs, baseline_probs,
                alternative="greater" if direction == "rescue" else "less",
            )
        except ValueError:
            # All differences are zero
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    # 2. Binomial test: is flip rate significantly above 0?
    #    H0: flip probability = 0.05 (chance-level baseline)
    if n_eligible > 0:
        binom_result = binomtest(n_flipped, n_eligible, p=0.05, alternative="greater")
        binom_p = binom_result.pvalue
    else:
        binom_p = float("nan")

    return {
        "n_flipped": n_flipped,
        "n_eligible": n_eligible,
        "flip_rate": n_flipped / max(n_eligible, 1),
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "wilcoxon_scale": scales[best_scale_idx],
        "binom_p": float(binom_p),
        "binom_null": 0.05,
    }


def save_plot(correct_batch, wrong_batch, feature_id, out_path):
    """Save side-by-side suppression/rescue plot."""
    scales = np.array(correct_batch.scales)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Correct — suppression
    cc = np.array(correct_batch.mean_correct_ans_prob)
    cw = np.array(correct_batch.mean_wrong_ans_prob)
    ax1.plot(scales, cc, "o-", color="#2d5016", linewidth=2, label="P(correct)")
    ax1.plot(scales, cw, "o-", color="#c45a1a", linewidth=2, label="P(wrong-rule)")
    ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Intervention strength (scale)")
    ax1.set_ylabel("Mean next-token probability")
    ax1.set_title(f"Correct prompts (n={correct_batch.n_prompts}) — suppression")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")

    # Wrong — rescue
    wc = np.array(wrong_batch.mean_correct_ans_prob)
    ww = np.array(wrong_batch.mean_wrong_ans_prob)
    ax2.plot(scales, wc, "o-", color="#2d5016", linewidth=2, label="P(correct)")
    ax2.plot(scales, ww, "o-", color="#c45a1a", linewidth=2, label="P(wrong-rule)")
    ax2.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Intervention strength (scale)")
    ax2.set_ylabel("Mean next-token probability")
    ax2.set_title(f"Wrong prompts (n={wrong_batch.n_prompts}) — rescue")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")

    fig.suptitle(f"Feature {feature_id} — batch causal intervention", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch causal feature intervention")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--rule", type=str, default="ABA", choices=["ABA", "ABB"])
    parser.add_argument("--width", type=str, default=RES_SAE_WIDTH)
    parser.add_argument("--n-correct", type=int, default=100)
    parser.add_argument("--n-wrong", type=int, default=100)
    parser.add_argument("--min-activation", type=float, default=0.01)
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[-6, -4, -2, -1, 0, 0.5, 1, 2, 4, 6, 8, 10],
    )
    args = parser.parse_args()

    t0 = time.time()

    # Positions
    pos = get_token_positions(has_bos=True, n_shot=args.shots)
    c_positions = [ex["C"] for ex in pos["examples"]]
    query_pos = pos["query"]["sep2"]
    logger.info(f"C positions: {c_positions}, query: {query_pos}")

    # Load model + SAE
    logger.info("Loading model...")
    model = load_model()
    print_memory_usage()

    sae_layer = args.layer - 1
    logger.info(f"Loading SAE layer {sae_layer} (width={args.width})...")
    sae = load_residual_sae(sae_layer, width=args.width, device="cpu")

    np_url = neuronpedia_url(sae_layer, args.feature, args.width)
    logger.info(f"Feature {args.feature}: {np_url}")

    # Load prompts
    correct_prompts, wrong_prompts = load_prompts(args.shots, args.rule)
    correct_prompts = correct_prompts[:args.n_correct]
    wrong_prompts = wrong_prompts[:args.n_wrong]

    # Run batches
    logger.info(f"\n{'='*60}")
    logger.info(f"Running correct batch ({len(correct_prompts)} prompts)...")
    correct_batch = run_batch_intervention(
        model=model, sae=sae, prompts=correct_prompts,
        layer=args.layer, head=args.head, feature_id=args.feature,
        positions=c_positions, position_type="all_C",
        intervention_side="key", query_positions=[query_pos],
        key_positions=c_positions, rule=args.rule,
        scales=args.scales, min_feature_activation=args.min_activation,
    )
    logger.info(f"  {correct_batch.n_prompts} prompts with activation >= {args.min_activation}")

    logger.info(f"Running wrong batch ({len(wrong_prompts)} prompts)...")
    wrong_batch = run_batch_intervention(
        model=model, sae=sae, prompts=wrong_prompts,
        layer=args.layer, head=args.head, feature_id=args.feature,
        positions=c_positions, position_type="all_C",
        intervention_side="key", query_positions=[query_pos],
        key_positions=c_positions, rule=args.rule,
        scales=args.scales, min_feature_activation=args.min_activation,
    )
    logger.info(f"  {wrong_batch.n_prompts} prompts with activation >= {args.min_activation}")

    # Compute flip stats
    suppress_stats = compute_flip_stats(correct_batch, args.scales, "suppress")
    rescue_stats = compute_flip_stats(wrong_batch, args.scales, "rescue")

    # Output directory
    out_dir = RESULTS_DIR / "causal_interventions" / f"{args.shots}shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"L{args.layer}H{args.head}_feat{args.feature}_{args.rule}_batch"

    # Save results
    output = {
        "layer": args.layer,
        "head": args.head,
        "feature_id": args.feature,
        "sae_layer": sae_layer,
        "sae_width": args.width,
        "n_shot": args.shots,
        "rule": args.rule,
        "scales": args.scales,
        "neuronpedia_url": np_url,
        "correct_batch": {
            "n_prompts": correct_batch.n_prompts,
            "mean_correct_ans_prob": correct_batch.mean_correct_ans_prob,
            "mean_wrong_ans_prob": correct_batch.mean_wrong_ans_prob,
            "suppress_stats": suppress_stats,
        },
        "wrong_batch": {
            "n_prompts": wrong_batch.n_prompts,
            "mean_correct_ans_prob": wrong_batch.mean_correct_ans_prob,
            "mean_wrong_ans_prob": wrong_batch.mean_wrong_ans_prob,
            "rescue_stats": rescue_stats,
        },
    }

    # Pick top-activation example from each batch for illustration
    def _top_example(batch):
        if not batch.per_prompt:
            return None
        return max(batch.per_prompt, key=lambda pp: pp["feature_activation"])

    output["top_suppress_example"] = _top_example(correct_batch)
    output["top_rescue_example"] = _top_example(wrong_batch)

    json_path = out_dir / f"{fname}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Save plot
    plot_path = out_dir / f"{fname}.png"
    save_plot(correct_batch, wrong_batch, args.feature, plot_path)

    # Print summary
    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS — Feature {args.feature} at L{args.layer}H{args.head}, {args.shots}-shot {args.rule}")
    logger.info(f"{'='*60}")
    logger.info(f"Correct prompts (suppression):")
    logger.info(f"  {suppress_stats['n_flipped']}/{suppress_stats['n_eligible']} broken ({100*suppress_stats['flip_rate']:.0f}%)")
    logger.info(f"  Mean P(correct) drop: {suppress_stats['mean_delta']:.4f}")
    logger.info(f"  Wilcoxon (scale={suppress_stats['wilcoxon_scale']}x vs 1x): p={suppress_stats['wilcoxon_p']:.4g}")
    logger.info(f"  Binomial (flip rate > 5% chance): p={suppress_stats['binom_p']:.4g}")
    logger.info(f"Wrong prompts (rescue):")
    logger.info(f"  {rescue_stats['n_flipped']}/{rescue_stats['n_eligible']} rescued ({100*rescue_stats['flip_rate']:.0f}%)")
    logger.info(f"  Mean P(correct) increase: {rescue_stats['mean_delta']:.4f}")
    logger.info(f"  Wilcoxon (scale={rescue_stats['wilcoxon_scale']}x vs 1x): p={rescue_stats['wilcoxon_p']:.4g}")
    logger.info(f"  Binomial (flip rate > 5% chance): p={rescue_stats['binom_p']:.4g}")
    logger.info(f"\nFeature {args.feature} is NECESSARY for {100*suppress_stats['flip_rate']:.0f}% of correct predictions")
    logger.info(f"Feature {args.feature} is SUFFICIENT to rescue {100*rescue_stats['flip_rate']:.0f}% of failures")
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
