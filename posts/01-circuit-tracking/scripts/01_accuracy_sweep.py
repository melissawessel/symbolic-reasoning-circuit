"""Accuracy sweep across shot counts for gemma-2-2b.

Tests n_shot = 1, 2, 3, 4, 5, 6, 8, 10 with 500 prompts per rule per shot count.
Measures how ABA/ABB accuracy evolves with more in-context examples.

Estimated runtime: ~15 minutes on MPS (8000 forward passes total).
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import json
import logging
import time

import torch
import numpy as np
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint

from src.config import DEVICE, SHOT_COUNTS, SHOT_SWEEP_DIR, MODEL_NAME
from src.model_utils import load_model, get_prediction, get_answer_prob, print_memory_usage
from src.prompt_generation import load_vocab, validate_vocab, generate_eval_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_PER_RULE = 500  # prompts per rule per shot count


def evaluate_at_shot_count(model, vocab, n_shot, n_per_rule=N_PER_RULE):
    """Evaluate ABA and ABB accuracy at a specific shot count."""
    results = {}

    for rule in ["ABA", "ABB"]:
        # Unique seed per (shot_count, rule) to avoid confounding
        seed = n_shot * 100 + (0 if rule == "ABA" else 1)

        prompts = generate_eval_prompts(
            vocab, model, n_per_rule, rule,
            n_shot=n_shot, seed=seed,
        )

        correct = 0
        probs = []

        for p in tqdm(prompts, desc=f"  {rule} @ {n_shot}-shot", leave=False):
            tokens = model.to_tokens(p["prompt"])
            logits = model(tokens)
            ans_id = model.to_single_token(p["correct_ans"])
            pred_id = get_prediction(logits)
            prob = get_answer_prob(logits, ans_id)

            correct += int(pred_id == ans_id)
            probs.append(prob)

        accuracy = correct / len(prompts)
        ci_low, ci_high = proportion_confint(correct, len(prompts), method="wilson")

        results[rule] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(prompts),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "mean_prob": float(np.mean(probs)),
            "median_prob": float(np.median(probs)),
            "prob_above_50": int(sum(1 for p in probs if p >= 0.5)),
            "prob_above_70": int(sum(1 for p in probs if p >= 0.7)),
            "prob_above_90": int(sum(1 for p in probs if p >= 0.9)),
        }

    return results


def main():
    start_time = time.time()

    logger.info(f"Shot-count accuracy sweep for {MODEL_NAME}")
    logger.info(f"Shot counts: {SHOT_COUNTS}")
    logger.info(f"Prompts per rule per shot: {N_PER_RULE}")
    logger.info(f"Total forward passes: {len(SHOT_COUNTS) * 2 * N_PER_RULE}")

    logger.info("Loading model...")
    model = load_model()
    print_memory_usage()

    logger.info("Loading and validating vocabulary...")
    vocab = load_vocab()
    vocab = validate_vocab(vocab, model)

    all_results = {}

    for n_shot in SHOT_COUNTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {n_shot}-shot...")
        shot_start = time.time()

        results = evaluate_at_shot_count(model, vocab, n_shot)
        all_results[str(n_shot)] = results

        shot_elapsed = time.time() - shot_start
        logger.info(
            f"  {n_shot}-shot: ABA={results['ABA']['accuracy']:.1%}, "
            f"ABB={results['ABB']['accuracy']:.1%}  ({shot_elapsed:.0f}s)"
        )

    # Save results
    SHOT_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": MODEL_NAME,
        "shot_counts": SHOT_COUNTS,
        "n_per_rule": N_PER_RULE,
        "results": all_results,
    }

    output_file = SHOT_SWEEP_DIR / "accuracy_sweep.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved results to {output_file}")

    # Print summary table
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY (total time: {elapsed:.0f}s)")
    logger.info(f"{'N-shot':>8} | {'ABA':>8} | {'ABB':>8} | {'ABA prob':>10} | {'ABB prob':>10}")
    logger.info("-" * 55)
    for n_shot in SHOT_COUNTS:
        r = all_results[str(n_shot)]
        logger.info(
            f"{n_shot:>8} | {r['ABA']['accuracy']:>7.1%} | {r['ABB']['accuracy']:>7.1%} | "
            f"{r['ABA']['mean_prob']:>10.4f} | {r['ABB']['mean_prob']:>10.4f}"
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
