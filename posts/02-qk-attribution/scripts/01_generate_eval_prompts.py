"""Generate pre-evaluated prompts at arbitrary shot counts for SAE analysis.

Adapts script 02 to support multiple shot counts. Generates 1000 prompts per
(shot-count, rule), evaluates each with the model, and saves with
predicted_correct and answer_prob fields.

Usage:
    python posts/02-qk-attribution/scripts/01_generate_eval_prompts.py --shots 4 10
    python posts/02-qk-attribution/scripts/01_generate_eval_prompts.py --shots 4 --n-prompts 500

Estimated runtime: ~5 min per (shot-count, rule) combination on MPS.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
import time

import torch
from tqdm import tqdm

from src.config import DEVICE, PROMPTS_DIR
from src.model_utils import load_model, get_prediction, get_answer_prob, print_memory_usage
from src.prompt_generation import load_vocab, validate_vocab, generate_eval_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_prompts(model, prompts: list[dict]) -> dict:
    """Evaluate model predictions on prompts, annotating each with results."""
    correct = 0
    total = 0

    for p in tqdm(prompts, desc=f"Evaluating {prompts[0]['rule']}"):
        tokens = model.to_tokens(p["prompt"])
        logits = model(tokens)

        ans_id = model.to_single_token(p["correct_ans"])
        pred_id = get_prediction(logits)
        prob = get_answer_prob(logits, ans_id)

        p["predicted_correct"] = bool(pred_id == ans_id)
        p["answer_prob"] = float(prob)

        correct += int(pred_id == ans_id)
        total += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-evaluated prompts for SAE analysis"
    )
    parser.add_argument(
        "--shots", type=int, nargs="+", required=True,
        help="Shot counts to generate prompts for (e.g., 4 10)"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=1000,
        help="Number of prompts per (shot-count, rule). Default: 1000"
    )
    parser.add_argument(
        "--rules", type=str, nargs="+", default=["ABA"],
        choices=["ABA", "ABB"],
        help="Rules to generate. Default: ABA"
    )
    args = parser.parse_args()

    logger.info("Loading model...")
    model = load_model()
    print_memory_usage()

    logger.info("Loading and validating vocabulary...")
    vocab = load_vocab()
    vocab = validate_vocab(vocab, model)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    for n_shot in sorted(args.shots):
        for rule in args.rules:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating {args.n_prompts} {rule} prompts at {n_shot}-shot")
            logger.info(f"{'='*60}")

            start = time.time()

            # Use seed that encodes shot count to avoid overlap with 2-shot prompts
            seed = n_shot * 1000 + (0 if rule == "ABA" else 1)

            prompts = generate_eval_prompts(
                vocab, model, args.n_prompts, rule,
                n_shot=n_shot, seed=seed,
            )
            logger.info(f"Generated {len(prompts)} valid prompts")

            result = evaluate_prompts(model, prompts)
            elapsed = time.time() - start

            n_correct = sum(1 for p in prompts if p["predicted_correct"])
            n_wrong = sum(1 for p in prompts if not p["predicted_correct"])

            logger.info(f"  Accuracy: {result['accuracy']:.1%}")
            logger.info(f"  Correct: {n_correct}, Wrong: {n_wrong}")
            logger.info(f"  Time: {elapsed:.0f}s")

            # Save: include shot count in filename
            out_file = PROMPTS_DIR / f"eval_{rule.lower()}_{n_shot}shot_prompts.json"
            serializable = [
                {k: v for k, v in p.items() if not isinstance(v, torch.Tensor)}
                for p in prompts
            ]
            with open(out_file, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"  Saved to {out_file}")

    logger.info("\nDone!")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
