"""CMA sweep at key shot counts to trace the three-stage circuit in gemma-2-2b.

Run AFTER scripts/01_accuracy_sweep.py.

For each shot count, runs all three CMA head types (symbol abstraction,
symbolic induction, retrieval) and permutation tests.

Uses filter_mode="correct" instead of probability threshold so that
CMA works even at low shot counts where model confidence is low.

Usage:
    python scripts/02_cma_sweep.py                          # defaults
    python scripts/02_cma_sweep.py --shots 2 5 10           # specify shots
    python scripts/02_cma_sweep.py --shots 2 10 --pairs 50  # fewer pairs (faster)

Estimated runtime: ~2-4 hours per shot count with 100 pairs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
import time

import torch

from src.config import (
    DEVICE, N_CMA_PAIRS, N_PERMUTATIONS,
    get_patch_positions, get_shot_results_dir, MODEL_NAME,
)
from src.model_utils import load_model, print_memory_usage
from src.prompt_generation import load_vocab, validate_vocab, generate_cma_context_pairs
from src.cma import run_cma_experiment
from src.stats import run_permutation_test, extract_significant_heads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cma_for_shot_count(model, vocab, n_shot, n_pairs, has_bos):
    """Run full CMA pipeline for a single shot count."""
    shot_start = time.time()
    logger.info(f"\n{'='*70}")
    logger.info(f"CMA SWEEP: {n_shot}-shot")
    logger.info(f"{'='*70}")

    patch_pos = get_patch_positions(has_bos, n_shot)
    shot_dir = get_shot_results_dir(n_shot)

    # Create result directories
    cma_abstract_dir = shot_dir / "cma" / "abstract"
    cma_token_dir = shot_dir / "cma" / "token"
    for d in [cma_abstract_dir, cma_token_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n_per_direction = n_pairs // 2
    all_significant = {}

    # ---- Abstract CMA (symbol abstraction + symbolic induction) ----
    for base_rule in ["ABA", "ABB"]:
        logger.info(f"\nGenerating abstract context pairs (base={base_rule}, n_shot={n_shot})...")
        pairs = generate_cma_context_pairs(
            vocab, model, n_per_direction, "abstract", base_rule,
            n_shot=n_shot,
            seed=n_shot * 1000 + (0 if base_rule == "ABA" else 1),
        )

        # Symbol Abstraction
        logger.info(f"  Symbol abstraction CMA (base={base_rule})...")
        logger.info(f"    Patch positions: {patch_pos['symbol_abstraction']}")
        sa_scores, sa_valid = run_cma_experiment(
            model, pairs, patch_pos["symbol_abstraction"], DEVICE,
            filter_mode="correct",
        )
        sa_file = cma_abstract_dir / f"symbol_abstraction_scores_base_{base_rule.lower()}.pt"
        torch.save(sa_scores, sa_file)
        logger.info(f"    {sa_valid} valid pairs, saved to {sa_file}")

        # Permutation test for symbol abstraction
        if sa_valid > 1:
            sa_perm = run_permutation_test(sa_scores)
            sa_heads = extract_significant_heads(sa_perm)
            all_significant[f"symbol_abstraction_base_{base_rule.lower()}"] = [
                {"layer": l, "head": h, "score": s} for l, h, s in sa_heads
            ]
            logger.info(f"    {len(sa_heads)} significant symbol abstraction heads")

        # Symbolic Induction
        logger.info(f"  Symbolic induction CMA (base={base_rule})...")
        logger.info(f"    Patch positions: {patch_pos['symbolic_induction']}")
        si_scores, si_valid = run_cma_experiment(
            model, pairs, patch_pos["symbolic_induction"], DEVICE,
            filter_mode="correct",
        )
        si_file = cma_abstract_dir / f"symbolic_induction_scores_base_{base_rule.lower()}.pt"
        torch.save(si_scores, si_file)
        logger.info(f"    {si_valid} valid pairs, saved to {si_file}")

        if si_valid > 1:
            si_perm = run_permutation_test(si_scores)
            si_heads = extract_significant_heads(si_perm)
            all_significant[f"symbolic_induction_base_{base_rule.lower()}"] = [
                {"layer": l, "head": h, "score": s} for l, h, s in si_heads
            ]
            logger.info(f"    {len(si_heads)} significant symbolic induction heads")

    # ---- Token CMA (retrieval) ----
    for base_rule in ["ABA", "ABB"]:
        logger.info(f"\nGenerating token context pairs (base={base_rule}, n_shot={n_shot})...")
        pairs = generate_cma_context_pairs(
            vocab, model, n_per_direction, "token", base_rule,
            n_shot=n_shot,
            seed=n_shot * 1000 + 100 + (0 if base_rule == "ABA" else 1),
        )

        logger.info(f"  Retrieval CMA (base={base_rule})...")
        logger.info(f"    Patch positions: {patch_pos['retrieval']}")
        ret_scores, ret_valid = run_cma_experiment(
            model, pairs, patch_pos["retrieval"], DEVICE,
            filter_mode="correct",
        )
        ret_file = cma_token_dir / f"retrieval_scores_base_{base_rule.lower()}.pt"
        torch.save(ret_scores, ret_file)
        logger.info(f"    {ret_valid} valid pairs, saved to {ret_file}")

        if ret_valid > 1:
            ret_perm = run_permutation_test(ret_scores)
            ret_heads = extract_significant_heads(ret_perm)
            all_significant[f"retrieval_base_{base_rule.lower()}"] = [
                {"layer": l, "head": h, "score": s} for l, h, s in ret_heads
            ]
            logger.info(f"    {len(ret_heads)} significant retrieval heads")

    # Save all significant heads for this shot count
    sig_file = shot_dir / "significant_heads.json"
    with open(sig_file, "w") as f:
        json.dump(all_significant, f, indent=2)
    logger.info(f"\nSaved significant heads to {sig_file}")

    elapsed = time.time() - shot_start
    total_sig = sum(len(v) for v in all_significant.values())
    logger.info(f"{n_shot}-shot CMA complete: {total_sig} significant heads total ({elapsed:.0f}s)")

    return all_significant


def main():
    parser = argparse.ArgumentParser(description="CMA sweep at key shot counts")
    parser.add_argument("--shots", type=int, nargs="+", default=[2, 4, 10],
                        help="Shot counts to run CMA at (default: 2 4 10)")
    parser.add_argument("--pairs", type=int, default=N_CMA_PAIRS // 2,
                        help=f"CMA pairs per rule direction (default: {N_CMA_PAIRS // 2}, doubled internally for both directions)")
    args = parser.parse_args()

    sweep_shots = sorted(args.shots)
    n_pairs = args.pairs * 2  # pairs per direction → total

    logger.info(f"CMA Sweep for {MODEL_NAME}")
    logger.info(f"Shot counts: {sweep_shots}")
    logger.info(f"CMA pairs per head type: {n_pairs} ({n_pairs // 2} per rule direction)")
    logger.info(f"Filter mode: correct (argmax must match answer)")

    logger.info("\nLoading model...")
    model = load_model()
    has_bos = model.cfg.default_prepend_bos
    print_memory_usage()

    logger.info("Loading and validating vocabulary...")
    vocab = load_vocab()
    vocab = validate_vocab(vocab, model)

    start_time = time.time()
    summary = {}

    for n_shot in sweep_shots:
        sig_heads = run_cma_for_shot_count(model, vocab, n_shot, n_pairs, has_bos)
        summary[n_shot] = {
            head_type: len(heads) for head_type, heads in sig_heads.items()
        }

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL CMA SWEEPS COMPLETE ({elapsed:.0f}s = {elapsed/3600:.1f}h)")
    logger.info(f"{'='*70}")

    for n_shot in sweep_shots:
        logger.info(f"\n{n_shot}-shot significant heads:")
        for head_type, count in summary[n_shot].items():
            logger.info(f"  {head_type}: {count}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
