"""Causal Mediation Analysis (CMA) implementation.

Implements the CMA procedure from Yang et al. (2025), Section 3.1 and Algorithm 1.

Notation (following the paper):
- base context (c2): the context whose activations are patched INTO the exp context
- exp context (c1): the context that receives patched activations
- patched exp context (c1*): exp context after patching

CMA score = (patched_logit_diff) - (unpatched_logit_diff)
where logit_diff = logits[causal_ans] - logits[exp_ans]
"""

import logging
from functools import partial

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
import transformer_lens.utils as tl_utils

from src.config import N_LAYERS, N_HEADS, PROB_THRESHOLD
from src.model_utils import hook_z_filter, get_answer_prob, get_prediction

logger = logging.getLogger(__name__)


def _patch_head_hook(
    activation: torch.Tensor,
    hook,
    head_idx: int,
    token_pos: int | list[int],
    base_cache,
):
    """Hook function that patches a single head's activation at specified positions.

    activation shape for hook_z: [batch, seq, n_heads, d_head]
    """
    cache_key = hook.name
    if isinstance(token_pos, int):
        token_pos = [token_pos]
    for pos in token_pos:
        activation[:, pos, head_idx, :] = base_cache[cache_key][:, pos, head_idx, :]
    return activation


def _patch_all_heads_hook(
    activation: torch.Tensor,
    hook,
    token_pos: int | list[int],
    base_cache,
):
    """Batched hook: batch element b gets head b patched from base_cache.

    Input is batched as [n_heads, seq, n_heads, d_head] — one copy per head.
    For batch element h, we patch head h at the specified positions.
    """
    cache_key = hook.name
    if isinstance(token_pos, int):
        token_pos = [token_pos]
    n_heads = activation.shape[2]
    for pos in token_pos:
        for h in range(n_heads):
            activation[h, pos, h, :] = base_cache[cache_key][0, pos, h, :]
    return activation


def compute_logit_diff(
    logits: torch.Tensor,
    model: HookedTransformer,
    causal_ans: str,
    original_ans: str,
) -> torch.Tensor:
    """Compute logit difference: logits[causal_ans] - logits[original_ans] at final position."""
    causal_id = model.to_single_token(causal_ans)
    original_id = model.to_single_token(original_ans)
    return logits[0, -1, causal_id] - logits[0, -1, original_id]


def _check_pair_validity(
    logits: torch.Tensor,
    model: HookedTransformer,
    answer_token: str,
    filter_mode: str,
    prob_threshold: float,
    context_label: str,
) -> bool:
    """Check whether a context passes the validity filter.

    Args:
        filter_mode: "correct" (argmax matches answer) or "threshold" (prob >= threshold)
    """
    ans_id = model.to_single_token(answer_token)

    if filter_mode == "correct":
        pred_id = get_prediction(logits)
        if pred_id != ans_id:
            logger.debug(f"Skipping pair: {context_label} prediction wrong")
            return False
    else:  # "threshold"
        prob = get_answer_prob(logits, ans_id)
        if prob < prob_threshold:
            logger.debug(f"Skipping pair: {context_label} prob {prob:.3f} < {prob_threshold}")
            return False

    return True


def run_cma_single_pair(
    model: HookedTransformer,
    base_prompt: str,
    exp_prompt: str,
    base_ans: str,
    exp_ans: str,
    causal_ans: str,
    patch_positions: list[int],
    device: str,
    filter_mode: str = "threshold",
    prob_threshold: float = PROB_THRESHOLD,
) -> torch.Tensor | None:
    """Run CMA for a single context pair across all layers and heads.

    Args:
        filter_mode: "correct" filters to pairs where argmax is correct answer.
                     "threshold" filters to pairs where P(correct) >= prob_threshold.

    Returns:
        Tensor of shape [n_layers, n_heads] with CMA scores, or None if pair is filtered out.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Tokenize
    base_ids = model.to_tokens(base_prompt)
    exp_ids = model.to_tokens(exp_prompt)

    # Run base context and cache hook_z activations
    base_logits, base_cache = model.run_with_cache(
        base_ids, names_filter=hook_z_filter
    )

    # Check base context validity
    if not _check_pair_validity(base_logits, model, base_ans, filter_mode, prob_threshold, "base"):
        del base_cache
        return None

    # Run exp context (no cache needed)
    exp_logits = model(exp_ids)

    # Check exp context validity
    if not _check_pair_validity(exp_logits, model, exp_ans, filter_mode, prob_threshold, "exp"):
        del base_cache
        return None

    # Compute unpatched logit diff for exp context
    exp_logit_diff = compute_logit_diff(exp_logits, model, causal_ans, exp_ans)

    # Precompute token IDs for scoring
    causal_id = model.to_single_token(causal_ans)
    original_id = model.to_single_token(exp_ans)

    # Batched sweep: one forward pass per layer (all heads in parallel)
    scores = torch.zeros(n_layers, n_heads, device=device)
    exp_ids_batched = exp_ids.expand(n_heads, -1)  # [n_heads, seq_len]

    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name("z", layer)
        hook_fn = partial(
            _patch_all_heads_hook,
            token_pos=patch_positions,
            base_cache=base_cache,
        )
        patched_logits = model.run_with_hooks(
            exp_ids_batched,
            fwd_hooks=[(hook_name, hook_fn)],
        )
        # patched_logits: [n_heads, seq, vocab]
        patched_logit_diffs = patched_logits[:, -1, causal_id] - patched_logits[:, -1, original_id]
        scores[layer, :] = patched_logit_diffs - exp_logit_diff

    # Clean up
    del base_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return scores


def run_cma_experiment(
    model: HookedTransformer,
    context_pairs: list[dict],
    patch_positions: list[int],
    device: str,
    filter_mode: str = "threshold",
    prob_threshold: float = PROB_THRESHOLD,
) -> tuple[torch.Tensor, int]:
    """Run CMA across all valid context pairs.

    Args:
        context_pairs: List of dicts from generate_cma_context_pairs
        patch_positions: Token positions to patch at
        device: Device string
        filter_mode: "correct" or "threshold" (see run_cma_single_pair)
        prob_threshold: Only used when filter_mode="threshold"

    Returns:
        (all_scores, n_valid) where all_scores is [n_valid, n_layers, n_heads]
    """
    all_scores = []
    n_valid = 0

    for i, pair in enumerate(tqdm(context_pairs, desc="CMA pairs")):
        scores = run_cma_single_pair(
            model=model,
            base_prompt=pair["base_prompt"],
            exp_prompt=pair["exp_prompt"],
            base_ans=pair["base_ans"],
            exp_ans=pair["exp_ans"],
            causal_ans=pair["causal_ans"],
            patch_positions=patch_positions,
            device=device,
            filter_mode=filter_mode,
            prob_threshold=prob_threshold,
        )
        if scores is not None:
            all_scores.append(scores)
            n_valid += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(context_pairs)} pairs, {n_valid} valid")

    if n_valid == 0:
        logger.error("No valid context pairs! Check model accuracy on the task.")
        return torch.zeros(0, model.cfg.n_layers, model.cfg.n_heads), 0

    stacked = torch.stack(all_scores, dim=0)  # [n_valid, n_layers, n_heads]
    logger.info(f"CMA complete: {n_valid} valid pairs out of {len(context_pairs)}")
    return stacked, n_valid


# ---------------------------------------------------------------------------
# Rescue patching: patch from a correct prompt into a wrong prompt (same rule)
# ---------------------------------------------------------------------------


def run_rescue_single_pair(
    model: HookedTransformer,
    correct_prompt: str,
    wrong_prompt: str,
    correct_ans: str,
    patch_positions: list[int] | None,
    device: str,
) -> dict:
    """Patch each head's activations from a correct prompt into a wrong prompt.

    Unlike standard CMA (which swaps between rules), this patches between two
    same-rule prompts where the model succeeds on one and fails on the other.
    This identifies which heads carry the information that makes the difference
    between success and failure.

    Args:
        correct_prompt: Prompt the model gets right.
        wrong_prompt: Prompt the model gets wrong (same rule).
        correct_ans: The correct answer token for both prompts.
        patch_positions: Token positions to patch at. If None, patches ALL positions.
        device: Device string.

    Returns:
        Dict with:
            prob_delta: [n_layers, n_heads] — P(correct|patched) - P(correct|unpatched)
            flipped: [n_layers, n_heads] — bool, did argmax flip from wrong to correct?
            baseline_prob: float — P(correct) on the unpatched wrong prompt
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    ans_id = model.to_single_token(correct_ans)

    # Run correct prompt and cache hook_z
    correct_ids = model.to_tokens(correct_prompt)
    _, correct_cache = model.run_with_cache(
        correct_ids, names_filter=hook_z_filter
    )

    # Run wrong prompt (baseline)
    wrong_ids = model.to_tokens(wrong_prompt)
    wrong_logits = model(wrong_ids)
    baseline_prob = get_answer_prob(wrong_logits, ans_id)
    seq_len = wrong_ids.shape[1]

    # If no positions specified, patch all positions
    if patch_positions is None:
        patch_positions = list(range(seq_len))

    # Batched sweep: one forward pass per layer (all heads in parallel)
    prob_delta = torch.zeros(n_layers, n_heads, device=device)
    flipped = torch.zeros(n_layers, n_heads, dtype=torch.bool, device=device)
    wrong_ids_batched = wrong_ids.expand(n_heads, -1)  # [n_heads, seq_len]

    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name("z", layer)
        hook_fn = partial(
            _patch_all_heads_hook,
            token_pos=patch_positions,
            base_cache=correct_cache,
        )
        patched_logits = model.run_with_hooks(
            wrong_ids_batched,
            fwd_hooks=[(hook_name, hook_fn)],
        )
        # patched_logits: [n_heads, seq, vocab]
        patched_probs = torch.softmax(patched_logits[:, -1, :], dim=-1)[:, ans_id]
        patched_preds = patched_logits[:, -1, :].argmax(dim=-1)

        prob_delta[layer, :] = patched_probs - baseline_prob
        flipped[layer, :] = (patched_preds == ans_id)

    # Clean up
    del correct_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "prob_delta": prob_delta,
        "flipped": flipped,
        "baseline_prob": baseline_prob,
    }


def run_rescue_experiment(
    model: HookedTransformer,
    rescue_pairs: list[dict],
    patch_positions: list[int] | None,
    device: str,
) -> dict:
    """Run rescue patching across all correct→wrong pairs.

    Args:
        rescue_pairs: List of dicts with keys:
            correct_prompt, wrong_prompt, correct_ans, rule
        patch_positions: Positions to patch, or None for all positions.
        device: Device string.

    Returns:
        Dict with:
            mean_prob_delta: [n_layers, n_heads] — mean probability rescue per head
            flip_rate: [n_layers, n_heads] — fraction of pairs where head rescued the answer
            n_pairs: int — number of pairs processed
            baseline_probs: list of floats — P(correct) on each wrong prompt before patching
    """
    all_prob_deltas = []
    all_flipped = []
    baseline_probs = []

    for i, pair in enumerate(tqdm(rescue_pairs, desc="Rescue pairs")):
        result = run_rescue_single_pair(
            model=model,
            correct_prompt=pair["correct_prompt"],
            wrong_prompt=pair["wrong_prompt"],
            correct_ans=pair["correct_ans"],
            patch_positions=patch_positions,
            device=device,
        )
        all_prob_deltas.append(result["prob_delta"])
        all_flipped.append(result["flipped"])
        baseline_probs.append(result["baseline_prob"])

        if (i + 1) % 10 == 0:
            n_done = i + 1
            mean_flip = torch.stack(all_flipped).float().mean(dim=0).max().item()
            logger.info(
                f"  Processed {n_done}/{len(rescue_pairs)} pairs, "
                f"best flip rate so far: {mean_flip:.1%}"
            )

    n_pairs = len(all_prob_deltas)
    if n_pairs == 0:
        logger.error("No rescue pairs processed!")
        return {
            "mean_prob_delta": torch.zeros(model.cfg.n_layers, model.cfg.n_heads),
            "flip_rate": torch.zeros(model.cfg.n_layers, model.cfg.n_heads),
            "n_pairs": 0,
            "baseline_probs": [],
        }

    stacked_deltas = torch.stack(all_prob_deltas)  # [n_pairs, n_layers, n_heads]
    stacked_flipped = torch.stack(all_flipped).float()

    return {
        "mean_prob_delta": stacked_deltas.mean(dim=0),
        "flip_rate": stacked_flipped.mean(dim=0),
        "n_pairs": n_pairs,
        "baseline_probs": baseline_probs,
        "all_prob_deltas": stacked_deltas,
    }
