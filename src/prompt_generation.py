"""Prompt generation for ABA/ABB identity rule tasks.

Follows the methodology from Yang et al. (2025):
- Randomly sample English tokens from the model vocabulary
- Construct N-shot prompts with separator ^ and newline between examples
- Validate tokenization integrity
"""

import random
import logging
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.config import SEPARATOR, N_SHOT, VOCAB_DIR

logger = logging.getLogger(__name__)


def load_vocab(vocab_path: Optional[Path] = None) -> list[str]:
    """Load the English vocabulary file."""
    if vocab_path is None:
        vocab_path = VOCAB_DIR / "gemma2_english_vocab.txt"
    with open(vocab_path) as f:
        vocab = [line.rstrip() for line in f.readlines()]
    logger.info(f"Loaded {len(vocab)} vocab entries from {vocab_path}")
    return vocab


def validate_vocab(vocab: list[str], model: HookedTransformer) -> list[str]:
    """Filter vocab to only tokens that encode to a single token ID.

    Also validates that token + separator doesn't merge during tokenization.
    """
    valid = []
    tokenizer = model.tokenizer
    sep_tokens = tokenizer.tokenize(SEPARATOR)

    for token in vocab:
        # Must encode to exactly 1 token
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        # token + separator must be 2 tokens (no merging)
        combined = tokenizer.tokenize(token + SEPARATOR)
        if len(combined) != 2:
            continue
        valid.append(token)

    logger.info(f"Validated vocab: {len(valid)}/{len(vocab)} tokens passed")
    return valid


def generate_eval_prompt(
    vocab: list[str],
    rule: str,
    n_shot: int = N_SHOT,
    sep: str = SEPARATOR,
) -> tuple[str, str]:
    """Generate a single evaluation prompt.

    Returns:
        (prompt_string, correct_answer_token)
    """
    assert rule in ("ABA", "ABB")
    n_tokens = (n_shot + 1) * 2
    tokens = random.sample(vocab, k=n_tokens)

    examples = []
    for i in range(n_shot):
        a, b = tokens[i * 2], tokens[i * 2 + 1]
        if rule == "ABA":
            example = sep.join([a, b, a])
        else:
            example = sep.join([a, b, b])
        examples.append(example)

    # Query (incomplete)
    query_a, query_b = tokens[-2], tokens[-1]
    query = query_a + sep + query_b + sep
    examples.append(query)

    prompt = "\n".join(examples)
    correct_ans = query_a if rule == "ABA" else query_b
    return prompt, correct_ans


def validate_prompt_tokenization(
    prompt: str,
    model: HookedTransformer,
    n_shot: int = N_SHOT,
    sep: str = SEPARATOR,
) -> bool:
    """Check that the prompt tokenizes to the expected number of tokens.

    Expected: 3 tokens per token-triple * 2 (n_shot examples) + 4 tokens for query
    = 3 * 2 * n_shot + 4 (without BOS)

    Also checks that separators/newlines appear at odd positions.
    """
    tokenizer = model.tokenizer
    proc_tokens = tokenizer.tokenize(prompt)
    expected_len = 3 * 2 * n_shot + 4

    if len(proc_tokens) != expected_len:
        return False

    # Check separators at odd positions
    newline_token = tokenizer.tokenize("\n")[0]
    sep_set = {sep, newline_token}
    for i in range(1, len(proc_tokens), 2):
        if proc_tokens[i] not in sep_set:
            return False

    return True


def generate_eval_prompts(
    vocab: list[str],
    model: HookedTransformer,
    n_prompts: int,
    rule: str,
    n_shot: int = N_SHOT,
    seed: int = 0,
) -> list[dict]:
    """Generate validated evaluation prompts.

    Returns list of dicts with keys: prompt, correct_ans, rule
    """
    random.seed(seed)
    np.random.seed(seed)

    prompts = []
    seen = set()
    attempts = 0
    max_attempts = n_prompts * 10

    while len(prompts) < n_prompts and attempts < max_attempts:
        attempts += 1
        prompt, correct_ans = generate_eval_prompt(vocab, rule, n_shot)

        if prompt in seen:
            continue

        if not validate_prompt_tokenization(prompt, model, n_shot):
            continue

        seen.add(prompt)
        prompts.append({
            "prompt": prompt,
            "correct_ans": correct_ans,
            "rule": rule,
        })

    logger.info(f"Generated {len(prompts)}/{n_prompts} valid {rule} prompts in {attempts} attempts")
    return prompts


def generate_cma_context_pairs(
    vocab: list[str],
    model: HookedTransformer,
    n_pairs: int,
    context_type: str,
    base_rule: str = "ABA",
    n_shot: int = N_SHOT,
    sep: str = SEPARATOR,
    seed: int = 0,
) -> list[dict]:
    """Generate context pairs for CMA experiments.

    Args:
        context_type: "abstract" (for symbol abstraction / symbolic induction)
                      or "token" (for retrieval heads)
        base_rule: The rule for the base context ("ABA" or "ABB")

    Returns list of dicts with keys:
        base_prompt, exp_prompt, base_ans, exp_ans, causal_ans
    """
    assert context_type in ("abstract", "token")
    assert base_rule in ("ABA", "ABB")

    random.seed(seed)
    np.random.seed(seed)

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 10

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        n_tokens = (n_shot + 1) * 2
        tokens = random.sample(vocab, k=n_tokens)

        base_examples = []
        exp_examples = []

        for i in range(n_shot):
            a, b = tokens[i * 2], tokens[i * 2 + 1]

            if base_rule == "ABA":
                base_triple = [a, b, a]
                if context_type == "abstract":
                    # Swap first two tokens, keep third the same
                    exp_triple = [b, a, a]
                else:
                    exp_triple = base_triple[:]
            else:  # ABB
                base_triple = [a, b, b]
                if context_type == "abstract":
                    exp_triple = [b, a, b]
                else:
                    exp_triple = base_triple[:]

            base_examples.append(sep.join(base_triple))
            exp_examples.append(sep.join(exp_triple))

        # Query tokens
        qa, qb = tokens[-2], tokens[-1]
        base_ans_idx = 0 if base_rule == "ABA" else 1
        base_ans = [qa, qb][base_ans_idx]

        # Base query: qa^qb^
        base_query = qa + sep + qb + sep
        base_examples.append(base_query)
        base_prompt = "\n".join(base_examples)

        # Exp query: qb^qa^ (swapped)
        exp_query = qb + sep + qa + sep
        exp_examples.append(exp_query)
        exp_prompt = "\n".join(exp_examples)

        # Determine answers
        if context_type == "abstract":
            # Exp has the opposite rule due to token swapping
            exp_ans_idx = 1 - base_ans_idx
            causal_ans_idx = base_ans_idx  # After patching, should follow base rule
            exp_ans = [qb, qa][exp_ans_idx]
            causal_ans = [qb, qa][causal_ans_idx]
        else:  # token
            # Same rule, but query tokens swapped
            exp_ans_idx = base_ans_idx
            exp_ans = [qb, qa][exp_ans_idx]
            causal_ans = base_ans

        # Validate tokenization for both prompts
        key = (base_prompt, exp_prompt)
        if key in seen:
            continue

        valid = True
        for p in [base_prompt, exp_prompt]:
            if not validate_prompt_tokenization(p, model, n_shot):
                valid = False
                break
        if not valid:
            continue

        seen.add(key)
        pairs.append({
            "base_prompt": base_prompt,
            "exp_prompt": exp_prompt,
            "base_ans": base_ans,
            "exp_ans": exp_ans,
            "causal_ans": causal_ans,
            "base_rule": base_rule,
        })

    logger.info(
        f"Generated {len(pairs)}/{n_pairs} valid {context_type} "
        f"context pairs (base={base_rule}) in {attempts} attempts"
    )
    return pairs


def generate_rescue_pairs(
    vocab: list[str],
    model: HookedTransformer,
    n_pairs: int,
    rule: str,
    n_shot: int = N_SHOT,
    seed: int = 0,
    precomputed_prompts: list[dict] | None = None,
) -> list[dict]:
    """Generate correct→wrong prompt pairs for rescue patching.

    Can either load pre-evaluated prompts (from script 02 or 07) or generate
    fresh ones. Pre-evaluated prompts should have "predicted_correct" field.

    Args:
        n_pairs: Target number of rescue pairs.
        rule: "ABA" or "ABB"
        n_shot: Number of in-context examples.
        precomputed_prompts: Optional list of dicts with "prompt", "correct_ans",
            and "predicted_correct" fields. If provided, skips generation + eval.

    Returns:
        List of dicts with keys:
            correct_prompt, wrong_prompt, correct_ans, rule
    """
    from src.model_utils import get_prediction

    if precomputed_prompts is not None:
        # Use pre-evaluated prompts — no model inference needed for splitting
        logger.info(f"Using {len(precomputed_prompts)} pre-evaluated {rule} prompts")
        correct_pool = [p for p in precomputed_prompts if p.get("predicted_correct")]
        wrong_pool = [p for p in precomputed_prompts if not p.get("predicted_correct")]
    else:
        # Generate fresh prompts and evaluate them
        pool_size = n_pairs * 6  # oversample since we need both categories
        logger.info(f"Generating {pool_size} {rule} prompts to find rescue pairs...")
        prompts = generate_eval_prompts(vocab, model, pool_size, rule, n_shot=n_shot, seed=seed)

        correct_pool = []
        wrong_pool = []

        for p in prompts:
            tokens = model.to_tokens(p["prompt"])
            logits = model(tokens)
            ans_id = model.to_single_token(p["correct_ans"])
            pred_id = get_prediction(logits)

            if pred_id == ans_id:
                correct_pool.append(p)
            else:
                wrong_pool.append(p)

    logger.info(
        f"Pool split: {len(correct_pool)} correct, {len(wrong_pool)} wrong "
        f"({len(correct_pool)/(len(correct_pool)+len(wrong_pool)):.0%} accuracy)"
    )

    if not correct_pool or not wrong_pool:
        logger.error("Need both correct and wrong prompts for rescue pairs!")
        return []

    # Pair them up: shuffle and zip
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(correct_pool)
    rng.shuffle(wrong_pool)

    pairs = []
    n_available = min(len(correct_pool), len(wrong_pool), n_pairs)
    for i in range(n_available):
        pairs.append({
            "correct_prompt": correct_pool[i]["prompt"],
            "wrong_prompt": wrong_pool[i]["prompt"],
            "correct_ans": wrong_pool[i]["correct_ans"],
            "rule": rule,
        })

    logger.info(f"Created {len(pairs)} rescue pairs for {rule} @ {n_shot}-shot")
    return pairs
