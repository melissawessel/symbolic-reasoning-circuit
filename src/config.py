"""Centralized configuration for the symbolic reasoning circuit project."""

import torch
import pathlib

# Model
MODEL_NAME = "gemma-2-2b"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.bfloat16

# Gemma-2-2b architecture
N_LAYERS = 26
N_HEADS = 8
N_KV_HEADS = 4
D_MODEL = 2304
D_HEAD = 256

# Task
SEPARATOR = "^"
EXAMPLE_SEP = "\n"
N_SHOT = 2  # default number of in-context examples
RULES = ["ABA", "ABB"]

# Shot-count sweep
SHOT_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10]

# Evaluation
N_EVAL_PROMPTS = 2000  # total (1000 per rule)
PROB_THRESHOLD = 0.9   # only use prompts model gets right with high confidence

# CMA
N_CMA_PAIRS = 200      # total context pairs (100 per rule direction)
N_PERMUTATIONS = 5000
P_VALUE_THRESHOLD = 0.05

# Paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VOCAB_DIR = DATA_DIR / "vocab"
PROMPTS_DIR = DATA_DIR / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
SHOT_SWEEP_DIR = RESULTS_DIR / "shot_sweep"


def get_shot_results_dir(n_shot: int) -> pathlib.Path:
    """Return results directory for a specific shot count."""
    return SHOT_SWEEP_DIR / f"{n_shot}shot"


def get_token_positions(has_bos: bool, n_shot: int = N_SHOT) -> dict:
    """Return position indices for an N-shot prompt.

    Prompt structure (without BOS), e.g. 2-shot:
      A1 ^ B1 ^ C1 \\n A2 ^ B2 ^ C2 \\n A3 ^ B3 ^
      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15

    Each in-context example occupies 6 tokens: A ^ B ^ C \\n
    The query occupies 4 tokens: A ^ B ^

    Total tokens without BOS: 6 * n_shot + 4
    With BOS prepended, add 1 to all positions.

    Returns:
        {
            "examples": [{"A": int, "sep1": int, "B": int, "sep2": int, "C": int, "newline": int}, ...],
            "query": {"A": int, "sep1": int, "B": int, "sep2": int},
            "total_len": int,
        }
    """
    offset = 1 if has_bos else 0

    examples = []
    for i in range(n_shot):
        base = offset + 6 * i
        examples.append({
            "A": base,
            "sep1": base + 1,
            "B": base + 2,
            "sep2": base + 3,
            "C": base + 4,
            "newline": base + 5,
        })

    query_base = offset + 6 * n_shot
    query = {
        "A": query_base,
        "sep1": query_base + 1,
        "B": query_base + 2,
        "sep2": query_base + 3,  # final position
    }

    return {
        "examples": examples,
        "query": query,
        "total_len": offset + 6 * n_shot + 4,
    }


def get_patch_positions(has_bos: bool, n_shot: int = N_SHOT) -> dict:
    """Return patch positions for each CMA type.

    Args:
        has_bos: Whether the model prepends a BOS token.
        n_shot: Number of in-context examples.

    Returns:
        Dict with keys "symbol_abstraction", "symbolic_induction", "retrieval",
        each mapping to a list of token position indices.
    """
    pos = get_token_positions(has_bos, n_shot)
    return {
        # Symbol abstraction: final content token of each in-context example
        "symbol_abstraction": [ex["C"] for ex in pos["examples"]],
        # Symbolic induction: final position (query separator)
        "symbolic_induction": [pos["query"]["sep2"]],
        # Retrieval: final position
        "retrieval": [pos["query"]["sep2"]],
    }


def get_token_labels(rule: str, has_bos: bool, n_shot: int = N_SHOT) -> list[str]:
    """Return human-readable labels for each token position.

    Args:
        rule: "ABA" or "ABB"
        has_bos: Whether the model prepends a BOS token.
        n_shot: Number of in-context examples.

    Returns:
        List of string labels, one per token position.
    """
    labels = []
    if has_bos:
        labels.append("[BOS]")

    for i in range(1, n_shot + 1):
        c_label = f"A{i}" if rule == "ABA" else f"B{i}"
        labels.extend([f"A{i}", "^", f"B{i}", "^", c_label, "\\n"])

    q = n_shot + 1
    labels.extend([f"A{q}", "^", f"B{q}", "^"])
    return labels
