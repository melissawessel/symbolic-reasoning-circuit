"""Model loading and cache utilities."""

import torch
from transformer_lens import HookedTransformer

from src.config import MODEL_NAME, DEVICE, DTYPE


def load_model(device: str = DEVICE, dtype: torch.dtype = DTYPE) -> HookedTransformer:
    """Load Gemma-2 model with TransformerLens."""
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def hook_z_filter(name: str) -> bool:
    """Filter to cache only attention output (hook_z) activations."""
    return name.endswith("hook_z")


def get_answer_logits(logits: torch.Tensor, answer_token_id: int) -> float:
    """Get the logit for a specific answer token at the final position."""
    return logits[0, -1, answer_token_id].item()


def get_answer_prob(logits: torch.Tensor, answer_token_id: int) -> float:
    """Get the softmax probability for a specific answer token at the final position."""
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    return probs[answer_token_id].item()


def get_prediction(logits: torch.Tensor) -> int:
    """Get the argmax prediction at the final position."""
    return logits[0, -1, :].argmax(dim=-1).item()


def print_memory_usage():
    """Print current memory usage (MPS or CPU)."""
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"MPS memory allocated: {allocated:.2f} GB")
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss / 1e9
    print(f"Process RSS: {rss:.2f} GB")
