"""QK and OV feature attribution using Gemma Scope residual stream SAEs.

Decomposes attention head computation into sparse feature interactions:
- QK attribution: which (query feature, key feature) pairs drive attention scores
- OV attribution: which source features map to which destination features
- Handoff analysis: feature overlap between adjacent circuit stages

Uses residual stream SAEs (gemma-scope-2b-pt-res-canonical) to decompose
the residual stream at layer inputs into sparse features, then traces these
features through W_Q, W_K, W_V, W_O weight matrices.

Reference: Anthropic, "Tracing Attention Computation Through Feature Interactions"
"""

import json
import logging
import math
from pathlib import Path

import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

from src.config import (
    N_HEADS, N_KV_HEADS, D_HEAD, D_MODEL,
    get_token_positions, SHOT_SWEEP_DIR, RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Gemma Scope residual stream SAE config
RES_SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
RES_SAE_WIDTH = "16k"  # default; 65k also available for all layers
AVAILABLE_WIDTHS = ("16k", "65k")  # widths with full layer coverage (0-25)

# Output directory
QK_OV_DIR = RESULTS_DIR / "qk_ov_attribution"


def compute_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute RMS normalization scale factor.

    Args:
        x: [..., d_model] tensor.
        eps: Epsilon for numerical stability.

    Returns:
        [..., 1] RMS scale factors.
    """
    return (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()


def get_kv_head_idx(head_idx: int) -> int:
    """Map query head index to its KV head index under GQA.

    Gemma-2-2b: 8 query heads, 4 KV heads. Heads 0,1 share KV 0; 2,3 share KV 1; etc.

    Note: TransformerLens duplicates KV heads so W_K/W_V have shape [n_heads, ...].
    You can index directly by head_idx. This function is kept for documentation
    and for code that works with the raw (non-duplicated) weights.
    """
    heads_per_kv = N_HEADS // N_KV_HEADS  # 2
    return head_idx // heads_per_kv


def load_residual_sae(layer: int, width: str = RES_SAE_WIDTH, device: str = "cpu") -> SAE:
    """Load a Gemma Scope residual stream SAE for a given layer.

    Args:
        layer: Residual stream layer index (0-25).
        width: SAE width — "16k" (d_sae=16384) or "65k" (d_sae=65536).
        device: Device to load onto.
    """
    sae_id = f"layer_{layer}/width_{width}/canonical"
    logger.info(f"Loading residual SAE: {RES_SAE_RELEASE} / {sae_id}")
    sae = SAE.from_pretrained(
        release=RES_SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )
    return sae


def load_significant_heads(shots_dir: Path) -> dict:
    """Load CMA-significant heads and build head→stage mapping.

    Returns:
        {(layer, head): {"stage": str, "score": float, "rules": list[str]}}
        where stage is one of: "symbol_abstraction", "symbolic_induction", "retrieval"
    """
    sig_path = shots_dir / "significant_heads.json"
    with open(sig_path) as f:
        sig_data = json.load(f)

    heads = {}
    for key, head_list in sig_data.items():
        # key format: "{stage}_base_{rule}"
        parts = key.rsplit("_base_", 1)
        stage = parts[0]
        rule = parts[1] if len(parts) > 1 else "unknown"

        for entry in head_list:
            lh = (entry["layer"], entry["head"])
            if lh not in heads or entry["score"] > heads[lh]["score"]:
                heads[lh] = {
                    "stage": stage,
                    "score": entry["score"],
                    "rules": [rule],
                }
            else:
                if rule not in heads[lh]["rules"]:
                    heads[lh]["rules"].append(rule)

    logger.info(f"Loaded {len(heads)} significant heads from {sig_path}")
    for stage in ["symbol_abstraction", "symbolic_induction", "retrieval"]:
        count = sum(1 for v in heads.values() if v["stage"] == stage)
        logger.info(f"  {stage}: {count} heads")

    return heads


def get_position_pairs(
    stage: str, has_bos: bool, n_shot: int, rule: str = "ABA",
) -> list[dict]:
    """Get (query_pos, key_pos) pairs for a given stage.

    Returns list of dicts with "query_pos" and "key_positions" keys.
    """
    pos = get_token_positions(has_bos, n_shot)

    if stage == "symbol_abstraction":
        # SA heads: query from C, keys are A and B of the same example
        pairs = []
        for ex in pos["examples"]:
            pairs.append({
                "query_pos": ex["C"],
                "key_positions": [ex["A"], ex["B"]],
            })
        return pairs

    elif stage == "symbolic_induction":
        # SI heads: query from final position, keys are all C positions
        final = pos["query"]["sep2"]
        c_positions = [ex["C"] for ex in pos["examples"]]
        return [{"query_pos": final, "key_positions": c_positions}]

    elif stage == "retrieval":
        # Retrieval heads: query from final, keys are answer token positions
        final = pos["query"]["sep2"]
        if rule == "ABA":
            # Answer is the A token — A positions in examples
            key_positions = [ex["A"] for ex in pos["examples"]]
        else:
            key_positions = [ex["B"] for ex in pos["examples"]]
        return [{"query_pos": final, "key_positions": key_positions}]

    else:
        raise ValueError(f"Unknown stage: {stage}")


def encode_residual_at_positions(
    sae: SAE,
    resid_tensor: torch.Tensor,
    positions: list[int],
) -> dict[int, torch.Tensor]:
    """Encode residual stream through SAE at specific token positions.

    Args:
        sae: Loaded residual stream SAE.
        resid_tensor: [n_prompts, seq_len, d_model] residual stream.
        positions: Token positions to encode.

    Returns:
        {pos: feature_activations [n_prompts, n_features]}
    """
    result = {}
    for pos in positions:
        x = resid_tensor[:, pos, :]  # [n_prompts, d_model]
        with torch.no_grad():
            feat_acts = sae.encode(x.to(sae.device).to(sae.dtype))
        result[pos] = feat_acts.cpu()
    return result


def compute_qk_interactions(
    model: HookedTransformer,
    feature_acts_q: torch.Tensor,
    feature_acts_k: torch.Tensor,
    sae_decoder: torch.Tensor,
    layer: int,
    head: int,
    top_k: int = 50,
    **kwargs,
) -> dict:
    """Compute pairwise QK feature interactions for one head.

    For each pair of active features (one query-side, one key-side), computes:
        interaction = act_q[i] * act_k[j] * (dec_i @ W_Q) . (dec_j @ W_K) / sqrt(d_head)

    Note: TransformerLens `from_pretrained` folds RMSNorm gamma into W_Q/W_K,
    so decoder directions can be projected directly through the (folded) weights.
    The per-prompt 1/rms(x) scalar is a common factor that doesn't affect rankings.

    Args:
        model: The HookedTransformer model.
        feature_acts_q: [n_prompts, n_features] query feature activations.
        feature_acts_k: [n_prompts, n_features] key feature activations.
        sae_decoder: [n_features, d_model] SAE decoder weight matrix.
        layer: Attention layer index.
        head: Query head index.
        top_k: Number of top interactions to return.

    Returns:
        Dict with "top_interactions" (list of triples) and "total_score" (sum).
    """
    n_prompts = feature_acts_q.shape[0]

    # Get weight matrices [d_model, d_head]
    # Note: TransformerLens duplicates KV heads, so W_K[head] works directly.
    # These already include folded RMSNorm gamma weights.
    W_Q = model.blocks[layer].attn.W_Q[head].detach().float().cpu()
    W_K = model.blocks[layer].attn.W_K[head].detach().float().cpu()
    scale = 1.0 / math.sqrt(D_HEAD)

    # Find features active in >5% of prompts on either side
    q_active_mask = (feature_acts_q > 0).float().mean(0) >= 0.05
    k_active_mask = (feature_acts_k > 0).float().mean(0) >= 0.05
    q_active_idxs = torch.where(q_active_mask)[0]
    k_active_idxs = torch.where(k_active_mask)[0]

    if len(q_active_idxs) == 0 or len(k_active_idxs) == 0:
        return {"top_interactions": [], "total_score": 0.0, "n_q_active": 0, "n_k_active": 0}

    # Precompute decoder projections through W_Q and W_K
    # (gamma is already folded into weights by TransformerLens)
    dec_q_proj = (sae_decoder[q_active_idxs].float() @ W_Q)  # [n_q_active, d_head]
    dec_k_proj = (sae_decoder[k_active_idxs].float() @ W_K)  # [n_k_active, d_head]

    # QK dot product matrix: [n_q_active, n_k_active]
    qk_matrix = (dec_q_proj @ dec_k_proj.T) * scale

    # Mean activations across prompts
    mean_acts_q = feature_acts_q[:, q_active_idxs].float().mean(0)  # [n_q_active]
    mean_acts_k = feature_acts_k[:, k_active_idxs].float().mean(0)  # [n_k_active]

    # Interaction = mean_act_q * mean_act_k * qk_dot
    interaction_matrix = mean_acts_q.unsqueeze(1) * mean_acts_k.unsqueeze(0) * qk_matrix

    # Total reconstructed score (sum of all interactions)
    total_score = interaction_matrix.sum().item()

    # Get top-k by absolute value
    flat = interaction_matrix.flatten()
    n_total = flat.shape[0]
    k = min(top_k, n_total)
    top_vals, top_flat_idxs = flat.abs().topk(k)

    top_interactions = []
    for idx in range(k):
        flat_idx = top_flat_idxs[idx].item()
        qi = flat_idx // len(k_active_idxs)
        ki = flat_idx % len(k_active_idxs)
        q_feat = q_active_idxs[qi].item()
        k_feat = k_active_idxs[ki].item()
        interaction_val = interaction_matrix[qi, ki].item()
        top_interactions.append({
            "query_feat": q_feat,
            "key_feat": k_feat,
            "interaction": interaction_val,
            "mean_act_q": mean_acts_q[qi].item(),
            "mean_act_k": mean_acts_k[ki].item(),
            "qk_dot": qk_matrix[qi, ki].item(),
        })

    return {
        "top_interactions": top_interactions,
        "total_score": total_score,
        "n_q_active": len(q_active_idxs),
        "n_k_active": len(k_active_idxs),
    }


def compute_ov_output_features(
    model: HookedTransformer,
    feature_acts_src: torch.Tensor,
    sae_decoder_src: torch.Tensor,
    sae_decoder_dest: torch.Tensor,
    layer: int,
    head: int,
    top_k: int = 50,
    **kwargs,
) -> dict:
    """Compute OV output features: what does this head write to the residual stream?

    For each active source feature i, computes the OV output vector:
        ov_i = dec_i @ W_V @ W_O  (d_model vector)
    Then projects onto destination SAE decoder directions:
        alignment(i, j) = act_src[i] * (ov_i . dec_dest_j)

    Note: TransformerLens `from_pretrained` folds RMSNorm gamma into W_V,
    so decoder directions project directly through the folded weights.

    Args:
        model: The HookedTransformer model.
        feature_acts_src: [n_prompts, n_features] source feature activations.
        sae_decoder_src: [n_features, d_model] source SAE decoder.
        sae_decoder_dest: [n_features_dest, d_model] destination SAE decoder.
        layer: Attention layer index.
        head: Query head index.
        top_k: Number of top alignments to return.

    Returns:
        Dict with "top_alignments" list and summary stats.
    """
    # Get weight matrices (TransformerLens duplicates KV heads)
    # These already include folded RMSNorm gamma weights.
    W_V = model.blocks[layer].attn.W_V[head].detach().float().cpu()  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head].detach().float().cpu()  # [d_head, d_model]

    # OV matrix: [d_model, d_model]
    W_OV = W_V @ W_O

    # Find active source features
    src_active_mask = (feature_acts_src > 0).float().mean(0) >= 0.05
    src_active_idxs = torch.where(src_active_mask)[0]

    if len(src_active_idxs) == 0:
        return {"top_alignments": [], "n_src_active": 0}

    # Compute OV output for each active source feature
    # (gamma already folded into W_V by TransformerLens)
    ov_outputs = sae_decoder_src[src_active_idxs].float() @ W_OV  # [n_src_active, d_model]

    # Project onto destination decoder directions
    # alignment_matrix[i, j] = ov_outputs[i] . dec_dest[j]
    # Shape: [n_src_active, n_dest_features]
    alignment_matrix = ov_outputs @ sae_decoder_dest.float().T

    # Weight by mean source activations
    mean_acts_src = feature_acts_src[:, src_active_idxs].float().mean(0)  # [n_src_active]
    weighted_alignment = mean_acts_src.unsqueeze(1) * alignment_matrix

    # Get top-k by absolute value
    flat = weighted_alignment.flatten()
    k = min(top_k, flat.shape[0])
    _, top_flat_idxs = flat.abs().topk(k)

    n_dest = sae_decoder_dest.shape[0]
    top_alignments = []
    for idx in range(k):
        flat_idx = top_flat_idxs[idx].item()
        si = flat_idx // n_dest
        di = flat_idx % n_dest
        src_feat = src_active_idxs[si].item()
        dest_feat = di
        top_alignments.append({
            "src_feat": src_feat,
            "dest_feat": dest_feat,
            "weighted_alignment": weighted_alignment[si, di].item(),
            "raw_alignment": alignment_matrix[si, di].item(),
            "mean_act_src": mean_acts_src[si].item(),
        })

    return {
        "top_alignments": top_alignments,
        "n_src_active": len(src_active_idxs),
    }


def validate_qk_reconstruction(
    feature_acts_q: torch.Tensor,
    feature_acts_k: torch.Tensor,
    sae_decoder: torch.Tensor,
    model: HookedTransformer,
    layer: int,
    head: int,
    actual_scores: torch.Tensor,
    resid_q: torch.Tensor | None = None,
    resid_k: torch.Tensor | None = None,
    sae_b_dec: torch.Tensor | None = None,
    **kwargs,
) -> dict:
    """Validate that feature interactions reconstruct the actual attention scores.

    Computes per-prompt: sum of all feature interactions vs actual attention score.
    When resid_q/resid_k are provided, applies RMSNorm correction:
        1. Reconstruct residual from SAE features (+bias)
        2. Divide by actual RMS (from true residual) to match the ln1 normalization
        3. Project through W_Q, W_K (which already include folded gamma) and dot product

    Note: TransformerLens `from_pretrained` folds RMSNorm gamma into W_Q/W_K.
    The ln1 at runtime only does x/rms(x). So we only need the per-prompt 1/rms
    correction, not a separate gamma multiplication.

    Args:
        feature_acts_q: [n_prompts, n_features]
        feature_acts_k: [n_prompts, n_features]
        sae_decoder: [n_features, d_model]
        model: HookedTransformer
        layer: int
        head: int
        actual_scores: [n_prompts] actual pre-softmax attention scores
        resid_q: Optional [n_prompts, d_model] actual residual at query positions
            (for computing RMS normalization factor).
        resid_k: Optional [n_prompts, d_model] actual residual at key positions.
        sae_b_dec: Optional [d_model] SAE decoder bias.

    Returns:
        Dict with correlation, mean_error, and per-prompt scores.
    """
    n_prompts = feature_acts_q.shape[0]
    scale = 1.0 / math.sqrt(D_HEAD)

    # W_Q/W_K already include folded RMSNorm gamma
    W_Q = model.blocks[layer].attn.W_Q[head].detach().float().cpu()
    W_K = model.blocks[layer].attn.W_K[head].detach().float().cpu()

    # Precompute per-prompt RMS normalization if residual is available
    use_rms = resid_q is not None and resid_k is not None
    if use_rms:
        rms_q_all = compute_rms(resid_q)  # [n_prompts, 1]
        rms_k_all = compute_rms(resid_k)  # [n_prompts, 1]

    reconstructed = torch.zeros(n_prompts)

    for p in range(n_prompts):
        acts_q = feature_acts_q[p].float()  # [n_features]
        acts_k = feature_acts_k[p].float()

        # Active features for this prompt
        q_active = torch.where(acts_q > 0)[0]
        k_active = torch.where(acts_k > 0)[0]

        if len(q_active) == 0 or len(k_active) == 0:
            continue

        # Reconstruct residual from SAE features: x_hat = sum(act_i * dec_i) + b_dec
        q_resid_hat = (acts_q[q_active].unsqueeze(1) * sae_decoder[q_active].float()).sum(0)
        k_resid_hat = (acts_k[k_active].unsqueeze(1) * sae_decoder[k_active].float()).sum(0)

        if sae_b_dec is not None:
            q_resid_hat = q_resid_hat + sae_b_dec.float()
            k_resid_hat = k_resid_hat + sae_b_dec.float()

        # Apply RMS normalization: x_norm = x_hat / rms(x_actual)
        # (gamma is already folded into W_Q/W_K by TransformerLens)
        if use_rms:
            q_resid_hat = q_resid_hat / rms_q_all[p]
            k_resid_hat = k_resid_hat / rms_k_all[p]

        # Project through W_Q, W_K (with folded gamma) and dot product
        q_total = q_resid_hat @ W_Q  # [d_head]
        k_total = k_resid_hat @ W_K  # [d_head]
        reconstructed[p] = (q_total @ k_total) * scale

    actual_np = actual_scores.float().cpu().numpy()
    recon_np = reconstructed.numpy()

    # Pearson correlation
    if np.std(actual_np) > 1e-8 and np.std(recon_np) > 1e-8:
        corr = float(np.corrcoef(actual_np, recon_np)[0, 1])
    else:
        corr = 0.0

    mean_error = float(np.mean(np.abs(actual_np - recon_np)))
    mean_ratio = float(np.mean(recon_np / (actual_np + 1e-8)))

    return {
        "correlation": corr,
        "mean_abs_error": mean_error,
        "mean_ratio": mean_ratio,
        "reconstructed_scores": recon_np.tolist(),
        "actual_scores": actual_np.tolist(),
    }


def compute_handoff(
    ov_results: dict,
    qk_results: dict,
    sae_layer_src: int,
    sae_layer_dest: int,
    top_k: int = 20,
) -> dict:
    """Find feature overlap between one stage's OV output and the next stage's QK input.

    Identifies features that appear as both:
    - Destination features in OV output (what upstream heads write)
    - Key features in QK interactions (what downstream heads read)

    Note: This only works when the SAE layers match (src OV dest = dest QK key).

    Args:
        ov_results: Dict with "top_alignments" from compute_ov_output_features.
        qk_results: Dict with "top_interactions" from compute_qk_interactions.
        sae_layer_src: SAE layer used for OV destination features.
        sae_layer_dest: SAE layer used for QK key features.
        top_k: Number of top overlapping features to return.

    Returns:
        Dict with overlapping features and their combined scores.
    """
    if sae_layer_src != sae_layer_dest:
        logger.warning(
            f"SAE layers don't match (OV dest={sae_layer_src}, QK key={sae_layer_dest}). "
            "Handoff analysis requires matching layers for meaningful comparison."
        )
        return {"overlapping_features": [], "note": "SAE layer mismatch"}

    # Collect destination features from OV output
    ov_dest_feats = {}
    for entry in ov_results.get("top_alignments", []):
        feat = entry["dest_feat"]
        score = abs(entry["weighted_alignment"])
        if feat not in ov_dest_feats or score > ov_dest_feats[feat]:
            ov_dest_feats[feat] = score

    # Collect key features from QK interactions
    qk_key_feats = {}
    for entry in qk_results.get("top_interactions", []):
        feat = entry["key_feat"]
        score = abs(entry["interaction"])
        if feat not in qk_key_feats or score > qk_key_feats[feat]:
            qk_key_feats[feat] = score

    # Find overlap
    shared = set(ov_dest_feats.keys()) & set(qk_key_feats.keys())
    overlapping = []
    for feat in shared:
        overlapping.append({
            "feature": feat,
            "ov_write_score": ov_dest_feats[feat],
            "qk_read_score": qk_key_feats[feat],
            "combined_score": ov_dest_feats[feat] * qk_key_feats[feat],
        })

    overlapping.sort(key=lambda x: x["combined_score"], reverse=True)
    overlapping = overlapping[:top_k]

    return {
        "overlapping_features": overlapping,
        "n_ov_dest_features": len(ov_dest_feats),
        "n_qk_key_features": len(qk_key_feats),
        "n_shared": len(shared),
        "jaccard": len(shared) / max(len(ov_dest_feats | qk_key_feats), 1),
    }


def compute_handoff_cosine(
    ov_results: dict,
    qk_results: dict,
    sae_decoder_ov_dest: torch.Tensor,
    sae_decoder_qk_key: torch.Tensor,
    sae_layer_ov_dest: int,
    sae_layer_qk_key: int,
    top_k: int = 20,
    cos_threshold: float = 0.3,
) -> dict:
    """Cross-layer handoff using decoder direction cosine similarity.

    For each OV destination feature, finds the closest QK key feature in a
    (potentially different) SAE layer by cosine similarity of decoder directions.
    This enables handoff analysis across non-adjacent layers where feature
    indices aren't directly comparable.

    Args:
        ov_results: Dict with "top_alignments" from compute_ov_output_features.
        qk_results: Dict with "top_interactions" from compute_qk_interactions.
        sae_decoder_ov_dest: [n_features, d_model] OV destination SAE decoder.
        sae_decoder_qk_key: [n_features, d_model] QK key SAE decoder.
        sae_layer_ov_dest: Layer index for OV destination SAE.
        sae_layer_qk_key: Layer index for QK key SAE.
        top_k: Number of top matches to return.
        cos_threshold: Minimum cosine similarity to count as a match.

    Returns:
        Dict with matched feature pairs and similarity scores.
    """
    # Collect destination features from OV output with scores
    ov_dest_feats = {}
    for entry in ov_results.get("top_alignments", []):
        feat = entry["dest_feat"]
        score = abs(entry["weighted_alignment"])
        if feat not in ov_dest_feats or score > ov_dest_feats[feat]:
            ov_dest_feats[feat] = score

    # Collect key features from QK interactions with scores
    qk_key_feats = {}
    for entry in qk_results.get("top_interactions", []):
        feat = entry["key_feat"]
        score = abs(entry["interaction"])
        if feat not in qk_key_feats or score > qk_key_feats[feat]:
            qk_key_feats[feat] = score

    if not ov_dest_feats or not qk_key_feats:
        return {
            "matches": [],
            "n_ov_dest_features": len(ov_dest_feats),
            "n_qk_key_features": len(qk_key_feats),
            "sae_layer_ov_dest": sae_layer_ov_dest,
            "sae_layer_qk_key": sae_layer_qk_key,
        }

    ov_feat_ids = sorted(ov_dest_feats.keys())
    qk_feat_ids = sorted(qk_key_feats.keys())

    # Get decoder directions and L2-normalize
    ov_dirs = sae_decoder_ov_dest[ov_feat_ids].float()  # [n_ov, d_model]
    qk_dirs = sae_decoder_qk_key[qk_feat_ids].float()  # [n_qk, d_model]

    ov_dirs_norm = ov_dirs / (ov_dirs.norm(dim=1, keepdim=True) + 1e-8)
    qk_dirs_norm = qk_dirs / (qk_dirs.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity matrix: [n_ov, n_qk]
    cos_sim = ov_dirs_norm @ qk_dirs_norm.T

    # For each OV dest feature, find best matching QK key feature
    matches = []
    for i, ov_feat in enumerate(ov_feat_ids):
        best_j = cos_sim[i].argmax().item()
        best_cos = cos_sim[i, best_j].item()

        if best_cos >= cos_threshold:
            qk_feat = qk_feat_ids[best_j]
            matches.append({
                "ov_dest_feat": ov_feat,
                "qk_key_feat": qk_feat,
                "cosine_similarity": best_cos,
                "ov_write_score": ov_dest_feats[ov_feat],
                "qk_read_score": qk_key_feats[qk_feat],
                "combined_score": (
                    ov_dest_feats[ov_feat] * qk_key_feats[qk_feat] * best_cos
                ),
                "ov_neuronpedia": neuronpedia_url(sae_layer_ov_dest, ov_feat),
                "qk_neuronpedia": neuronpedia_url(sae_layer_qk_key, qk_feat),
            })

    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    matches = matches[:top_k]

    # Also check bidirectional: for each QK key feat, best OV dest feat
    n_bidirectional = 0
    if matches:
        matched_ov = {m["ov_dest_feat"] for m in matches}
        matched_qk = {m["qk_key_feat"] for m in matches}
        for j, qk_feat in enumerate(qk_feat_ids):
            if qk_feat not in matched_qk:
                continue
            best_i = cos_sim[:, j].argmax().item()
            if ov_feat_ids[best_i] in matched_ov:
                n_bidirectional += 1

    return {
        "matches": matches,
        "n_matches_above_threshold": len(matches),
        "n_bidirectional": n_bidirectional,
        "n_ov_dest_features": len(ov_dest_feats),
        "n_qk_key_features": len(qk_key_feats),
        "sae_layer_ov_dest": sae_layer_ov_dest,
        "sae_layer_qk_key": sae_layer_qk_key,
        "cos_threshold": cos_threshold,
        "same_layer": sae_layer_ov_dest == sae_layer_qk_key,
    }


def neuronpedia_url(layer: int, feature_id: int, width: str = RES_SAE_WIDTH) -> str:
    """Generate Neuronpedia URL for a Gemma Scope residual feature."""
    return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-{width}/{feature_id}"


def add_neuronpedia_urls(results: dict, sae_layer: int) -> dict:
    """Add Neuronpedia URLs to QK or OV results."""
    if "top_interactions" in results:
        for entry in results["top_interactions"]:
            entry["query_neuronpedia"] = neuronpedia_url(sae_layer, entry["query_feat"])
            entry["key_neuronpedia"] = neuronpedia_url(sae_layer, entry["key_feat"])
    if "top_alignments" in results:
        for entry in results["top_alignments"]:
            entry["src_neuronpedia"] = neuronpedia_url(sae_layer, entry["src_feat"])
            # dest features use the destination layer SAE
            # (caller should pass appropriate layer)
    return results
