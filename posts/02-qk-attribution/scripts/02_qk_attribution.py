"""QK and OV feature attribution for all CMA-significant heads.

Computes feature-level decompositions using Gemma Scope residual stream SAEs:
- QK: which (query feature, key feature) pairs drive attention at each head
- OV: which source features map to which destination features
- Handoff: feature overlap between adjacent circuit stages (SA→SI, SI→Ret)

Loads all CMA-significant heads from the significant_heads.json for the
specified shot count, processes them data-driven by stage.

Results are saved under results/qk_ov_attribution/width_{width}/.

Usage:
    python posts/02-qk-attribution/scripts/02_qk_attribution.py --shots 10
    python posts/02-qk-attribution/scripts/02_qk_attribution.py --shots 10 --width 65k
    python posts/02-qk-attribution/scripts/02_qk_attribution.py --shots 10 --mode qk --n-prompts 50

Prerequisite: Run posts/02-qk-attribution/scripts/01_generate_eval_prompts.py --shots 10 first.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from src.config import (
    DEVICE, PROMPTS_DIR, SHOT_SWEEP_DIR,
    get_token_positions,
)
from src.model_utils import load_model, get_prediction, print_memory_usage
from src.qk_ov_attribution import (
    load_residual_sae,
    load_significant_heads,
    get_position_pairs,
    encode_residual_at_positions,
    compute_qk_interactions,
    compute_ov_output_features,
    validate_qk_reconstruction,
    compute_handoff,
    compute_handoff_cosine,
    add_neuronpedia_urls,
    neuronpedia_url,
    QK_OV_DIR,
    AVAILABLE_WIDTHS,
    RES_SAE_WIDTH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cache_residual_streams(model, prompts, target_layers, cache_attn_scores=True):
    """Run model on prompts and cache residual streams at target layers.

    Args:
        model: HookedTransformer model.
        prompts: List of prompt dicts with "prompt" key.
        target_layers: Set of layer indices to cache hook_resid_pre.
        cache_attn_scores: Also cache hook_attn_scores for validation.

    Returns:
        resid_cache: {layer: tensor [n_prompts, seq_len, d_model]}
        attn_cache: {layer: tensor [n_prompts, n_heads, seq_len, seq_len]} or None
    """
    def name_filter(name):
        for layer in target_layers:
            if name == f"blocks.{layer}.hook_resid_pre":
                return True
            if cache_attn_scores and name == f"blocks.{layer}.attn.hook_attn_scores":
                return True
        return False

    resid_cache = {l: [] for l in target_layers}
    attn_cache = {l: [] for l in target_layers} if cache_attn_scores else None

    for p in tqdm(prompts, desc="Caching residual streams"):
        tokens = model.to_tokens(p["prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=name_filter)

        for layer in target_layers:
            resid_cache[layer].append(
                cache[f"blocks.{layer}.hook_resid_pre"].cpu()
            )
            if cache_attn_scores:
                attn_cache[layer].append(
                    cache[f"blocks.{layer}.attn.hook_attn_scores"].cpu()
                )

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    resid_cache = {l: torch.cat(resid_cache[l], dim=0) for l in target_layers}
    if cache_attn_scores:
        attn_cache = {l: torch.cat(attn_cache[l], dim=0) for l in target_layers}

    return resid_cache, attn_cache


def run_qk_attribution(
    model, sig_heads, resid_cache, attn_cache,
    n_shot, rule, out_dir, top_k=50, width=RES_SAE_WIDTH,
):
    """Run QK attribution for all significant heads."""
    has_bos = True
    qk_dir = out_dir / "qk"
    qk_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    all_validations = {}

    # Group heads by layer to share SAE loading
    layer_heads = {}
    for (layer, head), info in sig_heads.items():
        layer_heads.setdefault(layer, []).append((head, info))

    for layer in sorted(layer_heads.keys()):
        # SAE for layer L's input = layer L-1's residual stream
        sae_layer = layer - 1
        if sae_layer < 0:
            logger.warning(f"Skipping layer {layer} (no SAE for layer -1)")
            continue

        if layer not in resid_cache:
            logger.warning(f"Layer {layer} not in residual cache, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer}: loading SAE for layer {sae_layer} (width={width})")
        sae = load_residual_sae(sae_layer, width=width, device="cpu")
        sae_decoder = sae.W_dec.detach().cpu()  # [n_features, d_model]
        sae_b_dec = sae.b_dec.detach().float().cpu()  # [d_model]

        resid = resid_cache[layer]  # [n_prompts, seq_len, d_model]

        # Precompute all position encodings for this layer
        all_positions = set()
        for head, info in layer_heads[layer]:
            for pair in get_position_pairs(info["stage"], has_bos, n_shot, rule):
                all_positions.add(pair["query_pos"])
                all_positions.update(pair["key_positions"])

        logger.info(f"  Encoding {len(all_positions)} positions through SAE...")
        feat_cache = encode_residual_at_positions(sae, resid, sorted(all_positions))

        for head, info in layer_heads[layer]:
            stage = info["stage"]
            score = info["score"]
            logger.info(f"  L{layer}H{head} ({stage}, CMA={score:.2f})")

            pos_pairs = get_position_pairs(stage, has_bos, n_shot, rule)
            head_results = {
                "head": [layer, head],
                "stage": stage,
                "cma_score": score,
                "n_shot": n_shot,
                "rule": rule,
                "sae_layer": sae_layer,
                "per_key_pos": {},
            }

            all_recon_corrs = []

            for pair in pos_pairs:
                q_pos = pair["query_pos"]
                for k_pos in pair["key_positions"]:
                    qk_result = compute_qk_interactions(
                        model=model,
                        feature_acts_q=feat_cache[q_pos],
                        feature_acts_k=feat_cache[k_pos],
                        sae_decoder=sae_decoder,
                        layer=layer,
                        head=head,
                        top_k=top_k,
                    )

                    # Add Neuronpedia URLs
                    for entry in qk_result["top_interactions"]:
                        entry["query_neuronpedia"] = neuronpedia_url(
                            sae_layer, entry["query_feat"], width
                        )
                        entry["key_neuronpedia"] = neuronpedia_url(
                            sae_layer, entry["key_feat"], width
                        )

                    # Validation against actual attention scores
                    if attn_cache is not None and layer in attn_cache:
                        actual = attn_cache[layer][:, head, q_pos, k_pos]
                        val_result = validate_qk_reconstruction(
                            feature_acts_q=feat_cache[q_pos],
                            feature_acts_k=feat_cache[k_pos],
                            sae_decoder=sae_decoder,
                            model=model,
                            layer=layer,
                            head=head,
                            actual_scores=actual,
                            resid_q=resid[:, q_pos, :],
                            resid_k=resid[:, k_pos, :],
                            sae_b_dec=sae_b_dec,
                        )
                        qk_result["reconstruction_corr"] = val_result["correlation"]
                        qk_result["reconstruction_error"] = val_result["mean_abs_error"]
                        all_recon_corrs.append(val_result["correlation"])

                    key = f"q{q_pos}_k{k_pos}"
                    head_results["per_key_pos"][key] = qk_result

            # Summary stats
            if all_recon_corrs:
                head_results["mean_reconstruction_corr"] = float(np.mean(all_recon_corrs))
                all_validations[f"L{layer}H{head}"] = {
                    "mean_corr": float(np.mean(all_recon_corrs)),
                    "min_corr": float(np.min(all_recon_corrs)),
                    "max_corr": float(np.max(all_recon_corrs)),
                    "n_pairs": len(all_recon_corrs),
                }

            n_pairs = sum(
                len(v.get("top_interactions", []))
                for v in head_results["per_key_pos"].values()
            )
            logger.info(f"    {n_pairs} total interaction entries across "
                       f"{len(head_results['per_key_pos'])} position pairs")

            out_path = qk_dir / f"L{layer}H{head}.json"
            with open(out_path, "w") as f:
                json.dump(head_results, f, indent=2)

        del sae, sae_decoder, feat_cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save validation summary
    with open(val_dir / "reconstruction.json", "w") as f:
        json.dump(all_validations, f, indent=2)

    return all_validations


def run_ov_attribution(
    model, sig_heads, resid_cache,
    n_shot, rule, out_dir, top_k=50, width=RES_SAE_WIDTH,
):
    """Run OV attribution for all significant heads."""
    has_bos = True
    ov_dir = out_dir / "ov"
    ov_dir.mkdir(parents=True, exist_ok=True)

    # Group heads by layer
    layer_heads = {}
    for (layer, head), info in sig_heads.items():
        layer_heads.setdefault(layer, []).append((head, info))

    for layer in sorted(layer_heads.keys()):
        sae_layer_src = layer - 1
        sae_layer_dest = layer

        if sae_layer_src < 0:
            logger.warning(f"Skipping layer {layer} OV (no SAE for layer -1)")
            continue

        if layer not in resid_cache:
            logger.warning(f"Layer {layer} not in residual cache, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer}: loading SAEs for layers {sae_layer_src} (src) and {sae_layer_dest} (dest) (width={width})")

        sae_src = load_residual_sae(sae_layer_src, width=width, device="cpu")
        sae_dest = load_residual_sae(sae_layer_dest, width=width, device="cpu")

        sae_decoder_src = sae_src.W_dec.detach().cpu()
        sae_decoder_dest = sae_dest.W_dec.detach().cpu()

        resid = resid_cache[layer]

        # Determine source positions for OV analysis
        # SA heads: OV output at C positions (what they write at the third element)
        # SI/Ret heads: OV output at the position they attend FROM (final position)
        for head, info in layer_heads[layer]:
            stage = info["stage"]
            logger.info(f"  L{layer}H{head} ({stage}) OV analysis")

            pos_pairs = get_position_pairs(stage, has_bos, n_shot, rule)

            head_results = {
                "head": [layer, head],
                "stage": stage,
                "cma_score": info["score"],
                "sae_layer_src": sae_layer_src,
                "sae_layer_dest": sae_layer_dest,
                "per_source_pos": {},
            }

            # For SA: source positions are the C positions (query positions in SA pairs)
            # For SI/Ret: source positions are the key positions (what they attend to)
            if stage == "symbol_abstraction":
                src_positions = [pair["query_pos"] for pair in pos_pairs]
            else:
                # SI/Ret heads attend FROM final, TO key positions
                # The OV output is what gets written at the destination (final pos)
                # using info from the source (key positions)
                # Actually: the head reads from key positions via V, writes via O to query pos
                # So for OV: source = what the head attends to (key positions)
                src_positions = []
                for pair in pos_pairs:
                    src_positions.extend(pair["key_positions"])
                src_positions = sorted(set(src_positions))

            # Encode source positions
            feat_cache = encode_residual_at_positions(sae_src, resid, src_positions)

            for src_pos in src_positions:
                ov_result = compute_ov_output_features(
                    model=model,
                    feature_acts_src=feat_cache[src_pos],
                    sae_decoder_src=sae_decoder_src,
                    sae_decoder_dest=sae_decoder_dest,
                    layer=layer,
                    head=head,
                    top_k=top_k,
                )

                # Add Neuronpedia URLs
                for entry in ov_result["top_alignments"]:
                    entry["src_neuronpedia"] = neuronpedia_url(
                        sae_layer_src, entry["src_feat"], width
                    )
                    entry["dest_neuronpedia"] = neuronpedia_url(
                        sae_layer_dest, entry["dest_feat"], width
                    )

                head_results["per_source_pos"][str(src_pos)] = ov_result

            out_path = ov_dir / f"L{layer}H{head}.json"
            with open(out_path, "w") as f:
                json.dump(head_results, f, indent=2)

            del feat_cache

        del sae_src, sae_dest, sae_decoder_src, sae_decoder_dest
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_handoff_analysis(out_dir, sig_heads, width=RES_SAE_WIDTH):
    """Compute feature handoff between adjacent circuit stages.

    Uses two methods:
    - Same-layer pairs: direct feature index matching (original method)
    - Cross-layer pairs: decoder direction cosine similarity (new method)
    """
    ov_dir = out_dir / "ov"
    qk_dir = out_dir / "qk"

    if not ov_dir.exists() or not qk_dir.exists():
        logger.warning("Need both QK and OV results for handoff analysis")
        return

    # Organize heads by stage
    stage_heads = {"symbol_abstraction": [], "symbolic_induction": [], "retrieval": []}
    for (layer, head), info in sig_heads.items():
        stage_heads[info["stage"]].append((layer, head))

    # Cache loaded SAE decoders to avoid reloading
    sae_decoder_cache = {}

    def get_sae_decoder(sae_layer):
        if sae_layer not in sae_decoder_cache:
            sae = load_residual_sae(sae_layer, width=width, device="cpu")
            sae_decoder_cache[sae_layer] = sae.W_dec.detach().cpu()
            del sae
        return sae_decoder_cache[sae_layer]

    def compute_stage_handoff(upstream_heads, downstream_heads, upstream_label, downstream_label):
        """Compute handoffs between two stages."""
        handoffs = []

        for up_l, up_h in upstream_heads:
            ov_path = ov_dir / f"L{up_l}H{up_h}.json"
            if not ov_path.exists():
                continue
            with open(ov_path) as f:
                ov_data = json.load(f)

            sae_layer_ov_dest = ov_data.get("sae_layer_dest", -1)

            for dn_l, dn_h in downstream_heads:
                qk_path = qk_dir / f"L{dn_l}H{dn_h}.json"
                if not qk_path.exists():
                    continue
                with open(qk_path) as f:
                    qk_data = json.load(f)

                sae_layer_qk_key = qk_data.get("sae_layer", -1)

                # Aggregate results across positions
                all_qk_interactions = []
                for key, pair_data in qk_data.get("per_key_pos", {}).items():
                    all_qk_interactions.extend(pair_data.get("top_interactions", []))

                all_ov_alignments = []
                for pos, pos_data in ov_data.get("per_source_pos", {}).items():
                    all_ov_alignments.extend(pos_data.get("top_alignments", []))

                if sae_layer_ov_dest == sae_layer_qk_key:
                    # Same layer: use direct feature matching (precise)
                    handoff = compute_handoff(
                        {"top_alignments": all_ov_alignments},
                        {"top_interactions": all_qk_interactions},
                        sae_layer_src=sae_layer_ov_dest,
                        sae_layer_dest=sae_layer_qk_key,
                    )
                    handoff["method"] = "exact_match"
                else:
                    # Cross-layer: use cosine similarity of decoder directions
                    if sae_layer_ov_dest < 0 or sae_layer_qk_key < 0:
                        handoff = {"matches": [], "note": "invalid SAE layers"}
                    else:
                        dec_ov = get_sae_decoder(sae_layer_ov_dest)
                        dec_qk = get_sae_decoder(sae_layer_qk_key)
                        handoff = compute_handoff_cosine(
                            {"top_alignments": all_ov_alignments},
                            {"top_interactions": all_qk_interactions},
                            sae_decoder_ov_dest=dec_ov,
                            sae_decoder_qk_key=dec_qk,
                            sae_layer_ov_dest=sae_layer_ov_dest,
                            sae_layer_qk_key=sae_layer_qk_key,
                        )
                        handoff["method"] = "cosine_similarity"

                handoff[f"{upstream_label}_head"] = [up_l, up_h]
                handoff[f"{downstream_label}_head"] = [dn_l, dn_h]
                handoffs.append(handoff)

        return handoffs

    # SA → SI handoff
    logger.info("\nComputing SA → SI handoff...")
    sa_si_handoffs = compute_stage_handoff(
        stage_heads["symbol_abstraction"],
        stage_heads["symbolic_induction"],
        "sa", "si",
    )
    n_exact = sum(1 for h in sa_si_handoffs if h.get("method") == "exact_match")
    n_cosine = sum(1 for h in sa_si_handoffs if h.get("method") == "cosine_similarity")
    logger.info(f"  {len(sa_si_handoffs)} SA→SI pairs ({n_exact} exact, {n_cosine} cosine)")

    with open(out_dir / "handoff_sa_si.json", "w") as f:
        json.dump(sa_si_handoffs, f, indent=2)

    # SI → Ret handoff
    logger.info("Computing SI → Ret handoff...")
    si_ret_handoffs = compute_stage_handoff(
        stage_heads["symbolic_induction"],
        stage_heads["retrieval"],
        "si", "ret",
    )
    n_exact = sum(1 for h in si_ret_handoffs if h.get("method") == "exact_match")
    n_cosine = sum(1 for h in si_ret_handoffs if h.get("method") == "cosine_similarity")
    logger.info(f"  {len(si_ret_handoffs)} SI→Ret pairs ({n_exact} exact, {n_cosine} cosine)")

    with open(out_dir / "handoff_si_ret.json", "w") as f:
        json.dump(si_ret_handoffs, f, indent=2)

    # Clean up SAE cache
    sae_decoder_cache.clear()


def main():
    parser = argparse.ArgumentParser(
        description="QK/OV feature attribution for CMA-significant heads"
    )
    parser.add_argument(
        "--shots", type=int, default=10,
        help="Shot count (default: 10)"
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=["qk", "ov", "both"],
        help="Which attribution to compute (default: both)"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=100,
        help="Number of prompts to process (default: 100)"
    )
    parser.add_argument(
        "--rule", type=str, default="ABA", choices=["ABA", "ABB"],
        help="Rule to analyze (default: ABA)"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Number of top interactions to save per position pair (default: 50)"
    )
    parser.add_argument(
        "--width", type=str, default=RES_SAE_WIDTH,
        choices=list(AVAILABLE_WIDTHS),
        help=f"SAE width (default: {RES_SAE_WIDTH}). '65k' gives 4x more features."
    )
    args = parser.parse_args()

    n_shot = args.shots
    rule = args.rule.upper()
    width = args.width

    # Load prompts
    prompt_file = PROMPTS_DIR / f"eval_{rule.lower()}_{n_shot}shot_prompts.json"
    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        logger.error(f"Run: python scripts/13_generate_eval_prompts.py --shots {n_shot}")
        sys.exit(1)

    with open(prompt_file) as f:
        all_prompts = json.load(f)

    # Filter to correct prompts only
    correct_prompts = [p for p in all_prompts if p.get("predicted_correct", True)]
    logger.info(f"Loaded {len(all_prompts)} prompts, {len(correct_prompts)} correct")

    # Sample
    rng = np.random.default_rng(42)
    n_sample = min(args.n_prompts, len(correct_prompts))
    sample_idxs = rng.choice(len(correct_prompts), n_sample, replace=False)
    prompts = [correct_prompts[i] for i in sample_idxs]
    logger.info(f"Sampled {n_sample} prompts")

    # Load significant heads
    shots_dir = SHOT_SWEEP_DIR / f"{n_shot}shot"
    sig_heads = load_significant_heads(shots_dir)

    # Determine which layers we need residual streams for
    target_layers = set()
    for (layer, head) in sig_heads:
        target_layers.add(layer)
    logger.info(f"Target layers: {sorted(target_layers)}")

    # Load model
    logger.info("Loading model...")
    model = load_model()
    print_memory_usage()

    # Cache residual streams
    start = time.time()
    resid_cache, attn_cache = cache_residual_streams(
        model, prompts, target_layers, cache_attn_scores=(args.mode in ("qk", "both"))
    )
    cache_time = time.time() - start
    logger.info(f"Caching took {cache_time:.0f}s")
    for l in sorted(target_layers):
        logger.info(f"  Layer {l}: resid shape {list(resid_cache[l].shape)}")
    print_memory_usage()

    # Output directory — separate subdirectory per SAE width and rule
    out_dir = QK_OV_DIR / f"width_{width}" / rule.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Save config
    config = {
        "n_shot": n_shot,
        "rule": rule,
        "n_prompts": n_sample,
        "mode": args.mode,
        "top_k": args.top_k,
        "sae_width": width,
        "significant_heads": {
            f"L{l}H{h}": {"stage": info["stage"], "score": info["score"]}
            for (l, h), info in sig_heads.items()
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run QK attribution
    if args.mode in ("qk", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING QK ATTRIBUTION")
        logger.info("=" * 60)
        start = time.time()
        validations = run_qk_attribution(
            model, sig_heads, resid_cache, attn_cache,
            n_shot, rule, out_dir, args.top_k, width=width,
        )
        qk_time = time.time() - start
        logger.info(f"\nQK attribution took {qk_time:.0f}s")
        if validations:
            corrs = [v["mean_corr"] for v in validations.values()]
            logger.info(f"Reconstruction correlations: "
                       f"mean={np.mean(corrs):.3f}, "
                       f"min={np.min(corrs):.3f}, "
                       f"max={np.max(corrs):.3f}")

    # Run OV attribution
    if args.mode in ("ov", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING OV ATTRIBUTION")
        logger.info("=" * 60)
        start = time.time()
        run_ov_attribution(
            model, sig_heads, resid_cache,
            n_shot, rule, out_dir, args.top_k, width=width,
        )
        ov_time = time.time() - start
        logger.info(f"\nOV attribution took {ov_time:.0f}s")

    # Handoff analysis (needs both QK and OV)
    if args.mode == "both":
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING HANDOFF ANALYSIS")
        logger.info("=" * 60)
        run_handoff_analysis(out_dir, sig_heads, width=width)

    logger.info("\nDone! Results saved to: " + str(out_dir))
    print_memory_usage()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
