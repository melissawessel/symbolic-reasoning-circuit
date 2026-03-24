"""Causal feature intervention via SAE residual stream patching.

Validates QK/OV feature attributions by intervening on individual SAE features
in the residual stream and measuring the effect on attention scores and output
probabilities. Inspired by Anthropic's "Tracing Attention Computation Through
Feature Interactions" (2025).

Core intervention:
    x_modified = x + (scale - 1) * act_f * decoder[f]

At scale=0 the feature is ablated; at scale=1 it's the clean run.
Sweeping scale from negative to positive traces the causal effect.
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

from src.config import D_HEAD

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Results from a single feature intervention sweep."""

    layer: int
    head: int
    feature_id: int
    positions: list[int]  # all positions intervened on simultaneously
    sae_layer: int
    scales: list[float]
    attn_scores: dict[str, list[float]]  # {f"{query_pos}_{key_pos}": [score_per_scale]}
    answer_probs: dict[str, list[float]]  # {token_str: [prob_per_scale]}
    correct_ans_prob: list[float]  # P(correct_ans) per scale — rule-agnostic
    wrong_ans_prob: list[float]  # P(wrong_rule_ans) per scale
    baseline_attn_scores: dict[str, float]
    baseline_answer_probs: dict[str, float]
    baseline_prediction: str
    prompt_text: str
    feature_activation: float  # mean clean-run activation across positions
    intervention_side: str  # "query" or "key"
    correct_ans: str
    wrong_ans: str
    rule: str

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "feature_id": self.feature_id,
            "positions": self.positions,
            "sae_layer": self.sae_layer,
            "scales": self.scales,
            "attn_scores": self.attn_scores,
            "answer_probs": self.answer_probs,
            "correct_ans_prob": self.correct_ans_prob,
            "wrong_ans_prob": self.wrong_ans_prob,
            "baseline_attn_scores": self.baseline_attn_scores,
            "baseline_answer_probs": self.baseline_answer_probs,
            "baseline_prediction": self.baseline_prediction,
            "prompt_text": self.prompt_text,
            "feature_activation": self.feature_activation,
            "intervention_side": self.intervention_side,
            "correct_ans": self.correct_ans,
            "wrong_ans": self.wrong_ans,
            "rule": self.rule,
        }


@dataclass
class BatchInterventionResult:
    """Results aggregated across multiple prompts."""

    layer: int
    head: int
    feature_id: int
    position_type: str  # e.g. "all_C", "query_sep2"
    sae_layer: int
    intervention_side: str
    rule: str
    scales: list[float]
    n_prompts: int
    # Per-scale means across prompts (rule-agnostic)
    mean_attn_scores: dict[str, list[float]]
    mean_correct_ans_prob: list[float]
    mean_wrong_ans_prob: list[float]
    # Individual prompt results
    per_prompt: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "head": self.head,
            "feature_id": self.feature_id,
            "position_type": self.position_type,
            "sae_layer": self.sae_layer,
            "intervention_side": self.intervention_side,
            "rule": self.rule,
            "scales": self.scales,
            "n_prompts": self.n_prompts,
            "mean_attn_scores": self.mean_attn_scores,
            "mean_correct_ans_prob": self.mean_correct_ans_prob,
            "mean_wrong_ans_prob": self.mean_wrong_ans_prob,
            "per_prompt": self.per_prompt,
        }


def _make_resid_patch_hook(
    sae: SAE,
    feature_id: int,
    positions: int | list[int],
    scale: float,
):
    """Create a hook that patches a single SAE feature at one or more positions.

    The intervention modifies the residual stream at each position:
        x[pos] = x[pos] + (scale - 1) * act_f[pos] * decoder[f]

    This adds/removes the feature's contribution without full SAE encode/decode.
    """
    if isinstance(positions, int):
        positions = [positions]
    decoder_dir = sae.W_dec[feature_id].detach().float()  # [d_model]

    def hook_fn(activation, hook):
        # activation: [batch, seq, d_model] on model device (e.g. MPS)
        device = activation.device
        dec = decoder_dir.to(device)  # [d_model]

        for pos in positions:
            x = activation[:, pos, :].float()  # [batch, d_model]

            # Encode to get the feature's activation (SAE may be on CPU)
            with torch.no_grad():
                feat_acts = sae.encode(x.to(sae.device).to(sae.dtype))
            act_f = feat_acts[:, feature_id].float().to(device)  # [batch]

            # Patch: x_new = x + (scale - 1) * act_f * decoder[f]
            delta = (scale - 1) * act_f.unsqueeze(1) * dec.unsqueeze(0)
            activation[:, pos, :] = activation[:, pos, :] + delta.to(activation.dtype)

        return activation

    return hook_fn


def run_feature_intervention(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    layer: int,
    head: int,
    feature_id: int,
    positions: int | list[int],
    intervention_side: str = "key",
    query_positions: list[int] | None = None,
    key_positions: list[int] | None = None,
    answer_tokens: list[str] | None = None,
    correct_ans: str | None = None,
    wrong_ans: str | None = None,
    rule: str = "ABA",
    scales: list[float] | None = None,
) -> InterventionResult:
    """Run a sweep of feature interventions on a single prompt.

    Args:
        model: HookedTransformer model.
        sae: Residual stream SAE for the layer *input* (layer L-1 for layer L attention).
        prompt: The prompt string.
        layer: Attention layer to measure.
        head: Attention head to measure.
        feature_id: SAE feature index to intervene on.
        positions: Token position(s) at which to intervene. Can be a single int
            or a list of ints to intervene at all simultaneously.
        intervention_side: "query" or "key" — which side of the attention the positions are on.
        query_positions: Positions to track attention FROM (as query).
            If None, uses positions when side="query".
        key_positions: Positions to track attention TO (as key).
            If None, uses positions when side="key".
        answer_tokens: Additional token strings to track output probabilities for.
        correct_ans: The correct answer token for this prompt's rule.
        wrong_ans: The wrong-rule answer token. For ABA prompts this is query_b;
            for ABB prompts this is query_a. If None, inferred from the prompt.
        rule: "ABA" or "ABB" — used to infer wrong_ans if not provided.
        scales: List of scaling factors. Default: [-6, -5, ..., -1, 0, 0.5, 1, 1.5, 2].

    Returns:
        InterventionResult with per-scale attention scores and answer probabilities.
    """
    if isinstance(positions, int):
        positions = [positions]
    if scales is None:
        scales = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]

    tokens = model.to_tokens(prompt)

    # Infer correct/wrong answers from the prompt tokens if not provided.
    # Prompt ends with: ... query_a ^ query_b ^
    # With BOS: tokens[0]=BOS, last 4 content tokens are query_a, ^, query_b, ^
    if correct_ans is None or wrong_ans is None:
        seq_len = tokens.shape[1]
        query_a_tok = model.to_string([tokens[0, seq_len - 4].item()])
        query_b_tok = model.to_string([tokens[0, seq_len - 2].item()])
        if correct_ans is None:
            correct_ans = query_a_tok if rule == "ABA" else query_b_tok
        if wrong_ans is None:
            wrong_ans = query_b_tok if rule == "ABA" else query_a_tok

    correct_id = model.to_single_token(correct_ans)
    wrong_id = model.to_single_token(wrong_ans)

    # --- Baseline (clean) forward pass ---
    attn_hook = f"blocks.{layer}.attn.hook_attn_scores"
    resid_hook = f"blocks.{layer}.hook_resid_pre"

    with torch.no_grad():
        baseline_logits, baseline_cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n in (attn_hook, resid_hook),
        )

    # Get mean baseline feature activation across all intervention positions
    clean_acts = []
    for pos in positions:
        resid_at_pos = baseline_cache[resid_hook][0, pos, :].float()
        with torch.no_grad():
            feat_acts = sae.encode(resid_at_pos.unsqueeze(0).to(sae.device).to(sae.dtype))
        clean_acts.append(feat_acts[0, feature_id].item())
    clean_act = sum(clean_acts) / len(clean_acts)

    # Determine positions to track attention at
    if query_positions is None:
        query_positions = positions if intervention_side == "query" else []
    if key_positions is None:
        key_positions = positions if intervention_side == "key" else []

    # Baseline attention scores
    baseline_attn = baseline_cache[attn_hook][0, head]  # [seq, seq]
    baseline_attn_scores = {}
    for qp in query_positions:
        for kp in key_positions:
            key = f"{qp}_{kp}"
            baseline_attn_scores[key] = baseline_attn[qp, kp].item()

    # Baseline answer probabilities
    baseline_probs = torch.softmax(baseline_logits[0, -1, :].float(), dim=-1)
    baseline_pred_id = baseline_logits[0, -1, :].argmax().item()
    baseline_prediction = model.to_string([baseline_pred_id])

    # Always track correct and wrong answer; optionally also track extras
    all_answer_tokens = [correct_ans, wrong_ans]
    if answer_tokens:
        for tok in answer_tokens:
            if tok not in all_answer_tokens:
                all_answer_tokens.append(tok)

    baseline_answer_probs = {}
    for tok_str in all_answer_tokens:
        tid = model.to_single_token(tok_str)
        baseline_answer_probs[tok_str] = baseline_probs[tid].item()

    del baseline_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Intervention sweep ---
    attn_scores_per_scale = {k: [] for k in baseline_attn_scores}
    answer_probs_per_scale = {tok: [] for tok in all_answer_tokens}
    correct_ans_prob_per_scale = []
    wrong_ans_prob_per_scale = []

    for s in scales:
        if s == 1.0:
            # No intervention needed — use baseline
            for k, v in baseline_attn_scores.items():
                attn_scores_per_scale[k].append(v)
            for tok, v in baseline_answer_probs.items():
                answer_probs_per_scale[tok].append(v)
            correct_ans_prob_per_scale.append(baseline_probs[correct_id].item())
            wrong_ans_prob_per_scale.append(baseline_probs[wrong_id].item())
            continue

        patch_hook_fn = _make_resid_patch_hook(sae, feature_id, positions, s)

        # Capture attention scores via a second hook
        captured_attn = {}
        def _capture_attn(activation, hook, _store=captured_attn):
            _store["scores"] = activation.detach().cpu()
            return activation

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (resid_hook, patch_hook_fn),
                    (attn_hook, _capture_attn),
                ],
            )

        patched_attn = captured_attn["scores"][0, head]  # [seq, seq]
        for qp in query_positions:
            for kp in key_positions:
                key = f"{qp}_{kp}"
                attn_scores_per_scale[key].append(patched_attn[qp, kp].item())

        patched_probs = torch.softmax(patched_logits[0, -1, :].float(), dim=-1)
        for tok_str in all_answer_tokens:
            tid = model.to_single_token(tok_str)
            answer_probs_per_scale[tok_str].append(patched_probs[tid].item())

        correct_ans_prob_per_scale.append(patched_probs[correct_id].item())
        wrong_ans_prob_per_scale.append(patched_probs[wrong_id].item())

        del captured_attn
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return InterventionResult(
        layer=layer,
        head=head,
        feature_id=feature_id,
        positions=positions,
        sae_layer=layer - 1,
        scales=scales,
        attn_scores=attn_scores_per_scale,
        answer_probs=answer_probs_per_scale,
        correct_ans_prob=correct_ans_prob_per_scale,
        wrong_ans_prob=wrong_ans_prob_per_scale,
        baseline_attn_scores=baseline_attn_scores,
        baseline_answer_probs=baseline_answer_probs,
        baseline_prediction=baseline_prediction,
        prompt_text=prompt,
        feature_activation=clean_act,
        intervention_side=intervention_side,
        correct_ans=correct_ans,
        wrong_ans=wrong_ans,
        rule=rule,
    )


def run_batch_intervention(
    model: HookedTransformer,
    sae: SAE,
    prompts: list[dict],
    layer: int,
    head: int,
    feature_id: int,
    positions: int | list[int],
    position_type: str = "",
    intervention_side: str = "key",
    query_positions: list[int] | None = None,
    key_positions: list[int] | None = None,
    rule: str = "ABA",
    scales: list[float] | None = None,
    min_feature_activation: float = 0.0,
) -> BatchInterventionResult:
    """Run feature intervention across multiple prompts and aggregate.

    Aggregates using rule-agnostic metrics (correct_ans_prob, wrong_ans_prob)
    so that prompts with different token identities can be compared.

    Args:
        prompts: List of prompt dicts with "prompt" and "correct_ans" fields.
        positions: Token position(s) to intervene at simultaneously.
        position_type: Label for the positions (e.g. "all_C", "query_sep2").
        rule: "ABA" or "ABB" — determines which query token is correct/wrong.
        min_feature_activation: Skip prompts where the mean feature activation is below this.
        (other args same as run_feature_intervention)

    Returns:
        BatchInterventionResult with per-scale means and per-prompt details.
    """
    if scales is None:
        scales = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]

    per_prompt = []
    skipped = 0

    for p in prompts:
        result = run_feature_intervention(
            model=model,
            sae=sae,
            prompt=p["prompt"],
            layer=layer,
            head=head,
            feature_id=feature_id,
            positions=positions,
            intervention_side=intervention_side,
            query_positions=query_positions,
            key_positions=key_positions,
            correct_ans=p.get("correct_ans"),
            rule=rule,
            scales=scales,
        )

        if result.feature_activation < min_feature_activation:
            skipped += 1
            continue

        per_prompt.append(result.to_dict())

    if skipped > 0:
        logger.info(f"Skipped {skipped}/{len(prompts)} prompts (feature activation below {min_feature_activation})")

    n = len(per_prompt)
    if n == 0:
        logger.warning("No prompts passed the activation threshold")
        return BatchInterventionResult(
            layer=layer, head=head, feature_id=feature_id,
            position_type=position_type, sae_layer=layer - 1,
            intervention_side=intervention_side, rule=rule, scales=scales,
            n_prompts=0, mean_attn_scores={}, mean_correct_ans_prob=[],
            mean_wrong_ans_prob=[],
        )

    # Aggregate attention scores (position keys are shared across prompts)
    attn_keys = set(per_prompt[0]["attn_scores"].keys())
    for pp in per_prompt[1:]:
        attn_keys &= set(pp["attn_scores"].keys())

    mean_attn = {}
    for k in attn_keys:
        vals = np.array([pp["attn_scores"][k] for pp in per_prompt])  # [n, n_scales]
        mean_attn[k] = vals.mean(axis=0).tolist()

    # Aggregate correct/wrong answer probs (rule-agnostic, works across different tokens)
    correct_vals = np.array([pp["correct_ans_prob"] for pp in per_prompt])  # [n, n_scales]
    wrong_vals = np.array([pp["wrong_ans_prob"] for pp in per_prompt])

    return BatchInterventionResult(
        layer=layer,
        head=head,
        feature_id=feature_id,
        position_type=position_type,
        sae_layer=layer - 1,
        intervention_side=intervention_side,
        rule=rule,
        scales=scales,
        n_prompts=n,
        mean_attn_scores=mean_attn,
        mean_correct_ans_prob=correct_vals.mean(axis=0).tolist(),
        mean_wrong_ans_prob=wrong_vals.mean(axis=0).tolist(),
        per_prompt=per_prompt,
    )
