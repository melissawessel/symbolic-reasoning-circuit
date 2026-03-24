import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, json, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Causal Feature Intervention — L14H0

    Visualize batch causal intervention results for SAE features at L14H0
    (the SI decision head). Data produced by `scripts/03_batch_causal_intervention.py`.

    Three artifacts:
    1. **Sample suppress** — highest-activation correct prompt, feature suppressed
    2. **Sample rescue** — highest-activation wrong prompt, feature amplified
    3. **Batch summary table** — aggregate statistics
    """)
    return


@app.cell
def _(Path, mo):
    _results_dir = Path("results/causal_interventions/2shot")
    _files = sorted(_results_dir.glob("*.json"))

    # Build dropdown options from available result files
    _options = {}
    for _f in _files:
        _options[_f.stem] = str(_f)

    file_picker = mo.ui.dropdown(
        options=_options,
        label="Select batch result",
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, json, mo):
    mo.stop(file_picker.value is None, mo.md("*Select a result file above.*"))

    with open(file_picker.value) as _fh:
        batch_data = json.load(_fh)

    _feat = batch_data["feature_id"]
    _rule = batch_data["rule"]
    _layer = batch_data["layer"]
    _head = batch_data["head"]
    _url = batch_data.get("neuronpedia_url", "")

    mo.md(f"""
    **L{_layer}H{_head} — Feature {_feat} — {_rule} rule**
    {"[View on Neuronpedia](" + _url + ")" if _url else ""}

    Correct batch: {batch_data['correct_batch']['n_prompts']} prompts |
    Wrong batch: {batch_data['wrong_batch']['n_prompts']} prompts
    """)
    return (batch_data,)


@app.cell
def _(batch_data, mo):
    _suppress = batch_data.get("top_suppress_example", {})
    _text = _suppress.get("prompt_text", "")
    mo.md(f"""### Suppress example prompt
    **Correct answer:** {_suppress.get('correct_ans', '?')} | **Wrong answer:** {_suppress.get('wrong_ans', '?')} | **Baseline prediction:** {_suppress.get('baseline_prediction', '?')}

    ```
    {_text}
    ```
    """)
    return


@app.cell
def _(batch_data, mo, np, plt):
    _has_examples = "top_suppress_example" in batch_data
    mo.stop(not _has_examples, mo.md(
        "*No per-prompt examples in this file. Re-run the batch script to generate them.*"
    ))

    _suppress = batch_data["top_suppress_example"]
    _scales = np.array(batch_data["scales"])

    _mask = _scales <= 1.0
    _s_scales = _scales[_mask]
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _sc = np.array(_suppress["correct_ans_prob"])[_mask]
    _sw = np.array(_suppress["wrong_ans_prob"])[_mask]
    _ax.plot(_s_scales, _sc, "o-", color="#00C885", linewidth=2)
    _ax.plot(_s_scales, _sw, "o-", color="#c0c0c0", linewidth=2)
    _ax.text(_s_scales[-1] + 0.3, _sc[-1], _suppress['correct_ans'],
             color="#00C885", fontsize=12, fontweight="bold", va="center")
    _ax.text(_s_scales[-1] + 0.3, _sw[-1], _suppress['wrong_ans'],
             color="#c0c0c0", fontsize=12, fontweight="bold", va="center")
    _ax.set_xlabel("Intervention strength (scale)")
    _ax.set_ylabel("Next token probability")
    _ax.set_title(f"Suppress — feature {batch_data['feature_id']} (act={_suppress['feature_activation']:.2f})")
    _ax.set_yscale("log")
    _ax.set_xticks([_s for _s in _s_scales if _s == int(_s)] + [1])
    _ax.set_xlim(-6.5, 2.8)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(batch_data, mo):
    _rescue = batch_data.get("top_rescue_example", {})
    _text = _rescue.get("prompt_text", "")
    mo.md(f"""### Rescue example prompt
    **Correct answer:** {_rescue.get('correct_ans', '?')} | **Wrong answer:** {_rescue.get('wrong_ans', '?')} | **Baseline prediction:** {_rescue.get('baseline_prediction', '?')}

    ```
    {_text}
    ```
    """)
    return


@app.cell
def _(batch_data, mo, np, plt):
    _has_examples = "top_rescue_example" in batch_data
    mo.stop(not _has_examples, mo.md(
        "*No per-prompt examples in this file. Re-run the batch script to generate them.*"
    ))

    _rescue = batch_data["top_rescue_example"]
    _scales = np.array(batch_data["scales"])

    _mask = _scales >= 1.0
    _r_scales = _scales[_mask]
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _rc = np.array(_rescue["correct_ans_prob"])[_mask]
    _rw = np.array(_rescue["wrong_ans_prob"])[_mask]
    _ax.plot(_r_scales, _rc, "o-", color="#00C885", linewidth=2)
    _ax.plot(_r_scales, _rw, "o-", color="#c0c0c0", linewidth=2)
    _ax.text(_r_scales[-1] + 0.3, _rc[-1], _rescue['correct_ans'],
             color="#00C885", fontsize=10, fontweight="bold", va="center")
    _ax.text(_r_scales[-1] + 0.3, _rw[-1], _rescue['wrong_ans'],
             color="#c0c0c0", fontsize=10, fontweight="bold", va="center")
    _ax.set_xlabel("Intervention strength (scale)")
    _ax.set_ylabel("Next token probability")
    _ax.set_title(f"Rescue — feature {batch_data['feature_id']} (act={_rescue['feature_activation']:.2f})")
    _ax.set_yscale("log")
    _ax.set_xticks([_s for _s in _r_scales if _s == int(_s)] + [1])
    _ax.set_xlim(0.5, 12.0)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(batch_data, mo):
    _scales = batch_data["scales"]
    _cb = batch_data["correct_batch"]
    _wb = batch_data["wrong_batch"]
    _ss = _cb.get("suppress_stats", {})
    _rs = _wb.get("rescue_stats", {})

    _base_idx = _scales.index(1.0)

    _rows = []
    for _si, _s in enumerate(_scales):
        _rows.append({
            "scale": f"{_s}x",
            "correct_mean_P(correct)": f"{_cb['mean_correct_ans_prob'][_si]:.4f}",
            "correct_mean_P(wrong)": f"{_cb['mean_wrong_ans_prob'][_si]:.4f}",
            "wrong_mean_P(correct)": f"{_wb['mean_correct_ans_prob'][_si]:.4f}",
            "wrong_mean_P(wrong)": f"{_wb['mean_wrong_ans_prob'][_si]:.4f}",
        })

    mo.vstack([
        mo.md(f"""### Batch Results — Feature {batch_data['feature_id']} ({batch_data['rule']})

    **Suppression** (correct prompts, n={_cb['n_prompts']}):
    flipped {_ss.get('n_flipped', '?')}/{_ss.get('n_eligible', '?')}
    ({100*_ss.get('flip_rate', 0):.1f}%),
    Wilcoxon p={_ss.get('wilcoxon_p', float('nan')):.2e},
    binomial p={_ss.get('binom_p', float('nan')):.3f}

    **Rescue** (wrong prompts, n={_wb['n_prompts']}):
    flipped {_rs.get('n_flipped', '?')}/{_rs.get('n_eligible', '?')}
    ({100*_rs.get('flip_rate', 0):.1f}%),
    Wilcoxon p={_rs.get('wilcoxon_p', float('nan')):.2e},
    binomial p={_rs.get('binom_p', float('nan')):.3f}
    """),
        mo.ui.table(_rows, label="Per-scale mean probabilities"),
    ])
    return


if __name__ == "__main__":
    app.run()
