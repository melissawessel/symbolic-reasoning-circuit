import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    from urllib.parse import quote

    BASE_DIR = Path("results/qk_ov_attribution")
    return BASE_DIR, defaultdict, json, mo, np, quote


@app.cell
def _(BASE_DIR, mo):
    # Discover available widths and heads from the results directory
    available_widths = sorted(
        [d.name.replace("width_", "") for d in BASE_DIR.glob("width_*") if d.is_dir()]
    )

    width_selector = mo.ui.dropdown(
        options=available_widths,
        value="65k" if "65k" in available_widths else available_widths[0],
        label="SAE width",
    )
    width_selector
    return (width_selector,)


@app.cell
def _(BASE_DIR, mo, width_selector):
    # Discover heads from whichever rule dir has qk results
    _qk_dir = BASE_DIR / f"width_{width_selector.value}" / "aba" / "qk"
    available_heads = sorted(
        [f.stem for f in _qk_dir.glob("L*H*.json")],
        key=lambda h: (int(h.split("H")[0][1:]), int(h.split("H")[1])),
    ) if _qk_dir.exists() else []

    head_selector = mo.ui.dropdown(
        options=available_heads,
        value="L14H0" if "L14H0" in available_heads else available_heads[0],
        label="Head",
    )
    head_selector
    return (head_selector,)


@app.cell
def _(aba_qk, abb_qk, defaultdict, mo, np, quote, sae_width):
    def compare_url(sae_layer, qf, kf):
        slug = f"gemma-2-2b/{sae_layer}-gemmascope-res-{sae_width}"
        left = quote(f"{slug}/{int(qf)}", safe="")
        right = quote(f"{slug}/{int(kf)}", safe="")
        return f"https://np-feature-compare.netlify.app/?left={left}&right={right}"

    def extract(qk_data):
        ints = []
        for key, pair_data in qk_data["per_key_pos"].items():
            for entry in pair_data.get("top_interactions", []):
                ints.append({**entry, "pos_pair": key})
        return ints

    def aggregate_pairs(ints, sae_layer):
        groups = defaultdict(list)
        for e in ints:
            groups[(e["query_feat"], e["key_feat"])].append(e["interaction"])
        rows = []
        for (qf, kf), scores in groups.items():
            rows.append({
                "query_feat": qf,
                "key_feat": kf,
                "mean_score": round(np.mean(scores), 4),
                "n_positions": len(scores),
                "np-feature-compare_link": mo.Html(f'<a href="{compare_url(sae_layer, qf, kf)}" target="_blank">view ↗</a>'),
            })
        rows.sort(key=lambda x: abs(x["mean_score"]), reverse=True)
        return rows

    def neuronpedia_url(layer, feat):
        return f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-{sae_width}/{feat}"

    def marginalize(ints, side, sae_layer):
        scores = defaultdict(float)
        for e in ints:
            scores[e[side]] += e["interaction"]
        rows = [{"feature": f, "marg_score": round(s, 4),
                 "neuronpedia_link": mo.Html(f'<a href="{neuronpedia_url(sae_layer, f)}" target="_blank">view ↗</a>')}
                for f, s in scores.items()]
        rows.sort(key=lambda x: abs(x["marg_score"]), reverse=True)
        return rows

    sae_layer = aba_qk["sae_layer"]
    aba_ints = extract(aba_qk)
    abb_ints = extract(abb_qk)
    all_ints = aba_ints + abb_ints

    pair_tables = {
        "All (ABA+ABB)": aggregate_pairs(all_ints, sae_layer),
        "ABA only": aggregate_pairs(aba_ints, sae_layer),
        "ABB only": aggregate_pairs(abb_ints, sae_layer),
    }

    marg_tables = {}
    for label, ints in [("All (ABA+ABB)", all_ints), ("ABA only", aba_ints), ("ABB only", abb_ints)]:
        marg_tables[label] = {
            "query": marginalize(ints, "query_feat", sae_layer),
            "key": marginalize(ints, "key_feat", sae_layer),
        }

    view_selector = mo.ui.dropdown(
        options=["All (ABA+ABB)", "ABA only", "ABB only"],
        value="All (ABA+ABB)",
        label="View",
    )
    n_rows = mo.ui.slider(5, 30, value=5, step=5, label="Top N")
    mo.hstack([view_selector, n_rows])
    return marg_tables, n_rows, pair_tables, view_selector


@app.cell
def _(BASE_DIR, head_selector, json, mo, width_selector):
    def load_qk(width, head, rule):
        path = BASE_DIR / f"width_{width}" / rule / "qk" / f"{head}.json"
        return json.loads(path.read_text())

    sae_width = width_selector.value
    head = head_selector.value

    aba_qk = load_qk(sae_width, head, "aba")
    abb_qk = load_qk(sae_width, head, "abb")

    # Load reconstruction validation
    recon = {}
    for rule in ("aba", "abb"):
        _val_path = BASE_DIR / f"width_{sae_width}" / rule / "validation" / "reconstruction.json"
        if _val_path.exists():
            _val_data = json.loads(_val_path.read_text())
            recon[rule] = _val_data.get(head, {})

    _aba_corr = recon.get("aba", {}).get("mean_corr")
    _abb_corr = recon.get("abb", {}).get("mean_corr")
    _corr_str = ""
    if _aba_corr is not None:
        _corr_str = f" | reconstruction r: ABA={_aba_corr:.3f}, ABB={_abb_corr:.3f}"

    mo.md(
        f"**{head}** — stage: {aba_qk['stage']}, "
        f"CMA score: {aba_qk['cma_score']:.2f}, "
        f"SAE layer: {aba_qk['sae_layer']} (width={sae_width})"
        f"{_corr_str}"
    )
    return aba_qk, abb_qk, sae_width


@app.cell(hide_code=True)
def _(marg_tables, mo, n_rows, pair_tables, view_selector):
    _view = view_selector.value
    _n = n_rows.value

    _pairs = pair_tables[_view][:_n]
    _mq = marg_tables[_view]["query"][:_n]
    _mk = marg_tables[_view]["key"][:_n]

    mo.vstack([
        mo.md(f"### Feature-feature pairs — {_view}"),
        mo.ui.table(_pairs, label=f"Top {_n} QK pairs by mean interaction"),
        mo.md("### Marginalized features"),
        mo.hstack([
            mo.ui.table(_mq, label="Query-side"),
            mo.ui.table(_mk, label="Key-side"),
        ]),
    ])
    return


if __name__ == "__main__":
    app.run()
