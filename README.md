# Symbolic Reasoning Circuits in Language Models

Companion code for a series of posts replicating and extending [Yang et al. (2025)](https://arxiv.org/abs/2502.20332)'s work on symbolic reasoning circuits in **gemma-2-2b**, using Anthropic's interpretability methods.

Yang et al. identified a three-stage attention head circuit (Symbol Abstraction → Symbolic Induction → Retrieval) that supports in-context learning of abstract identity rules (ABA/ABB). This series traces how that circuit behaves, breaks down, and can be understood at the feature level.

## Posts

### 01 — [Tracking a Symbolic Reasoning Circuit from Failure to Success](https://melissawessel6.substack.com/p/tracking-a-symbolic-reasoning-circuit)

Tracks the three-stage circuit across shot counts as gemma-2-2b goes from ~50% accuracy (2-shot) to ~99% (10-shot). Introduces rescue patching to identify bottleneck heads.

Key findings:
- The circuit topology is already present at 2-shot — more examples amplify signal, they don't construct the circuit.
- Symbolic Induction heads carry transferable abstract rule information and are the primary bottleneck. Retrieval heads carry token-specific information that hurts when transplanted.
- The single strongest rescuer is **L14H0** (SI), flipping 41% of failures at 4-shot.

## Setup

Requires Python 3.11 or 3.12 (TransformerLens constraint).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproducing results

Each post has its own `scripts/` and `notebooks/` directory under `posts/`. Pre-computed results are in `results/` so you can inspect data and regenerate figures without running the expensive experiments.

### Post 01

```bash
# 1. Accuracy sweep (~15 min on MPS)
python posts/01-circuit-tracking/scripts/01_accuracy_sweep.py

# 2. CMA at 2, 4, 10 shot (~2-4 hours per shot count)
python posts/01-circuit-tracking/scripts/02_cma_sweep.py --shots 2 4 10

# 3. Rescue patching at 2 and 4 shot (~2-3 hours per shot count)
python posts/01-circuit-tracking/scripts/03_rescue_patching.py --shots 2 4
```

Figures:
```bash
jupyter notebook posts/01-circuit-tracking/notebooks/publication_figures.ipynb
```

## Repository structure

```
src/                        Shared source modules
├── config.py               Model config, token positions, patch positions
├── model_utils.py          TransformerLens model loading and inference helpers
├── prompt_generation.py    ABA/ABB prompt generation and CMA context pairs
├── cma.py                  Causal mediation analysis and rescue patching
└── stats.py                Significance testing (permutation test + Wilcoxon/FDR)

posts/
└── 01-circuit-tracking/
    ├── scripts/            Experiment scripts (numbered in order)
    │   ├── 01_accuracy_sweep.py
    │   ├── 02_cma_sweep.py
    │   └── 03_rescue_patching.py
    ├── notebooks/
    │   └── publication_figures.ipynb
    └── figures/            Generated publication figures (PNG + PDF)

results/                    Pre-computed results
└── shot_sweep/
    ├── accuracy_sweep.json
    ├── {2,4,10}shot/
    │   ├── significant_heads.json
    │   └── cma/
    └── {2,4}shot/rescue/

data/vocab/                 English vocabulary list (~72K tokens from Yang et al.)
```

## Acknowledgments

This project builds on [Yang et al. (2025)](https://arxiv.org/abs/2502.20332) and their [codebase](https://github.com/yukang123/LLMSymbMech). The vocabulary file (`data/vocab/gemma2_english_vocab.txt`) is from their repository. The CMA and permutation test implementations are independent reimplementations of their published methods using TransformerLens.

If you build on this work, please cite the original paper:

```
@inproceedings{yang2025emergent,
  title={Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models},
  author={Yang, Yukang and Campbell, Declan and Huang, Kaixuan and Wang, Mengdi and Cohen, Jonathan and Webb, Taylor},
  booktitle={ICML},
  year={2025}
}
```

## License

MIT — applies to the code in this repository. The vocabulary file (`data/vocab/gemma2_english_vocab.txt`) originates from [Yang et al.'s repository](https://github.com/yukang123/LLMSymbMech).
