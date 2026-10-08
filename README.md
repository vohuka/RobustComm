# RobustComm: Byzantine-Resilient Learned Communication for Cooperative Multi-Agent Reinforcement Learning

Reference implementation for the research paper:  
**"RobustComm: Byzantine-Resilient Learned Communication for Cooperative Multi-Agent Reinforcement Learning via Robust Message-Space Aggregation"** (AAMAS Submission).

---

## 📌 Overview

**RobustComm** introduces Byzantine resilience directly into inter-agent communication channels for cooperative Multi-Agent Reinforcement Learning (MARL). Instead of aggregating inbound messages using unconstrained linear means, RobustComm operates in message space via robust statistical aggregation operators (Coordinate-wise Median, Krum, $\beta$-Trimmed Mean), hardened through minimax adversarial training and policy ensembles without requiring a central trusted authority.

This repository provides:
1. **Core Algorithms**: Coordinate-wise Median, Krum, $\beta$-Trimmed Mean, and standard Mean message aggregators.
2. **Adversarial Attack Suite**: Multi-step FGSM/PGD, Backward Pass Differentiable Approximation (BPDA), Simultaneous Perturbation Stochastic Approximation (SPSA), and operator-specific attacks (boundary-pushing, clustering).
3. **MPE Particle Communication Benchmark**: Multi-seed evaluation on cooperative communication with $n=7$ agents under varying Byzantine fractions $\phi \in \{0.0, 0.14, 0.29, 0.43\}$.
4. **Baselines**: Mean (DIAL), No-Communication, Krum-only, Median-only, Adv-training-only, Sun et al. (Ablation Ensemble), and R-MACRL (Autoencoder Filter).
5. **Full Test Suite & Pre-computed Artifacts**: 55 automated unit and integration tests passing deterministically.

---

## 📂 Repository Structure

```text
RobustComm/
├── configs/                  # Experiment configuration files
│   ├── default.yaml
│   ├── mpe.yaml              # Benchmark configuration for MPE communication
│   ├── synthetic_demo.yaml
│   └── experiment_template.yaml
├── docs/                     # Technical documentation & notes
│   ├── baseline_adapters.md
│   ├── data_format.md
│   ├── ec2_experiment_guide.md
│   ├── implementation_notes.md
│   └── reproducibility.md
├── examples/                 # Minimal usage examples
│   ├── example_config.yaml
│   └── synthetic_data/
├── outputs/                  # Pre-computed evaluation logs, tables & plots
│   ├── mpe_run/              # Multi-seed MPE benchmark results (Tables 4, 5, 6)
│   ├── rebuttal/             # Extended sweeps, predator-prey, and attack ablations
│   └── synthetic_demo/       # Smoke-test demo outputs
├── scripts/                  # Executable entry points
│   ├── run_mpe_experiment.py         # Main MPE benchmark reproduction script
│   ├── run_rebuttal_experiments.py   # Extended analysis & ablations
│   ├── run_synthetic_demo.py         # Fast smoke test
│   ├── run_experiment.py
│   └── make_plots.py
├── src/
│   └── robustcomm/           # Core library
│       ├── aggregators/      # Krum, Median, Trimmed Mean, Mean, Registry
│       ├── attacks/          # FGSM, BPDA, SPSA, boundary-pushing, camouflage
│       ├── baselines/        # DIAL, No-Comm, SOTA defenses (Sun et al., R-MACRL)
│       ├── data/             # Schemas, topologies, synthetic generator
│       ├── envs/             # MPE communication, predator-prey, synthetic envs
│       ├── experiments/      # Evaluation, ablation, and plotting runners
│       ├── metrics/          # Reward-under-attack, latency, bootstrap CI
│       ├── policy/           # Surrogate policies, policy ensembles
│       ├── theory/           # Deviation bounds, Lipschitz estimation
│       ├── training/         # Minimax adversarial communication training
│       └── utils/            # I/O, logging, seeding, validation
├── tests/                    # 55 test modules (aggregators, attacks, metrics, envs, etc.)
├── Makefile                  # Helper targets
├── pyproject.toml            # Project packaging metadata
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 💻 Installation

The implementation is pure Python / NumPy / Pandas, requiring no specialized hardware or external network access.

```bash
# Clone the repository
git clone <anonymous-repo-url>
cd RobustComm

# Install dependencies
pip install -r requirements.txt

# (Optional) Install the package in editable mode
pip install -e .
```

### Requirements
- Python $\ge$ 3.10
- `numpy >= 1.24`
- `pandas >= 1.5`
- `pyyaml >= 6.0`
- `matplotlib >= 3.6`
- `pytest >= 7.0` (for testing)

---

## 🚀 Reproducing Paper Experiments

### 1. Main MPE Benchmark (Tables 4, 5, and 6)
To reproduce the full multi-seed evaluation across seeds `[42, 43, 44, 45, 46]` reported in the main paper:

```bash
python scripts/run_mpe_experiment.py --seeds 42 43 44 45 46
```

To immediately aggregate the bundled, verified experimental outputs without re-running:
```bash
python scripts/run_mpe_experiment.py --aggregate-only
```
This prints the exact numerical tables, latency profiling (microseconds per step), and Student-$t$ hypothesis tests, writing updated `.tex` tables directly to `outputs/mpe_run/`.

### 2. Extended Rebuttal & Ablation Experiments
To run the extended sweeps (predator-prey benchmark, attack strength $\tau \in \{0.5, 0.2, 0.1\}$, and network topology tolerance):

```bash
python scripts/run_rebuttal_experiments.py
```
Outputs are exported to `outputs/rebuttal/`.

### 3. Fast Synthetic Smoke-Test
To verify the complete pipeline end-to-end within seconds:

```bash
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml
```

---

## 🧪 Running Unit & Integration Tests

All 55 tests can be executed locally via `pytest`:

```bash
pytest -q
```
Expected output:
```text
55 passed in ~3s
```

---

## 📊 Pre-computed Results Inspection

For rapid auditing by reviewers, pre-computed outputs from the experimental runs are preserved in `outputs/`:
- **`outputs/mpe_run/mpe_table4.tex`**: LaTeX code for Table 4 (Reward-under-attack across Byzantine fractions $\phi$).
- **`outputs/mpe_run/mpe_table5.tex`**: Component ablation table with Student-$t$ 95% confidence intervals.
- **`outputs/mpe_run/mpe_table6_latency.tex`**: Wall-clock latency profiling table.
- **`outputs/mpe_run/evaluation_all_seeds.csv`**: Raw episode returns across all 5 seeds.
- **`outputs/rebuttal/per_seed_raw_mpe.csv`**: Detailed per-seed raw traces.

---

## 🔒 Anonymity Notice

This repository is formatted for double-blind peer review in accordance with the conference guidelines. Author names, affiliations, email addresses, and institution-specific paths have been omitted.
