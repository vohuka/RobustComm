# EC2 Experiment Guide

This repository is **EC2-ready** but never downloads data or runs paper-level experiments on
its own. Follow this guide on an EC2 instance (or any server you control) to run real
experiments after provisioning the environments.

## 0. Prerequisites
- Python 3.10+ and the package installed: `pip install -e .`.
- Confirm the offline pipeline first: `python -m pytest -q` and
  `python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml`.

## 1. Provision an environment (choose one; none is downloaded automatically)

### MPE (easiest)
```bash
pip install "pettingzoo[mpe]"
```
Then wire `MPEEnvAdapter` in `src/robustcomm/data/adapters.py` to a PettingZoo env
(e.g. `simple_speaker_listener_v4`): map the communication message to `(n, d)`, define the
inbound neighbourhood, and return `{"return", "mean_deviation"}` from `rollout`.

### DIAL switch-riddle
Obtain a DIAL / switch-riddle implementation (Foerster et al., 2016) and wire
`DIALEnvAdapter` to it. Expose per-step emitted/consumed messages and the team reward.
Set `env.data_root` to its config/log directory.

### SMAC (license required)
Install StarCraft II and the SMAC map pack (registration/license required). Set
`env.data_root` to the SC2 install and wire `SMACEnvAdapter`. **Do not commit maps or the
game binary.**

## 2. Configure
Copy `configs/experiment_template.yaml`, set:
- `env.name` = `mpe` | `dial` | `smac`;
- `env.data_root` = provisioned path;
- `env.n_agents`, `topology`, `msg_dim`, `discrete` per the sweep you want;
- `aggregator.operator`, `attack.*`, `training.*`, `eval.*`.

## 3. Run
```bash
python scripts/run_experiment.py --config configs/my_experiment.yaml --stage evaluate
python scripts/run_experiment.py --config configs/my_experiment.yaml --stage ablate
python scripts/make_plots.py --input outputs/experiment --output outputs/experiment/plots
```

## 4. External baselines (optional)
To compare against Sun et al. (2023) / Yu et al. (2024) / Yuan et al. (2024), install their
repos and wrap them with `ExternalDefenseAdapter` (see `docs/baseline_adapters.md`). Record
commit hashes in `TODO_IMPLEMENTATION.md`.

## 5. Provenance
Archive `output.dir` with `run_metadata.json`, a `requirements.lock` (`pip freeze`), and any
external-repo commit hashes. Only then are the numbers "paper-level results" with recorded
provenance (per the result-integrity policy).

## Manual-only steps (cannot be automated here)
- StarCraft II / SMAC license approval and map registration.
- Obtaining/pinning external-baseline repositories and pretrained models.
- Any GPU provisioning for large-scale MARL training (the synthetic core is CPU-only; real
  DIAL/MADDPG/QMIX training may benefit from a GPU).
