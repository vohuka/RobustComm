# Reproducibility

## Seeds
- Every run sets a single integer `seed` (config field). `utils.seeding.set_global_seed`
  seeds Python `random`, NumPy's global RNG, and `PYTHONHASHSEED`, and returns a dedicated
  `numpy.random.Generator` that library code threads through explicitly (no hidden global
  state). Aggregators, attacks, environments, and training all accept a `Generator`.
- Determinism is asserted by `tests/test_reproducibility.py` (identical env rollouts,
  identical trained weights, identical crafted attacks under a fixed seed).

## Config capture and provenance
- Configs are YAML (`configs/`) validated by `utils.validation.validate_config`.
- Each experiment writes `run_metadata.json` via `utils.io.write_run_metadata`, recording:
  resolved config, a short **config hash** (sha256 of the canonicalised config), the seed,
  Python/NumPy/pandas versions, platform, UTC timestamp, and a note flagging synthetic vs
  real runs.

## Outputs
- Results are CSV/JSON under `output.dir`. Plots are produced from those CSVs only
  (`experiments/plot_results.py`); nothing is hard-coded. Synthetic plots are titled
  "SYNTHETIC smoke-test - not a paper result".

## Environment
- Python 3.10+ (3.11+ recommended). Core deps: numpy, pandas, pyyaml, matplotlib.
- No GPU, no internet. Pin versions with `pip freeze > requirements.lock` for exact
  reproduction on EC2.

## Reproducing the demo
```bash
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml
# deterministic given seed=0 in the config; re-running yields identical CSVs.
```

## Reproducing a real run (EC2)
1. Provision the environment (`docs/ec2_experiment_guide.md`) and set `env.name` +
   `env.data_root`.
2. `python scripts/run_experiment.py --config <cfg> --stage all`.
3. Archive `output.dir` (including `run_metadata.json`) alongside the exact
   `requirements.lock` and, for external baselines, their commit hashes.

## Notes on stochastic components
- Adversarial training samples one ensemble member per episode and re-crafts the attack each
  step; both consume the threaded `Generator`, so the whole loop is reproducible.
- Bootstrap CIs (`metrics/confidence_intervals.py`) accept a `Generator` for reproducible
  resampling.
