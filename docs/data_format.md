# Data Formats and Environment Adapters

RobustComm operates on **inbound messages** (`R^d`) per agent per step. Two data sources
are supported: the built-in synthetic environment (offline, for tests/demo) and real
cooperative learned-communication environments (via adapters, EC2 only). **No data is ever
downloaded automatically.**

## 1. In-memory message schema (`data/schemas.py`)
- `CommRound`: `messages` `(n, d)` float array, `byzantine` `(n,)` bool mask, optional
  `honest_reference` `(d,)` and `target` `(d,)`.
- Message values are real vectors (not probabilities). Validation
  (`utils/validation.validate_messages`) requires 2D, finite, `>=1` sender.

## 2. Synthetic tabular schema (`data/synthetic.py`)
`generate_message_rounds(...)` produces a long-format DataFrame:

| column | type | meaning |
|---|---|---|
| `round` | int | communication step index |
| `sender` | int | sender id `0..n-1` |
| `byzantine` | bool | whether the sender is Byzantine |
| `target_k` (k=0..d-1) | float | per-round target signal `s[k]` |
| `message_k` (k=0..d-1) | float | emitted message `m[k]` |

Honest messages lie within an `l2` ball of radius `honest_dispersion` (=`B_i`, A1) around
the round target; Byzantine messages are large-magnitude outliers. This is synthetic data
for pipeline testing only.

## 3. Real environment adapters (`data/adapters.py`)
Each adapter must implement `EnvAdapter`: `n_agents`, `msg_dim`, and
`rollout(aggregator, attack, policy, byzantine_ids, rng) -> {"return", "mean_deviation", ...}`.
The `make_environment(cfg, rng)` factory returns the synthetic env for
`env.name == "synthetic_comm"` and the corresponding adapter otherwise. Set `env.data_root`
to the provisioned data directory.

### DIAL switch-riddle (`DIALEnvAdapter`) — `env.name: dial`
- **Requires:** the DIAL / switch-riddle environment (Foerster et al., 2016). No public
  license blocker, but no maintained pip package; wire to a reference implementation.
- **Expected layout under `env.data_root`:** environment config + per-episode logs exposing,
  at each step, the real-valued messages each agent emits and consumes, and the team reward.
- **Adapter must map:** emitted messages → `(n, d)`; the inbound neighbourhood per agent;
  the team reward per step.

### MPE particle communication (`MPEEnvAdapter`) — `env.name: mpe`
- **Requires:** `pip install pettingzoo[mpe]` (Lowe et al., 2017); e.g.
  `simple_speaker_listener_v4`, `simple_reference_v3`.
- **Expected layout:** none on disk — the simulator is installed as a package; `data_root`
  may point to a config directory. The adapter wraps the PettingZoo env, extracts the
  communication action/message vector, and exposes it as `(n, d)`.

### SMAC micromanagement (`SMACEnvAdapter`) — `env.name: smac`
- **Requires:** StarCraft II + SMAC maps (Samvelyan et al., 2019). **License/registration
  required** (Blizzard SC2 + SMAC map pack). No automatic download.
- **Expected layout under `env.data_root`:** installed SC2 + `SMAC_Maps/`.

## 4. Output format (produced by the runners)
- `evaluation.csv`: columns from `EvalRecord` (`operator, fraction, clean_reward,
  adv_reward, attack_success, aggregate_deviation, n_agents, msg_dim, topology, mode,
  colluding, seed, note`).
- `ablation.csv`: `variant, operator, adversarial_training, ensembles, discrete,
  clean_reward, adv_reward, note`.
- `run_metadata.json`: resolved config, config hash, seed, versions, timestamp.
- Every row carries `note = "synthetic smoke-test output, not a paper result"` when produced
  by the synthetic env.

## 5. Manual-provisioning summary
| Environment | Auto-download? | Manual step |
|---|---|---|
| synthetic_comm | n/a | none (built in) |
| dial | no | provide a DIAL/switch-riddle implementation and wire the adapter |
| mpe | no | `pip install pettingzoo[mpe]` and wire the adapter |
| smac | no | install SC2 + SMAC maps (license/registration) and wire the adapter |

See `docs/ec2_experiment_guide.md` for the end-to-end EC2 procedure.
