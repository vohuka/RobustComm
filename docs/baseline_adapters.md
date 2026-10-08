# Baselines and Adapters

Three clearly separated categories (Sec. 10.5 of the paper). The code never presents a dummy
as the real external method.

## 1. Real baselines — fully implemented from the paper
Defined as `BaselineSpec`s and built by `baselines.base.build_defense`:

| name | operator | adv. train | ensembles | paper |
|---|---|---|---|---|
| `no_defense` | mean | no | no | B1 (DIAL mean, undefended) |
| `no_comm` | nocomm | no | no | B2 (lower reference) |
| `krum_only` | krum | no | no | B3 (robust agg, no training) |
| `adv_only` | mean | yes | yes | B4 (mean + adversarial training) |
| `robustcomm_krum` | krum | yes | yes | full method |
| `robustcomm_median` | median | yes | yes | full method |

These require no external code and are exercised by the demo and `tests/test_defense_pipeline.py`.

## 2. Adapter stubs — external methods, backend required
- **Message signing / authentication (B5)** — `baselines/message_signing.py`.
  The real comparator needs a signature scheme and an availability-cost model. The provided
  `MessageSigningComparator` is a **DUMMY oracle signer** (it is told which senders are
  Byzantine, drops them, and drops an `availability_cost` fraction of honest messages). It
  logs a warning and is for pipeline testing only. **TODO:** replace with a real scheme.
- **External comm-robustness defenses** — `baselines/external_wrappers.py`.
  `ExternalDefenseAdapter(name, backend)` wraps a real backend implementing
  `.aggregate(messages) -> vector`. Without a backend it raises `NotImplementedError` with an
  actionable message. Candidates named in the paper:
  - Sun et al. (2023), certified message-ensemble (ICLR 2023) — needs their repo/model.
  - Yu et al. (2024), active/reliability-weighting defense (AAAI 2024) — needs their repo.
  - Yuan et al. (2024), multi-view message certification — needs their repo.

## 3. Dummy baselines — tests only
- `baselines/external_wrappers.DummyExternalDefense` (coordinate-wise median) — used ONLY to
  test the adapter plumbing. It is explicitly **not** the real external method.

## How to plug in a real external defense (EC2)
```python
from robustcomm.baselines.external_wrappers import ExternalDefenseAdapter

class SunEtAl2023Backend:          # you implement this around the authors' released code
    def aggregate(self, messages): # (n, d) -> (d,)
        ...

adapter = ExternalDefenseAdapter("sun2023", backend=SunEtAl2023Backend())
m_hat = adapter.aggregate(messages)
```
Record any external repo you integrate in `audits/implementation_gaps.md` and
`TODO_IMPLEMENTATION.md`, including its commit hash, for provenance.
