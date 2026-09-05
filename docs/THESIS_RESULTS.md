# Producing The Thesis Result Families

One config per result family in `config/thesis/`, one command each, and a
statement of which thesis claim the family is evidence for. If a claim is not in
this table, there is no run producing evidence for it.

Structure assumed (agreed 2026-09-05): **estimator-led, resource management
second**; **TR 38.901 Indoor Factory for all factory claims** with UMi retained
as a comparison arm; **FR1 numerology in the body with an FR3 mini-slot
section**; **reinforcement learning trained for real, with behaviour cloning
kept as an ablation**.

## Running a family

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc \
  "pip install -e . -q && python -m factory6g.cli.run --config config/thesis/<family>.json"
```

Enable `system.graph_mode` (already on in these configs) — it is bit-identical to
eager and roughly seven times faster, which is what makes 100-batch runs
affordable.

## The families

| Config | Claim it supports | Channel | Numerology | Notes |
|---|---|---|---|---|
| `estimators_inf_s.json` | Estimator comparison in a small factory hall | InF-DH, 15x15x5 m, 5 machines | FR1 | 4 users. Primary evidence for the lead contribution. |
| `estimators_inf_m.json` | Scaling with hall size and device count | InF-DH, 25x25x6 m, 10 machines | FR1 | 8 users. |
| `estimators_inf_l.json` | Scaling continued | InF-DH, 40x40x8 m, 20 machines | FR1 | 16 users. |
| `estimators_umi.json` | Comparison arm against the prior UMi evidence | TR 38.901 UMi | FR1 | Keeps the existing weekly-report results interpretable. |
| `estimators_fr3_minislot.json` | The 6G section: methods carry to FR3 and hit sub-ms latency | InF-DH | **FR3**, 13 GHz, 120 kHz, 4-symbol mini-slot | AGV mobility 3 m/s, CSI delay 4 slots, HARQ 3. Measured p99.9 latency 0.116 ms. |
| `resource_managers_inf.json` | Resource-management chapter | InF-DH, 25x25x6 m | FR1 | 8 users, CSI ageing on, `static_subset` as the equal-load control, `drl` and `rl` both present for the ablation. |

Because the hall geometry now reaches the propagation model, the S/M/L sweep is
a genuine factory-size study: room dimensions and machine count set the InF
clutter density and LOS probability. Before this change `--factory-size` altered
only the user count.

## Reading the output

Each run writes `stage_results_v2.json` per stage. The metrics that carry thesis
claims:

| Metric | Use it for |
|---|---|
| `bler`, `bler_upper_confidence` | Headline reliability. Exact Clopper-Pearson; codewords are near-independent, bits within one are not. |
| `worst_user_bler` | Factory URLLC reliability — the weakest device sets the system guarantee. |
| `latency_p999_ms` | The URLLC latency claim. Never the mean. |
| `nmse_db` | Estimator accuracy, isolated from the decoder. |
| `err_var_calibration` | **Check this before comparing estimators.** Declared error variance over actual; 1.0 is honest. Values far from 1.0 mean an estimator's BER is being flattered by an over-confident error declaration fed to the equalizer. |
| `paired_comparisons` | Per-batch differences against `paired_reference` with a bootstrap CI. Quote results as "improves BLER by X (95% CI [a, b])". |
| `num_scheduled_users` | Confirms methods are compared at equal load. |
| `run_provenance`, `manager_provenance` | Git commit, library versions, and whether each learned policy actually loaded. |

## Reliability the budget can reach

Every run prints its evidence ceiling at startup. At 100 batches x 20 x 4 users
that is about 12.3M bits and 8,000 codewords per Eb/No point, so the smallest
resolvable BLER is roughly 4e-3 and BER roughly 2.4e-6.

**Do not claim reliability below the ceiling from a measurement.** Use
`factory6g.sim.evidence.extrapolate_bler`, which fits the log-linear tail and
returns a prediction interval, and label it an extrapolation.

## Training the learned resource managers

```bash
# 1. Behaviour cloning on oracle labels — the ablation baseline.
python scripts/tools/generate_dataset.py --config config/thesis/resource_managers_inf.json \
    --label-repeats 4 --output data/rm_inf.parquet
python scripts/tools/train_drl_resource_manager.py --data data/rm_inf.parquet \
    --config config/thesis/resource_managers_inf.json \
    --output-dir models/drl_resource_manager_policy

# 2. Reinforcement learning, warm-started from it — the contribution.
python scripts/tools/train_rl_resource_manager.py \
    --config config/thesis/resource_managers_inf.json --iterations 300 \
    --initial-checkpoint models/drl_resource_manager_policy \
    --output-dir models/rl_resource_manager_policy
```

The RL trainer evaluates a max-SNR baseline through the identical path before and
after training and stores both in the checkpoint metadata, so any claimed gain is
reported against a measured baseline. Present `drl` (imitation) against `rl`
(reinforcement) against the heuristics as a three-way ablation — that is the
comparison an examiner will ask for, and the checkpoint metadata distinguishes
them (`training_method: supervised_imitation` versus
`reinforce_with_value_baseline`).

## What the evidence already establishes

Two results are recorded under `reports/evidence/` and are ready to write up:

1. **The TR 38.901 error floor is channel-estimation error, not a receiver
   artifact** (`llr_clip_floor/`). Perfect CSI removes it entirely; LLR clipping
   is irrelevant to it. This gives the estimator chapter a measured lower bound
   to work against.
2. **NMSE does not predict coded BER** (`estimator-floor-tr38901/`). LMMSE has
   the best NMSE at every Eb/No point and the worse BER; DFT and the adaptive
   hybrid close about 2.5 orders of magnitude of the floor. Verified to survive
   correcting — and then reversing — the error-variance confound.

## Known gaps

* **The adaptive branch policy is tuned on the wrong objective.** It prefers
  LMMSE at low SNR, which finding 2 shows is the weaker choice at every SNR.
  Retuning `quality_low` / `quality_high` / `leakage_reference` against BLER is
  the cheapest available improvement to the lead contribution.
* **LS and LMMSE still understate their error variance** by three to four times.
  Not distorting enough to change the ranking, but it should be stated when the
  calibration table is presented.
* **Importance sampling for the deep tail is not implemented.** The extrapolation
  path with intervals is the honest interim.
* **All numbers so far were produced under TensorFlow 2.21 / Sionna 1.2.1**, not
  the pinned 2.15 in the Docker image. Regenerate in the container before
  quoting.
