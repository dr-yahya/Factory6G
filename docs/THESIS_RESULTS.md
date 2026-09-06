# Producing The Thesis Result Families

One config per result family in `config/thesis/`, one command each, and a
statement of which thesis claim the family is evidence for. If a claim is not in
this table, there is no run producing evidence for it.

Structure (agreed 2026-09-05): **estimator-led, resource management second**;
**TR 38.901 Indoor Factory for all factory claims** with UMi retained as a
comparison arm; **reinforcement learning trained for real, with behaviour cloning
kept as an ablation**.

The numerology decision was revised once the delay-spread model was measured.
The factory families now run on a **4-symbol mini-slot at 120 kHz over 512
subcarriers** — 61.4 MHz, the only configuration in this project where a factory
hall's delay profile is actually resolvable (see the bandwidth section). FR1 is
retained as a deliberate narrowband control rather than as the body numerology.

## Running a family

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc \
  "pip install -e . -q && python -m factory6g.cli.run --config config/thesis/<family>.json"
```

Enable `system.graph_mode` (already on in these configs) — it is bit-identical to
eager and roughly seven times faster, which is what makes 100-batch runs
affordable.

## The families

| Config | Claim it supports | Hall | Bandwidth | Selectivity | Notes |
|---|---|---|---|---|---|
| `estimators_inf_s.json` | **Lead contribution** — estimator comparison in a factory hall | 15×15×5 m, 5 machines | 61.4 MHz | 7.3 | 4 users. Mini-slot FR3, 13 GHz. |
| `estimators_inf_m.json` | Scaling with hall size and device count | 25×25×6 m, 10 machines | 61.4 MHz | 9.2 | 8 users. |
| `estimators_inf_l.json` | Scaling continued | 40×40×8 m, 20 machines | 61.4 MHz | 12.1 | 16 users. Longest delay spread, most selective. |
| `estimators_inf_narrowband.json` | **Control** — a CP-length window is furthest from optimal when the channel is flat | 15×15×5 m | 3.8 MHz | 0.45 | FR1. Reinterpreted 2026-09-06: this is where the adaptive window gains most (+11.5 dB NMSE), not least. |
| `estimators_umi.json` | Comparison arm; where the estimation floor lives | — (UMi) | 3.8 MHz | selective | Keeps the existing weekly-report evidence interpretable. |
| `resource_managers_inf.json` | Resource-management chapter | 25×25×6 m | 61.4 MHz | 9.2 | 8 users, AGV mobility 3 m/s, CSI delay 4 slots, HARQ 3. `static_subset` is the equal-load control; `drl` and `rl` give the imitation-vs-RL ablation. |

Selectivity is signal bandwidth over coherence bandwidth. It rises with hall size
because a larger hall reverberates longer, so the factory-size sweep is now a
sweep over a real propagation property rather than over the user count alone.

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

## Bandwidth: can the carrier see the factory channel?

An Indoor Factory hall has a short delay spread — TR 38.901 ties it to the hall's
volume-to-surface ratio, giving 23.7 ns for the 15×15×5 m hall and 39.4 ns for
40×40×8 m. Short delay spread means *wide* coherence bandwidth, 5–8 MHz here. A
carrier narrower than that sees a flat channel, and every frequency-domain
estimator converges to the same answer no matter how good it is.

> **Correction (2026-09-06): the table below describes the channel, but until now
> the code did not build it.** `system.inf_hall_volume_m3` and
> `system.inf_hall_surface_m2` were plain floats defaulting to the *small* hall,
> and `asdict()` always materialised them, so the geometry fallback in
> `ChannelModel._apply_inf_large_scale` was unreachable — every hall size
> simulated a single delay spread. `frequency_selectivity_report()` meanwhile
> derived its numbers from the room dimensions, which is where the varying
> figures quoted here came from. Reported and simulated disagreed.
>
> `config/config.json` also carried a hand-entered `inf_hall_surface_m2: 900.0`
> against the 750 m² its own room dimensions imply, making the simulated delay
> spread 20.8 ns rather than 23.7 ns — 12% short.
>
> Both keys are now optional overrides defaulting to `None`, derived from
> `factory_scenario.room_dimensions` when unset, and the channel and the
> selectivity report share one resolution path. The numbers in this document were
> always correct as a description of intent; the simulation now matches them.
> **Every InF result produced before this fix used the wrong delay spread and
> must be regenerated.**

The current FR1 numerology is narrower than that. Measured:

| Numerology | Bandwidth | Selectivity ratio | RMS taps | LDPC block | Usable? |
|---|---|---|---|---|---|
| fft 128 × 14 sym @ 30 kHz *(current FR1)* | 3.8 MHz | 0.45 | 0.02 | k = 1536 | **flat — estimators cannot differ** |
| fft 512 × 14 sym @ 30 kHz | 15.4 MHz | 1.82 | 0.39 | k = 6144 | mildly selective |
| fft 1024 × 14 sym @ 30 kHz | 30.7 MHz | 3.64 | 0.78 | k = 12288 | **exceeds the 5G LDPC limit (8448)** |
| fft 128 × 4 sym @ 120 kHz *(current FR3)* | 15.4 MHz | 1.82 | 0.39 | k = 256 | mildly selective |
| fft 512 × 4 sym @ 120 kHz | **61.4 MHz** | **7.27** | **1.75** | k = 1024 | **strongly selective** |

Selectivity ratio is signal bandwidth over coherence bandwidth; below about 1 the
carrier cannot resolve the delay profile at all. Figures are for the small hall —
the large hall is roughly 1.7× more selective at any given bandwidth.

Two consequences worth carrying into the write-up:

* **Mini-slots buy bandwidth.** A 4-symbol TTI produces a codeword 3.5× shorter
  than a 14-symbol slot, so 512 subcarriers fit inside the 5G LDPC maximum where
  14 symbols would not. The FR3 mini-slot configuration is therefore not only the
  6G section — it is the only configuration in this project where frequency-domain
  channel estimation is genuinely exercised in a factory hall.
* **Wideband FR1 needs code-block segmentation.** Real NR splits a transport block
  across multiple LDPC code blocks; this simulator maps the whole resource grid to
  one codeword per user, which caps FR1 at roughly fft 512. Implementing
  segmentation is the prerequisite for a 100 MHz FR1 study.

Check `frequency_selectivity_report()` on the channel, or the `err_var_calibration`
and `nmse_db` metrics, before concluding anything about estimator ranking: on a
flat channel the ranking is not informative.

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
3. **The truncation window, not the smoother, is the quantity worth adapting**
   (`adaptive-window-estimator/`). A clairvoyant DFT-versus-LMMSE branch
   selector beats plain DFT by 0.2% on a factory channel; sizing the window to
   the measured delay profile instead lands within 0.1 dB of an exhaustive
   search over window length on every factory point. Unlike finding 2, this one
   reaches the decoder: on the small hall it cuts BLER significantly at every
   Eb/No from 0 to 14 dB, closing **28-43%** of the gap between fixed DFT and
   the perfect-CSI bound of finding 1, and improving worst-user BLER with it.
   The winner is also the better-calibrated estimator, so it is not the
   over-confidence artifact finding 2 had to be cleared of.

## Known gaps

* **Only the small hall has run at BLER.** Narrowband, medium, large and UMi are
  queued behind it; finding 3's cross-family claim is not yet evidenced.
* **The adaptive window's NMSE-versus-oracle table predates the hall-geometry
  fix** and reads a few tenths of a dB optimistic as a result. The BLER table
  does not — it was rerun on the corrected channel.
* **The adaptive window's precondition fails on UMi.** Its noise fit assumes no
  channel energy past the cyclic prefix — 0.000% in both halls, 3.4% on UMi at
  this numerology — so UMi loses up to 1.1 dB at high SNR. Stated rather than
  detected; an overrun detector was tried and fires spuriously on the large hall.
* **The adaptive branch policy is tuned on the wrong objective.** It prefers
  LMMSE at low SNR, which finding 2 shows is the weaker choice at every SNR.
  Given finding 3, the branch selector is probably not worth retuning at all —
  the clairvoyant version of it is worth 0.2%.
* **LS and LMMSE still understate their error variance** by three to four times.
  Not distorting enough to change the ranking, but it should be stated when the
  calibration table is presented.
* **The FR1 body cannot exercise frequency-domain estimation.** At 3.8 MHz the
  factory channel is flat (see the bandwidth section above). Either move to
  fft 512, lead with the mini-slot configuration, or reframe the estimator
  contribution around noise averaging rather than frequency interpolation.
* **No LDPC code-block segmentation.** One codeword per user per grid caps FR1
  bandwidth at about fft 512, which is what blocks a 100 MHz study.
* **Importance sampling for the deep tail is not implemented.** The extrapolation
  path with intervals is the honest interim.
* **All numbers so far were produced under TensorFlow 2.21 / Sionna 1.2.1**, not
  the pinned 2.15 in the Docker image. Regenerate in the container before
  quoting.
