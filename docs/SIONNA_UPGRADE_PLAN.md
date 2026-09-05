# Sionna Upgrade Plan — What The Current Library Offers The Thesis

Status: **proposal** (2026-09-05). Nothing here is implemented.

## How this was produced

The documentation site `nvlabs.github.io` is blocked by this session's egress
proxy — the gateway answers 403 to CONNECT, a policy denial rather than a
transient failure. Two routes around it were used instead:

* **The doc sources themselves.** `raw.githubusercontent.com` is reachable, and
  the site is built from `doc/source/*.rst` and `tutorials/**/*.ipynb` in the
  Sionna repo. Both were read at tag **`v1.2.2`** (the TensorFlow line we are
  on) and at `main` (the 2.x PyTorch line), so the recipes quoted below are the
  official ones for our backend.
* **The shipped package source**: the `sionna` 1.2.1 and 2.0.1 wheels and the
  `sionna-rt` 1.2.1 wheel, plus PyPI release metadata.

Every API named below was confirmed present in the **1.2.1 wheel we already
have installed**. Nothing was executed — this container has no Docker daemon and
no Sionna install — so this is a code- and docs-reading result, not a
measurement.

Objectives assumed, from `CLAUDE.md`: 6G / B5G in **smart factory**
environments, **AI/ML** applied to **PHY reliability** and **resource
management**. Gap references are to `docs/SIMULATION_REVIEW.md` and the
"Known gaps" list in `docs/THESIS_RESULTS.md`.

---

## 0. Where the pinned stack sits

`requirements.txt` pins `sionna==1.2.1` (released 2025-10-16) on
`tensorflow==2.15.0`. The current releases are:

| Release | Date | Backend | Notes |
|---|---|---|---|
| 1.2.1 | 2025-10-16 | TensorFlow | what we pin |
| **1.2.2** | 2026-03-19 | **TensorFlow** | last TF release; `sionna-rt==1.2.2` |
| 2.0.0 / 2.0.1 | 2026-03-19 / 2026-03-31 | **PyTorch** | backend migration; PHY/SYS require torch >= 2.9 |

**Recommendation: move to 1.2.2, do not move to 2.x for this thesis.**

1.2.2 is a same-backend patch. Its dependency metadata is
`tensorflow!=2.16,!=2.17,>=2.14`, `numpy>=1.26` (the `<2.0` cap is gone), and it
pulls `sionna-rt==1.2.2`, which pins **`mitsuba==3.8.0` and `drjit==1.3.1`** —
exactly the two packages the `Dockerfile` currently force-overrides because
`sionna-rt==1.2.1` pins `mitsuba==3.7.1` with no arm64 wheel. Upgrading deletes
the `--override /tmp/overrides.txt` hack and its comment. Low risk, and it
should be done before the results are regenerated in the container (the open
item in `THESIS_RESULTS.md`: every number so far came from TF 2.21 outside
Docker).

2.0 is a rewrite onto PyTorch. Every `tf.function`, the `graph_mode` work from
§3.4/§4.1, `sionna.phy.config.seed`, the Keras `NeuralChannelEstimator`, and the
DRL/CNN policy stack would all need porting, and the review's own conclusion was
that graph mode is the load-bearing speedup. The upside is a faster ecosystem,
not a new capability the thesis needs. Migrating mid-write-up buys nothing and
risks invalidating the evidence base. Worth one paragraph in future work.

Note the `numpy<2.0` cap in `requirements.txt` is now imposed by TF 2.15, not by
Sionna. Dropping it needs TF >= 2.18 (Sionna forbids 2.16 and 2.17).

---

## 1. Sionna SYS — the resource-management chapter is the intended use case

**This is the largest single win, and it is already installed.** `sionna.sys`
ships inside the pinned 1.2.1 wheel; the repo imports `sionna.phy` and
`sionna.rt` and has never touched it.

What is in `sionna/sys/`:

| Component | What it gives us |
|---|---|
| `PHYAbstraction` | SINR + MCS -> BLER/TBLER, HARQ feedback, decoded bits, from precomputed 3GPP tables. Ships PUSCH tables 1-2, so **uplink is covered**. |
| `EESM` / `EffectiveSINR` | Exponential effective SINR mapping — the standard link-to-system compression. |
| `OuterLoopLinkAdaptation`, `InnerLoopLinkAdaptation` | OLLA: picks the highest MCS holding TBLER under target, adjusting an SINR offset on ACK/NACK. |
| `PFSchedulerSUMIMO` | A reference proportional-fair scheduler. |
| `open_loop_uplink_power_control`, `downlink_fair_power_control` | 3GPP-shaped power control. |
| `get_pathloss`, `is_scheduled_in_slot`, `spread_across_subcarriers` | Glue between a PHY channel and a slot-level scheduler. |

Three gaps it closes:

* **§3.4, the Monte Carlo budget.** The harness gets ~1600 codewords per Eb/No
  point, so BLER below ~2e-2 is unresolvable, against a URLLC target of 1e-5 to
  1e-6. Physical-layer abstraction is the standard answer: run the full link
  chain once to characterise, then run tens of thousands of slots through the
  BLER tables. The evidence-ceiling report and the interval-bearing
  extrapolation stay as the honesty layer for the link-level results; a SYS arm
  is what actually reaches the regime the thesis is about.
* **The missing standard baseline.** Every scheduler in
  `models/resource_manager.py` is hand-written, so "our DRL beats PF" is
  measured against our own PF. `PFSchedulerSUMIMO` plus OLLA is a citable
  reference implementation, and the existing paired-bootstrap machinery
  (`compare_methods_paired`) can be pointed straight at it.
* **Fixed MCS.** The whole sweep runs at `num_bits_per_symbol: 2`,
  `coderate: 0.5`. Real schedulers pick MCS per user per slot; OLLA does that
  and closes the loop on HARQ, which `harq_max_rounds` already models.

Shape of the work: a new stage next to `sim/stages/resource_managers.py` that
holds the topology and channel, computes post-equalisation SINR, and drives
`PHYAbstraction` + `OuterLoopLinkAdaptation` per slot. The learned policies
already emit per-user scheduling directives, so they can drive the same loop and
be compared against `PFSchedulerSUMIMO` on identical slots. Report BLER and
worst-user BLER exactly as now.

**There is an official blueprint for exactly this, on our backend.** The
`SYS_Meets_RT` tutorial at tag `v1.2.2` builds the whole loop — ray-traced
scene, per-slot scheduling, power control, SINR, link adaptation, PHY
abstraction — in about 120 lines. Its `step()` is one
`@tf.function(jit_compile=True)`, which also confirms XLA works end to end on
this line. The full call chain is reproduced in the appendix. Two deltas for
us: it is downlink, so `downlink_fair_power_control` becomes
`open_loop_uplink_power_control`; and it derives noise physically from
`BOLTZMANN_CONSTANT * temperature * subcarrier_spacing` with `scene.bandwidth`
set, rather than sweeping Eb/No. The second is arguably the more honest system
model for a factory reliability claim, but it is a reporting change — decide
deliberately whether the SYS arm keeps the Eb/No x-axis for comparability with
the link-level families.

Caveat worth stating in the methodology: PHY abstraction inherits 3GPP's
LDPC/PUSCH tables, so the abstracted arm cannot also carry the custom-estimator
contribution. It is the *resource-management* arm; the estimator chapter stays
link-level. Cross-validating one overlapping operating point between the two
arms is the standard way to defend the split.

---

## 2. Feed the ray-traced factory channel into the main loop — and get mobility free

`docs/SIMULATION_REVIEW.md` §3.1 lists RT-based site-specific channels as
"deliberately not done — a research project rather than a fix". **Most of it is
already built, and the official recipe is simpler than the one I first
proposed.**

Already in place:

* `scripts/tools/generate_factory_dataset.py` ray-traces the hall — metal
  machines, concrete walls, `PathSolver` — and stores `paths_a` / `paths_tau`.
* `data/factory_dataset_1k.h5` is 51 MB of exactly that.
* `scripts/tools/train_estimator.py` already converts it with
  `cir_to_ofdm_channel`.

The missing adapter is **not** `CIRDataset`. `Paths` exposes `cfr()` directly:

```python
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)  # from sionna.rt
paths = p_solver(scene, max_depth=8)
h_freq = paths.cfr(frequencies=frequencies,
                   sampling_frequency=1/resource_grid.ofdm_symbol_duration,
                   num_time_steps=resource_grid.num_ofdm_symbols,
                   out_type="tf")
```

That returns the channel frequency response in the shape
`ChannelModel.sample_frequency_response()` already produces, so a
`channel_model_type: "rt"` branch in `components/channel.py` is a substitution,
not a new data path. (Confirmed: `Paths.cfr` is in the `sionna-rt` 1.2.1 wheel
we have, with `out_type="tf"`.) `CIRDataset` remains the right tool if we want a
*frozen* dataset for reproducibility; `cfr()` is the right tool for live
generation.

**The mobility trick is the part worth stealing.** `SYS_Meets_RT` models moving
users by adding a receiver at every future position up front, solving paths
once, then reshaping the result into slots:

```python
step = (ut_pos_end - ut_pos_start) / (num_slots - 1)
for slot in range(num_slots):
    pos = ut_pos_start + slot * step
    for ut in range(num_ut):
        scene.add(Receiver(f"ut{ut}_slot{slot}", position=pos[ut, :]))
# ... one solve ...
h_freq = tf.reshape(h_freq, [num_slots, num_ut] + h_freq.shape[2:])
```

This is a direct answer to §3.2. Right now mobility is a first-order Jakes
coefficient, `rho = J0(2*pi*f_d*tau)`, aging a static channel. With this, an AGV
traversing the hall produces channel evolution from **actual geometry** —
shadowing as it passes behind a machine, LOS/NLOS transitions at real
positions, Doppler from `Paths.cfr`'s own time evolution. "Our learned scheduler
handles AGV mobility" is a much stronger claim when the mobility is traced
rather than modelled by a correlation coefficient. It also gives the
worst-user-BLER metric something physical to be worst about.

Two things to size first: the existing H5 is 1k samples per profile at a single
BS position, which may not be enough independent realisations for a full sweep;
and it was traced at whatever `carrier_frequency` was configured then, so an FR3
family needs a re-trace at 13 GHz. Both are script re-runs, and the re-trace is
also when to turn diffraction on (§6).

## 3. A matched LMMSE baseline — the estimator claim currently under-tests itself

`components/estimators/lmmse_estimator.py` builds its covariance from an
**assumed** exponential correlation, `R = 0.98^|Δk|`, fixed at construction and
independent of the actual channel. Sionna ships the matched alternative:

* `sionna.phy.ofdm.LMMSEInterpolator` — LMMSE across frequency, time and space
  (it has an internal `SpatialChannelFilter`).
* `tdl_freq_cov_mat(model, subcarrier_spacing, fft_size, delay_spread, ...)` and
  `tdl_time_cov_mat(model, speed, carrier_frequency, ofdm_symbol_duration, ...)`
  — covariance matrices computed **from the delay spread and the UE speed**.

`components/channel.py` already computes the InF RMS delay spread per link
(`inf_delay_spread_seconds`), and `frequency_selectivity_report()` reports it, so
the inputs those two helpers need are in hand.

This matters for the lead contribution. `THESIS_RESULTS.md` records that LMMSE
has the best NMSE at every point and the worse BER, and that the adaptive
hybrid's LMMSE branch never wins. A referee's first question is whether that is
a property of LMMSE or of *this* LMMSE — a mismatched covariance and a
three-to-four-times understated error variance are both live explanations. Adding
a genie-covariance `LMMSEInterpolator` arm settles it either way:

* if the matched LMMSE also loses on BER, the NMSE-does-not-predict-BER finding
  gets much stronger, because it survives a correctly specified estimator;
* if it wins, we have found the real bound and the adaptive hybrid should be
  retuned against it — which is already the top item in the known-gaps list.

Cheap, and it changes what the headline claim is worth. Note the interpolator
also needs `err_var` handling consistent with the calibration table already
being reported.

---

## 4. Make InF a real `SystemLevelScenario` instead of a NumPy overlay

`components/inf_channel.py` + `ChannelModel._apply_inf_large_scale()` implement
TR 38.901 InF as a NumPy post-multiplication onto a `RayleighBlockFading` draw:
large-scale gain times an exponential-PDP frequency response. It was the right
fix for §3.1 and it made hall geometry matter. Its limits are structural:

* i.i.d. taps per antenna pair — **no spatial correlation**, so the 8-antenna BS
  array has no realistic angular structure and MIMO separability is fictional;
* the frequency response is broadcast over OFDM symbols — **no Doppler within a
  slot**, so the mobility story rests entirely on the separate Jakes CSI-ageing
  term;
* no spatial consistency between nearby users, and no LSP cross-correlation;
* it runs in NumPy on the CPU, which is a fixed cost per batch and works against
  the `graph_mode` speedup.

Sionna does not ship InF (confirmed: `phy/channel/tr38901/` has RMa, UMa, UMi,
TDL, CDL and nothing else in both 1.2.1 and 2.0.1) — but it ships the machinery.
`SystemLevelScenario` is abstract with a small required surface:
`los_probability`, `min_2d_in` / `max_2d_in`, `rays_per_cluster`, the LOS/NLOS/O2I
parameter file paths, `clip_carrier_frequency_lsp`, `_compute_lsp_log_mean_std`,
`_compute_pathloss_basic`. `UMiScenario` is 233 lines; the parameter files are
JSON tables under `phy/channel/tr38901/models/`. An `InFScenario` subclass plus
an `InF` wrapper over `SystemLevelChannel` (`umi.py` is 123 lines) would give
InF the full LSP -> rays -> coefficients pipeline: angular spreads, spatial
consistency, Doppler, cluster structure, all inside the TF graph and on the GPU.

The InF tables needed are TR 38.901 Table 7.4.2-1 (LOS probability), 7.4.1-1
(path loss) — both already transcribed in `inf_channel.py` — plus Table 7.5-6
Part 3 for the delay and angular spread distributions, which is the new work.

This is the most standards-defensible version of the factory channel, it is a
publishable artifact on its own ("an open InF scenario for Sionna"), and it is
strictly more work than items 1-3. Sequence it after them.

---

## 5. `TBEncoder` / `TBDecoder` — unblocks the wideband result family

Known gap: *"No LDPC code-block segmentation. One codeword per user per grid caps
FR1 bandwidth at about fft 512, which is what blocks a 100 MHz study."*

`sionna.phy.nr` ships `TBEncoder` and `TBDecoder`, which do exactly transport
block segmentation into code blocks with CRC, plus `TBConfig`,
`calculate_tb_size` and `decode_mcs_index`. `components/transmitter.py` uses
`LDPC5GEncoder` directly and `receiver.py` uses `LDPC5GDecoder`; swapping in the
TB pair is a localised change to those two files and to how BLER is counted
(transport-block error rate rather than per-codeword, which is also the metric
`PHYAbstraction` reports — so items 1 and 5 line up).

That removes the bandwidth cap and lets the FR3 mini-slot family run at a
carrier width a 6G claim can stand on, instead of the current 61.4 MHz ceiling
being an artifact of the codeword construction.

Also in `phy.nr`: `PUSCHTransmitter` / `PUSCHReceiver` / `PUSCHConfig` /
`PUSCHDMRSConfig`, a standards-compliant uplink chain with real DMRS patterns.
Worth a look for the estimator chapter — the current pilot pattern is
`pilot_ofdm_symbol_indices: [2, 11]`, which is a reasonable stand-in for DMRS but
is not DMRS. Optional; it would make the estimator results directly comparable to
published NR channel-estimation work.

---

## 6. Ray-tracing API drift — one function is dead, one line is inert

`visualization/factory_visualizer.py` was written against the Sionna 0.19 RT API
and never fully updated:

* **`render_coverage_map()` cannot work.** It calls `scene.coverage_map(...)`;
  `Scene` in `sionna-rt` 1.2.1 has no such method — radio maps moved to
  `RadioMapSolver`. The call sits outside the function's internal `try`, so it
  raises `AttributeError`, and the caller at line 429 catches it and logs
  "Coverage map failed — skipping". **The factory coverage map has silently not
  been produced.** Worth checking whether any figure in the thesis or the weekly
  reports is captioned as one. The current-API replacement, from `SYS_Meets_RT`:

  ```python
  rm_solver = RadioMapSolver()
  rm = rm_solver(scene, max_depth=8, cell_size=(1, 1), samples_per_tx=10_000_000)
  scene.render(camera=cam, radio_map=rm, rm_metric="sinr", rm_show_color_bar=True)
  ```

  Note `rm_metric` accepts `"path_gain" | "rss" | "sinr"`. An **SINR** map of the
  factory floor — showing where in the hall a device cannot meet its target — is
  a considerably better figure for a reliability thesis than the path-gain
  heatmap the dead function was trying to produce.
* **`scene.synthetic_array = True`** (`factory_visualizer.py:158`,
  `generate_factory_dataset.py:420`) sets an attribute `Scene` does not define.
  It is not a property in 1.2.x — `synthetic_array` is a `PathSolver.__call__`
  keyword. There are no `__slots__`, so the assignment silently creates an
  unused attribute. Behaviour happens to be correct because the solver default is
  `True`; the line should move to the solver call or go.
* **Diffraction is new and off by default.** `PathSolver.__call__` in 1.2 gained
  `diffraction`, `edge_diffraction` and `diffraction_lit_region`; the repo passes
  only `max_depth` and `samples_per_src`. In a hall full of metal boxes,
  diffraction around machine edges is precisely the mechanism that fills the
  shadow regions where the worst-user BLER lives. Turning it on for the dataset
  re-trace is a one-argument change with a real physical effect, and "we model
  edge diffraction around machinery" is a defensible sentence.

---

## 7. Smaller items

* **CPU pinning.** `components/channel.py` wraps channel generation, channel
  application and noise sampling in `with tf.device("/CPU:0")` (lines 133, 307,
  318). Whatever this worked around, it forces the heaviest tensor ops onto the
  CPU and works directly against `graph_mode` and the GPU service in
  `docker-compose.yml`. Re-test without it before the long runs.
* **`MMSEPICDetector`.** `sionna.phy.ofdm` ships MMSE-PIC — iterative detection
  and decoding. The `athirah/` JIDD-SCMA port is a joint iterative detection and
  decoding scheme; an MMSE-PIC arm on the OFDM chain is the natural
  non-SCMA comparator for it, and IDD is the standard way to buy back the
  reliability that estimation error costs (the measured cause of the TR 38.901
  floor).
* **Moving machines, not just moving users.** §2 covers AGV mobility via
  multi-position receivers. `Scene.edit()` additionally repositions scene
  *objects* between solves, so a robot arm or a shuttling AGV can be a moving
  metal scatterer rather than static clutter. That is a factory-specific
  propagation effect with no counterpart in UMi, and therefore a defensible
  novelty claim.
* **`out_type`** on both `Paths.cir` and `Paths.cfr` accepts `"tf"`, `"numpy"`,
  `"jax"`, `"torch"`, `"drjit"`. The dataset generator uses `"numpy"` and
  converts later; `"tf"` avoids the round trip.

---

## 8. Sionna Research Kit — the hardware validation path

Not a code upgrade, but it appears in the docs index at both `v1.2.2` and
`main` and is directly on-topic for the degree, so it belongs in the plan.

The **Sionna Research Kit** (`NVlabs/sionna-rk`, documented under
`nvlabs.github.io/sionna/rk/`) is a GPU-accelerated software-defined 5G RAN
built on OpenAirInterface, with O-RAN-compliant interfaces, for deploying
**trained AI/ML components into a real radio access network**. The pinned-line
docs cite the NVIDIA Jetson AGX Thor platform; `main` cites DGX Spark.

Why it matters here: the thesis trains a neural channel estimator and learned
schedulers and evaluates them only in simulation. The standing weakness of that
form of contribution is "does it survive contact with a real radio?" SRK is the
NVIDIA-supported answer, and it is the same team and the same component
abstractions, so a Sionna-trained block is the intended input.

Realistically this is out of scope for the remaining thesis timeline and needs
hardware we do not have. But it is the right **future work** paragraph — far
more concrete than "we plan to validate experimentally" — and if any hardware
becomes available, the neural estimator is the component to try first, because
it is a drop-in PHY block rather than a scheduler needing MAC integration.

---

## Appendix: the verified `SYS_Meets_RT` call chain

From tag `v1.2.2` (our TensorFlow line). Every symbol here was confirmed present
in the 1.2.1 wheels we already have. This is the skeleton for items 1 and 2.

```python
from sionna.rt import (load_scene, Transmitter, Receiver, PlanarArray,
                       RadioMapSolver, PathSolver, subcarrier_frequencies)
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (ResourceGrid, RZFPrecodedChannel,
                             LMMSEPostEqualizationSINR)
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.nr.utils import decode_mcs_index
from sionna.sys import (PHYAbstraction, OuterLoopLinkAdaptation,
                        PFSchedulerSUMIMO, downlink_fair_power_control)
from sionna.sys.utils import spread_across_subcarriers

phy_abs   = PHYAbstraction()
olla      = OuterLoopLinkAdaptation(phy_abs, num_ut=num_ut,
                                    bler_target=0.1, batch_size=[num_bs])
scheduler = PFSchedulerSUMIMO(num_ut, num_subcarriers, num_ofdm_symbols,
                              batch_size=[num_bs],
                              num_streams_per_ut=num_streams_per_ut, beta=.9)

@tf.function(jit_compile=True)
def step(h, harq_feedback, sinr_eff_feedback, num_decoded_bits):
    # scheduling -> power control -> SINR -> link adaptation -> PHY abstraction
    ...
```

Per-slot order inside `step()`: `scheduler(...)` returns `is_scheduled` per
resource element; `num_allocated_re` is reduced from it; power control produces
per-UT power, spread over subcarriers by `spread_across_subcarriers`; the
precoded channel gives post-equalisation SINR; OLLA picks the MCS from the last
effective-SINR feedback and the HARQ ACK/NACK history; `PHYAbstraction` returns
decoded bits, HARQ feedback, effective SINR and TBLER, which feed the next slot.

The loop over slots stays in Python and only the `step` is compiled — which is
also the structure `system.graph_mode` already uses, so the two fit together.

For our uplink, swap `downlink_fair_power_control` for
`open_loop_uplink_power_control`. Note this makes the per-UT power limit the
physically correct constraint, which is exactly the correction recorded in
`SIMULATION_REVIEW.md` §4.6.

---

## Suggested order

| # | Item | Why now | Size |
|---|---|---|---|
| 0 | `sionna` 1.2.2 + drop the Dockerfile mitsuba override | Do it before regenerating results in the container | Hours |
| 6 | Fix the RT API drift (radio map, `synthetic_array`, diffraction) | A figure is missing; the SINR map is a better one | Hours |
| 3 | Matched-covariance LMMSE arm | Directly hardens the lead contribution | Days |
| 2 | RT channel via `Paths.cfr` + traced AGV mobility | Data and scene code exist; also closes §3.2 properly | Days |
| 5 | `TBEncoder` / `TBDecoder` | Unblocks the wideband 6G family | Days |
| 1 | Sionna SYS resource-management arm | Largest gain; official blueprint exists for our backend | Weeks |
| 4 | `InFScenario(SystemLevelScenario)` | Most defensible statistical channel; publishable alone | Weeks |
| 8 | Sionna Research Kit | Future work; needs hardware | — |
| — | Sionna 2.x / PyTorch migration | Future work, not this thesis | — |

Items 2 and 1 now share a spine — `SYS_Meets_RT` does both in one loop — so if
the SYS arm is going to happen, build item 2 in the shape the tutorial uses and
the SYS arm becomes an extension rather than a rewrite.

None of this is verified by execution. Item 0 and item 6 should be confirmed in
the Docker image first, since they are the cheapest and they gate the
regeneration run that `THESIS_RESULTS.md` already says is outstanding.
