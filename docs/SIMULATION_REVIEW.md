# Factory6G Simulation Review — Gaps Against the Research Objectives

Status: static code review (2026-09-04). No Docker daemon was available in the
review environment, so nothing here was re-executed; every claim is traced to a
file and line, and the items marked **[verify]** need one short run to confirm
magnitude.

Research objectives assumed, from `CLAUDE.md` and `README.md`:
6G / B5G networks in **smart factory** environments, with **AI/ML** applied to
**physical-layer reliability** and **resource management**.

The harness itself is in good shape — fixed stage order, shared batch contexts,
confidence-aware BER, checkpoint/resume, a real config schema, and a test suite.
The problems below are not about code tidiness. They are about whether the
numbers the harness produces can support the claims the thesis wants to make.

---

## 1. Findings that contaminate results already reported

These change published curves. Fix and re-run before any of the current
evidence is defended.

### 1.1 LLR clipping at ±20 manufactures a high-SNR error floor

`src/factory6g/components/receiver.py:331`

```python
llr_clipped = tf.clip_by_value(llr, -20.0, 20.0)
```

The comment above it says a diagnostic found "27.5% of LLRs have |LLR| > 50,
which can cause decoder issues". That is not a decoder issue — at high Eb/No a
large fraction of LLRs *should* be large. Clipping them to ±20 destroys the
reliability information the BP decoder needs to correct the few genuinely weak
bits, and it does so exactly where the curve should be falling fastest.

This is the most likely origin of the "TR 38.901 BER floor" recorded in the
March and May weekly reports. **[verify]** Re-run one Eb/No sweep at ±20, ±200
and unclipped and compare the tail. If the floor lifts, every BER figure in the
thesis is affected and the fix is a one-line change plus a re-run.

Sionna's `LDPC5GDecoder` already guards its own internal dynamic range; an
external clip this tight is a debugging workaround, not a model.

### 1.2 The receiver is never told which users were scheduled

`src/factory6g/models/model.py:185`

Directives are applied at the transmitter (`transmitter.py:273-285`) and then
thrown away. `Receiver.equalize()` receives only `y`, `h_hat`, `err_var`, `no` —
it still equalizes and decodes **all** `num_ut` streams.

Consequences for a muted user: its pilots are zeroed with the rest of its grid,
so LS returns pure noise as its channel estimate; the LMMSE equalizer then
spends spatial degrees of freedom nulling interference that does not exist, and
noise-enhances the users that *are* scheduled. Meanwhile its all-noise decoded
bits are excluded from the statistics by `transmitted_ut_mask()`, so the cost is
invisible in the metrics but real in the BER of the active users.

Net effect: every RM that mutes users is systematically penalised relative to
`static`, which mutes none. That is a bias in the *baseline's* favour, so it
does not inflate the AI result — but it makes the whole RM comparison
unsound, and it hides how much the learned schedulers are actually worth.

Fix: thread `active_ut_mask` into `Receiver`, and restrict `StreamManagement` /
equalization / decoding to the scheduled streams.

### 1.3 `latency_ms` and `avg_power_w` measure the host CPU, not the system

`src/factory6g/models/model.py:189-242`, consumed at
`sim/stages/resource_managers.py:184`.

- `air_interface_latency_sec` = `(1/Δf)·(1+CP/N_FFT)·N_sym` — a constant of the
  numerology. For the shipped config: 0.540 ms, identical for every method, every
  Eb/No point, every scheduling decision.
- `runtime_latency_sec` = that constant **plus** `processing_latency_sec`, which
  is `time.perf_counter()` around eager TensorFlow equalize/demap/decode. On a
  batch of 20 in eager mode this dominates the 0.54 ms by one to two orders of
  magnitude.
- `_estimate_energy()` never sees the directives. Work through it: the `latency`
  factors cancel in the encode and decode terms, and `tx_energy`/`rx_energy` are
  fixed multiples of the constant air-interface time. Total energy per batch is
  therefore essentially a constant, varying only through the decoder iteration
  count.
- `avg_power_w = avg_energy / avg_latency` (`stages/common.py:258`) is then
  `constant / wall-clock`.

So `latency_vs_ebno.png` plots interpreter speed, and `power_vs_ebno.png` plots
its reciprocal. Neither figure carries information about the scheduler. In
particular **per-UT power allocation has literally zero effect on any reported
energy or power number** — WMMSE, queue-aware and the DRL policies all compute
power vectors that cannot show up in the results.

Fix: separate the three quantities that are currently fused.
- *Link latency* — model it: TTI duration × (1 + HARQ retransmissions) +
  queueing delay. Report the 99.9th/99.999th percentile, not the mean.
- *Energy* — `E = Σ_i P_i · T_symbol · N_active_symbols` from the actual
  directives, plus a fixed circuit term. This makes power control measurable.
- *Runtime* — keep it, but name it `runtime_sec` only and never mix it into a
  latency KPI.

### 1.4 Stateful schedulers keep state inconsistently, and lose it on resume

| Manager | State | Keyed by Eb/No? | Line |
|---|---|---|---|
| Round robin | `_current_index` | **no** | `resource_manager.py:62` |
| Proportional fair | `avg_rates` | **no** | `resource_manager.py:206` |
| Queue-aware | `_queues_by_state` | yes | `resource_manager.py:332` |
| DRL / reliability-DRL | `_avg_rate_by_state` | yes | `resource_manager.py:413` |

The sweep loop visits every Eb/No point inside every batch
(`stages/resource_managers.py:112-121`). Round robin's rotation pointer and PF's
average-rate filter are therefore driven by the *sweep order*, not by the time
evolution of one link: PF's fairness memory at +20 dB is polluted by what it saw
at 0 dB. The two learned managers do not have this problem. The baselines are
handicapped by a bookkeeping inconsistency, at exactly the comparison the thesis
turns on.

Worse, points drop out of the loop as they hit `target_block_errors`, so the
number of state updates a manager receives depends on how the *other* methods
are converging. Round robin's schedule is not even reproducible from the config.

Then `stages/checkpoint.py` serialises only counters — no RM internal state, and
no RNG state. A run resumed via the documented `--resume` workflow restarts
every scheduler from cold while its statistics continue accumulating. Long UMi
runs are precisely the ones that get resumed.

Fix: key all RM state by Eb/No point (or better, instantiate one manager
instance per point), and serialise RM state + RNG state in the checkpoint.

### 1.5 `static` is not a comparable baseline

`config/config.json` sets `num_active_users: 2` with `num_ut: 4`, but
`create_resource_manager()` builds `StaticResourceManager` with
`active_ut_mask=[1]*num_ut` (`resource_manager.py:527`) — all four users, full
power, `manager_kwargs` silently dropped.

So "static vs DRL" is a 4-user comparison against a 2-user comparison. Fewer
co-scheduled streams means less multi-user interference and better BER
essentially for free — that difference is a change of operating point, not a
scheduling result.

Fix: either give `static` the same `num_active`, or (better) report BER against
*served load* so the reader can see the throughput/reliability trade-off rather
than a single point. And stop dropping `manager_kwargs` for `static`.

### 1.6 The Rician channel is not Rician

`src/factory6g/components/channel.py:110-114`

```python
los = tf.cast(tf.sqrt(k / (k + 1.0)), tf.complex64)
h = los + nlos_scale * h
```

The LOS term is a **scalar**, added identically to every antenna pair,
subcarrier and OFDM symbol — and to every user. Total power is right
(`K/(K+1) + 1/(K+1) = 1`), but the spatial structure is wrong: with `K=1`, half
the channel energy of all four users is the *same* rank-one all-ones vector.
That makes the multi-user channel matrix artificially ill-conditioned. Whatever
the Rician results in the 2026-05-23 report show, they show that artifact.

A LOS component needs an array steering vector and a per-user phase:
`h_LOS = a_rx(θ_rx) a_tx(θ_tx)^H · e^{jφ}`. The cheap correct alternative is
Sionna's `TDL` with an explicit K-factor, or CDL-D/CDL-E, both of which carry
proper LOS geometry.

### 1.7 The DRL manager silently degrades to a hand-tuned heuristic

`src/factory6g/models/resource_manager.py:421-424`: if the checkpoint fails to
load, it prints a message and falls back to a heuristic actor. Nothing in
`stage_results_v2.json` records which one ran. A curve labelled `drl` in the
thesis may be a hand-written softmax rule, and there is no way to tell after
the fact from the artifacts.

`reliability_drl` also hardcodes a relative path
(`models/reliability_drl_resource_manager_policy`, line 524), so whether the
policy loads depends on the process working directory.

Fix: fail loudly by default (`strict=True`), keep the fallback behind an
explicit flag, resolve model paths against the config file's directory, and
record `policy_loaded`, checkpoint path and checkpoint hash in the stage output.

---

## 2. The "AI/ML" contribution is thinner than the naming suggests

### 2.1 The DRL resource manager is not reinforcement learning

`scripts/tools/train_drl_resource_manager.py` is `model.fit()` on a parquet of
oracle labels, with binary cross-entropy on the schedule head and MSE on power
and value. The metadata is honest about it —
`"checkpoint_type": "offline_actor_pretraining"`. There is no environment
interaction, no reward, no return, no temporal-difference target, no policy
gradient, no exploration, no discounting.

This is **behaviour cloning of a one-shot greedy oracle**. Two consequences:

1. It is bounded above by the oracle it imitates. It structurally cannot
   discover a policy better than the candidate search that labelled it, which is
   the whole point of an RL contribution.
2. Calling it DRL in a thesis is a terminology problem an examiner will find in
   the first ten minutes of reading the training script.

Either rename it honestly (`imitation`/`bc_policy`/`learned_scheduler`) and
frame the contribution as offline imitation of an oracle scheduler — which is a
legitimate, publishable framing — or implement actual RL. Given the harness
already exposes `prepare_batch_context` / `run_batch` / `ResourceDirectives`, a
contextual-bandit formulation is genuinely close: state = per-user channel
energy + Eb/No + fairness debt (already built), action = the top-k mask and
power vector, reward = a scalar you define from the batch outcome. That is a
REINFORCE or PPO loop over the existing simulator with maybe 200 lines. It also
gives you the ablation an examiner will ask for: imitation vs. RL vs. heuristic.

### 2.2 Train/serve feature skew on the fairness input

`train_drl_resource_manager.py:121` calls
`build_policy_training_inputs(channel_energy, ebno_db)` with no `fairness_debt`.
Following that through `drl_policy.py:218-226`, `:150` and `:186`: the fairness
feature is filled with ones, and `compute_policy_normalization` defaults
`fairness_mean = 1.0`, `fairness_std = 1.0`. The normalised fairness input the
network sees during training is therefore **constant zero**.

At inference, `resource_manager.py:466-468` computes a real normalised fairness
debt in [0, 1] and feeds it in, arriving at the network as [-1, 0].

The weights on that input were fit against a dead channel and are now driven by
a live signal. Whatever the fairness feature does at inference is untrained
behaviour. Either pass `fairness_debt` through at training time or drop the
feature — but the two paths must match. Worth adding a test that asserts the
training and inference state builders agree on a fixed input.

### 2.3 The oracle labels are selected on noise, and its utility is degenerate

`scripts/tools/generate_dataset.py:89-146` (`:105`):

```python
throughput_eff = throughput_bits / max(float(total_bits), 1.0)
utility = throughput_eff - latency_weight * latency_ms
```

`total_bits` counts only the **active** users (`ut_mask` is passed to
`extract_error_stats`), so `throughput_eff = 1 - BER` regardless of how many
users were scheduled. Scheduling one user scores as well as scheduling eight.
And `latency_ms` is the constant from §1.3. So the `utility` objective collapses
to "minimise BER, ignore throughput" — there is no reliability/throughput
trade-off in the labels at all. `throughput_bits` (the absolute figure that
*would* express the trade-off) is computed and then not used in the utility.
`--label-active-count` exists as a workaround for exactly this degeneracy.

Separately, each of the ~16 candidates is scored on **one** batch realisation
and the argmin-BER candidate is taken as the label. At the BERs involved that is
mostly a draw for luck — the winner's-curse problem. The label is the candidate
that got the favourable noise, not the best allocation.

Fix: (a) put absolute delivered bits in the utility so the objective actually
trades throughput against reliability; (b) average each candidate over several
independent noise draws with common random numbers across candidates before
ranking; (c) consider soft or ranking labels rather than a hard argmax.

### 2.4 The policy network's inductive bias fights the problem

`drl_policy.py:240-241` puts `Conv1D(kernel=3, padding="same")` over the **user**
axis. Users have no meaningful ordering — a 1-D convolution imposes a locality
prior between user 2 and user 3 that does not exist, and makes the policy
sensitive to user index permutation. The `GlobalAveragePooling1D` context branch
is correctly permutation-invariant; the conv branch undoes that.

Use a DeepSets or self-attention encoder over users instead. It is a small
change, it is the standard answer for set-structured scheduling, and it is easy
to defend in the thesis.

### 2.5 The adaptive estimator is SNR-switched, not channel-adaptive

`components/estimators/adaptive_estimator.py:54-58`:

```python
signal_power = tf.reduce_mean(tf.abs(h_ls) ** 2)
noise_power = tf.reduce_mean(tf.cast(no, tf.float32))
return float((signal_power / noise_power).numpy())
```

One scalar for the whole batch. Under `normalize_channel=True` that ratio is
essentially `1/no`, i.e. a deterministic function of the Eb/No point. So the
"adaptive hybrid" is a fixed SNR threshold switch between DFT and LMMSE, with
hand-picked breakpoints at 3.0 and 12.0. It does not adapt to the channel.

To make this a defensible contribution, key the branch on something the channel
actually varies: estimated delay-spread / power-delay-profile energy outside the
CP window, LS residual energy, or a per-user coherence-bandwidth estimate — and
decide **per user and per resource block**, not once per batch. That is a real
adaptive estimator and it gives you a mechanism to explain in Chapter 4.

---

## 3. The simulation is not a factory, and not yet 6G

This is the gap most likely to be raised in the viva, because it goes to whether
the whole evidence base is on-topic.

### 3.1 The factory scenario has no effect on any reported number

`config.factory_scenario` (room dimensions, machines, metal/concrete materials)
is consumed only by the ray-tracing visualiser. The channel used by both
benchmark stages is `RayleighBlockFading` or TR 38.901 **UMi**, with UT positions
drawn by `gen_single_sector_topology(..., "umi")` (`channel.py:81`) — an outdoor
urban-micro sector hundreds of metres across, unrelated to a 15×15 m hall.

And `--factory-size s|m|l|apple` only changes `num_ut` (4 / 8 / 16 / 8) —
`flow.py:19-49` and the duplicate table in `cli/run.py:91-129`. Room dimensions
and machine counts pass through to a config field nobody reads on the simulation
path.

So the "factory size" experiments are user-count experiments. A statement like
"BER degrades in larger factories" is really "BER degrades with more co-scheduled
users in a UMi cell", and an examiner who opens `flow.py` will see that.

This is the single highest-value thing to fix, and there are three routes:

1. **TR 38.901 InF (recommended).** Rel-16 added Indoor Factory scenarios —
   InF-SL, InF-DL, InF-SH, InF-DH, InF-HH — with clutter density, clutter height
   and hall-size parameters that map directly onto your S/M/L presets. Sionna
   does not ship InF, but you only need the LOS probability, path loss and
   delay-spread/angular-spread tables from TR 38.901 §7.2/§7.4 layered onto a
   CDL/TDL model. This is the standard-compliant answer and it makes
   `num_machines` and `room_dimensions` physically meaningful.
2. **Use the ray tracer you already have.** `factory_visualizer.py` and the
   `ray_tracing` config build a factory scene with metal machines. Generating
   CIRs from it and feeding them into the OFDM chain would give genuinely
   site-specific factory channels — the strongest option scientifically, and the
   most work.
3. **Minimum viable.** Turn on `enable_pathloss` and `enable_shadow_fading`
   (both `false` today, `config.json:146-147`) and constrain the topology
   generator to the hall footprint. Cheap, and at least removes the "no path
   loss in a metal hall" objection.

### 3.2 No mobility — which removes the reason to learn a scheduler

`min_ut_velocity: 0.0`, `max_ut_velocity: 0.0`. Every UT is static, so there is
no Doppler and no channel aging between the feedback probe and the data
transmission.

That matters more than it looks. With perfectly fresh CSI, a greedy max-SNR rule
is close to optimal and there is little for a learned policy to add. Channel
aging under AGV/robot-arm mobility (0.5–3 m/s) is exactly the regime where a
policy that has learned the temporal statistics beats an instantaneous-CSI
heuristic. Turning on mobility, and adding a feedback delay between
`prepare_batch_context`'s probe and the data slot, creates the problem your AI
contribution is supposed to solve.

### 3.3 The numerology is 5G FR1, not 6G, and not URLLC

- `carrier_frequency: 3.5 GHz` — NR FR1. A 6G thesis should show at least FR3
  (7–15 GHz) or mmWave. The `jidd_scma` block already lists 28 GHz, so the
  intent exists somewhere.
- `fft_size: 128` at 30 kHz SCS = **3.84 MHz** of bandwidth. That is very narrow
  for any 6G claim and it also limits frequency diversity, which biases the
  estimator comparison.
- `num_ofdm_symbols: 14` = a full slot. URLLC in factories uses **mini-slots**
  (2–7 symbols) and 60/120 kHz SCS to hit sub-millisecond latency. There is no
  short-TTI configuration anywhere.
- No HARQ, no retransmission, no packet deadline, no reliability-vs-latency
  trade-off.

### 3.4 The Monte Carlo budget cannot reach the reliability regime being claimed

With the shipped config: 12 data symbols × 128 subcarriers × 2 bits × 0.5 rate
≈ 1536 info bits per codeword; `batch_size: 20` × `num_ut: 4` = 80 codewords per
batch; `max_batches: 20` → at most **2.46 Mbit and 1600 codewords per point**.

- Smallest resolvable BER (the 30-error rule in `stages/common.py:16`): ~1.2e-5.
- Smallest observable BLER: 1/1600 ≈ 6e-4. Reaching the configured
  `target_block_errors: 100` requires BLER ≳ 6e-2.

URLLC in factory automation targets 1e-5 to 1e-6 residual error. **The harness
is two to three orders of magnitude short of the regime the research is about**,
which is why so many points come back `upper_bound_only`. No amount of
confidence-bound plotting fixes that; the evidence simply is not there.

Getting there needs all three of:
- **Speed** (§4.1) — nothing in the simulation path is graph-compiled.
- **BLER as the headline metric**, not BER. Blocks are the independent unit;
  factory URLLC specifies BLER; and BLER is what the confidence maths in
  `common.py` is actually valid for (see §4.2).
- **Importance sampling or multilevel splitting** for the deep tail, or an
  explicit, documented extrapolation with its uncertainty. This is a normal,
  citable technique for rare-event link simulation and it is the honest way to
  claim 1e-6 without 1e8 codewords.

---

## 4. Method and statistics

### 4.1 Nothing is graph-compiled — this is what makes the runs take hours

There is no `tf.function` and no `jit_compile` anywhere in `src/`. Every batch
runs eager, and `adaptive_estimator.py:58` calls `.numpy()` on the quality proxy
mid-forward-pass, which would break tracing even if a decorator were added.

Sionna is built for `@tf.function(jit_compile=True)`. Making `run_batch`
traceable — replace the `.numpy()` branch selection with `tf.where`, hoist the
`.numpy()` conversions in `model.py:194-206` out of the inner loop — is
plausibly a 10–100× speedup. That is not an optimisation nicety: it is the
difference between the Monte Carlo budget in §3.4 being impossible and being an
overnight run. **Treat this as the highest-leverage engineering item.**

Secondary: the estimator stage builds a full `Model` per method
(`stages/estimators.py:55-62`), each with its own channel model and LDPC
encoder, when only the estimator differs.

### 4.2 The BER confidence interval assumes independence that does not hold

`ber_upper_confidence_bound()` (`stages/common.py:116`) treats `total_bits`
Bernoulli trials as independent. LDPC errors are strongly bursty — a failed
codeword produces hundreds of correlated bit errors. The effective sample size
is closer to the **number of codewords** than the number of bits, so the
intervals are far too narrow, by roughly the square root of the average burst
size.

Two fixes, both easy:
- Compute the primary interval on **BLER** over codewords (Clopper–Pearson),
  where independence is defensible.
- For BER, use a batch-level bootstrap or jackknife over Monte Carlo batches.
  You already store per-batch quantities; resampling batches respects the
  correlation structure.

The zero-error bound (`zero_error_upper_bound`, the `-ln(α)/n` rule) is correct
as a *bit*-level rule of three but inherits the same independence problem.

### 4.3 You have common random numbers and are not using them

The shared `BatchContext` design means every method at a given (batch, Eb/No)
sees the identical channel, noise and source bits. That is textbook common
random numbers, and it is the single best variance-reduction property the
harness has — but the results are aggregated per method independently, and the
comparison is left to the reader's eye on a log plot.

Record the **paired per-batch differences** and report a paired bootstrap CI on
`BLER_drl − BLER_pf` at each Eb/No. Paired CIs are dramatically tighter than
the marginal ones, and they let you write "the learned policy improves BLER by
X% (95% CI [a, b], n = N batches)" instead of "the DRL curve is lower". For a
thesis defending an AI contribution over baselines, that sentence is worth more
than another 10× of Monte Carlo.

Caveat to handle: the early-stopping rule retires points at different batch
counts per method, breaking the pairing. Either run all methods to a common
batch count for the paired analysis, or truncate to the shared prefix.

### 4.4 Reproducibility gaps

- One global seed (`cli/run.py:364-367`) drives one RNG stream shared by every
  method and point. Add or remove a method from `enabled` and **every channel
  realisation changes**, so runs with different method lists are not comparable.
  Derive a per-(stage, method-set, Eb/No, batch) seed instead — then a point is
  reproducible in isolation, comparable across runs, and resumable exactly.
- Checkpoints carry no RNG state and no RM state (§1.4).
- `config_snapshot` is written per stage (`output.py:46`), which is good, but
  there is no git SHA, no dirty-tree flag, no Sionna/TF version, and no hash of
  the loaded policy checkpoint. For a thesis these belong in `summary_v2.json`.

### 4.5 Missing metrics for the claims being made

- **No NMSE anywhere.** A channel-estimator benchmark with no estimation-accuracy
  metric is a strange artifact — `run_batch` already returns both `channel` and
  `channel_hat` (`model.py:197-198`) and then never compares them. NMSE vs Eb/No
  is the standard figure in every channel-estimation paper, it isolates estimator
  quality from the LDPC/equalizer, and it is about ten lines of work.
- **No analytic complexity.** "Adaptive achieves near-LMMSE at lower complexity"
  needs a multiplication/FLOP count per estimate, not wall-clock — the current
  runtime comparison is dominated by eager overhead and structurally unfair to
  the NumPy-based estimators (PSO, ISTA).
- **No fairness metric.** PF, queue-aware and the fairness-weighted DRL heads all
  exist to trade throughput against fairness, and nothing reports Jain's index or
  per-user rate distribution.
- **No per-user reliability / outage.** In factory URLLC the *worst* device sets
  the system reliability. Aggregate BER hides it — report the per-user BLER
  distribution and the 99th-percentile user.
- **`pilot_reuse_factor` is a dead control surface.** Three managers set it;
  `grep` finds no consumer outside `resource_manager.py` and `model.py:111`.
  Either delete it or implement it — pilot contamination under reuse is a real
  and interesting factory-density question.

### 4.6 Smaller items

- `create_resource_manager()` dispatches by substring, ordered
  (`resource_manager.py:508-546`). `"max"` catches anything containing "max";
  `"pf"` and `"prop"` are separate substrings for one manager. Replace with an
  exact-name registry dict.
- `_normalize_power()` (`resource_manager.py:148`) peak-normalises so the largest
  active power is 1.0 and floors the rest at 0.15–0.85. There is **no sum-power
  constraint**, so muting users simply radiates less total power rather than
  redistributing it, and WMMSE's optimised power vector survives only as a ratio.
  Since power ∈ (0, 1] can only *reduce* SNR relative to full power, any manager
  that does power control is handicapped against one that does not. Impose a sum
  constraint (`Σ P_i = P_total`) so power allocation is a real trade-off.
- `_instantaneous_rate_per_user()` (`resource_manager.py:92`) uses
  `10**(ebno_db/10)` as a linear SNR, skipping the Eb/N0 → SNR conversion
  (`num_bits_per_symbol`, coderate, RG occupancy). Harmless for ranking, wrong if
  the value is ever read as a rate — and the queue-aware manager does use it as a
  served-bits proxy (`resource_manager.py:367`).
- `config.json` sets `stop_policy: "threshold"` while `README.md` and
  `docs/ARCHITECTURE.md` describe `sweep` as the policy for Eb/No sweeps.
- `_FACTORY_SIZE_PRESETS` is duplicated verbatim in `cli/run.py:91` and
  `flow.py:19`.
- `sim/results.py` and `sim/simulation.py` are legacy shims reachable only via
  lazy imports in `sim/__init__.py` and one test.

---

## 5. Suggested order of work

**Before defending any current result** — these invalidate published figures:

1. Remove or widen the ±20 LLR clip and re-run one sweep to quantify the floor (§1.1).
2. Pass `active_ut_mask` to the receiver (§1.2).
3. Give `static` the same active-user count, or plot against served load (§1.5).
4. Key RM state per Eb/No point; checkpoint RM + RNG state (§1.4).
5. Make the DRL loader strict and record `policy_loaded` in the outputs (§1.7).
6. Stop reporting `latency_ms` and `avg_power_w` until they mean something (§1.3).
7. Fix or withdraw the Rician results (§1.6).

**Then unblock the science:**

8. Graph-compile the batch path (§4.1). Everything below depends on this.
9. Add NMSE to the estimator stage (§4.5) — cheapest real gain in the repo.
10. Switch the headline metric to BLER with Clopper–Pearson intervals, and add
    paired bootstrap CIs between methods (§4.2, §4.3).
11. Fix the train/serve fairness skew and add a regression test (§2.2).

**Then close the objective gaps** — these are the chapters, not the fixes:

12. **InF channel** (§3.1). Without this the thesis is about UMi.
13. **Mobility + CSI feedback delay** (§3.2). This is what creates the problem
    the AI is meant to solve.
14. **Real RL, or honest renaming** (§2.1). Pick one; the current situation is
    neither.
15. **URLLC KPIs**: mini-slot numerology, HARQ, per-user BLER and latency
    percentiles (§3.3).
16. **Rare-event simulation** for the 1e-5..1e-6 regime (§3.4).
17. Fix the oracle utility and label averaging (§2.3), and swap the Conv1D user
    encoder for a permutation-equivariant one (§2.4).

Items 12, 13 and 14 are the ones that decide whether the work reads as
*6G AI for smart factories* or as *a UMi link-level study with a learned
scheduler*. Everything else is in service of them.
