# Thesis Result-Family Configs

The canonical configs behind the result families in `docs/THESIS_RESULTS.md`.
Each file is a complete config: run it with `--config config/thesis/<name>.json`.

These were reconstructed in September 2026. Before that, four of the six families
had no config in the repo at all, `estimators_inf_s.json` survived only as two
divergent copies inside `reports/evidence/`, and the one reproduction command in
`THESIS_RESULTS.md` pointed at a path that did not exist. `tests/test_thesis_configs.py`
now fails if a config named in the docs goes missing again.

## The families

| Config | Claim | Hall | Bandwidth | Selectivity |
|---|---|---|---|---|
| `estimators_inf_s.json` | **Lead contribution** — estimator comparison in a factory hall | 15×15×5 m, 5 machines, 4 UT | 61.44 MHz | 7.27 |
| `estimators_inf_m.json` | Scaling with hall size and device count | 25×25×6 m, 10 machines, 8 UT | 61.44 MHz | 9.15 |
| `estimators_inf_l.json` | Scaling continued | 40×40×8 m, 20 machines, 16 UT | 61.44 MHz | 12.11 |
| `estimators_inf_narrowband.json` | **Control** — estimators converge when the channel is flat | 15×15×5 m, 4 UT | 3.84 MHz | 0.45 |
| `estimators_umi.json` | Comparison arm — where the estimation floor lives | UMi | 3.84 MHz | — |
| `resource_managers_inf.json` | Resource-management chapter | 25×25×6 m, 8 UT | 61.44 MHz | 9.15 |

Selectivity is signal bandwidth over coherence bandwidth. The InF families are
FR3 mini-slot (13 GHz, 512 × 120 kHz, 4 OFDM symbols); the narrowband control and
the UMi arm are FR1 (3.5 GHz, 128 × 30 kHz, 14 symbols).

## Hall geometry sets the delay spread

None of these configs set `system.inf_hall_volume_m3` or
`system.inf_hall_surface_m2`, and new ones should not either. Omitted, they are
derived from `factory_scenario.room_dimensions`, so the S/M/L sweep is a sweep
over a real propagation property.

This used to be broken. Both keys were plain floats defaulting to the *small*
hall, and `asdict()` always materialised them, so the geometry fallback in
`ChannelModel._apply_inf_large_scale` never ran — every hall size simulated one
delay spread. The shipped `config/config.json` also carried a hand-entered
`inf_hall_surface_m2: 900.0` against the 750 m² its own room dimensions imply,
making the simulated delay spread 12% shorter than the figure in the docs. The
keys are now optional overrides defaulting to `None`, and must be set together or
not at all.

**Any InF evidence generated before this fix used the wrong delay spread and
needs regenerating.**

## Preconditions

`resource_managers_inf.json` enables `rl`, which loads
`models/rl_resource_manager_policy`. **That artifact does not exist in the
repository** — training a thesis-grade RL policy is item 5 of the outstanding
work in `docs/SIMULATION_REVIEW.md` §5. With `strict_policy_loading: true` the
run will fail until it is trained. Either train it first, or drop `rl` from
`resource_managers.enabled` to run the family without the imitation-vs-RL
ablation.

The `drl` arm loads `models/drl_resource_manager_policy`, which is present.
