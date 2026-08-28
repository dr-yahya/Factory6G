# Factory6G

A 6G smart-factory simulation research project that combines a reproducible
codebase with curated research artifacts, thesis material, and stakeholder
progress reports.

## Language

**Codebase**:
The Docker-first simulation software: CLI entrypoint, configuration, source,
tests, and training scripts used to run and extend experiments.
_Avoid_: Core, engine, platform

**Research evidence**:
Simulation outputs and derived artifacts that support claims in reports,
papers, or the thesis—summaries, key plots, comparison tables, and linked run
metadata—not every intermediate Monte Carlo dump.
_Avoid_: Results (when meaning the full raw run tree), data (when meaning
training datasets)

**Progress report**:
A dated stakeholder-facing summary of project status, findings, and next steps,
typically organized by week under `reports/`.
_Avoid_: Weekly update, status doc

**Thesis**:
Long-form academic writing and its source files (outlines, chapters, citations)
maintained alongside the project, distinct from weekly progress reports.
_Avoid_: Paper, dissertation (unless that is the formal degree term in use)

**Curated evidence**:
The subset of research evidence intentionally kept under version control because
it is cited, summarized, or needed to reproduce a published figure or table.
_Avoid_: Golden results, pinned outputs

**Evidence promotion**:
The act of copying or summarizing selected artifacts from a full run into a
tracked report or evidence folder; only citation- or milestone-relevant material
is promoted under the lean-git policy.
_Avoid_: Syncing results, archiving runs

**Weekly report package**:
The curated stakeholder bundle for one date under `reports/weekly/<date>/`:
messages, deck, tables, preview images, and assets for that reporting period.
_Avoid_: Sprint summary, status folder

**Evidence bundle**:
A folder of promoted artifacts plus a manifest describing the source full run;
week-local bundles live under `reports/weekly/<date>/assets/`, cross-cutting
bundles under `reports/evidence/<topic>/`.
_Avoid_: Snapshot, export folder

**Full run**:
A complete timestamped simulation output directory (logs, stage JSON/CSV, plots,
checkpoints) produced by `src/factory6g/cli/run.py`; treated as local workspace output under a
lean-git policy.
_Avoid_: Raw results, experiment folder

**Results untracking**:
Removing full runs from git index while keeping them on disk locally; promoted
curated copies live under `reports/` before untracking.
_Avoid_: Deleting results, archiving

**Strict results zone**:
`results/` contains only timestamped full-run directories and its README;
archives, loose plots, and non-run folders belong elsewhere or are removed.
_Avoid_: Mixed output folder, dump directory

**`outputs/`**:
Local gitignored home for regenerable script-generated artifacts before optional
promotion into `reports/`.
_Avoid_: results/, temp

**Lean git policy**:
Version control holds the codebase, documentation, curated evidence, and
lightweight thesis/report sources; large or regenerable artifacts stay local or
use Git LFS only when sharing binaries is required.
_Avoid_: Thin repo, minimal git

## Repository zones

**`results/`**:
Local workspace for full runs only—timestamped simulation output that can be
regenerated from the codebase and `config/config.json`.
_Avoid_: Evidence store, archive

**`reports/`**:
Stakeholder progress reports and curated research summaries derived from runs.
_Avoid_: Output folder, logs

**`reports/evidence/`**:
Cross-cutting curated evidence (plots, tables, manifests) cited in multiple
reports or the thesis, promoted from full runs.
_Avoid_: results/, plots dump

**`thesis_writing/`**:
Thesis source and writing workspace; version-controlled text sources and cited
figures, local-only build outputs and bulk rendered assets.
_Avoid_: reports/, notes

**Thesis source**:
Editable thesis inputs kept in git: markdown/LaTeX, bibliography, and helper
scripts under `thesis_writing/tools/`.
_Avoid_: Draft, export

**Cited thesis figure**:
A plot or diagram embedded in the thesis, kept under a dedicated figures folder
in git; distinct from bulk rendered page images.
_Avoid_: Screenshot, render output
