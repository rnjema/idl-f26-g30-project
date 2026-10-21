# AGENTS.md — Guidance for AI coding agents

## Project overview

This is **idl_f26_g30_project** ("Team Support Vector", IDL F26 Group 30), a machine-learning
research project for the **iWildCam Challenge (WILDS benchmark)**: wildlife camera-trap image
classification under distribution shift across 323 camera locations. The goal is *selective
classification under shift* — a classifier that knows when to abstain — built on top of a frozen
CLIP backbone fine-tuned with FLYP.

The codebase follows a [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
layout. Filled-in so far: `iwild/dataset.py` (WILDS iWildCam download/verification, FLYP
fine-tuning CSV, location splits) and the AutoFT baseline lane in `iwild/modeling/` (see "AutoFT
baseline" below). `iwild/features.py` and `iwild/plots.py` are still template stubs; the selector
(CF-Δ) lane is not implemented yet. The research planning lives in three markdown documents at
the repo root:

- `BIRCH_iWildCam_refined.md` — the core plan: **CF-Δ**, using spherical BIRCH CF-trees
  (sufficient statistics `(N, LS, SS)` over L2-normalized embeddings) as a compressed
  correct-vs-wrong evidence index for a Neyman–Pearson-style abstention selector.
- `Regime_MoE_iWildCam.md` — stretch lane: regime-conditioned mixture-of-experts ViT where the
  CF-tree supplies the gating signal.
- `Credal_Regimes_iWildCam.md` — stretch lane: imprecise-probabilistic (credal set / IDM)
  soft mixtures for the same robustness goal.

Key benchmark facts that constrain all code (from the planning docs): iWildCam-WILDS has 203,029
labeled images, 182 classes (~35% are "empty"), and the official metric is **macro F1**, not
accuracy. All experiments use a **leakage-free location partition** (splits A/B by camera trap ID)
and tune only on the official OOD val split — OOD test is touched once. Embeddings must be
L2-normalized before any BIRCH/cosine work.

## Technology stack

- **Language:** Python 3.12 (`requires-python = "~=3.12"`).
- **Packaging:** `pyproject.toml` with the `flit_core` build backend; the importable package is
  named `iwild` (installed editable). A minimal `uv.lock` exists, but **uv is not the primary
  workflow — conda is** (see below).
- **CLI / scripting:** `typer` for command-line entry points, `loguru` for logging (wired through
  `tqdm.write` in `iwild/config.py`), `tqdm` for progress bars, `python-dotenv` for `.env` files.
- **Data science:** numpy, pandas, scikit-learn, matplotlib (from `environment.yml`).
- **Docs:** mkdocs (source in `docs/docs/`, config in `docs/mkdocs.yml`).

## Environment setup and build commands

**Workstation reality (2026-10):** conda is not installed on the dev machine; the canonical env
is the repo's `.venv` (Python 3.12.8, created by uv — it has **no pip by default**; bootstrap with
`python -m ensurepip` if needed). C: is 100% full and pip's cache lives there, so install with
`--no-cache-dir`. `environment.yml` + `make requirements` remain the definition of the base
data-science env for anyone setting up via conda elsewhere.

```bash
make requirements-train    # torch+torchvision from the PyTorch cu128 index, then requirements-train.txt
```

After that, run Python modules directly, e.g. `python iwild/dataset.py download`,
`python iwild/modeling/train.py autoft --smoke`. There is no test suite, no CI, and no build
step beyond installing the package. CLIP/open_clip weight downloads are pinned to
`models/hub/` on the project drive via `TORCH_HOME` in `iwild/config.py` (C: is full).

Other Makefile targets:

- `make lint` — flake8 + isort (black profile) + black, all checked against `iwild/`
- `make format` — apply black formatting
- `make data` — runs `python iwild/dataset.py download` (depends on `requirements`)
- `make clean` — delete `__pycache__` / `*.pyc`
- `make` / `make help` — list available rules
- Docs: `mkdocs build` / `mkdocs serve` from within `docs/`

## Code organization

Cookiecutter Data Science layout:

```
iwild/                  # the only Python package (installed editable)
├── config.py           # central paths (DATA_DIR/MODELS_DIR/REPORTS_DIR overridable via
│                       #   $IWILD_DATA_DIR / $IWILD_MODELS_DIR / $IWILD_REPORTS_DIR; Modal mounts
│                       #   and the workstation both rely on this) + loguru setup
├── dataset.py          # Typer CLI: download/verify WILDS iWildCam, make-ft-csv, location-split
├── features.py         # feature-generation CLI stub (selector lane goes here later)
├── plots.py            # plotting CLI stub
└── modeling/
    ├── train.py        # Typer CLI: zeroshot / autoft / flyp / eval / wisefmt commands
    ├── autoft.py       # Optuna outer loop + inner/full fine-tuning routines
    ├── autoft_model.py # CLIP wrapper (open_clip), zero-shot head, checkpoint IO
    ├── autoft_data.py  # WILDS split datasets, FLYP CsvDataset, class-balanced OOD-HP subset
    ├── autoft_losses.py# the 9 AutoFT loss terms + LearnedLoss
    ├── autoft_hparams_iwildcam.json  # pre-learned loss weights/lr/wd from the AutoFT paper
    ├── iwildcam_metadata/labels.csv  # y -> species mapping (from the official auto-ft repo)
    └── predict.py      # inference CLI stub (selection/predict lane, not implemented yet)
data/{raw,interim,processed,external}/   # datasets — all git-ignored except .gitkeep dirs
models/                 # trained model artifacts + hub/ (CLIP weight cache); git-ignored content
scripts/
├── modal_autoft.py     # Modal app: iwildcam-data + iwildcam-ckpts volumes, staged runs,
│                       # periodic volume commits, W&B via a 'wandb' Modal secret
notebooks/              # Jupyter notebooks; naming convention: `N.N-initials-short-description`
reports/figures/        # generated figures
references/             # papers, data dictionaries
docs/                   # mkdocs project
```

Conventions to follow when replacing the stubs:

- **Every script is a Typer app** with a `main` command taking `pathlib.Path` parameters that
  default to the canonical directories from `iwild.config`. Keep that pattern — it makes paths
  composable and overridable from the CLI.
- **Use the paths from `iwild.config`** (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`,
  `FIGURES_DIR`, ...) rather than hard-coding directory names.
- **Log with `loguru`** (`logger.info` / `logger.success`), not `print`; **wrap loops in `tqdm`**.
- `.env` is git-ignored; secrets/config go there and are loaded via `load_dotenv()` in
  `iwild/config.py`.

## Code style

- **Line length 99** everywhere (black in `pyproject.toml`, flake8 in `setup.cfg`).
- **black** for formatting, **isort with the black profile** (`force_sort_within_sections`,
  `known_first_party = ["iwild"]` configured under `[tool.ruff.lint.isort]`), **flake8** for lint
  (ignores E731, E266, E501, C901, W503). Always run `make lint` before considering work done.
- Match the existing import style: stdlib first, then third-party, then `from iwild...`.

## Testing

**There is no test framework configured yet** (no pytest, no CI, no test directory). Do not
assume tests exist or silently add a framework without checking with the team first; the current
quality gate is `make lint`. Experiment correctness is currently enforced by the research protocol
in the planning docs (location-disjoint splits, OOD-test-once) rather than by automated tests —
when writing experiment code, encode those protocol rules as assertions (e.g., assert no location
ID crosses split slabs), as the docs explicitly demand.

## Data and security considerations

- `data/` is git-ignored — datasets are large and must not be committed. Place raw data in
  `data/raw/`; never edit it in place (write derived data to `interim/` or `processed/`).
  The iWildCam v2.0 download is ~12 GB compressed / ~15 GB extracted — check free space first;
  the dev workstation's D: has been nearly full (18 GB free) and C: is 100% full.
- Download/verify: `python iwild/dataset.py download` fetches the CodaLab bundle directly
  (the `http(s)` endpoints share an expired TLS cert, so verification is disabled; the
  tarball is flat and is extracted into `iwildcam_v2.0/`, matching the WILDS layout, and
  leftover flat extractions are repaired in place). `verify` re-runs the split-size,
  class-count, and location-leakage assertions. `make-ft-csv` builds the FLYP fine-tuning
  CSV; `location-split` writes the seeded 80/20 trap split for the selector lane.
- `.env` (git-ignored) is for credentials such as dataset download tokens — never commit it and
  never print its contents.
- Model artifacts go in `models/` (content git-ignored); CLIP weight cache in `models/hub/`.

## Working notes for agents

- The three `*_iWildCam*.md` root documents are the authoritative research spec. Read
  `BIRCH_iWildCam_refined.md` first; it defines the experiment protocol, baselines (MSP,
  raw-logit margin, Δ-MDS/Δ-KNN reproduction), metrics (AURC, selective risk at 80%/90%
  coverage on OOD test), and the leakage rules that all experiment code must honor.
- Keep `README.md`'s "Project Organization" section and `AGENTS.md` in sync if you change the
  directory structure.

## AutoFT baseline (implemented 2026-10-18)

The classifier baseline is a native reimplementation of **AutoFT** (WILDS iWildCam leaderboard
#1, 52.0 OOD-test macro F1 with ViT-L/14@336 + WiSE-FT; FLYP 46.0). The leaderboard config does
not fit our RTX 4070 8 GB, so we target **CLIP ViT-B/16 @ 224px**; claims are always relative to
the FLYP run on the same backbone, never leaderboard entries.

Recipe (faithful to github.com/cchoi1/auto-ft @ 594e6a2): zero-shot head from 2 prompts × the
182 `labels.csv` classnames with `y < 99999` (matching WILDS v2.0's 182 classes; the 24
y ≥ 99999 rows are unused); FLYP FT CSV =
129,809 train images × 2 prompts; 9 loss terms (`ce` anchored at weight 1.0) tuned by Optuna TPE
+ MedianPruner(5,30,10) over loss weights / lr / wd / seed; objective = macro F1 on a
class-balanced 1000-example OOD-val subset; final fine-tune 20 epochs, best epoch by ID-val
macro F1, with per-epoch checkpoints so `--stage ft --resume` continues after a disconnect
(the pipeline splits into `--stage hopt` / `--stage ft`). Documented deviations: pandas (not
dask), state_dict checkpoints (not pickling),
single GPU (no DataParallel), batch 64×2 (VRAM), `--hopt-evals` default 100 (paper: 500), and the
class-balanced OOD-HP subset is actually applied (official flag is parsed but never wired).
Known ported quirks kept on purpose: scheduler LR comes from the base `--lr` (not the tuned
trial lr); final fine-tune steps every batch despite `accumulation_steps` loss scaling.

Driver: `notebooks/1.0-g30-autoft-baseline.ipynb` (local), `notebooks/1.1-...-colab.ipynb`
(Colab), `scripts/modal_autoft.py` (Modal volumes + W&B). W&B runs carry the leaderboard-entry
name/config (`submission` block in the run config). Smoke first:
`python iwild/modeling/train.py autoft --smoke --run-name smoke --no-init-hparams`.
