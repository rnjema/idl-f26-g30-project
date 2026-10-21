# idl_f26_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An attempt at scaling beyond the benchmark heights on the iWILD Cam Challenge

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         iwild and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── scripts
│   └── modal_autoft.py <- Modal app: persistent data/checkpoint volumes + W&B, no session limits
│
└── iwild   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes iwild a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── autoft.py             <- AutoFT Optuna outer loop + fine-tuning routines
    │   ├── autoft_model.py       <- CLIP wrapper, zero-shot head, checkpoint IO
    │   ├── autoft_data.py        <- WILDS split datasets, FLYP CSV dataset, OOD-HP subset
    │   ├── autoft_losses.py      <- AutoFT's 9 learned loss terms
    │   ├── autoft_wandb.py       <- Optional Weights & Biases logging (no-op without a key)
    │   ├── autoft_hparams_iwildcam.json <- Pre-learned loss weights/lr/wd (AutoFT paper)
    │   ├── iwildcam_metadata/    <- y -> species mapping (labels.csv)
    │   ├── predict.py            <- Code to run model inference with trained models
    │   └── train.py              <- CLI: zeroshot / autoft / flyp / eval / wisefmt
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Running on Modal (cloud GPUs, persistent volumes)

`scripts/modal_autoft.py` runs the same staged pipeline on Modal without Colab's
session limits. Data lives on the `iwildcam-data` volume (downloaded once), all
checkpoints on the `iwildcam-ckpts` volume (per-epoch resume survives preemption;
volumes are committed every 10 min inside the job), and W&B reporting is identical to
local/Colab — runs are named after the leaderboard entry (`AutoFT-<arch>-iWildCam-<run>`)
with the entry's paper/leaderboard config in `wandb.run.config.submission`.

```bash
pip install modal && modal setup
modal secret create wandb WANDB_API_KEY=... WANDB_PROJECT=... WANDB_ENTITY=...
# optional isolated world: modal environment create dev &&
#   modal secret create --environment dev wandb ... (then prefix runs with
#   MODAL_ENVIRONMENT=dev and add --environment dev)
modal run scripts/modal_autoft.py::download_data                      # once, CPU
modal run --gpu A100 scripts/modal_autoft.py::autoft_stage --stage hopt --run-name autoft_vitb16_official
modal run --gpu A100 scripts/modal_autoft.py::autoft_stage --stage ft   --run-name autoft_vitb16_official
```

Add `--detach` to any of these to release the terminal instantly (the job runs remotely;
watch it live in W&B or via `modal logs -f <job_id>`). Jobs are capped at 24 h — when a
stage hits the cap, re-launch the same command and `--resume` continues from the last
committed epoch.

Paths are steered by `IWILD_DATA_DIR` / `IWILD_MODELS_DIR` / `IWILD_REPORTS_DIR`
(see `iwild/config.py`), which the app points at the mounted volumes.

