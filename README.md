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

