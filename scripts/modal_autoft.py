"""Modal app for the AutoFT iWildCam baseline — persistent volumes + W&B, no session limits.

What this gives you over Colab:
- **Data volume** `iwildcam-data`: downloaded + verified once, reused by every run.
- **Checkpoint volume** `iwildcam-ckpts`: zeroshot models, per-epoch fine-tune
  checkpoints (`epoch_ckpt/last.pt`, `best.pt`), `hparams.json`, and `final_model.pt`
  persist across containers; `--resume` picks up where a preempted run stopped.
- Periodic volume commits every 10 min inside the job, so even a preemption/timeout
  loses at most the last few minutes — not the whole stage.
- W&B logging identical to the local/Colab path (env vars come from a Modal secret).

One-time setup:

    pip install modal
    modal setup
    modal secret create wandb \\
        WANDB_API_KEY=... WANDB_PROJECT=... WANDB_ENTITY=...

Environments (isolated dev/prod worlds): every resource below - the two volumes, the
'wandb' secret, and the data it downloads - is scoped to a Modal environment, default
'main'. To get an isolated 'dev' world:

    modal environment create dev
    modal secret create --environment dev wandb \\
        WANDB_API_KEY=... WANDB_PROJECT=... WANDB_ENTITY=...

Run from the repo root (the iwild package is baked into the image from your checkout),
always with MODAL_ENVIRONMENT matching the --environment flag:

    # 1) download + verify the dataset onto the volume (~1 hour, CPU is fine)
    modal run scripts/modal_autoft.py::download_data

    # 2) Stage A — official HPO: 500 trials x 100 inner steps (A100)
    modal run --gpu A100 scripts/modal_autoft.py::autoft_stage --stage hopt \\
        --run-name autoft_vitb16_official

    # 3) Stage B — resumable 20-epoch fine-tune (re-run to resume after preemption)
    modal run --gpu A100 scripts/modal_autoft.py::autoft_stage --stage ft \\
        --run-name autoft_vitb16_official

    # fast path: fine-tune straight from the paper's learned hparams
    modal run --gpu A100 scripts/modal_autoft.py::autoft_stage --stage ft \\
        --run-name autoft_vitb16_paperhparams --hparams-from-paper

    # FLYP baseline
    modal run --gpu A100 scripts/modal_autoft.py::flyp_stage --run-name flyp_baseline

('main' environment shown; for 'dev' prefix every command with
MODAL_ENVIRONMENT=dev and add --environment dev to modal run.)

GPU is chosen with the modal run `--gpu` flag (A100 / A100-80GB / L4 / T4).

Detached mode: every job runs remotely, so it survives closing the laptop. To get the
terminal back immediately, launch with `--detach` (prints a job ID and exits); watch
progress live in W&B, or re-attach logs with `modal logs -f <job_id>`. Note the 24 h
per-job cap still applies - when a stage hits it, re-launch the same command and
`--resume` continues from the last committed epoch.

Volume contents can be inspected from any job, e.g.:
`modal run scripts/modal_autoft.py::ls_volumes`.
"""

import os
import threading
import time

import modal

app = modal.App("idl-f26-g30-autoft")

# All Modal resources (volumes, the 'wandb' secret) are scoped to this environment.
# Modal sets MODAL_ENVIRONMENT inside containers; set it in your shell too so the
# local launcher and the remote job agree (see module docstring).
ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")

DATA_VOLUME_NAME = "iwildcam-data"
CKPT_VOLUME_NAME = "iwildcam-ckpts"
COMMIT_INTERVAL_S = 600

data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME, create_if_missing=True, environment_name=ENVIRONMENT
)
ckpt_volume = modal.Volume.from_name(
    CKPT_VOLUME_NAME, create_if_missing=True, environment_name=ENVIRONMENT
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned cu128 wheels from the PyTorch index — the same torch/torchvision as the
    # workstation .venv. (Unpinned PyPI torch currently resolves to CUDA-13 wheels,
    # which we have not validated.)
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_python_source("iwild")
    .run_commands(
        "python -c \"import pathlib, iwild; "
        "p = pathlib.Path(iwild.__file__).parent / 'modeling/iwildcam_metadata/labels.csv'; "
        "assert p.exists(), f'missing package data: {p}'\""
    )
)

ENV = {
    "IWILD_DATA_DIR": "/data",
    "IWILD_MODELS_DIR": "/ckpt/models",
    "IWILD_REPORTS_DIR": "/ckpt/reports",
    "TOKENIZERS_PARALLELISM": "false",
}

VOLUMES = {
    "/data": data_volume,
    "/ckpt": ckpt_volume,
}

SECRETS = [modal.Secret.from_name("wandb", environment_name=ENVIRONMENT)]

DAY = 24 * 60 * 60  # modal function timeout cap


def _start_periodic_commits() -> None:
    """Commit mounted volumes periodically so preemption loses only recent progress.

    Best-effort: any SDK drift here must never kill the training job.
    """

    def loop():
        while True:
            time.sleep(COMMIT_INTERVAL_S)
            for name in (DATA_VOLUME_NAME, CKPT_VOLUME_NAME):
                try:
                    modal.Volume.from_name(name, environment_name=ENVIRONMENT).commit()
                    print(f"[volume-commit] {name} committed", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[volume-commit] {name} skipped: {exc}", flush=True)

    threading.Thread(target=loop, daemon=True).start()


def _run_cli(args: list[str]) -> None:
    import subprocess
    import sys

    import iwild.modeling.train as train_cli

    cmd = [sys.executable, train_cli.__file__] + args
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    volumes=VOLUMES,
    env=ENV,
    secrets=SECRETS,
    timeout=DAY,
)
def download_data() -> None:
    """Download iWildCam v2.0 onto the data volume and build the FLYP FT CSV."""
    import subprocess
    import sys

    import iwild.dataset as dataset_cli

    for args in (["download"], ["make-ft-csv"], ["location-split", "--seed", "42"]):
        cmd = [sys.executable, dataset_cli.__file__] + args
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    data_volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    env=ENV,
    secrets=SECRETS,
    timeout=DAY,
)
def autoft_stage(
    stage: str,
    run_name: str,
    hopt_evals: int = 500,
    inner_steps: int = 100,
    batch_size: int = 128,
    accumulation_steps: int = 2,
    ft_epochs: int = 20,
    model: str | None = None,
    hparams_from_paper: bool = False,
    resume: bool = True,
    smoke: bool = False,
) -> None:
    """Run one AutoFT stage (hopt / ft / both) with checkpoints on the volume."""
    _start_periodic_commits()

    if hparams_from_paper:
        import json
        import shutil
        from pathlib import Path

        import iwild.modeling.train as train_cli

        paper = Path(train_cli.__file__).resolve().parent / "autoft_hparams_iwildcam.json"
        run_dir = Path("/ckpt/models/autoft") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(paper, run_dir / "hparams.json")
        print(f"Seeded {run_dir}/hparams.json from the paper's learned weights", flush=True)

    args = [
        "autoft",
        "--stage", stage,
        "--run-name", run_name,
        "--hopt-evals", str(hopt_evals),
        "--inner-steps", str(inner_steps),
        "--batch-size", str(batch_size),
        "--accumulation-steps", str(accumulation_steps),
        "--ft-epochs", str(ft_epochs),
        "--wandb",
    ]
    if model:
        args += ["--model", model]
    if resume:
        args.append("--resume")
    if smoke:
        args.append("--smoke")
    _run_cli(args)
    ckpt_volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    env=ENV,
    secrets=SECRETS,
    timeout=DAY,
)
def flyp_stage(
    run_name: str = "flyp_baseline",
    batch_size: int = 128,
    accumulation_steps: int = 2,
    ft_epochs: int = 20,
    model: str | None = None,
    resume: bool = True,
    smoke: bool = False,
) -> None:
    """Run the FLYP lower-baseline fine-tune, resumable, checkpoints on the volume."""
    _start_periodic_commits()

    args = [
        "flyp",
        "--run-name", run_name,
        "--batch-size", str(batch_size),
        "--accumulation-steps", str(accumulation_steps),
        "--ft-epochs", str(ft_epochs),
        "--wandb",
    ]
    if model:
        args += ["--model", model]
    if resume:
        args.append("--resume")
    if smoke:
        args.append("--smoke")
    _run_cli(args)
    ckpt_volume.commit()


@app.function(image=image, volumes=VOLUMES, env=ENV, secrets=SECRETS, timeout=DAY)
def eval_model(ckpt: str, batch_size: int = 128) -> None:
    """Evaluate a checkpoint from the volume on all four WILDS splits (logs to W&B)."""
    _run_cli(["eval", "--ckpt", ckpt, "--batch-size", str(batch_size), "--wandb"])
    ckpt_volume.commit()


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=60 * 10)
def ls_volumes() -> None:
    """Print what's on the volumes (quick sanity check)."""
    import os

    for root in ("/vol/data", "/vol/models", "/vol/reports"):
        print(f"== {root} ==")
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 2:
                dirnames[:] = []
                continue
            print(dirpath, "| dirs:", dirnames[:8], "| files:", filenames[:8])
