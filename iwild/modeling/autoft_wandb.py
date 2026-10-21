"""Optional Weights & Biases integration.

All wandb usage is lazy and guarded: if wandb is not installed or no run is active,
``wandb_log`` is a no-op and ``init_wandb`` returns None. The training pipeline therefore
runs identically (loguru + JSON artifacts) with or without wandb.

The wandb target resolves from (highest priority first): explicit argument, the
``WANDB_PROJECT`` / ``WANDB_ENTITY`` environment variables (e.g. injected via Colab
Secrets), then the project fallback ``idl_f26_g30_autoft`` and the personal default entity.
"""

import os
from typing import Optional

import torch

DEFAULT_WANDB_PROJECT = "idl_f26_g30_autoft"


def resolve_wandb_target(project: Optional[str] = None, entity: Optional[str] = None):
    """Resolve (project, entity): explicit arg > WANDB_PROJECT/WANDB_ENTITY env > defaults."""
    project = project or os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT
    entity = entity or os.environ.get("WANDB_ENTITY") or None
    return project, entity


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def init_wandb(
    enabled: bool,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
):
    """Start a wandb run if enabled and installed; otherwise return None."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        return None
    project, entity = resolve_wandb_target(project, entity)
    kwargs = {
        "project": project,
        "name": run_name,
        "config": config or {},
        "group": group,
        "tags": tags or [],
        "notes": notes,
    }
    if entity:
        kwargs["entity"] = entity
    run = wandb.init(**kwargs)
    # torch tensors don't serialize into wandb config/logs directly
    if config:
        run.config.update(
            {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in config.items()},
            allow_val_change=True,
        )
    return run


def wandb_summary(metrics: dict) -> None:
    """Write key=value pairs to the active run's summary table (sticky final metrics)."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value.item() if isinstance(value, torch.Tensor) else value


def wandb_log(metrics: dict, step: Optional[int] = None) -> None:
    """Log scalars to the active wandb run, if any. Missing wandb is silently ignored."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    clean = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
    wandb.log(clean, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
