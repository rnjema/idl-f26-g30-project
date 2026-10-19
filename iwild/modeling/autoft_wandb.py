"""Optional Weights & Biases integration.

All wandb usage is lazy and guarded: if wandb is not installed or no run is active,
``wandb_log`` is a no-op and ``init_wandb`` returns None. The training pipeline therefore
runs identically (loguru + JSON artifacts) with or without wandb.
"""

from typing import Optional

import torch


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def init_wandb(
    enabled: bool,
    project: str,
    run_name: str,
    config: Optional[dict] = None,
):
    """Start a wandb run if enabled and installed; otherwise return None."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        return None
    run = wandb.init(project=project, name=run_name, config=config or {})
    # torch tensors don't serialize into wandb config/logs directly
    if config:
        run.config.update(
            {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in config.items()},
            allow_val_change=True,
        )
    return run


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
