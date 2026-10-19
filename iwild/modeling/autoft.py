"""AutoFT outer loop (Optuna) and the inner/full fine-tuning routines."""

import copy
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import torch
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from iwild.modeling.autoft_data import evaluate_split
from iwild.modeling.autoft_losses import LearnedLoss, get_loss_weights
from iwild.modeling.autoft_model import (
    save_classifier,
    set_seed,
    snapshot_initial_params,
)
from iwild.modeling.autoft_wandb import wandb_log

ALL_LOSSES = ["ce", "dcm", "entropy", "flyp", "hinge", "l1init", "l1zero", "l2init", "l2zero"]
ID_VAL_METRIC = "F1-macro_all"


@dataclass
class AutoFTParams:
    """Training / hyperparameter-search configuration (CLI defaults)."""

    lr: float = 1e-5
    wd: float = 0.2
    batch_size: int = 64
    accumulation_steps: int = 2
    warmup_length: int = 500
    inner_steps: int = 10
    ft_epochs: int = 20
    hopt_evals: int = 100
    num_ood_hp_examples: int = 1000
    losses: tuple = field(default_factory=lambda: tuple(ALL_LOSSES))
    clip_gradient: bool = True
    seed: int = 0
    workers: int = 2
    max_train_examples: int | None = None


class HyperparameterSpace:
    """Optuna search space: loss weights (first sorted loss anchored at 1.0), seed, lr, wd."""

    def __init__(self, model, losses, dataset_name, orig_lr):
        self.model = model
        self.losses = sorted(losses)
        self.dataset_name = dataset_name
        self.orig_lr = orig_lr

    def build_space(self, trial) -> dict:
        hparams = {}
        for loss_type in self.losses:
            hparams[f"lossw_{loss_type}"] = trial.suggest_float(
                f"lossw_{loss_type}", 1e-4, 10, log=True
            )
        hparams[f"lossw_{self.losses[0]}"] = 1.0
        hparams["seed"] = trial.suggest_int("seed", 0, 10)
        hparams["lr"] = trial.suggest_float(
            "lr", 1e-2 * self.orig_lr, 1e2 * self.orig_lr, log=True
        )
        hparams["wd"] = trial.suggest_float("wd", 0.0, 1.0)
        return hparams


def create_optimizer(
    model, hparams: dict, base_lr: float, base_wd: float
) -> torch.optim.Optimizer:
    lr = hparams.get("lr", base_lr)
    wd = hparams.get("wd", base_wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    """Cosine schedule with linear warmup, assigning lr to each param group every step."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = base_lr * (step + 1) / warmup_length
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            param_group["lr"] = lr

    return _lr_adjuster


def maybe_dictionarize(batch) -> dict:
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")
    return batch


def _infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def evaluate_net(net, dataloader, dataset, device) -> dict:
    """Accuracy, cross-entropy and WILDS post-loop metrics on one dataloader."""
    net = net.to(device).eval()
    total_correct, total_samples = 0, 0
    all_labels, all_preds, all_metadata = [], [], []

    for batch in dataloader:
        data = maybe_dictionarize(batch)
        x, y = data["images"].to(device), data["labels"].to(device)
        outputs, _, _, _ = net(x)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == y).sum().item()
        total_samples += y.size(0)
        all_labels.append(y.cpu())
        all_preds.append(outputs.cpu())
        if "metadata" in data:
            all_metadata.extend(data["metadata"])

    accuracy = total_correct / total_samples
    all_labels, all_preds = torch.cat(all_labels), torch.cat(all_preds)
    xent = torch.nn.CrossEntropyLoss(reduction="none")(all_preds, all_labels).mean().item()
    metrics = {"acc": accuracy * 100, "xent": xent}
    metrics.update(dataset.post_loop_metrics(all_labels, all_preds, all_metadata))
    net.train()
    return metrics


def calculate_loss(
    loss_fn, model, logits, labels, image_features, text_features, logit_scale, accumulation_steps
):
    loss = loss_fn(model, logits, labels, image_features, text_features, logit_scale)
    return loss / accumulation_steps


def inner_finetune(
    model, loss_fn, optimizer, dataloaders, ood_hp_dataset, params: AutoFTParams, device
):
    """N inner steps on the FLYP stream, then OOD-HP scoring."""
    scheduler = cosine_lr(
        optimizer, params.lr, params.warmup_length * params.accumulation_steps, params.inner_steps
    )
    model.train()
    iterator = _infinite_iter(dataloaders["id"])

    for step in range(params.inner_steps * params.accumulation_steps):
        scheduler(step)
        batch = maybe_dictionarize(next(iterator))
        images, labels = batch["images"].to(device), batch["labels"].to(device)
        text = batch["metadata"].to(device) if "metadata" in batch else None
        logits, image_features, text_features, logit_scale = model(images, text)
        loss = calculate_loss(
            loss_fn,
            model,
            logits,
            labels,
            image_features,
            text_features,
            logit_scale,
            params.accumulation_steps,
        )
        loss.backward()
        if (step + 1) % params.accumulation_steps == 0:
            if params.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    val_metrics = {
        "meta_objective": evaluate_net(model, dataloaders["ood_hp"], ood_hp_dataset, device)
    }
    return model, val_metrics


def finetune(model, loss_fn, optimizer, dataloaders, params: AutoFTParams, device):
    """Full fine-tune over the FT CSV; best epoch by ID-val F1-macro_all is kept."""
    num_batches = len(dataloaders["id"])
    total_steps = params.ft_epochs * num_batches
    scheduler = cosine_lr(
        optimizer, params.lr, params.warmup_length * params.accumulation_steps, total_steps
    )
    model_copy = copy.deepcopy(model).cpu()
    best_metric = float("-inf")
    val_metrics = {}

    for epoch in range(params.ft_epochs):
        model.train()
        for i, batch in enumerate(dataloaders["id"]):
            step = i + epoch * num_batches
            scheduler(step)
            data = maybe_dictionarize(batch)
            images, labels = data["images"].to(device), data["labels"].to(device)
            text = data["metadata"].to(device) if "metadata" in data else None
            logits, image_features, text_features, logit_scale = model(images, text)
            loss = calculate_loss(
                loss_fn,
                model,
                logits,
                labels,
                image_features,
                text_features,
                logit_scale,
                params.accumulation_steps,
            )
            loss.backward()
            if params.clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        id_val = dataloaders["datasets"]["id_val"]
        metrics = evaluate_split(model, dataloaders["id_val"], id_val, device)
        val_metrics[epoch] = metrics
        curr_metric = metrics[ID_VAL_METRIC]
        logger.info(f"Epoch {epoch}: ID-val {ID_VAL_METRIC} = {curr_metric:.4f}")
        wandb_log(
            {
                "ft/epoch": epoch,
                f"ft/id_val_{ID_VAL_METRIC}": curr_metric,
                "ft/id_val_acc": metrics["acc"],
            }
        )
        if curr_metric > best_metric:
            best_metric = curr_metric
            model_copy = copy.deepcopy(model).cpu()
            for param in model_copy.parameters():
                param.requires_grad = False

    return model_copy, best_metric, val_metrics


def evaluate_hparams(
    net, hparams, dataloaders, ood_hp_dataset, params: AutoFTParams, device
) -> dict:
    """Score one hyperparameter draw: short inner fine-tune, macro F1 on OOD-HP subset."""
    current_net = copy.deepcopy(net).to(device)
    optimizer = create_optimizer(current_net, hparams, params.lr, params.wd)
    initial_params = snapshot_initial_params(net)
    loss_weights = get_loss_weights(list(params.losses), hparams).to(device)
    loss_fn = LearnedLoss(list(params.losses), loss_weights, initial_params, device=device)

    set_seed(hparams["seed"])
    current_net, all_metrics = inner_finetune(
        current_net, loss_fn, optimizer, dataloaders, ood_hp_dataset, params, device
    )
    set_seed(params.seed)
    all_metrics["meta_learning_objective"] = all_metrics["meta_objective"][ID_VAL_METRIC]
    return all_metrics


def clear_memory(study: optuna.study.Study, trial: optuna.trial.Trial):
    gc.collect()
    torch.cuda.empty_cache()


def auto_ft_iteration(
    model,
    dataloaders,
    ood_hp_dataset,
    params: AutoFTParams,
    device,
    run_dir: Path,
    init_hparams: dict | None = None,
) -> dict:
    """One Optuna study over loss weights / lr / wd, then the final full fine-tune."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def hp_objective_fn(trial):
        start_time = time.time()
        hspace = HyperparameterSpace(
            model=model, losses=params.losses, dataset_name="IWildCam", orig_lr=params.lr
        )
        hparams = hspace.build_space(trial)
        net = copy.deepcopy(model).to(device)
        results = evaluate_hparams(net, hparams, dataloaders, ood_hp_dataset, params, device)
        logger.info(
            f"AutoFT trial {trial.number} done in {time.time() - start_time:.1f}s, "
            f"objective={results['meta_learning_objective']:.4f}"
        )
        wandb_log(
            {
                "hopt/trial": trial.number,
                "hopt/objective": results["meta_learning_objective"],
                **{f"hopt/{k}": v for k, v in hparams.items()},
            }
        )

        trial_file = run_dir / f"trial_{trial.number}.json"
        first_result = copy.deepcopy(results)
        first_result["hparams"] = hparams
        trial_file.write_text(json.dumps(first_result, indent=2, default=float))
        # Maximize macro F1, prioritizing later trials in case of a tie
        return results["meta_learning_objective"] + trial.number * 1e-10

    if init_hparams is not None and list(params.losses) not in (["ce"], ["flyp"]):
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
        study = optuna.create_study(sampler=TPESampler(), direction="maximize", pruner=pruner)
        study.enqueue_trial(init_hparams)
        study.optimize(hp_objective_fn, n_trials=params.hopt_evals, callbacks=[clear_memory])
        best_hparams = dict(study.best_params)
    elif list(params.losses) == ["flyp"]:
        best_hparams = {"seed": 0, "lossw_flyp": 1.0, "lr": params.lr, "wd": params.wd}
    elif list(params.losses) == ["ce"]:
        best_hparams = {"seed": 0, "lossw_ce": 1.0, "lr": params.lr, "wd": params.wd}
    else:
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
        study = optuna.create_study(sampler=TPESampler(), direction="maximize", pruner=pruner)
        study.optimize(hp_objective_fn, n_trials=params.hopt_evals, callbacks=[clear_memory])
        best_hparams = dict(study.best_params)

    best_hparams["seed"] = int(best_hparams["seed"])
    if "ce" in params.losses and "lossw_ce" not in best_hparams:
        best_hparams["lossw_ce"] = 1.0
    logger.info(f"Best hyperparameters: {best_hparams}")
    (run_dir / "hparams.json").write_text(json.dumps(best_hparams, indent=2))

    initial_params = snapshot_initial_params(model)
    loss_weights = get_loss_weights(list(params.losses), best_hparams).to(device)
    loss_fn = LearnedLoss(list(params.losses), loss_weights, initial_params, device=device)
    optimizer = create_optimizer(model, best_hparams, params.lr, params.wd)
    set_seed(best_hparams["seed"])
    ft_model, val_metric, val_metrics = finetune(
        model, loss_fn, optimizer, dataloaders, params, device
    )
    save_classifier(ft_model, run_dir / "final_model.pt")
    results = {
        "best_hparams": best_hparams,
        "id_val_metric": val_metric,
        "id_val_metric_name": ID_VAL_METRIC,
        "epoch_metrics": {int(k): v for k, v in val_metrics.items()},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Saved AutoFT run to {run_dir}")
    return {
        "model": ft_model,
        "val_metric": val_metric,
        "hparams": best_hparams,
        "results": results,
    }


def auto_ft(
    model,
    dataloaders,
    ood_hp_dataset,
    params: AutoFTParams,
    device,
    run_dir: Path,
    init_hparams: dict | None = None,
) -> dict:
    """Run the AutoFT pipeline: hyperparameter search plus final fine-tune."""
    best = None
    for i in range(1):
        set_seed(np.random.randint(0, 1000))
        results = auto_ft_iteration(
            copy.deepcopy(model),
            dataloaders,
            ood_hp_dataset,
            params,
            device,
            run_dir,
            init_hparams,
        )
        if best is None or results["val_metric"] > best["val_metric"]:
            best = results
    return best


def wise_ft_interpolate(zeroshot_state: dict, ft_state: dict, alpha: float) -> dict:
    """WiSE-FT weight interpolation: (1 - alpha) * theta_0 + alpha * theta_1."""
    assert set(zeroshot_state.keys()) == set(ft_state.keys())
    return {
        key: (1 - alpha) * zeroshot_state[key] + alpha * ft_state[key]
        for key in zeroshot_state.keys()
    }
