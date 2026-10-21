"""Training CLI: zero-shot head, AutoFT, FLYP baseline, eval, and WiSE-FT interpolation."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
import typer
from loguru import logger

from iwild.config import AUTOFT_DIR, REPORTS_DIR, ZEROSHOT_DIR
from iwild.modeling.autoft import (
    ALL_LOSSES,
    ID_VAL_METRIC,
    AutoFTParams,
    auto_ft,
    create_optimizer,
    finetune,
    wise_ft_interpolate,
)
from iwild.modeling.autoft_data import IWildCamSplit, evaluate_split, get_dataloaders
from iwild.modeling.autoft_losses import LearnedLoss, get_loss_weights
from iwild.modeling.autoft_model import (
    ImageClassifier,
    build_zeroshot_model,
    get_device,
    load_classifier,
    save_classifier,
    set_seed,
    snapshot_initial_params,
)
from iwild.modeling.autoft_wandb import (
    DEFAULT_WANDB_PROJECT,
    finish_wandb,
    init_wandb,
    wandb_log,
    wandb_summary,
)

app = typer.Typer()
DEFAULT_ZEROSHOT_CKPT = ZEROSHOT_DIR / "clip_vitb16_iwildcam.pt"
DEFAULT_INIT_HPARAMS = Path(__file__).resolve().parent / "autoft_hparams_iwildcam.json"
SPLITS = {"IDVal": "id_val", "IDTest": "id_test", "OODVal": "val", "OODTest": "test"}


def _load_or_build_zeroshot(ckpt: Path, device) -> ImageClassifier:
    if Path(ckpt).exists():
        return load_classifier(ckpt, device=device)
    logger.info(f"No checkpoint at {ckpt}; building the zero-shot model from scratch.")
    model = build_zeroshot_model(device=device)
    save_classifier(model, ckpt)
    return model


def _build_params(
    lr: float,
    wd: float,
    batch_size: int,
    accumulation_steps: int,
    warmup_length: int,
    ft_epochs: int,
    inner_steps: int,
    hopt_evals: int,
    num_ood_hp_examples: int,
    losses: tuple,
    clip_gradient: bool,
    max_train_examples: Optional[int],
    smoke: bool,
    workers: int = 2,
) -> AutoFTParams:
    if smoke:
        hopt_evals, inner_steps, batch_size = 3, 5, 16
        num_ood_hp_examples, ft_epochs, max_train_examples = 200, 1, 2000
    return AutoFTParams(
        lr=lr,
        wd=wd,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        warmup_length=warmup_length,
        ft_epochs=ft_epochs,
        inner_steps=inner_steps,
        hopt_evals=hopt_evals,
        num_ood_hp_examples=num_ood_hp_examples,
        losses=losses,
        clip_gradient=clip_gradient,
        max_train_examples=max_train_examples,
        workers=workers,
    )


@app.command()
def zeroshot(
    arch: str = typer.Option(
        "ViT-B-16", "--arch", help="open_clip architecture, e.g. ViT-B-16 or ViT-L-14-336"
    ),
    pretrained: str = typer.Option("openai", "--pretrained"),
    out_dir: Path = typer.Option(ZEROSHOT_DIR, "--out-dir"),
    name: Optional[str] = typer.Option(None, "--name"),
):
    """Build the zero-shot CLIP model and save it to ZEROSHOT_DIR."""
    device = get_device()
    ckpt_name = name or f"clip_{arch.lower().replace('-', '')}_iwildcam"
    model = build_zeroshot_model(arch=arch, pretrained=pretrained, device=device)
    save_classifier(model, Path(out_dir) / f"{ckpt_name}.pt")


@app.command()
def autoft(
    model_ckpt: Path = typer.Option(DEFAULT_ZEROSHOT_CKPT, "--model"),
    run_name: str = typer.Option("default", "--run-name"),
    lr: float = typer.Option(1e-5, "--lr"),
    wd: float = typer.Option(0.2, "--wd"),
    batch_size: int = typer.Option(64, "--batch-size"),
    accumulation_steps: int = typer.Option(2, "--accumulation-steps"),
    warmup_length: int = typer.Option(500, "--warmup-length"),
    ft_epochs: int = typer.Option(20, "--ft-epochs"),
    inner_steps: int = typer.Option(10, "--inner-steps"),
    hopt_evals: int = typer.Option(100, "--hopt-evals"),
    num_ood_hp_examples: int = typer.Option(1000, "--num-ood-hp-examples"),
    losses: Optional[List[str]] = typer.Option(None, "--losses"),
    clip_gradient: bool = typer.Option(True, "--clip-gradient/--no-clip-gradient"),
    init_hparams: Optional[Path] = typer.Option(DEFAULT_INIT_HPARAMS, "--init-hparams"),
    no_init_hparams: bool = typer.Option(False, "--no-init-hparams"),
    max_train_examples: Optional[int] = typer.Option(None, "--max-train-examples"),
    smoke: bool = typer.Option(False, "--smoke"),
    workers: int = typer.Option(2, "--workers", help="DataLoader workers (raise to ~12 on A100)"),
    stage: str = typer.Option(
        "both", "--stage", help="hopt = search only | ft = fine-tune only | both = full pipeline"
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="ft stage: continue from the last epoch checkpoint instead of restarting",
    ),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb"),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Defaults to $WANDB_PROJECT or " + DEFAULT_WANDB_PROJECT
    ),
    wandb_entity: Optional[str] = typer.Option(
        None, "--wandb-entity", help="Defaults to $WANDB_ENTITY (else your personal entity)"
    ),
    wandb_run_name: Optional[str] = typer.Option(None, "--wandb-run-name"),
):
    """Full AutoFT pipeline: Optuna outer loop, then final fine-tune on the FT CSV.

    Use --stage hopt / --stage ft to split the two across sessions (Colab runtime
    limits); the ft stage resumes with --resume if a previous fine-tune disconnected.
    """
    if stage not in ("hopt", "ft", "both"):
        raise typer.BadParameter("--stage must be one of: hopt, ft, both")
    device = get_device()
    loss_list = tuple(sorted(losses)) if losses else tuple(ALL_LOSSES)
    params = _build_params(
        lr,
        wd,
        batch_size,
        accumulation_steps,
        warmup_length,
        ft_epochs,
        inner_steps,
        hopt_evals,
        num_ood_hp_examples,
        loss_list,
        clip_gradient,
        max_train_examples,
        smoke,
        workers,
    )
    model = _load_or_build_zeroshot(model_ckpt, device)
    arch = model.image_encoder.arch
    submission = {
        "algorithm": "AutoFT",
        "authors": "Choi, Lee, Chen, Zhou, Raghunathan, Finn",
        "paper": "arXiv:2401.10220",
        "leaderboard": "WILDS iWildCam",
        "leaderboard_url": "https://wilds.stanford.edu/leaderboard/#iwildcam",
        "leaderboard_entry": "AutoFT 52.0 (0.4) OOD-test macro F1, CLIP ViT-L/14@336, WiSE-FT ensembled",
        "dataset": "iWildCam v2.0 (WILDS benchmark)",
        "official_metric": "F1-macro_all (OOD test)",
        "protocol": "official WILDS splits; HPO objective on OOD val; OOD test touched once",
        "reproduction_backbone": f"CLIP {arch}",
    }
    run = init_wandb(
        wandb,
        wandb_project,
        wandb_run_name or f"AutoFT-{arch}-iWildCam-{run_name}",
        config={
            "submission": submission,
            "recipe": {
                "stage": stage,
                "hopt_evals": params.hopt_evals,
                "inner_steps": params.inner_steps,
                "ft_epochs": params.ft_epochs,
                "batch_size": params.batch_size,
                "accumulation_steps": params.accumulation_steps,
                "warmup_length": params.warmup_length,
                "base_lr": params.lr,
                "base_wd": params.wd,
                "num_ood_hp_examples": params.num_ood_hp_examples,
                "losses": list(params.losses),
                "clip_gradient": params.clip_gradient,
            },
            "command": "autoft",
            "run_name": run_name,
            "stage": stage,
            **asdict(params),
        },
        entity=wandb_entity,
        group=run_name,
        tags=[
            "AutoFT",
            "iwildcam",
            arch,
            stage,
            f"hopt{params.hopt_evals}",
            f"inner{params.inner_steps}",
            f"epochs{params.ft_epochs}",
            f"bs{params.batch_size}x{params.accumulation_steps}",
        ],
        notes=(
            "Reproduction of the WILDS iWildCam leaderboard entry 'AutoFT' (Choi et al., "
            "arXiv:2401.10220; leaderboard 52.0 macro F1 with ViT-L/14@336 + WiSE-FT). "
            f"Outer-loop objective: macro F1 on {params.num_ood_hp_examples} class-balanced "
            "OOD-val examples; best epoch by ID-val macro F1; recipe per the official auto-ft "
            "README (TPE over 9 loss weights + lr + wd, FLYP text stream)."
        ),
    )
    dataloaders = get_dataloaders(
        params.batch_size,
        model.train_preprocess,
        model.val_preprocess,
        num_ood_hp_examples=params.num_ood_hp_examples,
        max_train_examples=params.max_train_examples,
        num_workers=params.workers,
    )
    ood_hp_dataset = dataloaders["datasets"]["ood_hp"]

    init = None
    if not no_init_hparams and init_hparams is not None and Path(init_hparams).exists():
        init = json.loads(Path(init_hparams).read_text())
        logger.info(f"Seeding trial 0 with hyperparameters from {init_hparams}")
    elif init_hparams is not None and not no_init_hparams:
        logger.warning(f"Init hparams file {init_hparams} not found; starting fresh.")

    run_dir = AUTOFT_DIR / run_name
    results = auto_ft(
        model,
        dataloaders,
        ood_hp_dataset,
        params,
        device,
        run_dir,
        init_hparams=init,
        stage=stage,
        resume=resume,
    )
    if results["val_metric"] is not None:
        wandb_log(
            {
                "final/id_val_metric": results["val_metric"],
                **{f"final/{k}": v for k, v in results["hparams"].items()},
            }
        )
        wandb_summary({f"IDVal_best/{ID_VAL_METRIC}": results["val_metric"]})
    finish_wandb(run)
    logger.success(
        f"AutoFT {stage} done. Best ID-val {ID_VAL_METRIC}: {results['val_metric']} "
        f"(artifacts under {run_dir})"
    )


@app.command()
def flyp(
    model_ckpt: Path = typer.Option(DEFAULT_ZEROSHOT_CKPT, "--model"),
    run_name: str = typer.Option("flyp", "--run-name"),
    lr: float = typer.Option(1e-5, "--lr"),
    wd: float = typer.Option(0.2, "--wd"),
    batch_size: int = typer.Option(64, "--batch-size"),
    accumulation_steps: int = typer.Option(2, "--accumulation-steps"),
    warmup_length: int = typer.Option(500, "--warmup-length"),
    ft_epochs: int = typer.Option(20, "--ft-epochs"),
    max_train_examples: Optional[int] = typer.Option(None, "--max-train-examples"),
    smoke: bool = typer.Option(False, "--smoke"),
    workers: int = typer.Option(2, "--workers", help="DataLoader workers (raise to ~12 on A100)"),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Continue from the last epoch checkpoint instead of restarting",
    ),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb"),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Defaults to $WANDB_PROJECT or " + DEFAULT_WANDB_PROJECT
    ),
    wandb_entity: Optional[str] = typer.Option(
        None, "--wandb-entity", help="Defaults to $WANDB_ENTITY (else your personal entity)"
    ),
    wandb_run_name: Optional[str] = typer.Option(None, "--wandb-run-name"),
):
    """Plain FLYP fine-tune (ce=1, flyp=1, all other loss weights 0): lower baseline."""
    device = get_device()
    params = _build_params(
        lr,
        wd,
        batch_size,
        accumulation_steps,
        warmup_length,
        ft_epochs,
        10,
        0,
        1000,
        ("ce", "flyp"),
        True,
        max_train_examples,
        smoke,
        workers,
    )
    model = _load_or_build_zeroshot(model_ckpt, device)
    arch = model.image_encoder.arch
    run = init_wandb(
        wandb,
        wandb_project,
        wandb_run_name or f"FLYP-{arch}-iWildCam-{run_name}",
        config={
            "submission": {
                "algorithm": "FLYP (lower baseline in the AutoFT paper)",
                "paper": "arXiv:2401.10220 (Table 2)",
                "leaderboard": "WILDS iWildCam",
                "dataset": "iWildCam v2.0 (WILDS benchmark)",
                "official_metric": "F1-macro_all (OOD test)",
                "reproduction_backbone": f"CLIP {arch}",
            },
            "recipe": {"losses": ["ce", "flyp"], "loss_weights": {"ce": 1.0, "flyp": 1.0}},
            "command": "flyp",
            "run_name": run_name,
            **asdict(params),
        },
        entity=wandb_entity,
        group=run_name,
        tags=["FLYP", "iwildcam", arch, "baseline", f"epochs{params.ft_epochs}"],
        notes=(
            "FLYP baseline as reported in the AutoFT paper (Choi et al., arXiv:2401.10220, "
            "Table 2: 46.0 OOD-test macro F1 with ViT-L/14@336): ce=1, flyp=1 fixed loss "
            "weights, best epoch by ID-val macro F1."
        ),
    )
    dataloaders = get_dataloaders(
        params.batch_size,
        model.train_preprocess,
        model.val_preprocess,
        num_ood_hp_examples=params.num_ood_hp_examples,
        max_train_examples=params.max_train_examples,
        num_workers=params.workers,
    )
    hparams = {"seed": 0, "lossw_ce": 1.0, "lossw_flyp": 1.0, "lr": params.lr, "wd": params.wd}
    initial_params = snapshot_initial_params(model)
    loss_weights = get_loss_weights(list(params.losses), hparams).to(device)
    loss_fn = LearnedLoss(list(params.losses), loss_weights, initial_params, device=device)
    optimizer = create_optimizer(model, hparams, params.lr, params.wd)
    set_seed(hparams["seed"])
    run_dir = AUTOFT_DIR / run_name
    ft_model, best_metric, _ = finetune(
        model,
        loss_fn,
        optimizer,
        dataloaders,
        params,
        device,
        run_dir=run_dir,
        resume=resume,
    )
    save_classifier(ft_model, run_dir / "final_model.pt")
    wandb_log({f"final/id_val_{ID_VAL_METRIC}": best_metric})
    finish_wandb(run)
    logger.success(f"FLYP done. Best ID-val {ID_VAL_METRIC}: {best_metric:.4f}")


@app.command("eval")
def eval_model(
    ckpt: Path = typer.Option(..., "--ckpt", exists=True, readable=True),
    batch_size: int = typer.Option(64, "--batch-size"),
    num_workers: int = typer.Option(2, "--workers"),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb"),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Defaults to $WANDB_PROJECT or " + DEFAULT_WANDB_PROJECT
    ),
    wandb_entity: Optional[str] = typer.Option(
        None, "--wandb-entity", help="Defaults to $WANDB_ENTITY (else your personal entity)"
    ),
    wandb_run_name: Optional[str] = typer.Option(None, "--wandb-run-name"),
):
    """Evaluate a checkpoint on IDVal/IDTest/OODVal/OODTest; writes a JSON report."""
    device = get_device()
    model = load_classifier(ckpt, device=device)
    arch = model.image_encoder.arch
    run = init_wandb(
        wandb,
        wandb_project,
        wandb_run_name or f"eval_{Path(ckpt).stem}",
        config={"command": "eval", "ckpt": str(ckpt), "arch": arch},
        entity=wandb_entity,
        tags=["eval", "iwildcam", arch],
        notes="WILDS iWildCam v2.0 official eval splits; macro F1 + accuracy.",
    )
    results = {}
    for name, split in SPLITS.items():
        split_wrapper = IWildCamSplit(split, model.val_preprocess)
        loader = torch.utils.data.DataLoader(
            split_wrapper.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        metrics = evaluate_split(model, loader, split_wrapper, device)
        results[name] = metrics
        wandb_log(
            {
                f"eval/{name}/{ID_VAL_METRIC}": metrics[ID_VAL_METRIC],
                f"eval/{name}/acc": metrics["acc"],
            }
        )
        # sticky summary under the paper's split nomenclature (IDVal/IDTest/OODVal/OODTest)
        wandb_summary(
            {f"{name}/{ID_VAL_METRIC}": metrics[ID_VAL_METRIC], f"{name}/acc": metrics["acc"]}
        )
        logger.info(
            f"{name}: {ID_VAL_METRIC} = {metrics[ID_VAL_METRIC]:.4f}, "
            f"acc = {metrics['acc']:.2f}"
        )
    finish_wandb(run)
    out_path = REPORTS_DIR / f"autoft_eval_{Path(ckpt).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    logger.success(f"Wrote eval report to {out_path}")


@app.command()
def wisefmt(
    zeroshot_ckpt: Path = typer.Option(DEFAULT_ZEROSHOT_CKPT, "--zeroshot"),
    finetuned_ckpt: Path = typer.Option(..., "--finetuned"),
    run_name: str = typer.Option("wisefmt", "--run-name"),
    num_alphas: int = typer.Option(11, "--num-alphas"),
    batch_size: int = typer.Option(64, "--batch-size"),
):
    """Sweep WiSE-FT alpha in [0, 1]; select by ID-val macro F1, report OOD metrics."""
    device = get_device()
    alphas = [i / (num_alphas - 1) for i in range(num_alphas)]
    zeroshot_model = load_classifier(zeroshot_ckpt, device=device)
    finetuned_model = load_classifier(finetuned_ckpt, device=device)
    zeroshot_state = {k: v.clone() for k, v in zeroshot_model.state_dict().items()}
    ft_state = {k: v.clone() for k, v in finetuned_model.state_dict().items()}

    id_val = IWildCamSplit("id_val", zeroshot_model.val_preprocess)
    id_val_loader = torch.utils.data.DataLoader(
        id_val.dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )
    ood_val = IWildCamSplit("val", zeroshot_model.val_preprocess)
    ood_test = IWildCamSplit("test", zeroshot_model.val_preprocess)

    best_alpha, best_metric = None, float("-inf")
    for alpha in alphas:
        theta = wise_ft_interpolate(zeroshot_state, ft_state, alpha)
        zeroshot_model.load_state_dict(theta)
        metrics = evaluate_split(zeroshot_model, id_val_loader, id_val, device)
        logger.info(f"alpha={alpha:.2f}: ID-val {ID_VAL_METRIC} = {metrics[ID_VAL_METRIC]:.4f}")
        if metrics[ID_VAL_METRIC] > best_metric:
            best_alpha, best_metric = alpha, metrics[ID_VAL_METRIC]

    theta = wise_ft_interpolate(zeroshot_state, ft_state, best_alpha)
    zeroshot_model.load_state_dict(theta)
    for name, wrapper in (("OODVal", ood_val), ("OODTest", ood_test)):
        loader = torch.utils.data.DataLoader(
            wrapper.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        metrics = evaluate_split(zeroshot_model, loader, wrapper, device)
        logger.info(
            f"best alpha={best_alpha:.2f} {name}: {ID_VAL_METRIC} = "
            f"{metrics[ID_VAL_METRIC]:.4f}, acc = {metrics['acc']:.2f}"
        )

    run_dir = AUTOFT_DIR / run_name
    save_classifier(zeroshot_model, run_dir / f"wise_ft_alpha={best_alpha:.3f}.pt")
    report = {
        "best_alpha": best_alpha,
        "id_val_metric": best_metric,
        "id_val_metric_name": ID_VAL_METRIC,
    }
    (run_dir / "results.json").write_text(json.dumps(report, indent=2, default=float))
    logger.success(f"WiSE-FT done. Best alpha={best_alpha:.3f} (saved under {run_dir})")


if __name__ == "__main__":
    app()
