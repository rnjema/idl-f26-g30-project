"""Build midterm-report figures from reports/wandb_dump.json.

Outputs PDFs into reports/midterm/figures/:
- baselines_macrof1.pdf   grouped macro-F1 bars per split, all baseline models
- training_curves.pdf     2x2 training/eval curves from the logged histories
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = json.load(open("reports/wandb_dump.json", encoding="utf-8"))
OUT = Path("reports/midterm/figures")
OUT.mkdir(parents=True, exist_ok=True)


def find_run(name, user_substr=None, must_have=None):
    for r in DUMP:
        if r["name"] == name and (user_substr is None or user_substr.lower() in r["user"].lower()):
            if must_have is None or must_have in r["summary"]:
                return r
    return None


def pct(x):
    return 100 * x


# ---------------- numbers used in both figures and the report table ----------------
zeroshot = find_run("zeroshot_sanity", "Ramsey")["summary"]
autoft = find_run("eval_final_model", "Ramsey", must_have="OODTest/F1-macro_all")["summary"]
wisefmt = find_run("eval_wise_ft_alpha=0.800", "Ramsey")["summary"]
flyp = find_run("iwildcam_flyp_vitb16-seed0", "nouka", must_have="final/test_macro_f1")["summary"]
resnet_agg = find_run("resnet50-results-E5", "Emmilly")["summary"]
dfn = find_run("dfn-vith14-ft-lr1e-05-s0", "Emmilly")["summary"]

SPLITS = ["IDVal", "IDTest", "OODVal", "OODTest"]


def seed_mean(key_fn):
    vals = []
    for s in ("s0", "s1", "s2"):
        r = find_run(f"resnet50-absgd-lam1.1-{s}", "Emmilly")
        vals.append(r["summary"][key_fn(r["summary"])])
    return sum(vals) / len(vals)


def _pick(d, *keys):
    for k in keys:
        if k in d:
            return k
    raise KeyError(keys)


# Aggregate (3-seed means where per-split means are needed); "official" = the
# paper-style best-epoch checkpoint rule from the teammate's aggregate run.
resnet_faithful = [
    seed_mean(lambda d: _pick(d, "faithful/id_val/f1_macro")),
    resnet_agg["absgd/faithful/official/id_test_f1_mean"] / 100,
    seed_mean(lambda d: _pick(d, "faithful/val/f1_macro")),
    resnet_agg["absgd/faithful/official/ood_test_f1_mean"] / 100,
]
resnet_fixednorm = [
    seed_mean(lambda d: _pick(d, "fixednorm/id_val/f1_macro")),
    resnet_agg["absgd/fixednorm/official/id_test_f1_mean"] / 100,
    seed_mean(lambda d: _pick(d, "fixednorm/val/f1_macro")),
    resnet_agg["absgd/fixednorm/official/ood_test_f1_mean"] / 100,
]

MODELS = [
    ("Zero-shot CLIP ViT-B/16", [zeroshot[f"eval/{s}/F1-macro_all"] for s in SPLITS]),
    ("ResNet-50 + AbsGD (faithful)", resnet_faithful),
    ("ResNet-50 + AbsGD (fixed-norm)", resnet_fixednorm),
    ("FLYP ViT-B/16", [flyp["final/id_val_macro_f1"], flyp["final/id_test_macro_f1"],
                       flyp["final/val_macro_f1"], flyp["final/test_macro_f1"]]),
    ("AutoFT ViT-B/16 (ours)", [autoft[f"{s}/F1-macro_all"] for s in SPLITS]),
    ("AutoFT + WiSE-FT ViT-B/16 (ours)", [wisefmt[f"{s}/F1-macro_all"] for s in SPLITS]),
    ("DFN CLIP ViT-H/14", [dfn["id_val/f1_macro"], dfn["id_test/f1_macro"],
                           dfn["val/f1_macro"], dfn["test/f1_macro"]]),
]

print("== Table numbers (macro F1 %) ==")
for name, vals in MODELS:
    print(f"{name:36s} " + " ".join(f"{pct(v):6.1f}" for v in vals))

SPLITS = ["IDVal", "IDTest", "OODVal", "OODTest"]
COLORS = ["#9ecae1", "#6baed6", "#fdae6b", "#e6550d"]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
x = range(len(MODELS))
width = 0.2
for i, (split, color) in enumerate(zip(SPLITS, COLORS)):
    ax.bar([xx + i * width for xx in x], [pct(v[i]) for _, v in MODELS], width,
           label=split, color=color)
ax.set_xticks([xx + 1.5 * width for xx in x])
ax.set_xticklabels([m[0].replace(" + ", "+\n").replace(" (ours)", "\n(ours)")
                    for m in MODELS], fontsize=7.5)
ax.set_ylabel("Macro F1 (%)")
ax.legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 68)
fig.tight_layout()
fig.savefig(OUT / "baselines_macrof1.pdf")
plt.close(fig)

# ---------------- training curves ----------------
fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2))

flyp_run = find_run("iwildcam_flyp_vitb16-seed0", "nouka", must_have="final/test_macro_f1")
ax = axes[0][0]
hist = flyp_run["history"]
steps, loss = zip(*hist["train/flyp_loss"])
ax.plot(steps, loss, lw=0.6, color="#e6550d")
ax.set_xlabel("step")
ax.set_ylabel("FLYP loss")
ax.set_title("FLYP ViT-B/16 (N'nouka)", fontsize=9)

autoft_run = find_run("AutoFT-ViT-B-16-iWildCam-autoft_vitb16_paperhparams", "Ramsey")
ax = axes[0][1]
hist = autoft_run["history"]
if "ft/id_val_F1-macro_all" in hist:
    ep, val = zip(*hist["ft/id_val_F1-macro_all"])
    ax.plot(ep, [pct(v) for v in val], "o-", color="#3182bd")
    ax.axhline(pct(0.4943), ls="--", lw=0.8, color="gray")
ax.set_xlabel("epoch")
ax.set_ylabel("ID-val macro F1 (%)")
ax.set_title("AutoFT ViT-B/16 fine-tune (ours; resumed after preemption)", fontsize=9)

resnet_run = find_run("resnet50-absgd-lam1.1-s1", "Emmilly")
ax = axes[1][0]
hist = resnet_run["history"]
if "train/loss" in hist:
    st, ls = zip(*hist["train/loss"])
    ax.plot(st, ls, lw=0.7, color="#31a354")
ax.set_xlabel("step")
ax.set_ylabel("train loss")
ax.set_title("ResNet-50 + AbsGD s1 (Emmilly)", fontsize=9)

dfn_run = find_run("dfn-vith14-ft-lr1e-05-s0", "Emmilly")
ax = axes[1][1]
hist = dfn_run["history"]
if "test/f1_macro" in hist:
    ep, val = zip(*hist["test/f1_macro"])
    ax.plot(ep, [pct(v) for v in val], "o-", color="#756bb1")
if "id_test/f1_macro" in hist:
    ep, val = zip(*hist["id_test/f1_macro"])
    ax.plot(ep, [pct(v) for v in val], "s--", color="#bcbddc")
ax.set_xlabel("epoch")
ax.set_ylabel("macro F1 (%)")
ax.legend(["OOD test", "ID test"], fontsize=8, frameon=False)
ax.set_title("DFN CLIP ViT-H/14 fine-tune (Emmilly)", fontsize=9)

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "training_curves.pdf")
plt.close(fig)

print("wrote", OUT / "baselines_macrof1.pdf", "and", OUT / "training_curves.pdf")
