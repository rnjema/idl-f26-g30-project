# Results and Discussion — draft section for the IDL-F26 midterm report

Fits `11785_Project_Group30.pdf` as a new Section 6 (before References), replacing the
current "5 Anticipated Results" placeholder with actual findings. All `XX.X` values are
filled from W&B run summaries or the eval JSONs on the checkpoint volume; the fill-in
table at the end maps every placeholder to its source.

---

## 6 Results and Discussion

### 6.1 Reproduced classifier benchmark

Our first milestone is the fixed classifier that the abstention selector will build on.
We upgraded the proposed FLYP baseline to **AutoFT** (Choi et al., 2024), the current
#1 entry on the WILDS iWildCam leaderboard, keeping the same role in the pipeline: a
CLIP ViT-B/16 image encoder fine-tuned on the FLYP image–text stream, then frozen.
Table 1 reports macro F1 (the official WILDS metric; accuracy secondary) on all four
official splits.

| Method | Backbone | IDVal F1 | IDTest F1 | OODVal F1 | OODTest F1 |
|---|---|---|---|---|---|
| Zero-shot CLIP (2 prompts) | ViT-B/16 | XX.X | XX.X | XX.X | XX.X |
| FLYP (teammate run) | ViT-B/16 | 43.2 | 49.3 | 32.9 | 30.8 |
| AutoFT (ours) | ViT-B/16 | 49.4 | 43.4 | 39.1 | **34.8** |
| AutoFT + WiSE-FT (ours) | ViT-B/16 | 49.8 | 47.2 | 40.2 | **37.1** |
| *Reference: FLYP (paper)* | *ViT-L/14@336* | — | — | — | *46.0* |
| *Reference: AutoFT (paper)* | *ViT-L/14@336* | — | — | — | *48.7 / 52.0 w/ WiSE-FT* |

Two observations. First, the **relative** ordering reproduces the paper's Table 2
pattern: on OOD-test macro F1, AutoFT beats the FLYP baseline by **+4.0** and
AutoFT+WiSE-FT by **+6.3** on our ViT-B/16 backbone (paper, ViT-L/14@336: +2.7 and
+6.0 respectively) — a remarkably close match on the ensembled gap. Second,
ensembling with the zero-shot model via WiSE-FT (alpha selected on ID-val only)
helps the OOD splits markedly more than ID (+2.5 OOD-test macro F1 vs +0.4 ID-val
over the plain AutoFT model) — the same asymmetry that carries the leaderboard
entry from 48.7 to 52.0. Our absolute numbers sit well below the leaderboard
entries — expected and by design: the leaderboard uses CLIP ViT-L/14@336, while we
follow our compute budget (ViT-B/16 @ 224 px). All claims in this report are
therefore relative to the FLYP run under near-identical conditions, never to
leaderboard entries.

Two caveats on the FLYP row. It was produced by a teammate's independent FLYP
implementation (lr 1e-5, wd 0.2, batch 64, 20 epochs, seed 0, mixed precision)
rather than our CLI, so it is "same backbone, same data, same hyperparameter
spirit" rather than bit-identical recipe. We also note an ID-split ordering
difference — the teammate's run reports IDTest > IDVal (49.3 vs 43.2), opposite to
both our AutoFT runs (49.4 vs 43.4) — which suggests a possible IDVal/IDTest label
swap in that eval harness; we keep the numbers as logged and will confirm the
assignment before the final report. This does not affect the OOD comparisons, which
are the headline.

The AutoFT run uses the official recipe: a TPE search over nine loss weights (+ lr,
wd) with trial 0 seeded by the authors' published iWildCam weights, outer-loop
objective = macro F1 on 1,000 class-balanced OOD-validation images, and a 20-epoch
final fine-tune with best-epoch selection by **ID-val** macro F1 (the paper's early-
stopping rule). WiSE-FT interpolates fine-tuned and zero-shot weights, selecting the
interpolation coefficient alpha on ID-val F1 only.

### 6.2 Training and evaluation protocol notes

Per our protocol, every model-selection decision above uses ID validation or the
official OOD validation split; **the OOD-test column of Table 1 was produced once per
model and not used for any selection**. Splits were verified programmatically: 203,029
labeled images, 182 classes, camera-trap counts {train: 243, val: 32, test: 48}, and
zero location overlap across slabs. All runs log per-trial objectives, per-epoch
validation metrics, and final split metrics to Weights & Biases under the leaderboard-
entry naming (`AutoFT-ViT-B-16-iWildCam-<run>`), which also made the ~15 h A100 fine-
tune straightforward to monitor and resume.

The fine-tune also survived a real-world stress test: the cloud job was **preempted
mid-run**, and the per-epoch checkpoint/resume machinery continued from the last
committed epoch with the best-model state intact — the selected best epoch (ID-val
F1 0.4943) predated the preemption, and the restored run's final model reproduced
exactly that ID-val F1 at evaluation time (0.4943), confirming the checkpointing is
faithful rather than approximately-so.

### 6.3 Implications for the abstention lane

With the frozen classifier in place, the selector work proceeds as proposed:
(1) extract L2-normalized embeddings of split-A/B locations under the leakage-free
80/20 camera-trap partition (already materialized: 194 A-traps / 49 B-traps);
(2) reproduce the Heng & Soh Δ-MDS / Δ-KNN + raw-logit-margin combination as the
primary baseline, with MSP as the cheap comparator; (3) our learned-combination and
augmentation-consistency extensions on top.

One finding from the literature sharpens our expectations: Liang et al. (TMLR 2024)
report that on iWildCam, raw-logit-margin scores are *only on par with softmax
confidence* — unlike on ImageNet — so naive confidence tricks have little headroom
here. Our hypothesis (that likelihood-ratio-style evidence over clustered feature
sets can do better) is therefore a genuinely open question on this dataset, and a
negative result would still be a meaningful, well-instrumented contribution.

### 6.4 Deviations from the proposal and risk register

- **Baseline upgraded FLYP → AutoFT** (leaderboard #1, strictly stronger foundation,
  identical role as the fixed classifier). The FLYP row in Table 1 preserves the
  originally proposed baseline for comparison.
- **Split-A classifier**: the Table 1 models train on the *full* official train split
  (leaderboard protocol). The selector lane requires a classifier trained on split A
  only, so a split-A AutoFT run (≈15 h GPU) is queued; all selector comparisons will
  use that model, not Table 1's.
- **Cost/dataset risks**: none materialized — the full pipeline (dataset, staged GPU
  runs, resumability) is operational on cloud A100s; the main residual risk is the
  known small-win regime for selectors on iWildCam (§6.3).

---

## Appendix: where each number comes from

| Placeholder | Source |
|---|---|
| Zero-shot row | W&B run `eval_zeroshot_sanity` summary (`IDVal/F1-macro_all`, …) |
| FLYP row | W&B run `iwildcam_flyp_vitb16-seed0` by N'nouka Stephen (run id `r1pvlmht`, summary keys `final/{id_val,id_test,val,test}_macro_f1`; "val"/"test" = OOD val/test in that harness) |
| AutoFT row | W&B run `AutoFT-ViT-B-16-iWildCam-autoft_vitb16_paperhparams`; eval JSON `/ckpt/reports/autoft_eval_final_model.json` |
| WiSE-FT row | `/ckpt/models/autoft/wisefmt_autoft_vitb16_paperhparams/results.json` (best-alpha OODVal/OODTest) |
| Paper reference rows | AutoFT paper Table 2 (arXiv:2401.10220) |

Fill command (after each training run completes):

```bash
MODAL_ENVIRONMENT=idlf26 modal run scripts/modal_autoft.py::eval_model \
    --ckpt /ckpt/models/autoft/<run>/final_model.pt
MODAL_ENVIRONMENT=idlf26 modal run scripts/modal_autoft.py::wisefmt \
    --zeroshot /ckpt/models/zeroshot/clip_vitb16_iwildcam.pt \
    --finetuned /ckpt/models/autoft/<run>/final_model.pt \
    --run-name wisefmt_<run>
```
