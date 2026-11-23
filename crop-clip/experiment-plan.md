# Crop-CLIP: experiment plan

**Team Support Vector (IDL F26, Group 30)** · written 2026-11-22 · final report due **2026-12-04**

This is the reference for the final-phase experiment. It records what we decided, why, and the
tasks needed to finish. If a decision changes, update this file instead of keeping it in chat.

---

## 1. Goal

Beat the WILDS iWildCam leaderboard #1 (**AutoFT, 52.0 OOD-test macro F1**, CLIP ViT-L/14@336)
under the official WILDS protocol.

- **Primary metric:** official WILDS macro F1 on OOD test (the 48 cameras never seen in training).
- **Secondary:** ID-test macro F1 and average accuracy (both are leaderboard columns).
- **Abstention / selective classification** is no longer an experiment. It goes in the paper as
  Future Work (Section 10).

Where we start (all OOD-test macro F1):

| Model | OOD test |
|---|---|
| YOLO-CLIP (our midterm) | 10.7 |
| AutoFT + WiSE-FT, CLIP ViT-B/16 | 37.1 |
| DFN CLIP ViT-H/14, whole image (Emmilly's run, best epoch) | 45.75 |
| Same + logit adjustment + burst averaging (EDA, post-hoc) | 49.21 |
| **Target: AutoFT ViT-L/14@336 (leaderboard #1)** | **52.0** |

## 2. Why this direction (evidence from the EDA)

The EDA ("iWildCam Error Atlas", `EDA/`) shows that most errors come from the model reading
the **background** instead of the animal:

- **Camera 288:** 467 elephants (22% of all elephants) and 68% of warthogs are called *impala*,
  the species that dominates that site. None of the top 15 confusions are within a genus, so
  these are background shortcuts, not look-alike species.
- **Species seen at only 1–2 training cameras** score F1 ≈ 0.0 at new cameras (13 species,
  12.5 macro-F1 points lost). F1 rises steadily with the number of training cameras.
- **Empty errors:** 888 animal photos are called empty and 335 empty frames are called animals.
- **Night infrared:** 37% of OOD test but 62% of its errors; small, dark animals.
- **Errors are concentrated:** 4 of the 48 test cameras hold 57% of errors.

**Idea:** find the animal first and classify only the animal. Cropping removes the background
shortcut and enlarges small animals.

### Lessons from YOLO-CLIP (what we are changing)

| YOLO-CLIP problem | Crop-CLIP fix |
|---|---|
| YOLO-World (an open-vocabulary detector) boxed everything, including trees, which confused classification | **MegaDetector** looks only for animal / person / vehicle and was trained on about 3M camera-trap photos |
| YOLO's own small head did the classifying (only ~93K trainable parameters) | Detection and classification are **separate**: a full fine-tuned DFN ViT-H/14 classifies the crop |
| The empty threshold was tuned on OOD val, which has a different empty rate from test (16% vs 33%) | Empty comes from detector confidence; we check the val/test mismatch explicitly (Section 7) |

### Prior work and what we can claim

- **WILDS leaderboard:** no entry classifies detector crops. #6 (Gao et al., ICML 2023) used
  MegaDetector-derived **masks only to paste animals onto other cameras' empty frames during
  training**. Its classifier still sees the whole image at test time.
- **Outside WILDS:** iWildCam Kaggle competition winners (2020/2021) classified MegaDetector
  crops, blended crop and whole-image predictions, and averaged over bursts. Those competitions
  used different splits and metrics, so their numbers are not comparable to ours.
- **Our claim:** we don't claim the detect-then-classify idea. We claim (to our knowledge) the
  first evaluation of it under the WILDS OOD protocol with a modern CLIP backbone, plus the
  analysis of *why* it helps.

## 3. Locked decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | Detector | **MegaDetector, frozen (no fine-tuning)** | WILDS has no box labels to fine-tune with. MegaDetector is already trained for camera traps, including infrared |
| D2 | Who runs MegaDetector | **We run MegaDetector v5a ourselves** on all 203,029 images, with a very low confidence floor (0.005) so every threshold can be tuned later. We record the exact weights file (name + SHA-256), package version, input size and hardware | Full control and reproducibility: every box in the paper comes from a run we can describe and rerun. We don't depend on third-party outputs whose settings we can't verify |
| D3 | Classifier | **DFN CLIP ViT-H/14 @ 378px** (`ViT-H-14-378-quickgelu`, `dfn5b`, open_clip) | Strongest model we have measured (45.75). Higher input resolution helps small crops |
| D4 | What we train | **Full fine-tune of DFN** (backbone + head). Head initialised from text embeddings of the 182 class names | Standard FLYP/AutoFT-style recipe. Freezing the backbone underfits |
| D5 | Code | **Our own notebooks in `crop-clip/`**. We don't reuse the `notebooks/sclr` code | Its core module is only embedded as base64 |
| D6 | Splits | Official WILDS v2.0 splits, unchanged | Leaderboard protocol |
| D7 | Model selection | Checkpoints and every threshold/knob chosen on **OOD val**. **OOD test is run once per final configuration** | Leaderboard protocol. Test has already been seen too often in the EDA |
| D8 | Seeds | Final configuration × **3 seeds**, reported as mean (std) | Leaderboard requirement |
| D9 | Disclosure | State that we use MegaDetector (its training data is not public) and test-time burst metadata (if used) | WILDS lists MegaDetector entries separately. Burst use is non-standard |
| D10 | Headline numbers | Report **with and without** burst averaging. The "beats 52.0" claim must be clear about which one it is | Honesty about the test-time metadata |

## 4. The pipeline

```
photo ──► MegaDetector (frozen) ──► animal boxes with confidence
                                     │
            no box ≥ τ_det ──────────┼──► predict "empty"
                                     │
            boxes ≥ τ_det ───► crop each box (+ padding p, made square) ──► resize 378×378
                                     │
                                     ▼
                     DFN ViT-H/14 fine-tuned on crops ──► 182-class logits per crop
                                     │
            several boxes ──► keep the most confident crop prediction
                                     │
            (variant) blend with whole-image DFN logits, weight w
                                     │
            post-hoc: logit adjustment (τ_LA), burst averaging (β)
                                     ▼
                         one WILDS label per image
```

### Knobs, all chosen on OOD val

| Knob | Meaning | Search range |
|---|---|---|
| τ_det | MegaDetector confidence needed to trust a box (below it → empty) | 0.05 – 0.8 |
| p | Padding around the box, as a fraction of box size | 0, 0.1, 0.2, 0.4 |
| multi-box rule | Most confident crop vs probability average over crops | 2 options |
| w | Weight on crop logits vs whole-image logits (variant D, Section 6) | 0 – 1 |
| τ_LA | Logit adjustment: logits − τ·log(train class frequency) | 0 – 2 |
| β | Burst averaging: p ← (1−β)·p + β·mean over the burst | 0 – 1 |

### Training data for the crop classifier

- **Animal photos:** every MegaDetector animal box with confidence ≥ τ_train (start at 0.3, as
  in the Kaggle solutions) becomes a crop, labelled with the photo's species.
- **Animal photos with no box ≥ τ_train:** skip them. iWildCam labels are per burst, so many of
  these frames probably contain no visible animal. Skipping them removes label noise. Record
  how many we skip.
- **Empty photos:** low-confidence boxes on empty photos (MegaDetector false positives) become
  crops labelled **empty**. The classifier can then overrule a bad box.
- **Same augmentation for every crop:** random resized crop and flip. Start with RandAugment
  off to match the baseline. Grayscale / low-light augmentation is a stretch (Section 6).

### Fine-tuning recipe (starting point, same for whole-image and crop runs)

AdamW, lr 1e-5, weight decay 0.1, 500 warmup steps then cosine schedule, 10 epochs, batch 64,
bf16 autocast, gradient checkpointing. Evaluate on OOD val and ID val every epoch; keep the
best OOD-val macro-F1 epoch. **Estimated 5–10 h per run on a large GPU (unverified for our
code). ViT-H/14 needs a big card (~24 GB+), so measure on the first run.**

## 5. Notebooks to build (`crop-clip/`)

| Notebook | Contents | Outputs |
|---|---|---|
| `00_data_eda.ipynb` ✅ written | Download and verify the dataset, then a lighter EDA (Section 5.1) | `image_stats.csv.gz` (W&B artifact `iwildcam-image-stats`), `species_cameras.csv`, `figures/eda_*.png`, `eda_summary.json` |
| `01_boxes.ipynb` ✅ written | Run MegaDetector v5a on every image (resumable, in chunks, can be split across machines with `SHARD`), save all detections, box statistics, preview grids | `boxes/boxes_all.csv.gz` (W&B artifact `megadetector-v5a-boxes`), `megadetector_run.json`, `figures/md_*.png` |
| `02_crops.ipynb` | Build the crop dataset (padding, square crops, labels as in Section 4) | crop index + optional cached 378px crops |
| `03_train.ipynb` | One training notebook with `MODE = "whole"` or `"crop"`, seed and padding options. Checkpoints every epoch, resume after crash, W&B logging | checkpoints, per-epoch val logits |
| `04_eval.ipynb` | Combines everything: thresholds, multi-box rule, fusion, logit adjustment, burst averaging (all tuned on OOD val), final OOD test, error analysis | results tables, ablation table, figures |

Shared code (dataset download/checks, metadata, WILDS macro F1, MegaDetector helpers, square
crops, W&B helpers) lives in `crop-clip/common.py`, in plain readable Python. Each notebook's
first cell clones (or updates) the team repo to get it, so **push changes to `common.py` to
`main` before teammates run the notebooks**.

### 5.0 How we run the notebooks (decided 2026-11-23)

| Item | Choice |
|---|---|
| Platform | **Molab** (Marimo cloud, RTX PRO 6000 96 GB). Sessions stop after 12 h, so every long step resumes from files on disk |
| Format | **Jupyter `.ipynb`**, imported into Molab. Molab converts it to marimo, so every global name is defined in exactly one cell and temporaries start with `_`. Packages install through `python -m pip`, not `!pip` |
| Shared outputs | **W&B Artifacts** in `team-support-vectors-idlf26/iwilds` (group `crop-clip`): image stats, boxes, later crops index and checkpoints |
| Dataset | Team HF mirror `emmilly/iwildcam-wilds-v2` (unmodified official archive), pinned revision, checked by archive SHA-256, `metadata.csv` MD5 and official split sizes. Cite WILDS as the source |
| W&B server | **CoreWeave Forge** (`https://api.forge.coreweave.com`; the team W&B was migrated there). Default in `common.py` and in each notebook's settings cell |
| W&B runs | Team `team-support-vectors-idlf26`, project `iwilds`, group `crop-clip`, tags `crop-clip` + job type. Run names: `cropclip-<notebook>-<step>[-<variant>][-s<seed>]`, e.g. `cropclip-00-eda`, `cropclip-01-megadetector-shard0of3`, later `cropclip-03-train-B-crop-s0`, `cropclip-04-eval-E-logitadj`. Artifacts: `iwildcam-image-stats`, `megadetector-v5a-boxes-shard<i>of<n>`, `megadetector-v5a-boxes` |
| Keys | Pasted only in private Molab copies or set as environment variables; never committed |

Testing: both notebooks were run end to end (as Jupyter and as converted marimo scripts) on a tiny
fake dataset, including MegaDetector on CPU, a broken image, a two-machine shard split and
resuming after an interruption. They have **not** run on the real dataset or a GPU yet.

**Data gotcha (from the EDA bug):** `categories.csv` includes catalog rows with `y` up to
99999. Use **182 classes** (`y < 182`), never `len(categories)`.

### 5.1 EDA in `00_data_eda.ipynb` (lighter than Emmilly's, our own figures)

1. **Splits overview:** photos, cameras, bursts, classes, % empty, % infrared per split.
2. **Class distribution:** train images per class (log scale, long tail); which classes appear
   in OOD val / OOD test; top 20 classes per split.
3. **Image setting:** day vs infrared share per split; photos per hour of day; brightness;
   image sizes.
4. **Cameras:** photos per camera per split; species per camera; **number of training cameras
   each species appears at** (the EDA's strongest predictor of failure); empty rate per camera.
5. **Bursts:** burst length distribution; share of single-label bursts.
6. **Preview grids (train / ID val / OOD val only):** random photos per split, day vs infrared,
   the most common species, empty photos, and the species seen at only 1–2 training cameras.
   We don't show the OOD-test cameras the earlier EDA flagged (288, 120, 301, 101, 270): they
   are test photos.
7. **After `01_boxes`:** boxes per photo, box size relative to the image, confidence histogram,
   **share of animal photos with no box** (detector miss rate) by day/IR, and example grids of
   boxes on hard photos.

Use only train / ID val / OOD val images for figures that drive decisions. Label-count tables
for test are fine (the EDA already reported them), but don't look at test photos to make
choices.

## 6. Experiments (this is the whole list)

All rows use our own code and the same fine-tuning recipe. Seed 0 unless noted.

| ID | Run | What it shows | Training runs |
|---|---|---|---|
| **G0** | Go/no-go: **zero-shot** DFN on whole images vs on crops, OOD val | Do crops help before we spend GPU hours? Cheap; no training | 0 |
| **A** | Whole-image DFN fine-tune (our baseline) | Baseline in our own code; should land near 45.75 | 1 |
| **B** | Crop DFN fine-tune, empty decided by the classifier only | Effect of cropping | 1 |
| **C** | B + MegaDetector empty rule (τ_det) | Effect of the detector's empty decision | 0 (eval only) |
| **D** | C + blend with whole-image logits (A) | Does context help when the box is bad? | 0 (eval only) |
| **E** | Best of B–D + logit adjustment | Rare species | 0 |
| **F** | E + burst averaging | Test-time burst metadata (reported separately) | 0 |
| **Final** | Best configuration × 3 seeds (the classifiers it uses: crop, plus whole-image if D wins) | The headline number, OOD test once | 2–4 more |

Rows A → F are the **ablation table** for the report (the rubric asks for one).

**Go/no-go rule after G0:** if zero-shot crops are clearly worse than whole images on OOD val,
look at the box previews first (missed animals? too-tight crops?), fix padding or τ, and rerun
G0 before training B.

**Stretch (only if A–F are done by Nov 29):** grayscale / low-light augmentation in the crop
training (targets night errors); copy-paste of animals onto other cameras' empty frames.

## 7. Evaluation and reporting rules

- Use the official WILDS evaluator (macro F1 over classes present in the split; accuracy).
- Report ID test, OOD val, and OOD test, plus OOD-test macro F1 split by **day vs infrared** and
  the worst-camera list (the EDA's breakdowns), so we can show *where* cropping helps.
- **Val/test mismatch:** OOD val is 64% infrared and 16% empty; OOD test is 37% infrared and
  33% empty. After tuning τ_det on val, compare the predicted-empty rate with the true rate on
  val, and mention the mismatch in the paper. Do not retune on test.
- **OOD test is run only for final configurations.** Log every test run in the table below.
- Per-run record: config, seed, git commit, checkpoint path, val/test scores.

### Test-run log

| Date | Config | Seeds | OOD test macro F1 | Who |
|---|---|---|---|---|
| | | | | |

## 8. Tasks and timeline

Owners to be filled in at the team meeting. Several tasks can run in parallel; the critical
path is **T1 → T3 → T5 → T7 → T9**.

| ID | Task | Depends on | Target date | Owner |
|---|---|---|---|---|
| T1 | Dataset on each GPU machine; `common.py` (dataset, 182-class mapping, metrics, prompts) | — | Nov 23 | |
| T2 | `00_data_eda.ipynb`: EDA items 1–6 (Section 5.1) | T1 | Nov 25 | |
| T3 | `01_boxes.ipynb`: run MegaDetector v5a on all 203k images (time a 1,000-image sample first to estimate total run time); record version/weights hash; box statistics (EDA item 7) | T1 | Nov 24 | |
| T4 | `03_train.ipynb` whole-image mode; launch **run A** (seed 0) | T1 | start Nov 24 | |
| T5 | `02_crops.ipynb` + **G0** go/no-go | T3 | Nov 25 | |
| T6 | `04_eval.ipynb`: evaluator, knob sweeps on OOD val, logit adjustment, burst averaging, day/IR and per-camera breakdowns | T1 | Nov 27 | |
| T7 | Crop mode in `03_train`; **run B** (seed 0) | T5 | start Nov 26 | |
| T8 | Rows C–F on OOD val; pick the final configuration | T4, T6, T7 | Nov 29 | |
| T9 | **Final:** 2 more seeds of the chosen classifier(s); OOD test once | T8 | Nov 30 – Dec 1 | |
| T10 | Figures: ablation table, day/IR, worst cameras, example crops (good and bad) | T9 | Dec 2 | |
| T11 | Paper (8–10 pages): see outline below | all | Dec 2 – Dec 4 | all |

Watch out: Thanksgiving (Nov 26–27) falls in the middle. Start runs A and B before then so the
GPUs keep working.

## 9. Paper outline (final report rubric: 8–10 pages)

1. **Abstract / Introduction:** problem, goal (beat 52.0), what we found.
2. **Related work:** WILDS / iWildCam; CLIP fine-tuning (FLYP, AutoFT, WiSE-FT); DFN; detectors
   in camera-trap pipelines (MegaDetector, Gao et al. targeted augmentation, Kaggle crop
   solutions); YOLO-World / YOLIP.
3. **Dataset and EDA:** our EDA figures; the background-shortcut evidence that motivates cropping.
4. **Method:** pipeline diagram, input/output shapes, parameter counts, loss written out
   (cross-entropy over 182 classes; text-initialised head), knobs.
5. **Baselines:** our measured table (ResNet/ABSGD, FLYP, AutoFT, WiSE-FT, DFN whole-image,
   YOLO-CLIP) with reproduction evidence (W&B curves).
6. **Results:** final 3-seed numbers vs leaderboard; with/without bursts.
7. **Ablation (rows A–F)** and sensitivity to τ_det and padding.
8. **Failure analysis:** day vs infrared, worst cameras, detector misses, crops that still fail.
9. **Discussion:** limits (MegaDetector disclosure, val/test mismatch, burst metadata).
10. **Future work:** abstention / selective classification (the 42% of errors made at ≥ 0.9
    confidence, likelihood-ratio selectors); fine-tuning the detector; copy-paste augmentation.
11. **Team contributions**, GitHub link.

## 10. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| MegaDetector run slower than expected | 1,000-image timing implies > 1 day for 203k | Split the 203k images across two or three teammates' GPUs by index range and merge the outputs |
| MegaDetector misses small / night animals | High no-box rate on animal photos in OOD val | Lower τ_det; whole-image fallback (variant D); burst averaging |
| Crops lose useful context | G0 or B worse than A | Bigger padding; variant D blend |
| ViT-H too slow or out of memory | First epoch > 1.5 h, or OOM | Smaller batch + gradient accumulation; fall back to CLIP ViT-L/14@336 |
| A run crashes before the deadline | — | Checkpoint every epoch; resume |
| We fall short of 52.0 | Final < 52.0 | Still a full paper: ablation + failure analysis meets the rubric ("justified analysis of failures") |

## 11. References

- Beery, Morris, Yang (2019). *Efficient Pipeline for Camera Trap Image Review.*
  arXiv:1907.06772. (Main MegaDetector citation.)
- Hernandez et al. (2024). *Pytorch-Wildlife: A Collaborative Deep Learning Framework for
  Conservation.* arXiv:2405.12930. (MegaDetector v5/v6.)
- Beery et al. (2021). *The iWildCam 2021 Competition Dataset.* arXiv:2105.03494. (The
  iWildCam data that WILDS builds on.)
- Gao et al. (2023). *Out-of-Domain Robustness via Targeted Augmentations.* ICML.
  arXiv:2302.11861.
- Fang et al. (2023). *Data Filtering Networks.* (DFN-5B CLIP.)
- Koh et al. (2021). *WILDS.* ICML. · Choi et al. (2024). *AutoFT.* arXiv:2401.10220 ·
  Goyal et al. (2023). *FLYP.* · Wortsman et al. (2022). *WiSE-FT.*
- iWildCam 2021 1st-place solution: github.com/alcunha/iwildcam2021ufam
- WILDS leaderboard: https://wilds.stanford.edu/leaderboard/
