# YOLO-World on iWildCam-WILDS: implementation plan

## Goal and experiment lanes

Test an integrated YOLO-World-like vision-language detector for **one-label-per-image wildlife classification at unseen camera locations**. This is the `yolo-clip` lane; do not change teammates' AutoFT, FLYP, or selector work. The WILDS task supplies image labels and camera-location metadata, **not ground-truth animal boxes**. One image label does not imply exactly one animal or one valid detector proposal. [WILDS iWildCam task](https://wilds.stanford.edu/datasets/#iwildcam).

We will use two separate notebooks, imported directly into **Molab as `.ipynb` files**:

1. `yolo-clip-untrained.ipynb` - a **frozen, prompted YOLO-World reference** with no iWildCam weight updates. **Start here.** It validates the data pipeline, prompt mapping, candidate boxes, box-to-image rule, and inference cost before training is introduced.
2. `yolo-clip-notebook.ipynb` - the **WILDS-trained baseline**, initialized from the same pretrained model and evaluated under the same image-level protocol. Training components and the exact image-level loss are candidate choices below, to finalize before implementing training.

Both files are currently empty placeholders and must be made valid notebooks when coding begins. The [point-by-point final plan](yolo-clip-final-plan.md) records agreed decisions; this document gives their implementation order.

YOLO-World matches candidate visual features with embeddings of supplied class text **inside the detector**. It does not require a separate CLIP image-encoder pass over every crop. YOLIP is related but different: it adds language-aligned proposal features **and** CLIP visual processing of proposed regions. We are not claiming to reproduce YOLIP. [YOLO-World paper](https://arxiv.org/abs/2401.17270); [YOLIP review](lit-review.md).

## Ordered steps and completion gates

### 1. Prepare and inspect iWildCam-WILDS in the untrained notebook

- Run on Molab; the user will confirm dataset loading and actual available capacity in their environment. Record the Python/GPU and relevant package versions seen in the run. No notebook conversion is needed for Molab import.
- Load iWildCam-WILDS through its official interface. Inspect `train`, `id_val`, `val` (OOD validation), `id_test`, and `test` (OOD test) definitions; sample **train/validation** images, dimensions, image-level labels, class-name mapping (including `empty`), and camera-location metadata. Summarize class and location counts. Reuse existing read-only WILDS utilities if helpful; do not modify or relabel the supplied dataset.
- Preserve the official camera-location splits. Do not use test labels or images for prompt design, thresholds, training, or model choice. No box inventory is required to begin: WILDS is an image-level classification benchmark.

**Gate:** the first notebook section loads the dataset, displays representative non-test examples, and records the actual label/split mapping.

### 2. Establish frozen prompted YOLO-World inference

- Pin an official YOLO-World checkpoint, config, code revision, input resolution, preprocessing, and prompt template. Start with a supported model size that fits the observed Molab run; do not infer a model-size limit from the local workstation.
- Construct prompts from WILDS's existing animal-class names. The text encoder represents these supplied names; it does not generate new class labels. Map detector outputs back to WILDS class IDs.
- On a small set of **non-test** images, inspect candidate boxes, species scores, false boxes, missed animals, and empty images. Save a reproducible configuration and measure basic inference cost.

**Gate:** repeatable prompted single-image inference with a verified WILDS class-ID mapping.

### 3. Convert detections into the required image prediction

- Use the **agreed first rule**: output the class of the highest-scoring animal detection. If there is no detection, or its score is below a threshold selected on validation data, output the WILDS `empty` class. This must yield exactly one valid class ID for every image.
- Keep candidate boxes/scores alongside the image prediction for diagnosis. Do not assume a single true animal or box; high-scoring non-animal proposals are a failure mode to measure.

**Gate:** deterministic box-to-image predictions on non-test data, including `empty`.

### 4. Evaluate the frozen reference without contaminating the final test

- Evaluate on the official ID and OOD **validation** splits. Report the WILDS image-level **macro-F1** first, with accuracy and empty/non-empty, class, and location breakdowns as diagnostics. Record runtime, candidate-box counts, and representative failures.
- Do not report object-detection mAP from WILDS image labels; mAP requires box annotations. Preserve the OOD test split for final, once the model and analysis choices are fixed.

**Gate:** a reproducible untrained reference result and error breakdown exist.

### 5. Train the WILDS baseline in the second notebook

Both YOLO-World's and YOLIP's original region-level training depend on region/box supervision, whereas WILDS supplies an image label. **Adapting this mismatch is a research question**, not a reason to fabricate boxes. Our *candidate simple baseline* is:

- Initialize WILDS class embeddings from CLIP text prompts. Freeze the pretrained image backbone, fusion, localization/box branch, and CLIP text encoder; train only the class embeddings and possibly a small score scale/bias.
- For each animal class, take its maximum score over candidate regions to form an **image-level class score**. Add an `empty` score, then apply image-level cross-entropy using the WILDS training label. Use pre-filter candidate scores so training is not cut off by hard post-processing. This is compatible with the highest-scoring-detection inference rule.
- Use the same official splits and image-level evaluation as the frozen reference. Choose the exact trainable parameters, loss implementation, class imbalance handling, and validation threshold **before training code is written**. These are candidates, not yet agreed experimental results or claims of a novel loss.

**Gate:** the trained notebook produces one image label per sample, records its training configuration, and compares against the frozen reference at matched validation protocol.

### 6. Extensions and final evaluation

- Diagnose whether errors arise from missed localization, false boxes, wrong species matching, `empty` handling, or camera shift. Possible later experiments include unfreezing the semantic head/fusion layers, smooth or attention-based region aggregation, and cautiously filtered pseudo-box supervision. These are **options to revisit**, not baseline requirements.
- After all prompts, training choices, thresholds, and comparison rules are frozen using training/validation data, evaluate both fixed models on `id_test` and OOD `test` for final reporting. Keep teammates' classifiers as contextual comparisons with training/pretraining differences stated.
- If the course project later reconnects to abstention, evaluate selection on **unchanged image-level predictions** and report matched-coverage risk separately from classification macro-F1.

## Baseline finding: `empty` calibration and a future improvement

The trained baseline froze the YOLO-World detector and CLIP text encoder, trained only 181
CLIP-initialized animal class vectors, and used a fixed `empty` logit of 0 in its image-level
cross-entropy loss. Training pooled the maximum **pre-NMS** region score per class. Final inference
instead selected the highest-scoring **post-NMS** detection, predicting `empty` when its score was
below a threshold. That threshold was **not learned during training**: a sweep on the official
held-out-camera `val` split selected **0.080436** by macro-F1. The epoch-9 checkpoint and this
threshold were frozen before either final test split was evaluated. The tuned `val` macro-F1 is a
model-selection result, not an independent test estimate.

Using validation to choose a threshold is legitimate: the cutoff should transfer to the intended
evaluation setting, not necessarily reproduce the full training corpus. But a single threshold
chosen on one set of held-out cameras may not calibrate other cameras or class prevalences well.
The observed `empty` fractions illustrate the shift: `val` had **16.0%** true empty images and
**11.0%** predicted empty; `id_test` had **24.6%** versus **18.2%**; OOD `test` had **33.0%** versus
**20.2%**. These gaps are diagnostic, not proof that the threshold alone caused the test errors.
A threshold also cannot fix a confidently detected wrong species or a missed/poorly localized
animal. Freezing YOLO limits changes to detection/localization, but does **not** prevent learning
an empty-versus-animal calibration parameter.

For a **new, separately tracked experiment**, consider training an `empty` bias or a small
calibrator from image-level training labels while keeping YOLO frozen. Define its relationship to
the post-NMS image rule explicitly; the current pre-NMS training scores and post-NMS inference
scores are not interchangeable. Compare calibration and macro-F1 on development splits, and, if
tuning on `val`, keep a camera-disjoint portion for an untuned check where feasible. A threshold
fit on the training images alone may overfit that distribution. Do not retune this frozen run from
its final test outcomes; report any later variant as exploratory or under a new evaluation plan.

## Experiment discipline

- Record code/checkpoint versions, text prompts, class mapping, score threshold, aggregation rule, seed, split, hardware, and any deviations. Keep frozen-reference, trained-baseline, and later-extension results distinct.
- Store images, checkpoints, and full detections outside Git; commit only notebook/code, compact configuration, and summarized results. Do not change teammates' files.
- Our contribution is to investigate **image-level adaptation and evaluation of a region-level vision-language detector under camera-location shift**. Max-over-regions pooling is a candidate baseline, not by itself a proven novel method or state-of-the-art result.

## Immediate next action

Begin `yolo-clip-untrained.ipynb` with **Step 1 only**: create a valid notebook and implement dataset loading and exploratory checks. Review its observed labels, splits, and Molab behavior before adding YOLO-World inference.
