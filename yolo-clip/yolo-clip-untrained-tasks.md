# Untrained YOLO-World notebook: end-to-end tasks

This checklist is for `yolo-clip-untrained.ipynb` only. **Untrained** means the pretrained YOLO-World weights are never updated using iWildCam-WILDS. Choosing prompts and an `empty` threshold on validation data is allowed, but must be recorded as validation selection. The notebook produces a frozen-model **image-level classification reference**, not an object-detection benchmark or our later trained baseline. We will build it one stage at a time, reviewing each stage before continuing.

## 1. Make the notebook runnable on Molab

- [ ] Replace the current empty placeholder with a valid `.ipynb` containing a short purpose/protocol cell and clearly ordered code cells. Import it directly into Molab; no manual format conversion is planned.
- [ ] Record the observed Python, PyTorch/CUDA, GPU, and package versions. Install only the dependencies required for WILDS and the chosen official YOLO-World inference path; keep this lane separate from teammates' environments.
- [ ] Confirm the actual dataset-loading route and available capacity in the user's Molab session. Record what works instead of assuming a storage limit in advance.

**Checkpoint:** the notebook opens and its setup cells run in Molab without training a model.

## 2. Load and understand the official iWildCam-WILDS task

- [ ] Load iWildCam-WILDS through the WILDS interface. Identify `train`, `id_val`, `val` (unseen-location validation), `id_test`, and `test` (unseen-location test); keep the provided partitions unchanged.
- [ ] Inspect training/validation examples, image dimensions, labels, class-name mapping, camera-location metadata, and class/location counts. Confirm the actual number of classes and the `empty` class ID from the loaded version.
- [ ] Display a few **train/validation** photos, including `empty` and non-empty examples. Do not inspect test images or labels while designing the method. Do not infer that one image label means one animal or one ground-truth box.

**Checkpoint:** we can explain what the loader returns and map every WILDS class ID to its exact dataset name. [WILDS getting started](https://wilds.stanford.edu/get_started/).

## 3. Pin and smoke-test pretrained YOLO-World

- [ ] Smoke-test `yolov8s-worldv2.pt`, the Ultralytics-migrated S-size YOLO-World weights. Record the Ultralytics package version, weight source and SHA-256, image resolution, and preprocessing. This is not the authors' original MMYOLO runtime. Use the same starting checkpoint later in the trained notebook for a fair comparison.
- [ ] Run inference on a few non-test photos with simple animal prompts. Display proposed boxes, classes, and scores; verify coordinate scaling and class-score ordering. Keep all model weights frozen.
- [ ] Confirm which outputs are available for the image-level adapter and measure basic per-image runtime and GPU use. If the official checkpoint or dependencies fail in Molab, record the failure and choose a documented compatible alternative before a full run.

**Checkpoint:** repeatable prompted detection on individual WILDS photos. [Official YOLO-World code and model zoo](https://github.com/AILab-CVC/YOLO-World).

## 4. Build the WILDS prompt and label map

- [ ] Create a deterministic `class ID -> species name -> text prompt` mapping from WILDS training metadata, not from generated names. Check ambiguous or non-animal labels manually using the dataset's own mapping.
- [ ] Supply animal-class prompts to YOLO-World; handle WILDS `empty` as the **image-level no-animal outcome**, not as an object to box. Record the exact prompt template and order so detection scores map back to the correct WILDS IDs.
- [ ] Freeze one initial prompt set before broad validation. Prompt variants, if tried, must be explicit validation-only comparisons; never choose prompts after seeing test results.

**Checkpoint:** detected class indices translate to valid WILDS animal-class IDs without off-by-one or ordering errors.

## 5. Turn detections into one image-level prediction

- [ ] Apply our agreed initial rule: choose the class of the **highest-scoring animal detection**. If no detection exists or its score is below a threshold, predict WILDS `empty`.
- [ ] Choose the threshold using validation data and document the selected value and criterion. Keep the raw candidate boxes/scores and the final image prediction for later error analysis.
- [ ] Check edge cases: no boxes, several boxes, boxes on background/non-animals, tied scores, and a photo with more than one animal. Specify deterministic behavior rather than assuming exactly one valid box.

**Checkpoint:** every input photo produces exactly one valid WILDS image-level class ID.

## 6. Evaluate on validation splits and diagnose errors

- [ ] Run the frozen model on `id_val` and OOD `val` with the **same** checkpoint, prompts, preprocessing, and box-to-image rule. Use the WILDS evaluator to report its official image-level **macro-F1**, plus secondary accuracy.
- [ ] Report class-level and empty/non-empty breakdowns, the fraction predicted `empty`, and summaries by camera location. Plot a confusion matrix or compact error table; inspect representative missed animals, false animal boxes, and wrong species assignments on validation only.
- [ ] Record inference time, number of candidate boxes, and GPU usage. **Do not calculate detection mAP** from WILDS image labels, because WILDS does not provide ground-truth boxes for that metric.

**Checkpoint:** a reproducible, honestly labeled *untrained-on-WILDS* validation reference and failure analysis exist. [WILDS evaluation interface](https://wilds.stanford.edu/get_started/).

## 7. Freeze and hand off the reference configuration

- [ ] Save a compact run manifest: code/checkpoint version, prompts and class mapping, preprocessing, score definition, threshold, image-level rule, package versions, and validation metrics. Keep large datasets and full detections outside Git.
- [ ] Provide the same interface and configuration to `yolo-clip-notebook.ipynb` so its trained model is compared on the same image-level task and splits. **No WILDS weight updates occur in this notebook.**
- [ ] Keep `id_test` and OOD `test` execution behind an explicit final-evaluation switch. Do **not** run them until both the frozen reference and trained-baseline choices are finalized; then evaluate once for final reporting without retuning.

**End state:** one importable Molab notebook, a frozen YOLO-World WILDS reference on ID/OOD validation, diagnostic examples, and a locked final-test path. The separate trained notebook and any YOLIP-inspired or abstention extensions remain later work.

## Frozen-reference results and findings

The completed reference uses frozen `yolov8s-worldv2.pt` weights, the 181 exact non-`empty` WILDS class names as prompts, and the highest-scoring detection as the image label. No box, or a top score below **0.14**, yields `empty` (ID 0). This threshold maximized WILDS-style macro-F1 among 202 candidates on the official held-out-camera `val` split. The configuration and full-precision metrics are in [the frozen manifest](./yolo_world_reference_a52bf73dbd25.json).

| Official validation split | Images | Macro-F1 | Accuracy | True / predicted `empty` | Correct non-`empty` | Images at 20-box cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `id_val` (familiar cameras) | 7,314 | 0.0245 | 37.20% | 40.4% / 63.5% | 71 / 4,359 (1.6%) | 87.4% |
| `val` (held-out cameras; threshold-tuned) | 14,961 | 0.0224 | 15.05% | 16.0% / 54.0% | 151 / 12,562 (1.2%) | 86.6% |

Findings:

- The reference rarely assigns the correct non-`empty` class. On `val`, 5,981 non-`empty` images became `empty` and another 6,430 received the wrong non-`empty` class. Threshold tuning alone cannot correct the latter errors.
- Accuracy is below an always-`empty` predictor on both splits (40.4% on `id_val`; 16.0% on `val`). The accuracy gap between splits is partly explained by their different `empty` prevalence; macro-F1 is similarly low on both, so the failure cannot be attributed solely to camera-location shift.
- At the 0.001 detector confidence floor, every validation image produced a candidate box, while roughly 87% reached the 20-box cap. Thus `empty` predictions came from the image-level threshold, and stored box counts are truncated. In inspected examples, some boxes covered animals but assigned a wrong class, including a deer labeled hyena at score 0.949. This suggests a class/prompt-matching problem, but the examples do not isolate its cause.
- Exact dataset names include scientific names, broad labels, near-duplicates, and `motorcycle`. This is a deliberately fixed *untrained exact-label-prompt reference*, not a claim about YOLO-World's best achievable performance. Keep the prompt map fixed for the first trained comparison; test prompt wording as a separate ablation.
- The `val` macro-F1 is a **threshold-selection score**, not an independent performance estimate. Neither `id_test` nor `test` has been run. No detection mAP is reported because this task provides image-level labels, not ground-truth boxes.

Molab completed the 14,961-image `val` cache in about 65 seconds and the 7,314-image `id_val` cache in 32.8 seconds. The latter run reported 0.46 GiB peak **PyTorch-allocated** GPU memory, not total GPU usage.
