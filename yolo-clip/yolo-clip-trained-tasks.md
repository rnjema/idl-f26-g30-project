# YOLO-CLIP trained notebook: end-to-end tasks

This checklist is for `yolo-clip-notebook.ipynb`, the **trained YOLO-CLIP** lane. We will implement it one task at a time and review each Molab output before continuing. Every code cell must start with `# CELL N` (numbered from 1 in this notebook); Markdown cells are unnumbered. When a cell changes after the user's Molab copy has run, state exactly which cell changed and whether it must be rerun.

The main experiment is to **train** an integrated YOLO-CLIP model on iWildCam-WILDS image labels. We initialize it from `yolov8s-worldv2.pt`. The earlier frozen run is a lightweight contextual check, not the training objective or a prerequisite for this notebook.

## The simple trained baseline we intend to build

- **Integrated model:** Start with pretrained YOLO-World. Its detector produces candidate regions and compares their localized visual features with CLIP-derived class embeddings *inside the model*. We are not adding a separate CLIP image encoder that classifies YOLO crops.
- **Trainable parts:** Use the supplied WILDS class names in a simple text prompt to initialize one embedding for each of the 181 non-`empty` classes. These are starting vectors, not prompts we optimize or generate at inference. Freeze the image backbone, vision-language fusion, localization layers, and CLIP text encoder. Train the class embeddings, with at most a small shared score scale/bias if needed.
- **Image-level objective:** For each class, take its highest score across candidate regions as the image's class score. Add an `empty` score, then use cross-entropy against WILDS's single image label. Training needs differentiable candidate scores *before* hard filtering and NMS; no ground-truth boxes are assumed.
- **Image-level inference:** Use the agreed highest-scoring-detection rule: predict its WILDS class, or `empty` if there is no detection or its score falls below a validation-selected threshold. Evaluate the trained model on the official WILDS task.

This is **our proposed image-label adaptation**, not YOLO-World's original box-supervised training method. Its main limitation is that changing class embeddings cannot recover an animal the frozen localization path never proposes. The [implementation plan](./yolo-clip-implementation.md) and [point-by-point plan](./yolo-clip-final-plan.md) record this candidate; Task 2 checks whether the pinned implementation supports the necessary gradients, and Task 3 fixes the remaining score/`empty` details before a full run.

## 1. Prepare the training environment and WILDS data

- [ ] Make `yolo-clip-notebook.ipynb` a valid `.ipynb` importable directly into Molab. Record the observed Python, loaded and installed PyTorch versions, CUDA/GPU, Ultralytics, CLIP, WILDS, and other training dependencies. Do not assume the workstation's GPU or Molab storage limits apply to this session.
- [ ] Load the existing verified iWildCam-WILDS v2.0 data through WILDS; if the Molab session is new, repeat the verified dataset preparation without altering labels or official splits. Use only `train`, `id_val`, and `val` during development.
- [ ] Verify all 182 WILDS class IDs and names, including `empty` ID 0. Keep `motorcycle`, broad labels, and near-duplicate labels as distinct WILDS IDs. Do not substitute generated species names. Check the chosen checkpoint itself when loading the model in Task 2.

**Checkpoint:** the training dataset, split roles, and label map are correct; neither `id_test` nor `test` has been loaded.

## 2. Verify that the proposed integrated model is trainable with image labels

- [ ] Load a fresh copy of the **same** YOLO-World S-v2 checkpoint. Inspect the pinned Ultralytics implementation in Molab to locate class embeddings and differentiable region/class scores **before confidence filtering and NMS**. `set_classes()` working for inference does not by itself establish that its cached text embeddings can receive gradients.
- [ ] On one small training batch, confirm score tensor shapes, WILDS class order, the effect of changing one class embedding, and nonzero gradients to a trial embedding. Confirm that the image backbone, vision-language fusion, box/localization branch, and CLIP text encoder can remain frozen. Log trainable parameter names and counts.
- [ ] Keep this an **integrated detector**: do not quietly replace it with YOLO crops passed through a separate CLIP image encoder. Do not invent ground-truth boxes or use ordinary box-supervised YOLO loss on image-level WILDS labels. If the pinned port cannot expose the needed differentiable scores, stop at this gate and agree on a documented integrated-model alternative before training.

**Checkpoint:** a forward/backward smoke test proves the intended class-embedding training path is real; otherwise the blocker and alternative are explicit.

## 3. Lock the simple training recipe before the full run

- [ ] Specify the first baseline's trainable parameters: initialize one embedding per non-`empty` WILDS class from a simple class-name prompt, freeze the pretrained visual/fusion/box and CLIP-text components, and train only those embeddings plus an optional shared score scale/bias. Decide whether the `empty` score is a learned scalar. Record every deviation from this proposed recipe.
- [ ] Define the image-level objective using the actual differentiable outputs: for each non-`empty` class, take its maximum **pre-NMS region score**; combine the 181 class scores with an `empty` score and apply cross-entropy to the one WILDS image label. Hard filtering, NMS, and the 20-box inference cap must **not** sit inside this training gradient path. Specify how images with several animals and images labeled `empty` are handled without assuming one true box.
- [ ] Predeclare optimizer, learning rate, batch size/accumulation, epochs, seed, checkpoint interval, and class-imbalance policy. Recommended first pass: plain image-level cross-entropy, with class weighting as a separate controlled experiment if needed. Decide and record checkpoint selection (for example, `id_val` macro-F1) and use held-out-camera `val` for the image-level threshold; label any `val` result used for selection as tuned.
- [ ] Specify the trained model's image-level inference rule and select its own `empty` threshold on validation data. If a learned `empty` score is also used at inference, report that as a separate variant rather than silently changing the rule.

**Checkpoint:** the loss, trainable parameters, split roles, and primary inference rule are written down and approved before a long run.

## 4. Build and smoke-test the training pipeline

- [ ] Build a deterministic WILDS `train` loader with image-level labels and 640-pixel preprocessing compatible with the checkpoint. No synthetic box annotations are added. Verify `empty` and non-`empty` batches, class IDs, image shapes, and camera metadata.
- [ ] Run forward, finite-loss, backward, and optimizer-step checks on a tiny batch. Assert that only intended parameters change; verify the loss responds to labels and that a tiny subset can be overfit. Check that trained embeddings are not overwritten by a later `set_classes()` call.
- [ ] Save and reload a small checkpoint containing trainable weights, optimizer/scheduler state, step/epoch, seed, prompt map, and configuration. Measure actual GPU memory and step time in Molab before choosing full-run batch size. Make resume after interruption possible; keep checkpoints outside Git.

**Checkpoint:** a short supervised run genuinely updates the integrated model's class matching, survives reload, and produces one valid WILDS image-level label.

## 5. Train the first baseline

- [ ] Train only on the official `train` images. Save resumable checkpoints at planned intervals and record learning curves, loss components, GPU use, elapsed time, and any interruption/restart. Do not use test images or labels for any decision.
- [ ] Log the run to the team's existing CoreWeave Forge W&B project `team-support-vectors-idlf26/iwilds`: record the exact model/recipe and dataset split roles, periodic training loss and optimizer/throughput diagnostics, plus per-epoch `id_val` selection loss and checkpoint information. CELL 24 has a key placeholder for the user's private Molab copy; never print, share, or commit the filled-in copy. Retain local checkpoint/metric records if Molab or W&B disconnects, and resume the same W&B run ID when resuming a checkpoint.
- [ ] Evaluate each planned checkpoint on the predeclared selection split, then lock the selected checkpoint. Repeat the selected configuration across more than one seed if resources allow; do not present a best-of-many validation run as a single typical result.
- [ ] Freeze the resulting model for evaluation, recording the pretrained checkpoint hash and a new trained-checkpoint hash. Keep this baseline distinct from later prompt changes, class-weighting, pooling, or unfreezing experiments.

**Checkpoint:** a reproducible WILDS-trained checkpoint exists and differs from the starting checkpoint only in the declared trainable components.

## 6. Evaluate and compare on validation

- [ ] Apply the trained model to `id_val` and held-out-camera `val` with one image-level prediction per photo. Select its own threshold on `val` using WILDS-style macro-F1; do **not** reuse the frozen run's threshold of 0.14.
- [ ] Report the trained model's official WILDS **macro-F1** first and accuracy second. Include correct non-`empty` counts, false-`empty` and wrong-class errors, predicted-`empty` rate, class/location breakdowns, and inference cost. Add an always-`empty` sanity check and, if helpful, the earlier frozen result as secondary context. Note that the trained `val` result is selection-tuned.
- [ ] Inspect matched images and top boxes to separate apparent localization failures from wrong class matching and `empty`-threshold failures. Box counts at a fixed cap are truncated. Do not claim detection mAP: WILDS supplies image labels, not ground-truth boxes.

**Checkpoint:** the trained model has a complete *image-level* validation result, with limitations and failure cases documented.

## 7. Run limited, labeled extensions only if warranted

- [ ] Keep any optional prompt-wording experiment separate from the core trained model; it is not required for the first training run.
- [ ] Candidate later ablations: learned versus validation-threshold `empty` decision, class-weighted loss, smooth/attention pooling in place of max, or selectively unfreezing the semantic head/fusion layers. Change one major component at a time, log parameter/inference cost, and use validation rather than test results to choose among them.
- [ ] Pseudo-box supervision is optional and requires an explicit, checked box-generation method; it is **not** part of the first baseline. Neither the YOLO-World nor YOLIP box-level objective is being claimed as directly reproduced from image-only WILDS labels.

**Checkpoint:** any extension has its own configuration and ablation result, separate from the first trained baseline.

## 8. Freeze artifacts and keep final tests gated

- [ ] Write a compact trained-run manifest with code/library versions, source and trained checkpoint hashes, exact prompt map, loss/aggregation, trainable parameters, seed, split use, selected checkpoint and threshold, validation metrics, and known deviations. Save the notebook, manifest, and small summary plots; keep datasets, full detections, and large checkpoints outside Git.
- [ ] Keep both `id_test` and held-out-camera `test` disabled while model, loss, threshold, or evaluation choices are changing. Once the trained model is fixed, agree on and run a **single final evaluation**; report results without retuning.
- [ ] If the wider course project later studies abstention, evaluate it on unchanged image-level classifiers and report risk–coverage separately from WILDS classification macro-F1.

**End state:** a reproducible, image-label-trained integrated YOLO-CLIP model, documented ablations/failures, and a guarded path to final test reporting. The first notebook task is **Task 1 only**; training starts after the differentiable-interface and recipe gates pass.
