# YOLO-CLIP extensions after the first trained baseline

**Status:** Review draft. This records the two directions we discussed; it does not select an extension, change the completed baseline, or update the midterm report.

## Starting point

Our completed image-level baseline starts from pretrained YOLO-World S-v2. It freezes the visual detector and CLIP text encoder, initializes 181 animal-class vectors from WILDS class names, and trains only those vectors using image labels and maximum pre-NMS region scores. At inference, the highest-scoring post-NMS detection supplies one image label, or a validation-selected threshold produces the WILDS empty class. The frozen reference and this trained baseline are already distinct experiments.

We chose this manageable integrated model **for the first baseline**, rather than building a new base-YOLO-plus-CLIP fusion model immediately. That choice did not rule out the more involved architecture as a subsequent investigation. The two tracks below must remain separate: Track A modifies the current YOLO-World adaptation; Track B investigates a more substantial YOLO–CLIP design.

## Track A — smaller extensions to the YOLO-World baseline

These are candidate ablations, **not an agreed bundle of changes**. First inspect validation examples and class/location summaries to separate missed or poor localization, false animal proposals on empty images, wrong-species matching, and threshold errors. Each candidate should then change one major component at a time, with its own configuration and checkpoint.

1. **Learn the empty decision.** Test a trainable empty bias or small image-level calibrator against the current validation-selected threshold. Specify how a parameter learned from pre-NMS training scores connects to the post-NMS inference rule; the two score spaces are not interchangeable. Compare macro-F1 and empty/non-empty errors, not only the fraction predicted empty.
2. **Change region aggregation.** Test smooth or attention-based pooling in place of the hard maximum over candidate regions. This could distribute image-level supervision across several regions, but must not assume every region contains the labeled animal.
3. **Handle class imbalance.** Test a separately specified class-weighted image-level loss against the unweighted baseline, checking per-class effects and the official macro-F1 rather than accuracy alone.
4. **Fine-tune a limited part of the integrated detector.** Consider unfreezing the semantic matching head and/or vision–language fusion layers while initially retaining the pretrained backbone and localization branch. This is a larger change than calibrating empty; record exactly which parameters move and whether it improves species matching without destabilizing detection.
5. **Consider pseudo-box supervision only with an explicit source and audit.** WILDS supplies image labels, not animal boxes. A box-level loss cannot be added merely by treating the image label as a correct label for every proposal. Any pseudo-box method would need a generation rule, quality checks, and a separate ablation.

The first three items extend the image-level decision or loss. The fourth begins adapting internal visual–language matching. The fifth is conditional, not a requirement of this track. None is guaranteed to improve the baseline.

## Track B — a more involved YOLO–CLIP fusion architecture

This is the architectural direction we **deferred until after establishing the baseline**. Instead of only training class vectors inside pretrained YOLO-World, investigate a design built around a base YOLO detector with an explicit, trainable interface to CLIP representations. The aim is for localized YOLO features and species semantics to influence one another more deeply than they do in the frozen-detector baseline, while still producing one WILDS image-level label.

One candidate design is to project YOLO's multi-scale region features into a CLIP-aligned space, compare them with embeddings of the known WILDS species names, and train the projection and selected detector/fusion components using image-level supervision. A further option, informed by our [YOLIP review](lit-review.md), is proposal-level CLIP visual processing or learned interaction between proposal features and CLIP features. That would be materially different from the current text-vector-only adaptation. It is **not yet a chosen architecture**, nor a claim that we will reproduce YOLIP; a simple pipeline that sends YOLO crops to an independent CLIP classifier is also not automatically the intended fusion design.

Before implementing Track B, we must decide and document:

- Which base YOLO and CLIP checkpoints to use, and whether CLIP contributes text features only or also region-level visual features.
- Where the trainable projection/fusion occurs, which YOLO and CLIP components remain frozen, and what additional compute and parameters the design requires.
- How image labels supervise region matching or localization without ground-truth boxes; whether to use image-level multiple-instance learning, carefully checked pseudo-boxes, or another justified objective.
- How to score empty, aggregate multiple proposals into one image prediction, and align training scores with the inference rule.
- Which controlled comparisons isolate the value of the new fusion from changes in backbone, pretraining, resolution, or training budget.

The [point-by-point plan](yolo-clip-final-plan.md) records the decision to use an integrated YOLO-World-like model for the first baseline and names semantic-head/fusion unfreezing and alternative pooling as later options. The [implementation plan](yolo-clip-implementation.md) and [trained-task checklist](yolo-clip-trained-tasks.md) likewise defer these changes. Our earlier discussion of using a main YOLO model with ideas borrowed from YOLO-World is preserved here as a **post-baseline architectural direction**, not as a fully specified or approved training recipe.

## Evaluation and reporting guardrails for either track

- Keep the official WILDS image-level task, 182-class label mapping, familiar/unseen-camera splits, and macro-F1 as the primary metric. Do not report detection mAP without box annotations.
- Choose designs, checkpoints, and thresholds using training and validation data. The completed baseline's final familiar-camera and unseen-camera test results have already been seen; do not use them to tune extensions or present a later retuned score as a fresh, untouched test result.
- Compare each extension with the unchanged first baseline under a clearly stated protocol. Report class and empty errors, parameter/inference cost, and failure cases alongside aggregate scores.
- In the midterm report, describe Track A as candidate incremental ablations and Track B as the deferred architectural investigation. Neither track's effectiveness should be claimed before it is implemented and measured.
