# Regime-Conditioned MoE-ViT for iWildCam: Validation and Refined Design

**Prepared for:** Team Support Vector (11-785)
**Date:** 2026-09-28
**Scope:** Validation of the hypothesis — *MoE Vision Transformer with routers conditioned on distributional regimes discovered by a BIRCH-inspired module* — as an extension of the CF-$\Delta$ / abstention line.

---

## 0. Verdict, up front

The idea is worth building, but the two anchor papers don't say what you might think they say — read them as **architectural templates, not shift-robustness precedents**:

- **arXiv:2601.12483 = QuantumSMoE** — a MoE-ViT for *quantum surface-code decoding*. Its transferable content: MoE layer design inside a ViT and a novel load-balancing auxiliary loss. Nothing about distribution shift.
- **arXiv:2210.14793 = M³ViT** — *task*-conditioned sparse expert routing for multi-task ViTs with model–accelerator co-design. Its transferable content: conditioning routing on a discrete task variable and activating only the task-corresponding expert path at inference.

Your contribution is the substitution: **task ID → distributional regime, where the regime is not given but discovered online by a CF-tree**. That substitution is exactly where the novelty and the danger both live. The danger, in one sentence: M³ViT's task IDs are *known and permanent*, while your regimes are *inferred and drifting* — every failure mode below follows from that difference. And one more calibration before you build: **Meta-DMoE (NeurIPS 2022) already ran domain-expert mixtures on iWildCam**, and the literature contains direct evidence that domain experts *fail* on unseen intermediate regimes (§5.2) — so the unseen-regime fallback is not a garnish on this design, it is the contribution.

---

## 1. Steel-manning the idea

The strongest version of your hypothesis is not "experts memorize regimes" (that overfits to training cameras, the opposite of robustness). It is this:

> **A MoE-ViT decomposes the prediction problem into regime-local subproblems, and the CF-tree provides (a) the regime variable at training time, (b) soft responsibility weights at test time, and (c) an explicit detection mechanism for regimes with no trained expert.**

Three properties make this genuinely attractive on iWildCam specifically:

1. **Label shift is real across camera locations** — species frequencies differ drastically by site. Per-regime experts can carry per-regime label priors $\pi_k(y)$, and a prior-corrected logit $\tilde\ell_y = \ell_y - \log \pi_k(y)$ is a principled, cheap way to exploit regime information that a single global model cannot express.
2. **Soft routing by vMF responsibilities** $r_k(z) \propto \exp(\hat\kappa_k\, z \cdot \mu_k)$ requires *no router training at all* — the CF sufficient statistics define the gate analytically. This is the maverick path (Option B below), and it has a beautiful property: **the gate is exactly the original Jacobs–Jordan hierarchical mixture of experts with a nonparametric, streamable gating network**.
3. **Unseen regimes become detectable structure rather than silent failure**: $\max_k r_k(z) < \tau$ is an explicit "no expert owns this input" flag, which feeds directly into your abstention pipeline (gating entropy and max responsibility are free reliability signals for CF-$\Delta$).

---

## 2. Two router designs (safe vs maverick)

**Option A — distilled router (safe).** Build the CF-tree over train-location embeddings (background-masked, see §3); distill the regime assignments into an MLP router $g_\psi(z)$ trained with cross-entropy on cluster IDs plus a Switch-style load-balancing loss (this is where QuantumSMoE's aux-loss machinery is directly relevant). At inference, top-$k$ route to regime-specialized experts. Pros: differentiable end-to-end, standard training. Cons: the router can only express regimes it was trained on; unseen test locations get *forced* into the nearest trained regime — you inherit exactly the failure mode you're trying to solve unless you add the Option-B fallback.

**Option B — responsibilities as the gate (maverick).** Skip the router network. The CF-tree itself emits $r_k(z)$; experts are mixed $y = \sum_k r_k(z)\, E_k(x)$; a new test-time regime simply grows a new CF leaf, and if $\max_k r_k < \tau$ the system falls back to a uniform expert mixture, a generic expert, or abstention. Pros: zero-shot regime extension, streaming updates in $O(D)$ per leaf, principled probabilistic semantics. Cons: not end-to-end differentiable through the gate (experts train on weighted samples — weighted ERM, which is fine); the vMF assumption on gate shape is a modeling bet.

**Hybrid (recommended).** Option A for the trained-regime regime, Option B as the detection/fallback layer. The CF-tree persists at inference as the "do we even own this input?" arbiter.

### Where the experts live (decides feasibility)

| Variant | Cost | Notes |
|---|---|---|
| Per-regime linear heads on frozen backbone | ~hours | Tier 0 — do this first; it's GEORGE-with-soft-mixing and gives you the cleanest causal test of "does regime conditioning help at all" |
| Per-regime LoRA adapters + frozen trunk | ~1 GPU-day | Tier 1 — the realistic semester target |
| Full MoE-FFN inside ViT blocks | multi-GPU-weeks | Tier 2 — only if Tier 0/1 show signal; this is where M³ViT/QuantumSMoE layer designs matter |

---

## 3. Failure modes (ranked by lethality)

1. **Regime–species confounding.** Cluster full-frame embeddings and your "regimes" are species clusters; experts become per-species models and routing error becomes classification error. Mitigation: cluster **MegaDetector-background-masked or background-pooled** embeddings only. This is the single most important implementation decision.
2. **Routing collapse / expert starvation.** Standard MoE disease; load-balancing aux loss (QuantumSMoE's contribution, Switch-style) is mandatory in Option A, and in Option B you get balance for free from the data distribution but lose differentiability.
3. **Forced routing under shift.** Hard top-1 routing on an unseen location assigns it confidently to a wrong expert. Soft responsibility mixing + the $\max_k r_k$ fallback is the fix; measure routing-error-vs-regime-distance explicitly.
4. **Order sensitivity and drift of the CF-tree.** Same caveat as before: shuffle training inserts, report variance over 3 orders; in streaming mode, monitor leaf growth rate as a drift indicator.
5. **Metric mismatch.** MoE wins typically show in accuracy; the leaderboard is macro F1. With 35% "empty" images, per-regime prior correction can *hurt* macro F1 while helping accuracy — report both, and stratify per location.

---

## 4. The controls that make or break the claim

The entire scientific content of the project reduces to one comparison ladder, all on the same frozen backbone and same splits as the CF-$\Delta$ work:

1. **Single-head ERM** (no regime use) — floor.
2. **Per-camera-ID conditioning (oracle regime)** — ceiling for what regime information is worth. If oracle conditioning doesn't beat ERM, stop: regime information is not the bottleneck, and no clustering method will save the idea.
3. **k-means regimes, flat** — isolates the value of *discovery* over oracle IDs and the value of *BIRCH hierarchy* over flat clustering.
4. **CF-regime MoE (yours)** — Options A/B/hybrid.
5. **CF-regime MoE + label-prior correction** — isolates the prior-correction mechanism.

Metrics: OOD macro F1, worst-location macro F1, per-location breakdown; AURC if the abstention hook is enabled. Splits and leakage rules identical to the CF-$\Delta$ protocol (Fig. 3A of the companion document): regimes discovered on train locations only, selector/router tuning on OOD val, OOD test touched once.

---

## 5. Prior-art audit (verified 2026-09-28; two independent scouts)

### 5.1 The idea family is inhabited — know your neighbors

| Neighbor | Venue | What it does | Distance from your proposal |
|---|---|---|---|
| **Meta-DMoE** (arXiv:2210.03885) | NeurIPS 2022 | Domain-expert teachers + meta-learned transformer aggregator distilled to a student; evaluated on **iWildCam**, FMoW, Camelyon17, RxRx1 — SOTA on 4/5 WILDS image testbeds | **On your benchmark already.** You must cite and ideally compare. Reports a documented failure: without masking, the aggregator collapses into a domain classifier |
| **HMOE** (arXiv:2211.08253) | AAAI 2023 | Domain-label-free MoE that *self-discovers* domain clusters via embedding routing; hypernetwork generates expert weights | Closest single match to "experts specialize on discovered regimes"; clusters are emergent, not a streaming CF structure |
| **BECoTTA / MoDE** (arXiv:2402.08712) | ICML 2024 | Domain-adaptive routing of LoRA experts in ViT blocks for continual TTA; domain discriminator supplies pseudo-domain labels at test time | Closest to "router conditioned on inferred regime"; CTTA setting, adapter-scale experts |
| **GMoE** (arXiv:2206.04046) | ICLR 2023 Oral | Sparse MoE-ViT trained with plain ERM beats DG algorithms on DomainBed | Evidence MoE capacity alone helps DG — your baseline to beat |
| **MESSI** (arXiv:2606.25665) | 2026 preprint, *unreviewed* | Router induces latent domain subsets; per-expert alignment; reports +6.48 macro-F1 over GMoE on iWildCam | Potentially scooping; verify claims independently before citing |
| **c-BTM** (arXiv:2303.14177) | TACL 2023 | Balanced k-means discovers domains (NLP), one expert per cluster, cluster-distance gating, forest grows over time | Your proposal ≈ "c-BTM inside a single ViT with a probabilistic streaming gate" |
| **MoCE** (arXiv:2509.10513) | EMNLP 2025 | k-means cluster ID → expert-group routing (two-stage) | "Cluster assignment conditions expert selection," hard routing, text domain |
| **MMLD** (Matsuura & Harada) | AAAI 2020 | Cluster into latent domains → mixture of domain predictors with test-time posterior weighting | The original cluster-then-route in vision |

### 5.2 The skeptical results you must design against

- **Vision routers learn surprisingly little.** Liu, Puigcerver & Blondel (arXiv:2401.15969): learned routers in V-MoE barely beat fixed/hash routing. Your regime-conditioned gate must *demonstrably* beat a fixed regime→expert assignment and a hash baseline, or the routing mechanism is decoration.
- **Domain experts don't extrapolate.** A model-level domain MoE for detection (arXiv:2604.18256, 2026 preprint) found day/night experts degrade on unseen dawn/dusk subsets, with dense baselines more OOD-robust. This is the direct counterevidence to "regime experts generalize" — your unseen-regime fallback is not optional, it is the whole game.
- **Aggregator collapse.** Meta-DMoE documents the aggregator degenerating into a domain classifier without masking; analogous collapse modes exist for router-as-regime-classifier (Option A's router literally *is* a regime classifier — watch that its accuracy isn't bought at the expense of expert specialization).
- **Inferred environments can split on the wrong features.** EIIL/CnC lineage: environment inference is fragile and reference-model-dependent; clusters ≠ causal regimes. This is why §3's background-masking decision is load-bearing.
- **Posterior miscalibration under shift.** GMM/vMF responsibilities drift over-confident or uniform on far-OOD inputs; nothing in the literature validates responsibility entropy as a regime-novelty detector — treat it as a hypothesis to test, not a mechanism to assume.

### 5.3 What appears genuinely unclaimed

No direct occupant found for the full chain: **(i) streaming CF-tree regime discovery → (ii) soft vMF posterior gating at the image level in a ViT → (iii) per-regime experts with label-prior correction → (iv) explicit unseen-regime detection with mixture fallback or expert spawning → (v) gate statistics feeding an abstention selector.** Each ingredient exists separately (c-BTM, MoCE, DEMix, MoExtend/Lifelong-MoE, Moirai-MoE's centroid-guided gating); the composition — and specifically the *streaming sufficient-statistic memory* as the shared structure — appears open. Also apparently open: cluster→router distillation as an explicit mechanism (nearest relative: Moirai-MoE initializing gates from centroids, arXiv:2410.10469).

### 5.4 Maverick room — evidence the risky parts can work

- **Unseen-regime mixing beats hard routing**: DEMix (TACL 2022) showed parameter-free expert ensembling beats single-expert selection on unseen domains; MOPSA (arXiv:2505.24224) showed cluster-experts generalize to unseen speakers. Your Option B fallback is empirically grounded, not speculative.
- **Expert growth under streams exists**: Lifelong-MoE (ICML 2023) and MoExtend (ACL 2024) add experts online, initializing from the most-shifted existing expert — your "new CF leaf spawns/clones an expert" idea has working NLP precedents to transplant.
- **Retrieval-fallback routing exists**: kNN-MoE (arXiv:2601.02144) mixes a memory of past expert assignments with the parametric router, falling back when retrieval confidence is low — the closest analog to your "CF memory arbiters the gate" design.

*Uncertainty flags from the scouts, preserved: MESSI and other 2026 preprints are unreviewed; one candidate (DRMoET) had a date-inconsistent arXiv ID and was excluded; GEORGE's arXiv ID and BTM's venue should be double-checked before the final write-up.*

---

## 6. How this composes with the abstention project

The two threads unify into one system: **CF-tree as the shared regime/evidence memory**; MoE experts consume it as a gate; the abstention selector consumes it as features ($\max_k r_k$, gating entropy, CF-$\Delta$ score, $\bar R$). Suggested framing for the write-up: *"One streaming sufficient-statistic memory, three consumers: routing, prior correction, and abstention."* That is a coherent, semester-sized thesis with a defensible novelty sliver in each consumer and a clean shared evaluation.

## Figure

![Figure 4: Regime-MoE architecture](figs/fig4_regime_moe.png)
