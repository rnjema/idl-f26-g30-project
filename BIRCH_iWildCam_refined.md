# BIRCH Meets iWildCam: A Critique and Refined Research Plan

**Prepared for:** Team Support Vector (11-785 project)
**Date:** 2026-09-28
**Scope:** Validation and refinement of the "BIRCH Models for iWildCam Shift Robustness" brainstorm, reconciled with the team's selective-classification proposal ("Beyond Confidence: Learning to Abstain in Wildlife Classification Under Distribution Shift").

---

## 0. Verdict, up front

The brainstorm is a competent recombination of four established research lines, not a new direction — and it contains three outright technical errors plus two overclaims that would not survive review. But there is a salvageable core that is *more* valuable than the original pitch: a BIRCH Clustering-Feature (CF) tree is best understood not as a training-time robustness mechanism, but as a **streaming, memory-compressed evidence index for the likelihood-ratio selector your team is already building**. That reframing (a) merges the two documents into one coherent project, (b) requires no classifier retraining, (c) has a defensible novelty sliver, and (d) is feasible in one semester. The Group-DRO-over-latent-domains and streaming-TTA lanes are legitimate stretch goals, but both are crowded literatures where you must differentiate from GEORGE, DUA/NOTE, and DynProto respectively.

One framing correction before anything else: iWildCam-WILDS already ships camera-location labels, and its official metric is **macro F1** over 182 classes (with ~35% of images in the "empty" class), not accuracy. Any pitch that frames "domain discovery" as solving an unsolved problem on this dataset will be read as uninformed; the discovery claim has to be about *granularity* (sub-location structure: day/night, viewpoint, micro-habitat) and about what you do with the hierarchy, not about discovery per se.

---

## 1. Error ledger — what in the brainstorm is wrong or overclaimed

**E1. The $O(1)$ claims are false.** Inserting a point into a BIRCH CF-tree requires descending the tree, computing a distance to every child entry at each level — $O(B \cdot h \cdot D)$ for branching factor $B$, height $h$, dimension $D$ — and finding the nearest leaf costs $O(L \cdot D)$ over $L$ leaf entries in the flattened variant the brainstorm actually implements (a linear scan over `max_clusters` centroids). Memory is $O(L \cdot D)$, not constant. What *is* $O(D)$ is the sufficient-statistic update *given* the chosen cluster: $N \leftarrow N+1$, $LS \leftarrow LS + x$, $SS \leftarrow SS + x \odot x$. This distinction matters at test time: the per-sample cost of the streaming lane is real, and on 42,791 OOD-test images it is still trivial compared to one forward pass of ViT-B/16 — but the paper must state the complexity correctly.

**E2. "512–1024 dimensions is low-dimensional, where Euclidean distances become meaningful" is exactly backwards.** Distance concentration *worsens* with dimension. A quick simulation (800 random Gaussian points, relative dispersion std/mean of pairwise distances): $d{=}16$: 0.18 (raw $L_2$) / 0.25 (cosine); $d{=}128$: 0.06 / 0.09; $d{=}768$: **0.024 / 0.036**. At CLIP/DINOv2 embedding dimension, raw Euclidean pairwise distances concentrate to within ~2.4% of their mean — a fixed radius threshold like `radius_threshold=10.0` in the provided PyTorch module is uncalibrated noise, and the classic BIRCH radius/$T$ machinery is the *first* thing that breaks. The brainstorm's own spherical section is the correct repair, which leads to:

**E3. "Spherical BIRCH" is not a new metric — it is $L_2$-normalized Euclidean BIRCH.** On the unit hypersphere, $\|x-y\|^2 = 2 - 2\, x \cdot y$, so squared Euclidean rank-orders identically to cosine distance. The BETULA project's maintainers evaluated a native cosine CF and rejected it for exactly this reason (monotonic equivalence). The useful content of the spherical section is real but should be stated as: *$L_2$-normalize embeddings, keep CF $= (N, LS)$, centroid = mean direction $\mu_k = LS_k/\|LS_k\|$, tightness = mean resultant length $\bar R_k = \|LS_k\|/N_k \in (0,1]$*, which is the vMF concentration proxy ($\hat\kappa \approx \bar R(d-\bar R^2)/(1-\bar R^2)$, Banerjee et al.'s approximation). Also a notation trap the brainstorm missed: $1 - \cos\theta$ is **not a metric** (fails triangle inequality); $\arccos$ is. Thresholds should be tuned in cosine-similarity units, and all angular quantities in radians if you ever take $\arccos$.

**E4. Overclaim: "mathematically sound and highly effective."** Nothing here is effective until measured; Liang et al.'s iWildCam results (below) show that on this dataset even well-motivated margin scores only match plain softmax confidence. Prior probability of a win is modest; the write-up must be framed as a controlled investigation, which — credit where due — your team's proposal already does well.

**E5. Overclaim: "strong submission for NeurIPS D&B / CVPR."** See the novelty audit in §2. The individual mechanisms all exist; the *combination* on camera traps appears open, but combination-novelty at a top venue requires beating strong baselines convincingly. For a course project this is fine and even attractive; for venue ambitions, recalibrate.

**E6. Protocol confusion (minor but consequential).** The "weighted logit fusion (0.35 BBox + 0.35 mirrored + …)" recipe is an iWildCam *Kaggle competition* trick, not the WILDS protocol; WILDS evaluates per-image on full frames with sequence-disjoint splits. MegaDetector crops are allowed (leaderboard flags them) but add a pipeline dependency and a second model's failure modes. Treat as optional.

---

## 2. Novelty audit (verified against literature, 2026-09-28)

| Brainstorm sub-claim | Verdict | Closest prior art you must cite and differentiate from |
|---|---|---|
| Cluster embeddings → latent domains → Group DRO / CORAL | **Done, repeatedly** | GEORGE (Sohoni et al., NeurIPS 2020, arXiv:2011.12945) — cluster per-class embeddings → Group DRO, on WILDS datasets among others; MMLD (AAAI 2020); HRM (ICML 2021, arXiv:2105.03818); EIIL (ICML 2021); ZIN (NeurIPS 2022, arXiv:2203.05818 — important skeptic reference: clustering alone cannot identify invariance) |
| Streaming zero-backprop statistic adaptation (TTA) | **Crowded** | AdaBN; DUA (CVPR 2022); NOTE (NeurIPS 2022); prediction-time BN (Schneider et al., NeurIPS 2020); ADAPT (arXiv:2508.15568) and AdaPGC — the latter updates literally $(N, \sum x, \sum x^2)$ sufficient statistics online, i.e., the CF triplet in everything but name |
| CF-radius threshold for OOD / unseen-domain detection | **Concept done; one direct hit** | Mahalanobis OOD (Lee et al., NeurIPS 2018); kNN-OOD (Sun et al., ICML 2022, arXiv:2204.06507); **DynProto (2026 preprint) already uses BIRCH on cached deep features for online OOD detection** — verify its status before claiming novelty |
| Spherical / cosine BIRCH | **Anticipated + arguably redundant** | monotonic equivalence to $L_2$-normalized Euclidean (BETULA analysis); streaming spherical/vMF mixtures (Banerjee et al., SDM 2007) |
| Mahalanobis BIRCH | **Not found**, but note the CF triplet must be extended to a sum of outer products $SS = \sum x_i x_i^\top$ (the brainstorm does say this; at $D{=}768$ that is a $768{\times}768$ buffer per cluster — $O(KD^2)$ memory, diagonal-only in practice) | — |
| BIRCH/CF on camera-trap data; CF-tree as *abstention evidence index* | **Appears open** | Nearest neighbors on iWildCam: MABN (AAAI 2024), StreamTrap (2026 streaming camera-trap benchmark — a natural evaluation target for the TTA lane) |

*Caveat on the audit: 2026 preprints (DynProto, AdaPGC) are not peer-reviewed; confirm status before leaning on them. Negative results reflect search coverage, not proof of absence.*

**The defensible novelty sliver:** using one hierarchical CF structure as a *unifying memory* that (i) compresses the correct/wrong feature evidence sets for a Neyman–Pearson-style selector, (ii) supports multi-granularity domain summaries, and (iii) flags out-of-support test points — on camera-trap shift. That is a course-project-sized, honest contribution.

---

## 3. Verified benchmark facts (grounding for the experiment design)

- **iWildCam-WILDS:** 203,029 labeled images, 323 camera traps, 182 classes including "empty" (~34–35% of images; severe imbalance; known sequence-level label noise). Official metric **macro F1** (per-class F1 averaged over classes present in the split); average accuracy secondary. Current splits: train 129,809 img / 243 traps; OOD val 14,961 / 32; OOD test 42,791 / 48; ID val 7,314; ID test 8,154 (same traps as train, different days). Bursts are kept within one split; location, sequence, and timestamp metadata are available. (WILDS paper arXiv:2012.07421v3 App. E.1; WILDS 2.0, Sagawa et al. ICLR 2022.)
- **Leaderboard (live, low activity, not archived):** ranked by test-OOD macro F1. Without extra unlabeled data: AutoFT 52.0 (CLIP ViT-L/14@336, Jan 2024, asterisked for pretrained-model deviation), **FLYP 46.0 macro F1 / 76.2 OOD acc (CLIP ViT-L/14@336)**; best default-architecture entry ~33. Latest submission Feb 2025 (COSMO). Two claims floating around — "DIPS" and "Evidential Deep Learning" leaderboard entries — **do not exist on the official leaderboard**; do not cite them.
- **Heng & Soh (arXiv:2505.15008, ICLR 2026):** selector = likelihood ratio $p_c(x)/p_w(x)$ of "classifier correct" vs "wrong" (Neyman–Pearson). **Δ-MDS** fits per-class Gaussians separately on correctly- vs incorrectly-classified training features and scores the difference of Mahalanobis scores; **Δ-KNN** uses the difference of log kNN-distances to the correct vs wrong feature sets ($k \in [25,50]$); prescribed combination $t(x) = s_1(x) + \lambda\, s_2(x)$ with raw-logit margin (RLog). Benchmarks: ImageNet suite + Amazon-WILDS — **iWildCam confirmed absent**, so your reproduction is a genuine replication-gap contribution.
- **Liang et al. (TMLR 2024, arXiv:2405.05160):** raw-logit margin $z_{(1)} - z_{(2)}$ and geometric margin; on iWildCam-FLYP, full-coverage selective risk ≈ 0.29 (ID) / ≈ 0.31 (ID + covariate shift), and their margins are only **on par with softmax scores on iWildCam** — unlike ImageNet. Read: this dataset is a hard, honest testbed where naive confidence tricks don't dominate; headroom exists but wins will be small.

---

## 4. The refined approach — "CF-Δ": CF-trees as compressed evidence for abstention

The key observation connecting your two documents: **Δ-KNN is already a distance-based density-ratio score, and a CF-tree is exactly a streaming, memory-bounded compressor of a feature set.** So instead of "BIRCH for robustness" as a fourth unrelated project, make BIRCH the data structure inside your selector.

**Construction.** On a held-out subset of *training locations* (never seen by the classifier), compute embeddings $z = f_\theta(x)/\|f_\theta(x)\|$ of the frozen FLYP backbone. Partition them by whether the fixed classifier is correct ($\mathcal{A}_c$) or wrong ($\mathcal{A}_w$). Build one spherical CF-tree per set, optionally stratified per predicted class. Each leaf stores $(N_k, LS_k)$; centroids $\mu_k = LS_k/\|LS_k\|$; tightness $\bar R_k$.

**Score.** For a test embedding $z$:

$$s_{\mathrm{CF}\text{-}\Delta}(z) \;=\; \log d_{\cos}\big(z, \mu^{w}_{k^*}\big) \;-\; \log d_{\cos}\big(z, \mu^{c}_{k^*}\big),$$

a log-distance-ratio proxy for $\log p_c(x)/p_w(x)$ — the same estimand as Δ-KNN, but against $O(K)$ centroids instead of $O(|\mathcal{A}|)$ raw points. This gives you three things Δ-KNN does not: (1) **$O(KD)$ query cost** instead of an exact kNN search over all training features (matters on 42,791 test images — minor, but a selling point), (2) **a streaming update path** — new evidence folds into $(N, LS)$ in $O(D)$, enabling the TTA lane without retraining, and (3) **per-leaf tightness $\bar R_k$** as a free second signal: a point near a *tight* wrong-cluster is a more reliable abstain vote than one near a diffuse one.

**Augmented signal set for the learned combiner** (this is your "learned signal combination" experiment, now with CF features): CF-$\Delta$ score; $d_{\cos}$ to nearest correct-centroid; $\bar R$ of the winning clusters; raw-logit margin $z_{(1)}-z_{(2)}$; augmentation-consistency (prediction agreement over $m$ mild transforms, à la Bahat & Shakhnarovich). Combine with logistic regression and a 2-layer MLP, trained on split-B locations only, compared against Heng–Soh's prescribed $s_1 + \lambda s_2$.

**Multi-granularity option (the one place the *hierarchy* earns its keep).** Score the test point against CF-trees cut at two or three depths (leaf level = micro-environments; mid level = biome-like contexts). If coarse-level CF-$\Delta$ and fine-level CF-$\Delta$ disagree, that disagreement is itself a shift signal. This is the piece no flat k-means/kNN baseline has, and it is the honest answer to "why BIRCH and not k-means?"

### Why not the other two lanes as the core?

- **Lane 2 (latent-domain Group DRO / CORAL):** camera IDs are given, GEORGE already did cluster→DRO on WILDS, and clustering *full* embeddings will recover species clusters, not domains — you would need background-only embeddings (MegaDetector-masked or background-pooled), else DRO optimizes worst-species, not worst-domain. Legitimate stretch goal; requires retraining the classifier and a careful answer to ZIN's identifiability critique.
- **Lane 3 (streaming TTA):** transductive regime — not leaderboard-comparable. If pursued, evaluate on StreamTrap-style streaming protocol (time-ordered replay per location using the datetime metadata), report separately, and differentiate from DUA/NOTE/AdaPGC: your delta is *per-micro-cluster* whitening keyed to the CF tree, $\hat z = (z - \mu_k)/\sqrt{\sigma_k^2 + \epsilon}$ with diagonal $\sigma_k^2$ from $(N, LS, SS)$, versus their global or per-class statistics.

---

## 5. Experimental setup

**Fixed components.** Backbone: CLIP ViT-B/16 fine-tuned with FLYP per the official WILDS command (note: the leaderboard FLYP number uses ViT-L/14@336 — decide ViT-B/16 for compute and say so explicitly; your numbers will sit below 46.0 macro F1 and that is fine for a *selector* study, since all comparisons use the same fixed classifier). Freeze after training. All selector work is post-hoc.

**Leakage-free location partition (Fig. 3A).** Within the official train split (243 traps): split A (~80% of locations) trains the classifier; split B (~20%, disjoint locations) trains all selectors/combiners and hosts the CF evidence trees. Official OOD val (32 traps) tunes thresholds, $\lambda$, $K$, radius $T$; OOD test (48 traps) is touched once. No location ID crosses slabs. (Note: splitting the official train set means your classifier trains on fewer locations than leaderboard entries — another reason to frame all claims as *relative to the reproduced baselines under identical conditions*, never as leaderboard entries.)

**Baselines (all on the same frozen classifier):**
1. MSP (max softmax probability).
2. Raw-logit margin (Liang et al. RLconf-M) — cheap and specifically strong on CLIP.
3. Heng–Soh Δ-MDS + RLog and Δ-KNN + RLog, faithfully reproduced (primary baseline).
4. kNN-OOD distance (Sun et al.) as the uncompressed-distance ablation of CF-$\Delta$.
5. Your CF-$\Delta$ variants: single-granularity; multi-granularity; with/without $\bar R$ features; learned combiner (logistic, MLP) vs prescribed $\lambda$-combination; with/without augmentation-consistency features.

**Metrics.** Primary: AURC and selective risk at 80% / 90% coverage on OOD test, matched-coverage comparisons only. Secondary: partial AURC at low coverage (α = 0.1) since deployment cares about the high-confidence end; species- and location-stratified risk/coverage to catch uneven rejection (especially the "empty" class — check whether gains are just "reject empty images", which inflates macro-level improvements without helping animal ID); classifier macro F1 / accuracy for context. Report achieved coverage when thresholds are tuned on OOD val — val→test threshold transfer is itself a finding under shift.

**Ablations and stress tests.** $K$ (number of CF leaves) sweep $\{64, 128, 256, 512\}$; radius $T$ sweep on OOD val; per-class vs global CF trees; CF-$\Delta$ vs raw Δ-KNN at matched query budget (the compression–fidelity tradeoff is the quantitative story: plot AURC against evidence-store memory in MB); augmentation count $m \in \{0, 4, 8\}$ with wall-clock cost; 3 seeds for the learned combiner; location-balanced combiner training (your proposal's extension) retained as an ablation.

**Failure analysis.** Confidently-wrong accepted predictions stratified by location and species; correct-but-rejected (over-abstention); disagreement analysis between coarse- and fine-granularity CF-$\Delta$.

**Compute envelope (estimate, verify against your allocation):** FLYP ViT-B/16 fine-tune ≈ a few GPU-days on one A100-class card; feature extraction over 130k train + 58k val/test images ≈ hours; all selector training is CPU/1-GPU minutes. The entire core lane is feasible inside a semester with slack for the stretch lanes.

**Timeline suggestion:** weeks 1–2 splits + classifier + metrics harness; weeks 3–4 reproduce Δ-MDS/Δ-KNN + RLog baselines and RC curves (midterm); weeks 5–7 CF-$\Delta$ implementation + ablations; weeks 8–9 learned combiner + augmentation consistency + stratified analysis; weeks 10+ stretch lane (choose *one*: latent-domain DRO *or* streaming TTA — not both) and write-up.

---

## 6. Risks, honestly stated

1. **The win may be small or null.** Liang et al. found margins ≈ softmax on iWildCam; if CF-$\Delta$ also lands at parity, the paper becomes "a careful replication + negative result on compression-fidelity," which is still a fine course deliverable — pre-commit to reporting it that way.
2. **The "empty" class can poison everything.** 35% of images; if the selector learns "accept animals, abstain on empty-looking," stratified analysis will show it. Report per-class coverage from day one.
3. **Confounded evidence sets.** "Classifier wrong" correlates with hard locations; split-B must be location-disjoint or the combiner learns location ID, not reliability. (Already in the protocol — enforce it in code with an assertion on location IDs.)
4. **CF-tree order dependence.** BIRCH is insertion-order sensitive; shuffle split-B features before building, and report variance over 3 shuffles. If variance is large, that's a finding about the method, not a bug to hide.
5. **Threshold transfer.** Selectors tuned on 32 OOD-val traps may not transfer to 48 test traps; report achieved coverage vs target coverage.

---

## 7. Cross-course connections (since you're juggling three courses)

- The CF triplet is a **sufficient statistic** — the same trick as the EM E-step's expected sufficient statistics and as Kalman filtering's moment propagation (11-755): $(N, LS, SS)$ is to a Gaussian cluster what $(\hat x, P)$ is to a Kalman state. The streaming TTA lane is literally recursive moment matching; the Kalman/particle filtering unit gives you the vocabulary (and the failure modes: error accumulation under model mismatch = CoTTA's drift problem).
- The Neyman–Pearson selector and the HMM/CTC forward–backward recursions share the same skeleton: likelihood ratios of latent states from streamed evidence. If you frame CF-$\Delta$ as "forward-only sufficient-statistic updates for a correctness hypothesis test," the 11-755 audience will immediately see the structure.
- RPP: coordinate-frame discipline has a direct analog here — *state your feature-space convention every time* (normalized vs raw embeddings, cosine vs angular vs squared-Euclidean "distance", and radians if you take $\arccos$). Half the errors in §1 are convention errors.

---

## Figures

**Figure 1** — Three-lane architecture (core abstention lane; two dashed stretch lanes), with the protocol warning separating leaderboard-comparable from transductive regimes.

![Figure 1: refined architecture](figs/fig1_architecture.png)

**Figure 2** — (A) hierarchical CF-tree over background embeddings and where Group DRO should operate; (B) spherical CF geometry: mean direction, accept cone, tightness $\bar R$, OOD flag.

![Figure 2: CF-tree hierarchy and spherical geometry](figs/fig2_cftree_spherical.png)

**Figure 3** — (A) leakage-free location-partitioned protocol with hard rules; (B) hypothesized risk–coverage outcome (explicitly illustrative, not measured data).

![Figure 3: protocol and hypothesized risk-coverage](figs/fig3_protocol_riskcoverage.png)
