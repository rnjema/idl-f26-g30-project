# Credal Regimes: Imprecise-Probabilistic Soft Mixtures for iWildCam Shift Robustness

**Prepared for:** Team Support Vector (11-785) · **Date:** 2026-09-28
**Extends:** CF-$\Delta$ (companion doc 1) and Regime-MoE (companion doc 2)

---

## 0. Verdict, up front

This is the strongest version of your idea so far, and for a non-obvious reason: **imprecision is the correct mathematical object for "regimes that are inferred and drifting."** The previous iteration's weak point was that soft responsibilities $r_k(z)$ are *point estimates* — they pretend to know the regime mixture exactly even when a leaf has 25 samples. Credal sets fix this at the foundation rather than with a heuristic confidence correction. The proposal also lands on a beautiful freebie: an unseen regime ($N_k = 0$) yields the *vacuous* credal set — total ignorance — so abstention becomes a theorem of the representation rather than a bolted-on threshold.

Two cautions before the romance: (i) imprecise-probability methods are chronically accused of **conservativeness** — wide intervals everywhere, abstaining on everything — and you must show empirically that your credal widths are *informative* (correlate with error), not just wide; (ii) this stacks two uncertain estimates (gate imprecision + prior imprecision), and how you compose them (independence vs. dependence assumptions) is a real modeling decision that reviewers will poke.

---

## 1. The construction, layer by layer

Everything inherits the CF-tree substrate: frozen backbone, $z = f_\theta(x)/\|f_\theta(x)\|_2 \in \mathbb{S}^{767}$, spherical CF leaves $(N_k, LS_k)$, $\mu_k = LS_k/\|LS_k\|$, $\bar R_k = \|LS_k\|/N_k$, $\hat\kappa_k \approx \bar R_k(d - \bar R_k^2)/(1-\bar R_k^2)$.

**Layer 1 — imprecise gate.** $\hat\kappa_k$ is a point estimate from $N_k$ samples; treat concentration as interval-valued, $\kappa_k \in [\kappa_k^{lo}, \kappa_k^{hi}]$ (asymptotic CI or bootstrap over CF sub-clusters). The responsibility vector

$$r_k(z;\, \kappa) \;=\; \frac{\exp(\kappa_k\, z^\top \mu_k)}{\sum_j \exp(\kappa_j\, z^\top \mu_j)}$$

is then *set-valued*: $\mathcal{R}(z) = \{ r(z;\kappa) : \kappa \in \prod_k [\kappa_k^{lo},\kappa_k^{hi}] \}$ — a credal set over the regime simplex. Small leaves produce wide gates; that's honesty, not noise.

**Layer 2 — credal label priors per regime.** Regime $k$ has class counts $n_{k,y}$ (from split-B data). Replace the point prior $\pi_k(y)$ with the **Imprecise Dirichlet Model** (IDM) of strength $s$:

$$\Pi_k = \Big\{ p \;:\; p_y = \tfrac{n_{k,y} + s\, t_y}{N_k + s},\; t \in \Delta^{C-1} \Big\}, \qquad P_{lo}(y) = \tfrac{n_{k,y}}{N_k+s},\quad P_{up}(y) = \tfrac{n_{k,y}+s}{N_k+s}.$$

**Layer 3 — credal-mixture decision.** With expert logits $\ell_k(x)$, prior correction, and regime weights, the predictive class score for any $(r, p_1..p_K)$ in the joint credal set is $\sum_k r_k\,[\ell_{k,y}(x) - \log p_k(y)]$. Decide by **$\Gamma$-maximin** (maximize the lower expectation) when a unique winner exists; when **interval dominance fails** — no class's lower bound beats every rival's upper bound — emit the undominated **set-valued prediction** or abstain.

**Layer 4 — abstention as geometry.** Three abstention signals, all read off the credal structure, no extra machinery: (i) gate credal diameter at $z$; (ii) predictive interval width of the winning class; (iii) vacuity detection — a new CF leaf with $N_k \approx 0$ makes $\Pi_k$ the whole simplex, so the mixture's credal set inflates automatically. These feed straight into the CF-$\Delta$ selector as features.

**Layer 5 (theory tie-in) — Huber-robust CF-$\Delta$.** Heng–Soh's NP-optimality assumes the correct/wrong densities $p_c, p_w$ are *known*; under shift they're estimated from train locations. Wrap each in an $\varepsilon$-contamination neighborhood (itself a credal set) and Huber's **clipped likelihood-ratio test** is minimax-optimal for the composite problem. Your selector thus upgrades from "NP-optimal if the densities are right" to "minimax-optimal against density misspecification of size $\varepsilon$" — a defensible theoretical claim that directly answers the distribution-shift setting.

## 2. Why this is more than decoration

- **Unseen regime = vacuous prior.** $N_k{=}0$ in the IDM gives $P_{lo}(y)=0, P_{up}(y)=1$ for every class — the model *cannot* be confident. Compare: precise soft-gating assigns the input to the nearest regime and pretends it's fine.
- **Streaming shrinkage is interpretable.** As the CF-tree absorbs test-stream evidence, $N_k \uparrow$, intervals shrink, abstention rate decays. The plot "abstention rate vs. stream length per location" is a learning curve of epistemic uncertainty — a result figure no precise-mixture baseline can produce.
- **Conservativeness is tunable and testable.** IDM strength $s$ and contamination $\varepsilon$ are explicit knobs with sensitivity analysis as standard practice (report results over $s \in \{1,2,4,8\}$, $\varepsilon \in \{0, 0.05, 0.1\}$).

## 3. Pitfalls (ranked)

1. **Vacuous collapse.** If $s$ is large relative to typical $N_k$, everything is indeterminate and coverage craters. The $s$-sweep + coverage–risk curves are mandatory, not optional.
2. **Composing two credal sets.** Gate imprecision (Layer 1) and prior imprecision (Layer 2) — composing under independence is standard but strong; sensitivity to the dependence assumption should be probed (at minimum: natural extension vs. strong independence variants, or just Layer 2 alone vs both).
3. **Computational honesty.** $\Gamma$-maximin over $\prod_k[\kappa_k^{lo},\kappa_k^{hi}] \times \prod_k \Pi_k$ looks nasty but factorizes nicely: the objective is linear in $r$ and each $p_k$ separately, so lower bounds are attained at vertices — computable in closed form per class. Work this out in the writeup; don't hand-wave it.
4. **Metric mismatch persists.** Set-valued predictions don't map to macro F1 directly; report set-size vs. error-coverage tradeoff (standard in conformal/credal evaluation) alongside forced-point-prediction macro F1.

## 4. Experiments (delta over the Regime-MoE plan)

Same splits/leakage rules as before. New comparisons, all on the same frozen backbone:

1. **Precise soft mixture (Option B) vs credal mixture (this)** — OOD macro F1, worst-location F1, and risk at matched coverage.
2. **Credal diameter as an error detector** — AUROC/AURC of (gate diameter, predictive width, vacuity flag) vs. MSP and vs. CF-$\Delta$ score alone; then as *features* added to the selector.
3. **Set-valued evaluation** — average set size vs. empirical error coverage, plus the credal-classifier metrics **determinacy / single accuracy / set accuracy / utility-discounted accuracy** (NCC2 lineage); compare against split-conformal prediction at matched coverage (conformal is the precise-probability incumbent you must beat or complement — but note standard split conformal loses validity under shift, so report OOD coverage, not just set size).
4. **Streaming epistemic decay** — time-ordered replay per OOD location; plot credal width and abstention rate vs. images seen; compare against TTA baselines' error accumulation.
5. **Robust selector** — Huber-clipped CF-$\Delta$ ($\varepsilon$ sweep) vs. plain CF-$\Delta$ vs. Heng–Soh reproduction, AURC and risk@80/90%.

## 5. Prior-art audit (verified 2026-09-28; two independent scouts)

### 5.1 The headline: this quadrant is emptier than the MoE one

Two independent searches across multiple query formulations found **zero credal/imprecise-probability work on any WILDS benchmark** (iWildCam included), **no work combining credal sets with MoE gating**, and **no credal sets derived from streaming/online clustering of foundation-model embeddings** (the vMF-embedding OOD literature — CIDER, PALM, MoLAR — is entirely precise-probabilistic). Evidential deep learning has also never been run on WILDS, as far as we could verify. This is a genuinely open quadrant — rarer than the previous two ideas.

### 5.2 Theory anchors (all verified)

- **Walley (1991), *Statistical Reasoning with Imprecise Probabilities*** — credal sets as closed convex sets of distributions; coherent lower/upper previsions.
- **IDM: Walley (1996, JRSS-B 58(1):3–57, with discussion); Bernard (2005, IJAR 39:123–150)** — the formulas in §1 are the canonical ones: $P_{lo}(y) = n_y/(N+s)$, $P_{up}(y) = (n_y+s)/(N+s)$; vacuous at $N=0$.
- **Huber (1965, Ann. Math. Stat. 36:1753–1758) and Huber–Strassen (1973, Ann. Stat. 1:251–263)** — clipped likelihood-ratio test is minimax under $\varepsilon$-contamination; the 1973 paper is literally a **Neyman–Pearson lemma for capacities** (coherent upper probabilities), which upgrades your Layer-5 claim from analogy to theorem-shaped.
- **Banerjee, Dhillon, Ghosh & Sra (2005, JMLR 6:1345–1382)** — vMF mixture EM; concentration estimator $\hat\kappa = (\bar R d - \bar R^3)/(1-\bar R^2)$ (their Eq. 4.4), with Tanabe et al. (2007) bounds and Sra (2012) Newton refinement. (Matches the estimator used in the CF-layer; cite the four authors — some citation trails erroneously add a fifth.)
- **Troffaes (2007, IJAR 45:17–29)** — $\Gamma$-maximin, interval dominance, maximality; the decision-rule menu for §1 Layer 3.
- **Credal filtering:** Benavoli, Zaffalon & Miranda (2011, IEEE TAC 56:1567–1581) — Bayesian filtering over credal sets with an $\varepsilon$-contamination specialization outperforming KF under model error; Schweppe (1968) / Bertsekas–Rhodes (1971) — set-membership estimation. *Naming caveat: no canonical "credal Kalman filter" paper exists; use these anchors and say "credal-set filtering."*
- **DRO = credal:** Rahimian & Mehrotra (2022, OJMO 3:1–85) — ambiguity sets are credal sets; Group DRO's worst-group expectation is an upper prevision. This formally unifies your Lane-2 (Group DRO) with the credal framing.

### 5.3 Closest modern neighbors

1. **Structured Credal Learning (arXiv:2603.14070, 2026 preprint, unreviewed)** — constructs credal sets from covariate shift + label disagreement, bounds TV diameter, reduces robust learning over them to Group-DRO-style min-max. Closest in spirit; does *not* derive credal sets from latent embedding regimes.
2. **Credal Deep Ensembles (NeurIPS 2024) / Credal Wrapper (ICLR 2025, arXiv:2405.15047)** — per-class lower/upper probability bounds from ensembles, DRO-flavored training, strong OOD detection on CIFAR/ImageNet-O. Methodologically closest on the vision side; credal sets come from *model* ensembles, not *data* regimes.
3. **IDM-coupled imprecise mixtures (IJAR 2022 imprecise HMM/Gaussian-mixture classification; Caprio & Mukherjee, arXiv:2002.08409, finite admixtures via Choquet theory)** — the "mixture + imprecision" axis exists classically, but non-deep, non-streaming, non-embedding.
4. **The credal classifier lineage — NCC (Zaffalon 2002) and NCC2 (Corani & Zaffalon, JMLR 2008)** — is the direct precedent for set-valued prediction-as-abstention, and gives you ready-made evaluation metrics: **determinacy, single accuracy, set accuracy, utility-discounted accuracy** (Zaffalon, Corani & Mauá, IJAR 2012). Adopt these; don't reinvent.

### 5.4 Documented pitfalls of imprecise-prob ML (design against these)

- **Credal-set size is a contested uncertainty measure.** Sale, Caprio & Hüllermeier (UAI 2023) show volume/size measures fail desiderata; Hüllermeier, Destercke & Shaker (UAI 2022) find no consensus. Mitigation: report diameter-based signals alongside distance-based second-order measures (Sale et al., ICML 2024, arXiv:2312.00995) and validate by rank-correlation with error, not by assumption.
- **$\Gamma$-maximin is known to be over-conservative** (Seidenfeld's critique); offer E-admissibility/maximality variants (Troffaes 2007) as ablations.
- **$\varepsilon$-contamination can produce vacuous worst-case risk** — the motivation for bulk-calibrated credal ambiguity sets (Chen, Berrett, Damoulas & Caprio, ICML 2026 Spotlight, arXiv:2601.21324). Keep $\varepsilon$ small and sweep it.
- **Evidential DL is theoretically shaky** — non-identifiability of epistemic components (Bengs et al., NeurIPS 2022; Jürgens et al., ICML 2024); EDL ≈ energy-based OOD detector ("Mirage," NeurIPS 2024). This is an argument *for* your credal (set-based, rather than distributional-second-order) formulation — use it in the related-work section.
- **Conformal-credal hybrids inherit exchangeability assumptions** that break under shift (Kasa & Taylor, arXiv:2307.01088) — your split-conformal baseline (experiment 3) must be shift-aware or it will silently lose coverage on OOD test.

*Flags preserved: Structured Credal Learning and other 2026 preprints are unreviewed; one arXiv ID conflict (2512.05526) noted; Credal Deep Ensembles' arXiv preprint ID unconfirmed (venue confirmed via proceedings).*

## 6. Cross-course connections

- **11-755:** the IDM is Bayesian inference with a set of Dirichlet priors instead of one — you've done the precise version as conjugate updates; credal-set filtering (Benavoli–Zaffalon–Miranda 2011 — the imprecise analog of your Kalman unit; note there is no paper literally named "credal Kalman filter," cite this one) and Schweppe's set-membership estimation are the exact analogs for state estimation, and imprecise EM exists for the mixture fitting side.
- **11-785:** Huber's robust NP test turns your Heng–Soh baseline from a point-estimate selector into a minimax selector — same lemma, contamination neighborhoods around the densities.
- **RPP:** set-membership estimation is how robotics handles bounded-error sensors; your credal sets are the classification analog of guaranteed bounds rather than Gaussian overconfidence.

## Figure

![Figure 5: credal sets and decision geometry](figs/fig5_credal.png)
