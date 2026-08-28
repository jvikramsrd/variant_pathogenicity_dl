# Project Plan — Lynch Syndrome MMR Multi-Modal Pathogenicity Prediction

Companion to [LITERATURE_NOTES.md](LITERATURE_NOTES.md) — read that first for the "why," this is the "what to do, in order." Every design choice below cites the literature-notes section that justifies it, so you can trace any decision back to its source paper.

**Base papers** (full rationale in [BASE_PAPER.md](BASE_PAPER.md)): **[MVmamba (Zhang et al. 2025, IEEE BIBM)](papers/paywalled_needs_institutional_access/MVmamba_Deciphers_Missense_Variant_Pathogenicity_via_Enhanced_Bi-Mamba_and_Structure-Informed_Protein_Language_Model.pdf)** — primary, architecture for the ESM2 + fusion branches — plus [Tejura et al. 2024](base_papers/D2_Tejura2024_AJHG_GeneHeterogeneity_PMC11393694.pdf) (calibration pillar) and [ClinVar-BERT](base_papers/C1_ClinVarBERT_medRxiv2024.12.31.24319792.pdf) (LLM pillar).
**Chosen LLM approach:** fine-tuned local encoder, ClinVar-BERT-style, no API dependency in the deployed critical path.

---

## Phase 0 — Environment setup
*(Task #1)*

- [ ] Create a venv in `lynch-mmr-pathogenicity/`. Python 3.14 is present but very new for PyTorch wheel support — check `pip index versions torch` first; fall back to the existing `uv`-managed Python 3.13 install if 3.14 wheels aren't published yet.
- [ ] Install: `torch`, `transformers`, `huggingface_hub`, `biopython`, `pandas`, `numpy`, `scikit-learn`, `scipy` (for the bootstrap/binomial calibration math).
- [ ] Confirm GPU availability (`torch.cuda.is_available()`). Every fine-tuning recipe in the literature (ProPath, CSBJ, VariPred, ClinVar-BERT) ran on a single GPU (A100 or smaller) — this project does not need multi-GPU infrastructure.

---

## Phase 1 — Gene-specific dataset construction
*(Task #2, informed by [Literature Notes Part 0](LITERATURE_NOTES.md#part-0--non-negotiable-clinical-facts-read-this-first))*

Build one clean per-variant table for MLH1/MSH2/MSH6/PMS2 — not genome-wide dumps.

- [ ] **UniProt sequences**: P40692 (MLH1, 756aa), P43246 (MSH2, 934aa), P52701 (MSH6, 1360aa), P54278 (PMS2, 862aa).
- [ ] **ClinVar**: pull via NCBI E-utils for just these 4 genes (not the full XML dump). Retain star-rating per record.
- [ ] **InSiGHT/ClinGen VCEP** 3-star calls where available — flag these as highest-confidence tier, weight accordingly in training/eval.
- [ ] **gnomAD v4** allele frequencies for the 4 genes only, for BA1/BS1/PM2 filtering.
- [ ] **Data-quality gate (do this before anything downstream touches PMS2 data):** flag or exclude PMS2 variants in exons 11–15 (PMS2CL pseudogene homology region) unless orthogonally confirmed (long-range PCR/cDNA) — standard NGS calls there are untrustworthy. This is a hard filter, not a soft feature.
- [ ] **Circularity-safe split**: cluster at the sequence level (MMseqs2, 20% coverage/20% identity, per CSBJ's recipe) and additionally prepare a **leave-one-MMR-gene-out** split — the real generalization test given only 4 genes total.
- [ ] Build a **balanced-label diagnostic subset** per gene (VariPred's method) to later verify the fused model isn't shortcut-learning gene identity.

**Open decision to make before this phase:** whether functional-assay data (Phase 2) is fully held out from training or partially used — you flagged this as unresolved last time; resolve it now since it changes how Phase 1's split is built.

---

## Phase 2 — Orthogonal functional-assay data
*(Task #3, informed by [Part 0](LITERATURE_NOTES.md#part-0--non-negotiable-clinical-facts-read-this-first) and [Part 1](LITERATURE_NOTES.md#part-1--esm2--protein-embedding-branch))*

- [ ] **CIMRA assay data** (Rayner/Drost, validated for all 4 genes) — encode as a **numeric OddsPath feature**, not a binary flag, using the exact thresholds in Literature Notes Part 0 point 5. Exclude any variant whose primary hypothesized mechanism is splicing (CIMRA is cell-free, can't detect that).
- [ ] **MSH2 DMS dataset** (Jia et al. 2021, MaveDB, ~94% substitution coverage) — treat as first-class input for MSH2 specifically; per D6, it outperforms every computational predictor for MSH2 classification. No equivalent-scale dataset exists yet for MLH1/MSH6/PMS2 — search MaveDB/Atlas of Variant Effects for anything newer before assuming there's nothing.
- [ ] Keep this data **held out from ESM2/LLM branch training**, used only for validation — this is what makes later performance claims non-circular.

---

## Phase 3 — ESM2 protein-language-model branch
*(Task #4, informed by [Part 1](LITERATURE_NOTES.md#part-1--esm2--protein-embedding-branch), and now anchored on the **primary base paper**, MVmamba — see [BASE_PAPER.md](BASE_PAPER.md))*

1. **Primary recipe — implement MVmamba's feature extraction first**, since it's the base paper and the most complete, ablation-validated recipe available:
   - Generate wild-type (WT) and variant-type (VT) structures (AlphaFold for WT, FoldX for the point-mutation VT structure — or ESM2/ESMC's own structure module if avoiding the FoldX dependency).
   - Encode both through the structural PLM (ESMC, or ESM2 as the closest available substitute) to get per-residue embeddings for WT and VT separately.
   - Extract **global features** (mean-pooled over the full sequence) and **local features** (windowed around the mutation site, optimal window = mutation ±3 residues per MVmamba's own window-size sweep — Table I in the base paper) for both WT and VT — 4 feature vectors per variant.
   - Add **gnomAD allele frequency as an explicit input feature**, not just a filtering criterion — MVmamba's own ablation showed this improves every metric on top of a strong structure+sequence model (AUC 0.895→0.901).
2. **Baseline**: also implement true masked-marginal zero-shot scoring (mask position, read masked-LM softmax) for both **ESM-1b** and **ESM2-650M** as simpler baselines below the MVmamba-style recipe — don't assume ESM2 wins; two independent papers found ESM-1b better for clinical pathogenicity specifically. Compare on our MMR data.
3. **Sequence handling**: MSH6 (1360aa) needs truncation — VariPred's asymmetric-window recipe (1022 from nearest terminus, or 510/511 centered on the mutation). Other 3 genes fit natively.
4. **Fine-tuning strategy comparison** — benchmark MVmamba's recipe against the other three fine-tuning strategies in Part 1 (ProPath's Siamese/PLLR approach, LR 1e-5/batch 8/10 epochs; VariPred's frozen-embedding linear probe; CSBJ's per-feature token-classifier) — all four are independently validated but for different data regimes; don't commit without comparing on our own MMR data.
5. **Split**: leave-one-gene-out AND cluster-level (never variant-level-only, given the circularity risk documented in Part 1).
6. **Metric**: MCC primary, AUC/AUPR secondary (report both — MVmamba's own headline numbers are AUC/AUPR-based, so report on the same scale for direct comparability), bootstrap CIs (10,000 iterations). Tune decision threshold on validation data rather than assuming 0.5.
7. **Sanity-check against published anchors**: MVmamba's own numbers (AUC 0.901, AUPR 0.848, MCC 0.656 against 21 competitors on 18,731 variants) are the primary bar to clear or explain a gap against; PMS2 AUC should also approach EVE's 0.99; MSH2 should be checkable against the named agreement variants (S554N/T, D660G, I774V, E198G, G759E).

---

## Phase 4 — LLM clinical-reasoning branch
*(Task #5, informed by [Part 2](LITERATURE_NOTES.md#part-2--llm-clinical-reasoning-branch-fine-tuned-local-model--our-chosen-approach))*

1. **Corpus**: ClinVar free-text summaries for the 4 genes + PubMed/PMC Lynch syndrome case reports.
2. **Preprocessing (reuse ClinVar-BERT's pipeline near-verbatim)**:
   - MinHash + Jaccard(0.95) fuzzy dedup, grouped by lab × gene — target Ambry/GeneDx/Invitae/Color/Prevention Genetics templating specifically.
   - Train a SentenceClassifier (description / evidence / conclusion) and strip conclusion + boilerplate-description sentences before the main model ever sees them. **This step is not optional** — skipping it is the single most likely way to silently break the LLM branch (Literature Notes Master List #5).
   - Class-balance similarly to C1's 1:2:2 (B/LB : P/LP : VUS), adapted to our actual label distribution.
3. **Base model**: BioBERT-base, LR 2×10⁻⁵, weight decay 0.01, max_length 512. (Note: base-model choice mattered far less than the preprocessing pipeline in C1's own ablation — don't over-invest in searching for a fancier base encoder before the filtering pipeline is solid.)
4. **Ground the output in ACMG evidence codes, not just P/LP/B/LB/VUS**: use the RareDAI self-distillation pattern — offline, use a larger model (local or a one-time API call; doesn't matter, it's training-time only) to generate CoT rationales for labeled training examples, anchored to the [Plazzer/InSiGHT rule spec](papers/B_domain/B1_Plazzer_InSiGHT_ACMG_medRxiv2024.05.13.24307108.pdf) (PS3, PM2, PP3, PS4, PVS1, etc. — remembering PM1/PP2/BP1 are excluded for MMR genes entirely). Fine-tune the deployed model on (text, CoT, label) triples.
5. **Optional bounded literature-mining sub-module** (only if PS3/PS4 case-count/functional-evidence mining from primary literature turns out to be a real bottleneck): a reasoning-tier API model, architected as upstream feature generation only —
   - Default to "no decision" under any variant-identity ambiguity (C5's gate).
   - Never automate evidence-*strength* grading via this path (near-random even for frontier models).
   - Pin model version, re-benchmark on a schedule, treat output as probabilistic prior into the local classifier, never as final arbiter.
6. **Evaluation discipline**: hold out entire genes from training; validate against CIMRA/functional data (Phase 2) as the orthogonal check, exactly as C1 validated against DMS. Spot-check a blinded-expert-adjudicated sample before trusting raw accuracy numbers — ClinVar/InSiGHT labels themselves contain some fraction of stale calls.

---

## Phase 5 — Fusion
*(Task #6, informed by [Part 4](LITERATURE_NOTES.md#part-4--fusion-architecture) and the base paper's GateWave module — [BASE_PAPER.md](BASE_PAPER.md))*

1. **Start architecture (and mean it — this is not a placeholder)**: freeze both branch encoders → small linear/MLP projection per branch to a shared dimension → concatenate → shallow MLP head (BatchNorm → Linear → ReLU → Dropout → Linear → score).
2. **Mandatory baseline comparison**: ESM2-branch-only, LLM-branch-only, and fused — on the same held-out split, every time. Fusion helped in only 4/6 tasks in the most relevant benchmark paper (E3) and actively hurt in one — never assume it wins.
3. **Second fusion option to benchmark against plain concat+MLP — MVmamba's GateWave design**, a validated middle ground between "too simple" and cross-attention/MoE: a sigmoid gate balances two related inputs (in our case, ESM2-branch score vs. LLM-branch score, analogous to their WT-vs-VT gate) → a second softmax-based dynamic gate weights each modality's projected features → Gated Linear Unit → residual connection. Their own ablation (ours to replicate on our own data): removing the gate module cost ~0.001 AUC (small but real, on 18,731 real clinical variants), removing the wavelet-transform component cost more (~0.007 AUC) — quantified evidence for exactly how much complexity is worth adding, on a task adjacent to ours.
4. **Do not add cross-attention or Mixture-of-Experts by default.** Only escalate complexity if the simple-fusion ablation (or the GateWave comparison in step 3) shows clear underfitting, and even then expect a small (~1-2 point) ceiling per TriFit's own ablation, at 100-1000x more data than we'll have.
5. Reuse the **Tavtigian ACMG-AMP Bayesian point formula** (validated by two independent groups, EVE and CSBJ) as an alternative/complementary fusion mechanism to the learned MLP — treating each branch's output as one evidence category — worth comparing against both learned-fusion approaches since it's directly interpretable in clinical terms.

---

## Phase 6 — Gene-specific calibration & evaluation
*(Task #7, informed by [Part 3](LITERATURE_NOTES.md#part-3--calibration--evaluation-methodology-this-is-the-projects-core-evaluation-lens) and superseded/extended by [Part 5 — 2026 update](LITERATURE_NOTES.md#part-5--2026-update-new-papers-datasets-and-methods) — **the small-N calibration problem this phase was built to solve now has a published solution, use it before building anything from scratch**)*

**Revised as of the 2026 literature update ([F1, Chen et al., bioRxiv Feb 2026](papers/F_new_2026/F1_GeneDomainAwareCalibration_bioRxiv2026.02.17.706269.pdf)):** the 3-level hierarchy below used to be this project's own novel contribution to design from scratch. A UW/Mount Sinai group has since published exactly this class of solution — gene-specific calibration where data allows, falling back to a "domain-aggregate" pooling scheme (protein domains with similar score distributions pooled across genes, not just within the MMR family) where it doesn't — validated on 132 genes for REVEL/AlphaMissense/MutPred2, with **MSH2 used as their flagship worked example** (Fig. 3a–g) and MLH1/MSH2/MSH6/PMS2 all appearing in their prior-estimation comparison (Extended Data Fig. 3). This changes Phase 6 from "design a novel hierarchical method" to "check what's already published for our 4 genes, reuse where available, apply their method (open-source) where it isn't."

1. **Check PredictMD first** (https://igvf.mavedb.org/, their live tool) **and the GitHub repo** (https://github.com/yileevechen/VEP_calibration) **for MLH1/MSH2/MSH6/PMS2 before building anything**:
   - If a gene has full gene-specific calibration published (MSH2 almost certainly does — it's their headline example), **download and reuse those REVEL/AlphaMissense/MutPred2 thresholds directly** rather than re-deriving them.
   - If a gene only has domain-aggregate calibration (likely for MSH6/PMS2, which weren't shown as flagship examples), reuse that.
   - Supplementary Tables 1 (gene-specific priors, 648 genes), 2 (evidence assignments), 5 (cluster priors), and 6 (cluster thresholds) are hosted at their Zenodo archive (https://zenodo.org/uploads/18668684) — pull these directly rather than re-scraping ClinVar/gnomAD from scratch for the calibration step.
2. **For our own fused ESM2+LLM score** (which their tool doesn't cover — PredictMD only calibrates existing published VEPs, not a novel model), **apply their exact data-adaptive calibration framework**, now that it's open-source, instead of the from-scratch Pejaver-only approach originally planned:
   - Gene-specific prior via their adapted **DistCurve** algorithm (out-of-bag ensemble on MutPred2 features + PCA + random under-sampling — see their Methods) — more robust than plain DistCurve for small N.
   - Fit **all 11 calibration methods** (Platt, Weighted Platt, Beta, skew-normal/beta/truncated-normal mixtures, isotonic, smoothed isotonic, SplineCalib, MonoPostNN, plus Pejaver's original local-posterior method) per gene, and let their **3-stage miscalibration-aware selection procedure** (their open-source `MonoPostNN`/`PosteriorCalibration` repo: https://github.com/shajain/PosteriorCalibration) pick the best one — don't hand-pick a single method.
   - Their **eligibility tiers are directly reusable as our own go/no-go gate**: ≥50 variants needs 0.4≤pfrac≤0.7; ≥100 needs 0.1≤pfrac≤0.9; ≥300 needs 0.05≤pfrac≤0.99 + ≥10 benign. Apply this per MMR gene before attempting gene-specific calibration of our own score.
   - If a gene fails that gate (plausibly MSH6/PMS2 for our *novel* fused score, even if REVEL/AlphaMissense already clear it thanks to more historical data), **fall back to their domain-aggregation method**: cluster Pfam domains by Jensen-Shannon distance between score distributions, pool control variants within a cluster. This directly replaces the "MMR-family pooling" idea from the original plan with something more principled (domain-shape similarity, not just shared gene family) — and is a fully specified, implementable algorithm now, not a design gap.
3. **Their evaluation methodology is the field's new standard — adopt wholesale**: interval-based LR "win" comparison against expected Tavtigian thresholds (binomial test), TPR/FPR-difference scatterplots, MCC, and — critically — the **ClinGen Evidence Repository non-circular validation** (strip PP3/BP4 from expert panel calls, recompute classification from only non-computational evidence, then check whether recalibrated evidence agrees). This is a stronger non-circularity check than what was originally planned and should replace our own ad hoc "hold out DMS" validation as the primary evidence-accuracy metric (DMS/CIMRA remain valuable as an *additional*, independent axis, not the only one).
4. **Realistic ceiling, updated with real numbers**: their gene-specific calibration reduced VUS by 24% (135→103) in one worked ClinGen-repository test and increased determinate-evidence coverage from a median of 68%→80% across 132 genes — use these as the actual bar our fused model needs to clear to claim an improvement over just using their already-calibrated REVEL/AlphaMissense/MutPred2 thresholds directly.
5. **Baseline comparison, updated**: benchmark our fused model against F1's *already-calibrated* REVEL/AlphaMissense/MutPred2 for MLH1/MSH2/MSH6/PMS2 (not just their raw uncalibrated scores via dbNSFP as originally planned) — this is a strictly harder and more relevant bar, since a genome-wide-calibrated REVEL threshold is a much weaker baseline than F1's own MSH2-specific one.
6. **Independent validation axes** (still use both, neither alone): CAPS metric (D4, pooled across the 4 genes given per-gene sample limits) and DMS/AUBPRC (MSH2 especially, per D6/Livesey & Marsh).
7. **Calibration reporting**: reliability diagrams + ECE per gene (TriFit's Appendix D template), on top of — not instead of — the ACMG evidence-strength calibration.
8. **Set evaluation expectations honestly**: ~50–85% VUS resolution is realistic; near-100% is a bug signal, not a win.

---

## Phase 7 — Write-up
- Frame the contribution explicitly against the two base papers: Tejura et al. for *why* gene-specific calibration matters (with MSH2/MLH1 as literature-documented failure cases of genome-wide calibration), ClinVar-BERT for the LLM-branch architecture we extended.
- Report leave-one-gene-out performance prominently, not just in-distribution — it's the honest generalization number for a 4-gene project.
- Include the binomial power analysis (Phase 6, step 2) transparently — showing *which* genes/tiers could support gene-specific calibration and which had to fall back to family-level pooling is itself a legitimate methodological finding, not just a limitation to bury in a footnote.

---

## Decisions already locked in by the evidence
- LLM branch: fine-tuned local BioBERT-style model (not API) — confirmed correct by C4's drift findings.
- Fusion: simple concatenation + shallow MLP as the starting (and likely final) architecture — confirmed by E3/E4.
- Calibration: gene-specific where data allows, domain-aggregate fallback otherwise — **now an implementable, open-source method (F1)**, not a design problem we have to solve ourselves.
- LLM self-distillation teacher: MedGemma-27B-text-it, self-hosted — avoids both API cost-at-scale and the drift/determinism problems C4 documents.

## Open decisions still needing your input
1. **Functional-assay data (CIMRA/DMS)**: fully held out for validation only, or partially used in training? Affects Phase 1's split design.
2. **ESM-1b vs ESM2-650M backbone**: resolve empirically in Phase 3 step 1 — don't pre-commit.
3. **Whether to build the optional API-based literature-mining sub-module (Phase 4 step 5) at all**, given [AutoPM3](papers/F_new_2026/F3_AutoPM3_LLM_PM3Evidence_bioRxiv2024.10.29.621006.pdf) (Part 5) is now a working, MIT-licensed, open-source implementation of almost exactly this sub-module — the question is now "adopt AutoPM3 as-is" vs. "build our own," not "build from scratch."
4. **New:** now that F1 may have already-published gene-specific calibration for MSH2 (and possibly MLH1), do we want to spend Phase 6 effort trying to *beat* their calibrated REVEL/AlphaMissense/MutPred2 for those genes specifically, or focus novelty entirely on MSH6/PMS2 (where F1 likely only has domain-aggregate, not gene-specific, coverage) and the fused-model contribution itself?

Next concrete action: **Phase 0 environment setup**, unless you want to resolve open decision #1 first since it changes Phase 1.
