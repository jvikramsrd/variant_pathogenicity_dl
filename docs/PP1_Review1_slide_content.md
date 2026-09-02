# PP1 Review-1 — Slide Content Pack

Companion to **`~/Downloads/Batch_No_PP1_Review1_16-08-2026_FILLED.pptx`**
(27 slides). Every slide's text is reproduced here so you can rebuild or edit
any slide by hand. Rebuild from scratch with `~/Downloads/build_review1_deck.py`
(`pip install python-pptx` first).

**Sources used:** `~/Downloads/lynch-mmr-pathogenicity/` — `PROJECT_PLAN.md`,
`LITERATURE_NOTES.md`, `BASE_PAPER.md`, `papers/MANIFEST.md` (verified
citations) — plus the `variant_pathogenicity_dl` repo docs (`RUNLOG.md`,
`PAPER_DRAFT.md`) for the progress section.

- **Framing:** proposal + progress. Review-1 = problem / base paper /
  objectives / literature / proposed design / plan, plus a "built & measured
  so far" section (Phases 0–3 of the plan).
- **Placeholders you fill in:** presenter names + roll numbers, guide name +
  designation, batch number, Literature Survey Report link.
- Talk time ≈ 18–22 min.

---

## 1 — Title
**Multi-Modal Deep Learning for Missense Variant Pathogenicity Prediction in
Lynch Syndrome Mismatch-Repair Genes**
- Target genes: MLH1 · MSH2 · MSH6 · PMS2 — Batch No: `__`
- Under the Guidance of: `Supervisor Name`, `Designation`, IT, CBIT
- Presented by: `Name, Roll No` ×3

## 2 — Agenda
Introduction / Problem / Motivation · Base Paper(s) · Objectives & Plan ·
Literature Review (PLM · LLM · calibration · fusion) · Proposed Architecture &
Methodology · Datasets & Technology Stack · Progress & Preliminary Results ·
Challenges / Summary / Future Work · References

## 3 — Abstract
- Clinical genetics must label every missense variant pathogenic, benign, or a
  **VUS**; the VUS backlog is the largest bottleneck in returning results.
- In Lynch syndrome, a VUS in an MMR gene (MLH1/MSH2/MSH6/PMS2) leaves a family
  without cancer-surveillance guidance.
- We propose a multi-modal classifier: an **ESM-2 PLM branch** (WT/variant
  embeddings + PLLR, MVmamba-style global+local features) fused with an **LLM
  clinical-reasoning branch** (ClinVar-BERT-style, grounded in ACMG evidence
  codes), then **calibrated per gene**.
- Core lens: gene-specific calibration — genome-wide predictor thresholds are
  demonstrably wrong for MSH2 and MLH1 (Tejura et al., 2024) — on a
  leakage-audited pipeline (Grimm et al., 2015).
- Evaluation: leave-one-MMR-gene-out with held-out functional-assay validation.
  Realistic VUS-resolution ceiling ~50–85%.

## 4 — Problem Statement & Motivation
**Problem** — missense variant → ACMG/AMP call; most stay VUS · per-gene
clinical labels scarce, abundant labels are proxies · genome-wide calibration
hides per-gene heterogeneity (MSH2, MLH1 named) · overlapping public data →
circular evaluation (Grimm 2015).
**Motivation** — Lynch syndrome common; early classification changes
surveillance · the four genes are not interchangeable (penetrance, missense
burden, gene-specific ACMG rules) → one shared threshold is systematically
wrong.

## 5 — Base Paper(s)  *(table + notes)*

| Role | Paper | Venue | What we take |
|---|---|---|---|
| **Primary** | MVmamba — Zhang et al. (2025) | IEEE BIBM | ESM-2 + fusion architecture: WT/VT global+local embeddings, gated fusion, gnomAD AF as a feature |
| Secondary — calibration | Tejura et al. (2024) | Am J Hum Genet | Why gene-specific calibration is needed; MSH2/MLH1 as genome-wide-calibration failures |
| Secondary — LLM | ClinVar-BERT (2024/26) | Genome Medicine | LLM branch: dedup + sentence-level de-biasing pipeline, reused near-verbatim |

- MVmamba is primary because it already demonstrates SOTA-beating multi-modal
  fusion for this exact task — AUC 0.901 / AUPR 0.848 / MCC 0.656 on 18,731
  clinical variants vs 21 competitors, beating AlphaMissense and ESM-1b
  head-to-head.
- ESM-1b (Brandes 2023) and AlphaMissense (Cheng 2023) are the baselines
  MVmamba beats — background citations, not the base paper.
- No single paper covers all three pillars → one primary + one per pillar.

## 6 — Project Objectives
1. Gene-specific, leakage-audited variant table for MLH1/MSH2/MSH6/PMS2 (+80-gene
   pre-training panel) with a PMS2 pseudogene gate and full provenance.
2. ESM-2 PLM branch — MVmamba-style WT/VT global+local features + masked-marginal
   PLLR — benchmark 4 fine-tuning recipes and 2 backbones (ESM-1b vs ESM-2).
3. LLM clinical-reasoning branch (BioBERT, ClinVar-BERT pipeline) extended to
   per-ACMG-criterion evidence via RareDAI-style self-distillation.
4. Fuse the branches (concat-MLP start; GateWave / Tavtigian-points
   alternatives); compare against single-branch baselines every time.
5. Calibrate per gene — adopt the gene- & domain-aware method (Chen et al.,
   2026) where published, family-pooling where not.
6. Evaluate leave-one-MMR-gene-out with bootstrap CIs; validate against held-out
   CIMRA / DMS and the CAPS population-genetics metric.

## 7 — Project Plan & Timeline  *(table — adjust weeks to your calendar)*

| Phase | Duration | Status |
|---|---|---|
| 0 Environment & tooling | Week 1 | Done |
| 1 Gene-specific dataset + PMS2 gate | Weeks 2–4 | Done — 1.16 M variants, 12/12 audit |
| 2 Orthogonal functional-assay data (CIMRA/MaveDB) | Week 4 | Done — held out |
| 3 ESM-2 PLM branch + 4 fine-tuning recipes | Weeks 5–8 | Done — LOGO ROC-AUC ≈ 0.90 |
| 4 LLM clinical-reasoning branch (ACMG codes) | Weeks 8–11 | In progress |
| 5 Fusion (concat / GateWave / points) | Weeks 11–13 | Next |
| 6 Gene-specific calibration & evaluation | Weeks 13–15 | Next |
| 7 Write-up & thesis | Weeks 15–16 | Next |

## 8 — Section divider: **Literature Review**  (33 papers, by pillar)

## 9 — Literature Review — PLM Branch
- Zero-shot PLMs — ESM-2 (Lin 2023), ESM-1b (Brandes 2023), PLLR (Meier 2021);
  EVE (Frazer 2021); AlphaMissense (Cheng 2023) is the supervised prior to beat.
- **Surprising finding** — ESM-1b beats ESM-2-650M for clinical pathogenicity
  in two independent studies (VariPred, ProPath). Compare empirically.
- **Four non-redundant fine-tuning recipes:** VariPred (frozen linear probe) ·
  ProPath (Siamese full fine-tune, PLLR vs labels; Adam LR 1e-5, batch 8,
  10 epochs; +5–12 AUC over zero-shot on ~250–450 labels) · CSBJ (per-residue
  token classifier) · MVmamba (frozen WT/VT global + local ±3-aa pooled
  features + gated fusion — base paper).
- Engineering — homology-aware splitting mandatory (MMseqs2 20% id/cov,
  cluster-level); MSH6 (1360 aa) needs VariPred's asymmetric truncation window;
  MCC over AUROC; 10,000-iteration bootstrap CIs.
- External anchors — EVE reaches PMS2 AUC ≥ 0.99; the MSH2 DMS (Jia 2021, ~94%
  coverage) beats every computational predictor for MSH2.

## 10 — Literature Review — LLM Branch
- **ClinVar-BERT is the direct template — reuse its pipeline:** MinHash +
  Jaccard(0.95) fuzzy dedup, grouped by submitting-lab × gene (Ambry, GeneDx,
  Invitae, Color, Prevention Genetics) · **SentenceClassifier de-biasing (the
  critical step)** — strip conclusion + boilerplate sentences before training,
  or the model learns the lab template (F1 0.971 after filtering vs 0.376
  un-fine-tuned).
- **The gap it leaves** — outputs only a 3-way lean, not per-ACMG-criterion
  evidence (PS3, PM2, PP3, PS4 …).
- **RareDAI self-distillation closes it** — offline teacher generates CoT
  rationales anchored to ACMG codes; local student fine-tuned on (text, CoT,
  label) triples. CoT adds +5 pp; no API in the deployed path.
- Keep the core classifier local & deterministic — GPT-4 shows 1–9 pp
  run-to-run variance even at temperature 0 / fixed seed. Any API sub-module is
  upstream feature-generation only, never the arbiter.
- Do not LLM-grade evidence *strength* — near-random even for frontier reasoning
  models; cap at Supporting with human review.

## 11 — Literature Review — Calibration & Fusion
**Calibration** — Pejaver et al. (2022): the local-posterior LR⁺ formula + the
Supporting/Moderate/Strong/Very-Strong evidence table our score must map onto ·
Tejura et al. (2024): genome-wide calibration masks per-gene heterogeneity, 73%
of assessable gene-intervals "trending discordant", MSH2 & MLH1 named · Chen et
al. (2026) gene- & domain-aware calibration: the method Tejura said didn't exist
yet — open-source (PredictMD), MSH2 is their flagship worked example; adopt it,
reuse published MSH2/MLH1 thresholds · small-N gene calibration has no
off-the-shelf solution — binomial power test first, fall back to MMR-family /
domain pooling.
**Fusion** — start simple and mean it: E3 (Ko et al., 2026) hit SOTA with plain
concatenation; fusion helped only 4/6 tasks and hurt 2; TriFit's
MoE+contrastive apparatus bought ~1–2 AUROC at 100–1000× our data · frozen
encoders → per-branch projection → concat → shallow MLP; compare against
MVmamba GateWave and the Tavtigian ACMG-points formula.

## 12 — Gap Analysis & Our Approach  *(table)*

| Aspect | Common practice | This project |
|---|---|---|
| Calibration | genome-wide, one threshold | gene-specific (Chen 2026) → MMR-family pool → genome-wide fallback |
| Split discipline | residue-level only | MMseqs2 cluster-disjoint + leave-one-MMR-gene-out |
| Fine-tuning recipe | one, assumed | four recipes + two backbones benchmarked on identical splits |
| Functional-assay data | used in training | held out for validation by construction (CIMRA / DMS) |
| PMS2 pseudogene region | ignored | fail-closed data-quality gate (exons 11–15) |
| LLM branch | raw notes → label | sentence-level de-biasing + ACMG-code self-distillation |
| Evidence strength via LLM | attempted | not attempted — near-random; capped at Supporting |

Literature Survey Report: `<paste link here>`

## 13 — Section divider: **Proposed System**

## 14 — System Architecture / Design  *(diagram)*
```
Phase 1–2  Data: ClinVar (≥2★, VCEP tiers) · gnomAD v4 AF (PM2/BA1/BS1) ·
           UniProt + AlphaFold   ||   held out: CIMRA · MSH2 DMS · MaveDB
     │
Unified variant table — key (uniprot_id, pos, wt, mut) + provenance
     →  PMS2 pseudogene gate  →  12-check audit
     ├──────────────────────────────┬──────────────────────────────┐
ESM-2 branch (Phase 3)                     LLM clinical-reasoning branch (Phase 4)
WT + variant → ESM-2 / ESM-1b              ClinVar free-text → MinHash dedup
global (mean-pool) + local (±3 aa)         → SentenceClassifier de-biasing
z = [h_wt‖h_mut‖Δh‖|Δh|‖PLLR] + gnomAD AF  → BioBERT → ACMG evidence codes
4 fine-tuning recipes compared             RareDAI self-distilled CoT rationales
     └──────────────────────────────┴──────────────────────────────┘
Fusion (Phase 5) — concat + shallow MLP  vs  GateWave gated  vs  Tavtigian points
     │
Gene-specific calibration (Phase 6) — Chen 2026 gene/domain-aware →
     MMR-family pool → genome-wide fallback  ⇒  ACMG evidence strength
     │
Evaluation — leave-one-MMR-gene-out · MCC + AUBPRC · 10k bootstrap CIs ·
     held-out CIMRA / DMS · CAPS metric · VUS-resolution ceiling ~50–85%
```

## 15 — Methodology — MMR Gene Constraints  *(the non-negotiable clinical facts)*
- The four genes are not interchangeable — penetrance MLH1/MSH2 high, MSH6
  moderate, PMS2 low (11–20% lifetime CRC). No shared threshold set.
- Missense share of pathogenic variants: MLH1 40% · MSH2 30% · MSH6 50% ·
  PMS2 60% — hardest to predict exactly where it matters most.
- **PMS2 exons 11–15 (PMS2CL pseudogene)** — standard short-read NGS calls
  untrustworthy → orthogonal confirmation required. Fail-closed pipeline gate.
- **PVS1 is gene-specific** — initiation-codon loss is full PVS1 for MLH1,
  Strong only for MSH6/PMS2, NOT APPLICABLE for MSH2.
- **PM1 (hotspot) and PP2 (missense-intolerant gene) are NOT used for any MMR
  gene** — the VCEP found no recognised hotspots.
- CIMRA is the best-calibrated MMR functional assay (all 4 genes) but cannot
  detect splicing — never apply it to splice-hypothesised variants.
- Output language — "predisposing", not "pathogenic", for low-penetrance /
  PMS2 missense (disease needs a somatic second hit).

## 16 — Methodology — Splits, Leakage & Metrics
- Homology-aware splitting mandatory — MMseqs2 20% id / 20% cov, cluster-level;
  leave-one-MMR-gene-out is the real generalization test (>30%-identity pairs
  cost VariPred MCC 0.75→0.65).
- LLM template leakage — a few labs submit most Lynch variants; sentence-level
  evidence/conclusion filtering runs before training.
- Small-N gene calibration is the hardest unsolved problem — Tejura's binomial
  power test per gene first; MSH6 / PMS2 likely fall back to family / domain
  pooling.
- Functional data held out by construction — CIMRA / DMS never enter branch
  training → genuine non-circular check.
- Circularity control — exclude every constituent sub-tool's training variants;
  balanced-label per-gene subset to test for shortcut-learning of gene identity.
- Metrics — MCC + imbalance-corrected AUBPRC, 10,000-iteration bootstrap CIs,
  threshold tuned on an inner slice (not 0.5).
- Ground-truth caution — ClinVar / ClinGen labels can be stale; any accuracy
  number needs a blinded-expert spot-check.

## 17 — Datasets
- **Clinical labels:** ClinVar (NCBI, ≥ 2-star) · InSiGHT / ClinGen MMR-VCEP tiers
- **Functional evidence (held out — validation only):** CIMRA OddsPath (all 4
  genes) · MSH2 DMS (Jia 2021, ~94% coverage) · MLH1 abundance assay ·
  ProteinGym / MaveDB
- **Priors / features:** AlphaMissense · EVE / ESM-1b / 17 ProteinGym zero-shot
  scores · gnomAD v4 AF & constraint · AlphaFold pLDDT · InterPro / UniProt
  domains
- **Reference sequences (pinned):** MLH1 P40692 (756 aa) · MSH2 P43246 (934 aa)
  · MSH6 P52701 (1360 aa) · PMS2 P54278 (862 aa)

| Dataset composition (80-gene pre-training panel) | value |
|---|---|
| Unique missense substitutions | 1,156,625 |
| Labelled rows | 190,494  (82,149 P/LP · 108,345 B/LB) |
| Clinical-slice rows | ~5,152  (2.7% of labelled) |
| Rows dropped by WT-sequence validation | 50,304 |
| Independent integrity audit | 12 / 12 checks pass |

## 18 — Technology Stack

*Full detail + rationale for every choice: [`docs/TECH_STACK.md`](TECH_STACK.md).*

Python 3.13 · PyTorch · Hugging Face Transformers · ESM-2 650M
(facebook/esm2_t33_650M_UR50D) · ESM-1b · BioBERT · MedGemma-27B
(self-distillation teacher, training-time only) · BioPython / pandas / NumPy /
SciPy / scikit-learn · MMseqs2 · AlphaFold DB · gnomAD GraphQL · NCBI E-utils ·
MaveDB REST · PredictMD / VEP_calibration / PosteriorCalibration · CUDA GPU
(full fine-tune ~10 GiB VRAM with AMP + accumulation + checkpointing) ·
one-command reproducible pipeline · 82 passing unit tests · SHA-256 provenance
manifest.

## 19 — Section divider: **Progress & Preliminary Results**

## 20 — Work Done So Far
- **Phases 0–3 complete.**
- Unified 1.16 M-variant multi-source table — provenance-tracked, PMS2
  pseudogene gate, VCEP tiering; 12/12 independent integrity audit.
- Orthogonal functional evidence (CIMRA / MaveDB / MSH2 DMS) attached and held
  out from training by construction.
- ESM-2 PLM branch: four fine-tuning recipes + two backbones on identical
  splits; Stage-2b true backbone gradient fine-tune runs end-to-end.
- Reproducible one-command pipeline; 82 passing unit tests; full run log of
  every build failure and its root cause.
- Four circularity / silent-failure bugs found and fixed by instrumentation
  (e.g. a DMS feature equalling the label on 97.3% of rows → spurious
  ROC-AUC 0.9987).

## 21 — Preliminary Results
**Leave-one-MMR-gene-out (unseen gene) — ESM-2 Siamese fine-tune, 2026-08-28**

| Held-out gene | ROC-AUC (95% CI) | MCC | n |
|---|---|---|---|
| MLH1 | 0.899 (0.845–0.945) | 0.53 | 208 |
| MSH2 | 0.878 (0.835–0.916) | 0.30 | 335 |
| MSH6 | 0.907 (0.851–0.954) | 0.65 | 119 |
| **Mean** | **0.895** | — | 662 |

**Circularity waterfall (80-gene panel):** 0.9987 (target leakage) → **0.716**
(leakage-clean, all labels) → **0.963** (clinical slice). Clinical slice
0.963 / MCC 0.78 vs AlphaMissense 0.945 / 0.75.

**Honest caveats:** single seed, early-stop epoch 1–3 (frozen-backbone ablation
floor not yet run) · MCC-optimal threshold varies 0.06–0.43 across genes →
per-gene calibration required (the project thesis, restated) · PMS2 excluded by
the pseudogene gate → 3-gene result · AlphaMissense trained on ClinVar → the
comparison may favour the baseline.

## 22 — Challenges & Risks
Small-N gene-specific calibration (no off-the-shelf solution; MSH6/PMS2 fall
back to pooling) · PMS2 pseudogene → 3-gene coverage by default · ESM-1b vs
ESM-2 backbone unsettled → resolve empirically · LLM template leakage + stale
ClinVar ground truth → sentence filtering + blinded-expert spot-check ·
VUS-resolution ceiling ~50–85% (higher = overconfidence red flag) · 650M
fine-tune needs a CUDA GPU.

## 23 — Summary
VUS backlog is the bottleneck; genome-wide calibration is wrong for MSH2/MLH1 ·
multi-modal (ESM-2 + LLM) classifier on a leakage-audited, gene-aware pipeline ·
base paper MVmamba anchors the architecture, Tejura + ClinVar-BERT the other
pillars · progress: dataset + ESM-2 branch done, unseen-gene ROC-AUC ≈ 0.90,
clinical-slice 0.963 vs AlphaMissense 0.945 · remaining: LLM branch, fusion,
gene-specific calibration, broad-panel leave-one-protein-out.

## 24 — Future Work / Next Steps
Frozen-backbone ablation floor + ≥3-seed averaging · build the LLM branch
(ClinVar-BERT pipeline + RareDAI self-distillation) · implement & ablate fusion ·
check PredictMD for published MSH2/MLH1 calibrations, apply Chen et al. (2026)
elsewhere, run the binomial power test per gene · leave-one-protein-out CV +
MMseqs2 cluster-disjoint split; validate vs CIMRA / DMS and CAPS · evaluate
adopting AutoPM3 for the optional PM3 literature-mining sub-module.

## 25–26 — References  (37 — each is a clickable hyperlink in the .pptx)

*Slide 25 (1–16):*
1. Zhang H. et al. (2025). Deciphering missense variant pathogenicity via
   enhanced Bi-Mamba and a structure-informed protein language model (MVmamba).
   2025 IEEE Int. Conf. on Bioinformatics and Biomedicine (BIBM), 236–241.
   DOI 10.1109/BIBM66473.2025.11356763.  *[primary base paper]*
2. Tejura M. et al. (2024). Calibration of variant effect predictors on
   genome-wide data masks heterogeneous performance across genes. Am J Hum
   Genet. PMC11393694.  *[base paper — calibration]*
3. ClinVar-BERT (2024). From text to translation: using language models to
   prioritize variants for clinical review. medRxiv 2024.12.31.24319792 →
   Genome Medicine 2026.  *[base paper — LLM]*
4. Richards S. et al. (2015). Standards and guidelines for the interpretation of
   sequence variants (ACMG/AMP). Genet Med 17:405–424.
5. Grimm D.G. et al. (2015). The evaluation of tools used to predict the impact
   of missense variants is hindered by two types of circularity. Hum Mutat
   36:513–523.
6. Landrum M.J. et al. (2018). ClinVar: improving access to variant
   interpretations. Nucleic Acids Res 46:D1062.
7. Lin Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein
   structure with a language model (ESM-2). Science 379:1123–1130. (bioRxiv
   2022.07.20.500902)
8. Meier J. et al. (2021). Language models enable zero-shot prediction of the
   effects of mutations on protein function. NeurIPS.
9. Brandes N. et al. (2023). Genome-wide prediction of disease variant effects
   with a deep protein language model (ESM-1b). Nat Genet 55:1512–1522.
10. Cheng J. et al. (2023). Accurate proteome-wide missense variant effect
    prediction with AlphaMissense. Science 381:eadg7492.
11. Frazer J. et al. (2021). Disease variant prediction with deep generative
    models of evolutionary data (EVE). Nature 599:91–95.
12. Lin W., Wells J., Wang Z., Martin A.C.R. (2024). Enhancing missense variant
    pathogenicity prediction with protein language models using VariPred. Sci
    Rep. PMC10999449.
13. Saadat A., Fellay J. (2025). Fine-tuning the ESM2 protein language model to
    understand the functional impact of missense variants. Comput Struct
    Biotechnol J. arXiv:2410.10919.
14. Zhan H. et al. (2023). ProPath: Disease-Specific Protein Language Model for
    Variant Pathogenicity. arXiv:2311.03429.
15. Notin P. et al. (2023). ProteinGym: large-scale benchmarks for protein
    fitness prediction and design. NeurIPS Datasets & Benchmarks.
16. Orenbuch R. et al. (2025). Proteome-wide model for human disease genetics
    (popEVE). Nat Genet. (medRxiv 2023.11.27.23299062)

*Slide 26 (17–37):*
17. Plazzer J.P. et al. (InSiGHT / ClinGen VCEP) (2024). Mismatch repair gene
    specifications to the ACMG/AMP classification criteria. medRxiv
    2024.05.13.24307108.
18. Rayner E. et al. (2022). Predictive functional assay-based classification of
    PMS2 variants in Lynch syndrome (CIMRA). Hum Mutat. PMC9545740.
19. Lynch syndrome molecular mechanisms and variant classification review
    (2023). PMC9978028.
20. UMD locus-specific databases for MLH1 / MSH2 / MSH6 (2013). PMC3668602.
21. DNA mismatch repair gene variant classification: somatic mutations and
    MMR-deficient crypts/glands (2023). PMC10605939.
22. Jia X. et al. (2021). Massively parallel functional testing of MSH2 missense
    variants conferring Lynch syndrome risk. Am J Hum Genet 108:163–175.
23. Pejaver V. et al. (2022). Calibration of computational tools for missense
    variant pathogenicity classification and ClinGen PP3/BP4 recommendations.
    Am J Hum Genet. PMC9748256.
24. Bergquist T. et al. (2025). Calibration of additional computational tools
    expands ClinGen recommendation options for variant classification. Genet Med
    27:101402.
25. Chen Y.L., Fayer S., Jain S. et al. (2026). Gene- and domain-aware
    calibration increases the clinical utility of variant effect predictors.
    bioRxiv 2026.02.17.706269.
26. Gudkov M. et al. (2025). Benchmarking of variant pathogenicity prediction
    methods using a population genetics approach (CAPS). Bioinformatics
    Advances. PMC12579982.
27. Livesey B.J., Marsh J.A. (2023). Updated benchmarking of variant effect
    predictors using deep mutational scanning. Mol Syst Biol. PMC10407742.
28. Tavtigian S.V. et al. (2018). Modeling the ACMG/AMP variant classification
    guidelines as a Bayesian classification framework. Genet Med 20:1054–1060.
29. Brnich S.E. et al. (2020). Recommendations for application of the functional
    evidence PS3/BS3 criterion. Genome Med 12:3.
30. RareDAI (2026). Interpretable fine-tuned large language models facilitate
    genetic test decisions for rare diseases. npj Digital Medicine.
31. Preparing to integrate GPT-4 into genetic variant assessment workflows:
    performance, drift, and nondeterminism (2023). arXiv:2312.13521.
32. Saadat A., Fellay J. (2026). Large language models for variant-centric
    functional evidence mining. arXiv:2604.00075.
33. AutoPM3 (2025). Enhancing variant interpretation via LLM-driven PM3 evidence
    extraction from scientific literature. Bioinformatics. (bioRxiv
    2024.10.29.621006)
34. Ko J., Parkinson J., Wang W. (2026). Scalable embedding fusion with protein
    language models. Brief Bioinform. PMC12853110.
35. TriFit: Trimodal Fusion with Protein Dynamics for Mutation Fitness
    Prediction (2026). arXiv:2604.12026.
36. Chen S. et al. (2024). A genomic mutational constraint map using variation
    in 76,156 human genomes (gnomAD v4). Nature 625:92–100.
37. Varadi M. et al. (2022). AlphaFold Protein Structure Database. Nucleic Acids
    Res 50:D439–D444.

> Titles verified against the PDF metadata / first pages — see
> `~/Downloads/lynch-mmr-pathogenicity/PAPER_LINKS.md` for every paper with its
> full title and a working link. (`MANIFEST.md`'s "Jagota et al." for VariPred
> [12] was wrong — the authors are Lin W. et al.)

## 27 — Thank You
Batch No: `__` · `Name 1/2/3, Roll No` · Queries?
