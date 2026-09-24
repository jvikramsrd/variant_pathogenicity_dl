"""The experiment matrix, in code — so the plan, the commands and the docs cannot drift apart.

``vpdl-slm experiments --markdown`` renders the table in
docs/slm/GENOMIC_SLM_EXPERIMENT_PLAN.md from these entries. Every entry names
the hypothesis it tests, the split it runs on, the config that defines it and
the exact command. **Status is ``NOT RUN — REQUIRES DGX SPARK`` for all of
them**: nothing in this matrix has been executed, and no metric is recorded
here. Metrics arrive as files (``runs/slm_genomic/<id>/metrics.json``) written
by the run itself, plus one line per run in ``runs/slm_genomic/registry.jsonl``.

Seeds: three per arm (42, 43, 44). A single-seed difference smaller than the
seed spread is not a result — this project has paid for that lesson before
(docs/RUNLOG.md, 2026-09-06).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

__all__ = ["Experiment", "EXPERIMENTS", "HYPOTHESES", "CLINICAL_TEXT_ABLATIONS", "experiment",
           "experiments_markdown", "SEEDS", "STATUS"]

SEEDS = (42, 43, 44)
STATUS = "NOT RUN — REQUIRES DGX SPARK"

HYPOTHESES: dict[str, str] = {
    "H1": "Broad genomic continued pretraining improves transfer to unseen genes.",
    "H2": "Clinical text improves evidence-grounded reasoning over structured features alone.",
    "H3": "Structured genomic features improve classification over text alone.",
    "H4": "Evidence extraction improves explanation grounding.",
    "H5": "Multi-task learning improves evidence-aware classification.",
    "H6": "Explicit VUS training improves uncertainty quality.",
    "H7": "DL representations add complementary biological information.",
    "H8": "MMR adaptation improves MMR performance without destroying broad transfer.",
}


@dataclass
class Experiment:
    id: str
    name: str
    objective: str
    hypothesis: str | None
    split: str
    model: str
    tasks: tuple[str, ...] = ("classify",)
    features: str = "default"
    config: str | None = None
    command: str = ""
    depends_on: tuple[str, ...] = ()
    status: str = STATUS
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _e(*args, **kwargs) -> Experiment:
    return Experiment(*args, **kwargs)


_BASE = "vpdl-slm finetune --config configs/slm/{config}"

EXPERIMENTS: tuple[Experiment, ...] = (
    _e("EXP-001", "Majority", "the floor every arm must clear", None, "variant", "majority",
       command="vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet "
               "--kinds majority --out runs/slm_genomic/EXP-001"),
    _e("EXP-002", "TF-IDF + logistic regression", "linear text baseline", "H2", "variant", "tfidf_lr",
       command="vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet "
               "--kinds tfidf_lr --out runs/slm_genomic/EXP-002"),
    _e("EXP-003", "TF-IDF + linear SVM", "linear text baseline", "H2", "variant", "tfidf_svm",
       command="vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet "
               "--kinds tfidf_svm --out runs/slm_genomic/EXP-003"),
    _e("EXP-004", "BioBERT", "biomedical encoder, no genomic pretraining", "H1", "variant",
       "hf:dmis-lab/biobert-base-cased-v1.2", config="exp004_biobert.toml",
       command=_BASE.format(config="exp004_biobert.toml")),
    _e("EXP-005", "PubMedBERT (BiomedBERT)", "biomedical encoder, no genomic pretraining", "H1",
       "variant", "hf:microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
       config="exp005_pubmedbert.toml", command=_BASE.format(config="exp005_pubmedbert.toml")),
    _e("EXP-006", "BioClinicalBERT", "clinical-note encoder", "H1", "variant",
       "hf:emilyalsentzer/Bio_ClinicalBERT", config="exp006_bioclinicalbert.toml",
       command=_BASE.format(config="exp006_bioclinicalbert.toml")),
    _e("EXP-007", "ClinVar-BERT-style", "BioBERT + this project's conclusion/template pipeline",
       "H2", "text", "hf:dmis-lab/biobert-base-cased-v1.2", config="exp007_clinvarbert_style.toml",
       command=_BASE.format(config="exp007_clinvarbert_style.toml"),
       notes="The published ClinVar-BERT weights are not available; this reproduces its recipe "
             "(sentence filtering, lab-template dedup) on our data, and says so."),
    _e("EXP-008", "Existing project LLM", "the knowledge base's reader (medgemma:27b) on the same "
       "questions", None, "variant", "medgemma:27b via vpdl kb-ask",
       command="vpdl kb-eval --models medgemma:27b",
       notes="Different task by design: the KB reads passages and refuses; it never classifies a "
             "variant (docs/kb/DESIGN.md rule 4). Reported as context, not as a classification arm."),
    _e("EXP-009", "Custom SLM (no genomic pretraining)", "our architecture on a biomedical backbone",
       "H1", "variant", "best of EXP-004..006", config="exp009_custom_slm.toml",
       command=_BASE.format(config="exp009_custom_slm.toml"), depends_on=("EXP-004", "EXP-005", "EXP-006")),
    _e("EXP-010", "+ broad genomic continued pretraining", "does domain pretraining help?", "H1",
       "variant", "EXP-009 backbone after `vpdl-slm pretrain`", config="exp010_genomic_pretrained.toml",
       command="vpdl-slm pretrain --config configs/slm/pretrain_dgx.toml && "
               + _BASE.format(config="exp010_genomic_pretrained.toml"), depends_on=("EXP-009",)),
    _e("EXP-011", "+ structured features", "do genomic fields add to text?", "H3", "variant",
       "EXP-010", config="exp011_structured.toml", command=_BASE.format(config="exp011_structured.toml"),
       depends_on=("EXP-010",)),
    _e("EXP-012", "+ evidence extraction", "does unit-level supervision help?", "H4,H5", "variant",
       "EXP-011", ("classify", "evidence"), config="exp012_evidence.toml",
       command=_BASE.format(config="exp012_evidence.toml"), depends_on=("EXP-011",),
       notes="Evidence targets are rule-derived weak labels; their own accuracy is unmeasured "
             "until an annotated sample exists (DATA GAP)."),
    _e("EXP-013", "+ ACMG / ClinGen context", "does predicting criteria help?", "H4", "variant",
       "EXP-012", ("classify", "evidence", "acmg"), config="exp013_acmg.toml",
       command=_BASE.format(config="exp013_acmg.toml"), depends_on=("EXP-012",)),
    _e("EXP-014", "Multi-task", "all heads together vs each alone", "H5", "variant", "EXP-013",
       ("classify", "evidence", "acmg"), config="exp014_multitask.toml",
       command=_BASE.format(config="exp014_multitask.toml"), depends_on=("EXP-013",)),
    _e("EXP-015", "+ VUS objective", "VUS as a class, not 'low confidence'", "H6", "variant",
       "EXP-014", config="exp015_vus.toml", command=_BASE.format(config="exp015_vus.toml"),
       depends_on=("EXP-014",),
       notes="Scored by later reclassification (vpdl-slm evaluate --vus-old/--vus-new), which "
             "needs an archived ClinVar release."),
    _e("EXP-016", "+ explanation supervision", "grounded explanations", "H4", "variant", "EXP-015",
       config="exp016_explanation.toml", command=_BASE.format(config="exp016_explanation.toml"),
       depends_on=("EXP-015",),
       notes="The deterministic explainer needs no training; this arm exists for a generative "
             "explainer and is PROPOSED — no generative head is implemented."),
    _e("EXP-017", "+ teacher distillation", "does a local teacher's labelling help?", "H5", "variant",
       "EXP-014 + qwen3:32b labels", config="exp017_teacher.toml",
       command="vpdl-slm teacher --examples ... && " + _BASE.format(config="exp017_teacher.toml"),
       depends_on=("EXP-014",),
       notes="NOT RUNNABLE YET: there is no `vpdl-slm teacher` command and nothing writes "
             "teacher_units.parquet from vpdl.slm.teacher's records (audit 2026-09-24). "
             "Distillation here means training the evidence heads on filtered teacher labels "
             "(hard labels); there is no soft-label / KL loss."),
    _e("EXP-018", "+ DL representation", "complementary biology from the DL branch", "H7", "variant",
       "EXP-014 + DLRepresentation", config="exp018_dl.toml",
       command=_BASE.format(config="exp018_dl.toml"), depends_on=("EXP-014",),
       notes="Needs `vpdl-dl export`; the DL branch has not been trained yet."),
    _e("EXP-019", "+ MMR adapter", "specialise on MMR", "H8", "mmr", "EXP-014 + LoRA adapter",
       config="exp019_mmr_adapter.toml", command=_BASE.format(config="exp019_mmr_adapter.toml"),
       depends_on=("EXP-014",)),
    _e("EXP-020", "Final SLM", "the best configuration the ablations support", None, "variant",
       "decided by EXP-009..019", config="exp020_final.toml",
       command=_BASE.format(config="exp020_final.toml"),
       depends_on=tuple(f"EXP-{i:03d}" for i in range(9, 20))),
    _e("EXP-021", "Unseen genes", "transfer to genes never trained on", "H1", "gene", "EXP-020",
       config="exp021_gene_holdout.toml", command=_BASE.format(config="exp021_gene_holdout.toml")),
    _e("EXP-022", "Disease holdout", "transfer to unseen diseases", "H1", "disease", "EXP-020",
       config="exp022_disease_holdout.toml", command=_BASE.format(config="exp022_disease_holdout.toml")),
    _e("EXP-023", "Temporal holdout", "prospective prediction", "H1", "temporal", "EXP-020",
       config="exp023_temporal.toml", command=_BASE.format(config="exp023_temporal.toml")),
    _e("EXP-024", "Independent functional validation", "agreement with assay data never trained on",
       "H7", "functional", "EXP-020", config="exp024_functional.toml",
       command=_BASE.format(config="exp024_functional.toml"),
       notes="MSH2 (MaveDB 00000050-a-1) and MLH1 abundance (00001218-a-1); Spearman and class "
             "AUROC against the assay, not against ClinVar."),
    _e("EXP-025", "MMR / Lynch evaluation", "zero-shot transfer and adapted performance", "H8", "mmr",
       "EXP-020 (zero-shot) vs EXP-019 (adapter)", config="exp025_mmr.toml",
       command=_BASE.format(config="exp025_mmr.toml"), depends_on=("EXP-019", "EXP-020")),
)

# The clinical-text ablations (A-I): all on the same split, differing only in what the model reads.
CLINICAL_TEXT_ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "structured features only", "EXP-001/baselines structured_lr"),
    ("B", "clinical text only", "EXP-010"),
    ("C", "clinical text + structured features", "EXP-011"),
    ("D", "clinical text + literature (pretraining with PubMed)", "EXP-010 vs a corpus without PubMed"),
    ("E", "clinical text + evidence extraction", "EXP-012"),
    ("F", "clinical text + ACMG/ClinGen context", "EXP-013"),
    ("G", "clinical text + VUS objective", "EXP-015"),
    ("H", "clinical text + DL representation", "EXP-018"),
    ("I", "full system", "EXP-020"),
)


def experiment(identifier: str) -> Experiment:
    for item in EXPERIMENTS:
        if item.id == identifier:
            return item
    raise KeyError(f"no experiment {identifier!r}")


def experiments_markdown(items: Sequence[Experiment] = EXPERIMENTS) -> str:
    lines = [f"All experiments: **{STATUS}**. Seeds: {', '.join(map(str, SEEDS))} per arm.", "",
             "| id | experiment | hypothesis | split | model | tasks | status |",
             "|---|---|---|---|---|---|---|"]
    for item in items:
        lines.append(f"| `{item.id}` | {item.name} | {item.hypothesis or '—'} | {item.split} | "
                     f"{item.model} | {', '.join(item.tasks)} | {item.status} |")
    lines += ["", "### Commands", ""]
    for item in items:
        lines.append(f"- `{item.id}`: `{item.command}`" + (f"  \n  {item.notes}" if item.notes else ""))
    lines += ["", "### Clinical-text ablations", "", "| arm | what the model reads | experiment |",
              "|---|---|---|"]
    lines += [f"| {letter} | {what} | {where} |" for letter, what, where in CLINICAL_TEXT_ABLATIONS]
    lines += ["", "### Hypotheses", ""]
    lines += [f"- **{key}**: {value}" for key, value in HYPOTHESES.items()]
    return "\n".join(lines) + "\n"
