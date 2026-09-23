"""The whole pipeline, end to end, on tiny synthetic data — minutes, on any machine.

``vpdl-slm smoke`` runs every stage in order on an invented ClinVar release
(``vpdl.slm.build.synthetic``) with random-weight tiny models: records ->
statistics -> clusters -> split -> roles -> examples -> leakage audit ->
pretraining corpus -> token packing -> continued pretraining (2 steps) ->
fine-tuning (1 epoch) -> baselines -> predictions -> SLM export -> explanation
and its grounding check.

It proves the stages fit together and the contracts hold. It measures nothing
about genomics: the data is synthetic and the models are random, so no number
it prints is a result.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["run_smoke"]


def run_smoke(out_dir: str | None = None, seed: int = 7, keep: bool = False) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from vpdl.slm.baselines import run_baselines
    from vpdl.slm.build.clusters import document_clusters
    from vpdl.slm.build.examples import build_examples
    from vpdl.slm.build.leakage import AuditInputs, run_audit
    from vpdl.slm.build.pretrain_corpus import build_pretrain_corpus
    from vpdl.slm.build.records import build_records, load_tables, variant_view
    from vpdl.slm.build.roles import pretraining_exclusions, reserved_variants
    from vpdl.slm.build.splits import SplitConfig, make_split, split_frame, split_summary
    from vpdl.slm.build.stats import corpus_stats
    from vpdl.slm.build.synthetic import write_synthetic_clinvar
    from vpdl.slm.catalog import HOLDOUT_PUBLICATIONS, validate_catalog
    from vpdl.slm.evaluation.explain import check_grounding, explain
    from vpdl.slm.interface import read_slm_outputs
    from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
    from vpdl.slm.modeling.continued import PretrainConfig, pack_for_tokenizer, run_pretrain
    from vpdl.slm.modeling.finetune import FinetuneConfig, run_finetune
    from vpdl.slm.schema import CLASSES, validate_table

    started = time.time()
    temporary = None
    if out_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="vpdl-slm-smoke-")
        out_dir = temporary.name
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    steps: dict[str, Any] = {}
    try:
        steps["catalog_problems"] = validate_catalog()

        paths = write_synthetic_clinvar(root / "clinvar", variants_per_gene=6, seed=seed)
        manifest = build_records(root / "records", paths["variant_summary"],
                                 paths["submission_summary"], paths["var_citations"])
        tables = load_tables(root / "records")
        steps["records"] = manifest["counts"]
        steps["schema_problems"] = {name: validate_table(frame, name) for name, frame in tables.items()}
        steps["variant_view_keys"] = sorted(variant_view(tables, tables["variants"].variant_id.iloc[0]))

        steps["stats"] = {k: corpus_stats(tables)[k] for k in
                          ("documents", "unique_variants_with_text", "unique_genes", "characters")}

        clusters = document_clusters(tables["documents"])
        steps["clusters"] = {"near": int(clusters["near_dup_cluster"].nunique()),
                             "template": int(clusters["template_cluster"].nunique())}

        frame = split_frame(tables, clusters)
        split = make_split(frame, SplitConfig("variant", seed=seed))
        steps["split"] = split_summary(split, frame)["documents"]
        reserved = reserved_variants({"variant": split})
        exclusions = pretraining_exclusions(tables, reserved, HOLDOUT_PUBLICATIONS)
        steps["exclusions"] = {"documents": len(exclusions["documents"]), "pmids": len(exclusions["pmids"])}

        examples = build_examples(tables, split, "classify", HOLDOUT_PUBLICATIONS)
        units = build_examples(tables, split, "evidence_type", HOLDOUT_PUBLICATIONS)
        examples_dir = root / "examples"
        examples_dir.mkdir(exist_ok=True)
        examples.to_parquet(examples_dir / "classify.parquet", index=False)
        units.to_parquet(examples_dir / "units.parquet", index=False)
        steps["examples"] = {"classify": int(len(examples)), "units": int(len(units)),
                             "conclusions_removed": int(examples["conclusions_removed"].sum())}

        audit = run_audit(AuditInputs("variant", examples.merge(clusters, on="document_id", how="left"),
                                      tables["variants"], tables["documents"], tables["citations"],
                                      features=("consequence",),
                                      holdout_publications=HOLDOUT_PUBLICATIONS))
        steps["leakage"] = {"critical": len(audit.critical),
                            "findings": {f.check: f.severity for f in audit.findings}}

        corpus = build_pretrain_corpus(root / "pretrain_corpus", records_dir=root / "records",
                                       exclusions=exclusions, include_narratives=True,
                                       training_documents=split.loc[split["split"] == "train",
                                                                    "document_id"])
        steps["pretrain_corpus"] = {"sources": corpus["sources"], "decisions": corpus["decisions"]}

        backbone = load_backbone(BackboneSpec("tiny-bert"), seed=seed)
        pack = pack_for_tokenizer(root / "pretrain_corpus", backbone.tokenizer, root / "pretrain_tokens")
        steps["pretrain_pack"] = pack["splits"]

        pretrain_config = PretrainConfig(backbone="tiny-bert", corpus_dir=str(root / "pretrain_corpus"),
                                         token_dir=str(root / "pretrain_tokens"),
                                         out_dir=str(root / "pretrain"), context=64, micro_batch=2,
                                         total_steps=2, eval_every=2, checkpoint_every=2,
                                         warmup_steps=1, seed=seed)
        steps["pretrain_dry_run"] = run_pretrain(pretrain_config, dry_run=True)
        steps["pretrain"] = {k: v for k, v in run_pretrain(pretrain_config).items()
                             if k in ("step", "loss", "val_loss", "final", "objective")}

        # The smoke run's registry line goes in its own folder, never next to real runs.
        import vpdl.slm.modeling.finetune as finetune_module
        real_registry = finetune_module.REGISTRY
        finetune_module.REGISTRY = root / "registry.jsonl"
        finetune_config = FinetuneConfig(
            run_name="smoke", out_dir=str(root / "finetune"),
            examples=str(examples_dir / "classify.parquet"),
            unit_examples=str(examples_dir / "units.parquet"), records=str(root / "records"),
            backbone="tiny-bert", max_length=64, peft={"kind": "full", "allow_full": True},
            tasks=("classify", "evidence", "acmg"), embedding_dim=32,
            train={"epochs": 1, "batch_size": 4, "lr": 1e-3, "patience": 1},
            seed=seed, mc_samples=2, cache_dir=str(root / "cache"))
        steps["finetune_dry_run"] = {k: v for k, v in run_finetune(finetune_config, dry_run=True).items()
                                     if k in ("finite_loss", "probs_shape", "embedding_dim",
                                              "checkpoint_roundtrip_identical", "precision")}
        try:
            result = run_finetune(finetune_config)
        finally:
            finetune_module.REGISTRY = real_registry
        steps["finetune"] = {split_name: entry.get("five_class", {}).get("macro_f1")
                             for split_name, entry in result["metrics"].items()
                             if isinstance(entry, dict) and "five_class" in entry}

        baselines = run_baselines(examples, root / "baselines",
                                  ("majority", "tfidf_lr", "tfidf_svm"), seed=seed)
        steps["baselines"] = {kind: baselines[kind].get("test", {}).get("five_class", {}).get("macro_f1")
                              for kind in ("majority", "tfidf_lr", "tfidf_svm") if kind in baselines}

        predictions = pd.read_parquet(root / "finetune" / "predictions.parquet")
        embeddings = np.load(root / "finetune" / "embeddings_test.npy") if (
            root / "finetune" / "embeddings_test.npy").exists() else None
        if embeddings is not None and len(embeddings):
            from vpdl.slm.cli import main as cli_main
            code = cli_main(["embed", "--predictions", str(root / "finetune" / "predictions.parquet"),
                             "--embeddings", str(root / "finetune" / "embeddings_test.npy"),
                             "--examples", str(examples_dir / "classify.parquet"), "--split", "test",
                             "--model-version", "smoke@tiny-bert", "--feature-version", "smoke",
                             "--out", str(root / "export")])
            exported = read_slm_outputs(root / "export" / "slm_outputs.jsonl")
            steps["export"] = {"cli_exit": code, "records": len(exported),
                               "embedding_dim": int(exported[0].embedding.shape[0]) if exported else 0}

        document = tables["documents"].iloc[0]
        unit_rows = tables["evidence_units"]
        unit_rows = unit_rows.loc[unit_rows["document_id"] == document["document_id"]].to_dict("records")
        probabilities = predictions[[f"p_{c}" for c in CLASSES]].to_numpy()[0]
        explanation = explain(probabilities, unit_rows, predicted_codes=["PM2"])
        grounding = check_grounding(explanation["explanation"],
                                    {e["id"]: {"text": e["text"], "acmg_codes": e["acmg_codes"],
                                               "polarity": e["polarity"]}
                                     for e in explanation["key_evidence"]})
        steps["explanation"] = {"classification": explanation["classification"],
                                "evidence": len(explanation["key_evidence"]),
                                "missing_evidence": explanation["missing_evidence"][:3],
                                "unsupported_claims": len(grounding.unsupported),
                                "evidence_coverage": grounding.evidence_coverage}

        ok = (not steps["catalog_problems"] and steps["leakage"]["critical"] == 0
              and steps["finetune_dry_run"]["finite_loss"]
              and steps["finetune_dry_run"]["checkpoint_roundtrip_identical"]
              and steps["explanation"]["unsupported_claims"] == 0
              and all(not problems for problems in steps["schema_problems"].values()))
        return {"ok": bool(ok), "seconds": round(time.time() - started, 1),
                "out": str(root) if keep or temporary is None else "(temporary, removed)",
                "steps": steps,
                "note": "synthetic data and random weights: no number here is a genomics result"}
    finally:
        if temporary is not None and not keep:
            try:
                temporary.cleanup()
            except (OSError, PermissionError):
                pass


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, default=str))
