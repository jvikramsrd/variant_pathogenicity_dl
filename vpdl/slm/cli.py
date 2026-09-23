"""`vpdl-slm` — the genomic SLM branch's commands (docs/slm/GENOMIC_SLM_DGX_RUNBOOK.md).

    vpdl-slm hardware                what this machine is, vs the intended DGX Spark
    vpdl-slm catalog                 the knowledge-source catalogue (roles, terms, leakage risk)
    vpdl-slm inventory               what data already exists here, measured
    vpdl-slm build-records           ClinVar (+ ERepo) -> variants / documents / evidence / citations
    vpdl-slm stats                   corpus quantification
    vpdl-slm dedup                   exact / near-duplicate / laboratory-template clusters
    vpdl-slm splits                  one split scheme -> per-document assignment
    vpdl-slm roles                   data roles, reserved variants, pretraining exclusions
    vpdl-slm examples                task examples under the task's leakage policy
    vpdl-slm leakage                 the fifteen-check audit (exit 3 on a critical finding)
    vpdl-slm pretrain-corpus         continued-pretraining text, evaluation held out of it
    vpdl-slm pretrain-pack           that corpus -> token files for one backbone
    vpdl-slm pretrain                broad genomic continued pretraining (--dry-run, --benchmark)
    vpdl-slm finetune                supervised multi-task training (--dry-run)
    vpdl-slm baselines               majority / TF-IDF+LR / TF-IDF+SVM / structured-only
    vpdl-slm evaluate                metrics from saved predictions (+ VUS reclassification)
    vpdl-slm embed                   export SLMRepresentation records
    vpdl-slm sizing                  model-size candidates for this corpus and machine
    vpdl-slm experiments             the experiment matrix
    vpdl-slm smoke                   whole pipeline on tiny synthetic data, end to end

The `vpdl` command keeps the from-scratch pretraining pipeline
(`slm-corpus / slm-tokenizer / slm-pack / slm-train`) and the knowledge base;
nothing there is changed by this one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("vpdl-slm")

EXIT_LEAKAGE = 3


def _configure(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")      # Windows consoles default to cp1252
        except (AttributeError, ValueError):
            pass


def _print(value: Any, path: str | None = None) -> None:
    text = json.dumps(value, indent=2, default=str) if not isinstance(value, str) else value
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        print(f"written: {path}")
    else:
        print(text)


# -- commands -----------------------------------------------------------------------

def cmd_hardware(args) -> int:
    from vpdl.slm.hardware import hardware_report
    _print(hardware_report(smoke=args.smoke), args.out)
    return 0


def cmd_catalog(args) -> int:
    from vpdl.slm.catalog import catalog_json, catalog_markdown, validate_catalog
    problems = validate_catalog()
    if problems:
        for problem in problems:
            print(f"catalogue problem: {problem}", file=sys.stderr)
        return 2
    if args.markdown:
        _print(catalog_markdown(), args.out)
    else:
        _print(json.loads(catalog_json(args.root if args.hash else None)), args.out)
    return 0


def cmd_inventory(args) -> int:
    from vpdl.slm.build.inventory import inventory
    _print(inventory(args.root, args.limit, not args.no_clinvar), args.out)
    return 0


def cmd_build_records(args) -> int:
    from vpdl.slm.build.records import build_records
    manifest = build_records(
        args.out, args.variant_summary, args.submission_summary, args.citations, args.erepo,
        dl_canonical=args.dl_canonical, genes=args.genes, limit_documents=args.limit_documents,
        workers=args.workers)
    _print(manifest)
    return 0


def cmd_stats(args) -> int:
    from vpdl.slm.build.records import load_tables
    from vpdl.slm.build.stats import corpus_stats, stats_markdown
    tables = load_tables(args.records)
    tokenizer = None
    if args.tokenizer:
        from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
        backbone = load_backbone(BackboneSpec(args.tokenizer))
        tokenizer = lambda text: backbone.tokenizer(text)["input_ids"]      # noqa: E731
    stats = corpus_stats(tables, tokenizer=tokenizer)
    _print(stats, args.out)
    if args.markdown:
        _print(stats_markdown(stats), args.markdown)
    return 0


def cmd_dedup(args) -> int:
    from vpdl.slm.build.clusters import document_clusters
    from vpdl.slm.build.records import load_tables
    tables = load_tables(args.records, ["documents"])
    clusters = document_clusters(tables["documents"], args.near, args.template)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(args.out, index=False)
    print(json.dumps({"documents": int(len(clusters)),
                      "near_dup_clusters": int(clusters["near_dup_cluster"].nunique()),
                      "template_clusters": int(clusters["template_cluster"].nunique()),
                      "out": args.out}, indent=2))
    return 0


def cmd_splits(args) -> int:
    import pandas as pd

    from vpdl.slm.build.records import load_tables
    from vpdl.slm.build.splits import SplitConfig, make_split, split_frame, split_summary
    tables = load_tables(args.records, ["documents", "variants"])
    clusters = pd.read_parquet(args.clusters) if args.clusters else None
    frame = split_frame(tables, clusters)
    config = SplitConfig(args.scheme, seed=args.seed, val_fraction=args.val_fraction,
                         test_fraction=args.test_fraction, cutoff=args.cutoff,
                         temporal_mode=args.temporal_mode,
                         functional_variants=tuple(args.functional_variants or ()),
                         use_template_clusters=args.use_template_clusters)
    split = make_split(frame, config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(args.out, index=False)
    summary = split_summary(split, frame)
    _print(summary, args.summary)
    if not args.summary:
        print(f"written: {args.out}")
    return 0


def cmd_roles(args) -> int:
    import pandas as pd

    from vpdl.slm.build.records import load_tables
    from vpdl.slm.build.roles import assign_roles, pretraining_exclusions, reserved_variants
    from vpdl.slm.catalog import HOLDOUT_PUBLICATIONS
    tables = load_tables(args.records, ["documents", "citations"])
    splits = {Path(p).stem: pd.read_parquet(p) for p in args.splits}
    reserved = reserved_variants(splits)
    exclusions = pretraining_exclusions(tables, reserved, HOLDOUT_PUBLICATIONS,
                                        strict_literature=not args.no_strict_literature)
    roles = {name: assign_roles(frame).value_counts().to_dict() for name, frame in splits.items()}
    _print({"roles_per_split": roles, "reserved_variants": len(reserved),
            "exclusions": {"documents": len(exclusions["documents"]), "pmids": len(exclusions["pmids"]),
                           "strict_literature": exclusions["strict_literature"]},
            "exclusions_file": args.out}, None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(exclusions, indent=2))
    print(f"written: {args.out}")
    return 0


def cmd_examples(args) -> int:
    import pandas as pd

    from vpdl.slm.build.examples import build_examples, summarize_masking
    from vpdl.slm.build.records import load_tables
    from vpdl.slm.catalog import HOLDOUT_PUBLICATIONS
    tables = load_tables(args.records)
    split = pd.read_parquet(args.split)
    holdout = HOLDOUT_PUBLICATIONS if args.holdout_publications == "auto" else tuple(
        args.holdout_publications.split(",")) if args.holdout_publications else ()
    examples = build_examples(tables, split, args.task, holdout_publications=holdout,
                              min_chars=args.min_chars)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(args.out, index=False)
    summary = {"task": args.task, "out": args.out, "rows": int(len(examples)),
               "by_split": examples["split"].value_counts().to_dict() if len(examples) else {},
               "masking": summarize_masking(examples) if len(examples) else {}}
    if "target_label" in examples:
        summary["labels"] = examples["target_label"].value_counts().to_dict()
    _print(summary)
    return 0


def cmd_leakage(args) -> int:
    import pandas as pd

    from vpdl.slm.build.leakage import AuditInputs, run_audit
    from vpdl.slm.build.records import load_tables
    from vpdl.slm.catalog import HOLDOUT_PUBLICATIONS
    tables = load_tables(args.records, ["variants", "documents", "citations"])
    examples = pd.read_parquet(args.examples)
    if args.clusters:
        examples = examples.merge(pd.read_parquet(args.clusters), on="document_id", how="left")
    pretraining = json.loads(Path(args.pretraining).read_text()) if args.pretraining else None
    teacher = pd.read_parquet(args.teacher) if args.teacher else None
    retrieval = json.loads(Path(args.retrieval).read_text()) if args.retrieval else None
    report = run_audit(AuditInputs(
        scheme=args.scheme, examples=examples, variants=tables["variants"],
        documents=tables["documents"], citations=tables["citations"],
        features=tuple(args.features or ()), functional_variants=tuple(args.functional_variants or ()),
        holdout_publications=HOLDOUT_PUBLICATIONS, cutoff=args.cutoff, pretraining=pretraining,
        teacher=teacher, retrieval_documents=retrieval,
        strict_literature=args.strict_literature, strict_functional=not args.no_strict_functional))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report.to_markdown(), encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    print(report.to_markdown())
    if report.critical:
        print(f"CRITICAL leakage findings: {len(report.critical)} — training must not proceed.",
              file=sys.stderr)
        return EXIT_LEAKAGE
    return 0


def cmd_pretrain_corpus(args) -> int:
    from vpdl.slm.build.pretrain_corpus import build_pretrain_corpus
    exclusions = json.loads(Path(args.exclusions).read_text()) if args.exclusions else None
    training_documents = None
    if args.split:
        import pandas as pd
        split = pd.read_parquet(args.split)
        training_documents = split.loc[split["split"].isin(["train", "mmr_train"]), "document_id"]
    summary = build_pretrain_corpus(args.out, args.kb, args.pubmed, args.records, exclusions,
                                    include_narratives=args.include_narratives,
                                    training_documents=training_documents,
                                    limit_files=args.limit_files, workers=args.workers)
    _print(summary)
    return 0


def cmd_pretrain_pack(args) -> int:
    from vpdl.slm.modeling.backbone import BackboneSpec, load_backbone
    from vpdl.slm.modeling.continued import pack_for_tokenizer
    backbone = load_backbone(BackboneSpec(args.backbone))
    _print(pack_for_tokenizer(args.corpus, backbone.tokenizer, args.out,
                              limit_documents=args.limit_documents))
    return 0


def cmd_pretrain(args) -> int:
    from vpdl.slm.config import config_problems, load_pretrain_config
    from vpdl.slm.modeling.continued import run_pretrain
    config = load_pretrain_config(args.config)
    problems = config_problems(config)
    if problems:
        for problem in problems:
            print(f"config problem: {problem}", file=sys.stderr)
        return 2
    _print(run_pretrain(config, dry_run=args.dry_run, benchmark_steps=args.benchmark))
    return 0


def cmd_finetune(args) -> int:
    from vpdl.slm.config import config_problems, load_finetune_config
    from vpdl.slm.modeling.finetune import run_finetune
    overrides = {}
    if args.limit_rows:
        overrides["limit_rows"] = args.limit_rows
    if args.out_dir:
        overrides["out_dir"] = args.out_dir
    config = load_finetune_config(args.config, overrides)
    problems = config_problems(config)
    if problems:
        # A dry run checks the config AND the data it names; missing inputs are
        # exactly what it exists to catch, so it reports them and stops here.
        for problem in problems:
            print(f"config problem: {problem}", file=sys.stderr)
        return 2
    result = run_finetune(config, dry_run=args.dry_run)
    _print(result if args.dry_run else {"out": result["out"], "experiment_id": result["experiment_id"],
                                        "run_id": result["run_id"],
                                        "metrics_file": str(Path(result["out"]) / "metrics.json")})
    return 0


def cmd_baselines(args) -> int:
    import pandas as pd

    from vpdl.slm.baselines import run_baselines
    examples = pd.read_parquet(args.examples)
    features = None
    if args.records and "structured_lr" in (args.kinds or []):
        from vpdl.slm.build.features import feature_frame
        from vpdl.slm.build.records import load_tables
        tables = load_tables(args.records, ["variants", "documents"])
        features = feature_frame(examples, tables["variants"], tables["documents"])
    results = run_baselines(examples, args.out, tuple(args.kinds), features, backend=args.backend,
                            seed=args.seed)
    _print({kind: {split: value.get("five_class", {}).get("macro_f1")
                   for split, value in entry.items() if isinstance(value, dict) and "five_class" in value}
            for kind, entry in results.items() if kind in args.kinds}
           | {"written": str(Path(args.out) / "baselines.json")})
    return 0


def cmd_evaluate(args) -> int:
    import numpy as np
    import pandas as pd

    from vpdl.slm.evaluation.calibration import calibration_summary
    from vpdl.slm.evaluation.metrics import binary_metrics, five_class_metrics, pathogenic_score
    from vpdl.slm.evaluation.uncertainty import evaluate_uncertainty
    from vpdl.slm.schema import CLASSES
    predictions = pd.read_parquet(args.predictions)
    examples = pd.read_parquet(args.examples)[["example_id", "target_label", "target_binary", "gene"]]
    merged = predictions.merge(examples, on="example_id", how="left")
    probs = merged[[f"p_{c}" for c in CLASSES]].to_numpy()
    lookup = {name: i for i, name in enumerate(CLASSES)}
    y = merged["target_label"].map(lambda v: lookup.get(v, -1) if isinstance(v, str) else -1).to_numpy()
    out: dict[str, Any] = {}
    for split, rows in merged.groupby("split"):
        index = rows.index.to_numpy()
        entry = {"n": int(len(index)), "five_class": five_class_metrics(y[index], probs[index]),
                 "calibration": calibration_summary(probs[index], y[index]),
                 "uncertainty": evaluate_uncertainty(probs[index], y[index])}
        binary = pd.to_numeric(rows["target_binary"], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(binary)
        if usable.sum() and len(np.unique(binary[usable])) == 2:
            entry["binary"] = binary_metrics(binary[usable], pathogenic_score(probs[index])[usable],
                                             args.threshold, is_validation=str(split).endswith("val"))
        out[str(split)] = entry
    if args.vus_old and args.vus_new:
        from vpdl.slm.evaluation.vus import ranking_metrics, reclassification_outcomes, vus_scores
        old = pd.read_parquet(args.vus_old)
        new = pd.read_parquet(args.vus_new)
        outcomes = reclassification_outcomes(old, new)
        scores = vus_scores(probs)
        joined = merged.assign(priority=scores["priority"]).merge(
            outcomes, left_on="variant_id" if "variant_id" in merged else "example_id",
            right_on="variant_id", how="inner")
        out["vus_reclassification"] = ranking_metrics(joined["priority"], joined["outcome"])
    _print(out, args.out)
    return 0


def cmd_embed(args) -> int:
    import numpy as np
    import pandas as pd

    from vpdl.slm.interface import SLMRepresentation, write_slm_outputs
    from vpdl.slm.schema import CLASSES
    predictions = pd.read_parquet(args.predictions)
    if args.split:
        predictions = predictions.loc[predictions["split"] == args.split].reset_index(drop=True)
    embeddings = np.load(args.embeddings)
    if len(embeddings) != len(predictions):
        splits = sorted(predictions["split"].unique()) if "split" in predictions else []
        raise ValueError(
            f"{args.embeddings} has {len(embeddings)} rows for {len(predictions)} predictions"
            + (f"; the predictions hold splits {splits} — pass --split to choose the one the "
               "embedding file belongs to" if len(splits) > 1 else ""))
    examples = pd.read_parquet(args.examples).set_index("example_id")
    representations = []
    for row, record in enumerate(predictions.itertuples(index=False)):
        probabilities = {c: float(getattr(record, f"p_{c}")) for c in CLASSES}
        example = examples.loc[record.example_id] if record.example_id in examples.index else None
        variant_id = str(example["variant_id"]) if example is not None else str(record.example_id)
        representations.append(SLMRepresentation(
            variant_id=variant_id, embedding=embeddings[row],
            score=probabilities["pathogenic"] + probabilities["likely_pathogenic"],
            uncertainty=float(getattr(record, "entropy", float("nan"))),
            metadata={"gene": str(example["gene"]) if example is not None else "",
                      "model_version": args.model_version, "feature_version": args.feature_version,
                      "dataset_version": args.dataset_version, "split": str(record.split),
                      "uncertainty_method": "predictive_entropy"},
            class_probabilities=probabilities))
    paths = write_slm_outputs(representations, args.out, name=args.name)
    _print({k: str(v) for k, v in paths.items()} | {"records": len(representations)})
    return 0


def cmd_sizing(args) -> int:
    from vpdl.slm.modeling.sizing import size_report
    memory = args.memory_gib
    if memory is None:
        from vpdl.slm.hardware import hardware_report
        memory = hardware_report().get("checks", {}).get("memory_total_gib")
    _print(size_report(args.corpus_tokens, args.examples, memory, tflops=args.tflops,
                       context=args.context, micro_batch=args.micro_batch), args.out)
    return 0


def cmd_experiments(args) -> int:
    from vpdl.slm.experiments import EXPERIMENTS, experiment, experiments_markdown
    if args.id:
        _print(experiment(args.id).as_dict(), args.out)
    elif args.markdown:
        _print(experiments_markdown(), args.out)
    else:
        _print([e.as_dict() for e in EXPERIMENTS], args.out)
    return 0


def cmd_smoke(args) -> int:
    from vpdl.slm.smoke import run_smoke
    result = run_smoke(args.out, seed=args.seed)
    _print(result)
    return 0 if result.get("ok") else 1


# -- parser --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vpdl-slm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, function, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(function=function)
        p.add_argument("--out", default=None, help="write the result here instead of stdout")
        return p

    p = command("hardware", cmd_hardware, "detected hardware vs the intended DGX Spark")
    p.add_argument("--smoke", action="store_true", help="also build and run a tiny model")

    p = command("catalog", cmd_catalog, "the knowledge-source catalogue")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--hash", action="store_true", help="hash every local file (slow)")
    p.add_argument("--root", default=".")

    p = command("inventory", cmd_inventory, "what data already exists here, measured")
    p.add_argument("--root", default=".")
    p.add_argument("--limit", type=int, default=None, help="stop after N variants (a quick look)")
    p.add_argument("--no-clinvar", action="store_true", help="skip the variant_summary pass")

    p = command("build-records", cmd_build_records, "ClinVar files -> the SLM's tables")
    p.add_argument("--variant-summary", required=True)
    p.add_argument("--submission-summary", default=None, help="the narratives (Description column)")
    p.add_argument("--citations", default=None, help="var_citations.txt")
    p.add_argument("--erepo", default=None, help="ClinGen Evidence Repository export")
    p.add_argument("--dl-canonical", default=None, help="the DL branch's canonical table, for joins")
    p.add_argument("--genes", nargs="*", default=None)
    p.add_argument("--limit-documents", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(out="data/slm_genomic")

    p = command("stats", cmd_stats, "corpus quantification")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--markdown", default=None, help="also write a markdown summary here")
    p.add_argument("--tokenizer", default=None, help="backbone whose tokenizer counts tokens")

    p = command("dedup", cmd_dedup, "duplicate / near-duplicate / template clusters")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--near", type=float, default=0.8)
    p.add_argument("--template", type=float, default=0.6)
    p.set_defaults(out="data/slm_genomic/clusters.parquet")

    p = command("splits", cmd_splits, "one split scheme")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--scheme", required=True)
    p.add_argument("--clusters", default=None)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--cutoff", default=None, help="temporal: ISO date")
    p.add_argument("--temporal-mode", default="novel", choices=["novel", "reinterpretation"])
    p.add_argument("--functional-variants", nargs="*", default=None)
    p.add_argument("--use-template-clusters", action="store_true")
    p.add_argument("--summary", default=None)
    p.set_defaults(out="data/slm_genomic/splits/split.parquet")

    p = command("roles", cmd_roles, "roles, reserved variants and pretraining exclusions")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--no-strict-literature", action="store_true")
    p.set_defaults(out="data/slm_genomic/pretrain_exclusions.json")

    p = command("examples", cmd_examples, "task examples under the task's leakage policy")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--split", required=True)
    p.add_argument("--task", default="classify")
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--holdout-publications", default="auto")
    p.set_defaults(out="data/slm_genomic/examples/classify.parquet")

    p = command("leakage", cmd_leakage, "the fifteen-check leakage audit")
    p.add_argument("--records", default="data/slm_genomic")
    p.add_argument("--examples", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--clusters", default=None)
    p.add_argument("--features", nargs="*", default=None)
    p.add_argument("--functional-variants", nargs="*", default=None)
    p.add_argument("--cutoff", default=None)
    p.add_argument("--pretraining", default=None, help="pretraining exclusions / manifest JSON")
    p.add_argument("--teacher", default=None)
    p.add_argument("--retrieval", default=None, help="JSON list of document ids in the index")
    p.add_argument("--strict-literature", action="store_true")
    p.add_argument("--no-strict-functional", action="store_true")

    p = command("pretrain-corpus", cmd_pretrain_corpus, "continued-pretraining text")
    p.add_argument("--kb", default=None, help="data/kb (knowledge-base passages)")
    p.add_argument("--pubmed", default=None, help="data/raw/pubmed")
    p.add_argument("--records", default=None)
    p.add_argument("--exclusions", default=None)
    p.add_argument("--split", default=None, help="a split, to take TRAINING documents from")
    p.add_argument("--include-narratives", action="store_true")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(out="data/slm_genomic/pretrain_corpus")

    p = command("pretrain-pack", cmd_pretrain_pack, "corpus -> token files for one backbone")
    p.add_argument("--corpus", default="data/slm_genomic/pretrain_corpus")
    p.add_argument("--backbone", required=True)
    p.add_argument("--limit-documents", type=int, default=None)
    p.set_defaults(out="data/slm_genomic/pretrain_tokens")

    p = command("pretrain", cmd_pretrain, "broad genomic continued pretraining")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, help="measure N steps and save nothing")

    p = command("finetune", cmd_finetune, "supervised multi-task training")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-dir", default=None)

    p = command("baselines", cmd_baselines, "majority / TF-IDF / structured-only baselines")
    p.add_argument("--examples", required=True)
    p.add_argument("--records", default=None)
    p.add_argument("--kinds", nargs="*", default=["majority", "tfidf_lr", "tfidf_svm"])
    p.add_argument("--backend", default="scipy", choices=["scipy", "sklearn"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(out="runs/slm_genomic/baselines")

    p = command("evaluate", cmd_evaluate, "metrics from saved predictions")
    p.add_argument("--predictions", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--vus-old", default=None, help="variants table of an older ClinVar release")
    p.add_argument("--vus-new", default=None)

    p = command("embed", cmd_embed, "export SLMRepresentation records")
    p.add_argument("--predictions", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--split", default=None, help="the split the embedding file holds, e.g. test")
    p.add_argument("--name", default="slm_outputs")
    p.add_argument("--model-version", required=True)
    p.add_argument("--feature-version", required=True)
    p.add_argument("--dataset-version", default=None)
    p.set_defaults(out="runs/slm_genomic/export")

    p = command("sizing", cmd_sizing, "model-size candidates for this corpus and machine")
    p.add_argument("--corpus-tokens", type=float, default=None)
    p.add_argument("--examples", type=int, default=None)
    p.add_argument("--memory-gib", type=float, default=None)
    p.add_argument("--tflops", type=float, default=30.7)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=16)

    p = command("experiments", cmd_experiments, "the experiment matrix")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--id", default=None)

    p = command("smoke", cmd_smoke, "whole pipeline on tiny synthetic data")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(out=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure(args.verbose)
    try:
        return args.function(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
