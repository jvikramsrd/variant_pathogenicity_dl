"""One entry point. v1 had 23 separate scripts; this replaces them.

    vpdl device                          what hardware am I actually on
    vpdl sources                         what can be loaded, and what it gives
    vpdl build --sources clinvar         assemble one source into a table
    vpdl build --sources all             assemble the pooled table
    vpdl train --data T --model gbm      leave-one-gene-out over a built table
    vpdl compare --runs runs/            pool cells, provenance-gated
    vpdl kb-build / kb-ask / kb-eval     local knowledge base (docs/kb/)
    vpdl slm-corpus / -tokenizer / -pack / -train   small LM from scratch (docs/slm/)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("vpdl")

SOURCE_MODULES = {
    "clinvar": "vpdl.sources.clinvar",
    "pg_dms": "vpdl.sources.proteingym_dms",
    "alphamissense": "vpdl.sources.alphamissense",
    "gnomad": "vpdl.sources.gnomad",
    "uniprot": "vpdl.sources.uniprot",
}

# Sources fetched from an API rather than read from a local file.
API_SOURCES = {"gnomad"}

# The filenames each release is published under, so `--input-dir data/raw` is
# enough on its own. Override per source with --files '{"clinvar": "..."}'.
DEFAULT_FILES = {
    "clinvar": "variant_summary.txt.gz",
    "alphamissense": "AlphaMissense_aa_substitutions.tsv.gz",
    "pg_dms": "DMS_ProteinGym_substitutions.zip",
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_device(_: argparse.Namespace) -> int:
    from vpdl.device import log_summary
    info = log_summary()
    print(json.dumps(info.as_dict(), indent=2))
    if info.kind == "cpu" and info.gpu_present and info.torch_version is None:
        print(
            f"\nGPU present ({info.gpu_name}), but PyTorch is not installed, so "
            "no model can use it yet. That is expected before runbook Phase 10; "
            "the gbm arm does not need it.", file=sys.stderr,
        )
    elif info.kind == "cpu" and info.gpu_present:
        print(
            f"\nGPU present ({info.gpu_name}), but PyTorch {info.torch_version} "
            "cannot reach it. On aarch64 the default PyPI wheel is CPU-only — "
            "install a CUDA build or use NVIDIA's container (runbook Phase 10).",
            file=sys.stderr,
        )
    elif info.kind == "cpu":
        print(
            "\nNo GPU visible to the driver. Tabular arms (gbm) run fine here; "
            "the mlp and bilstm arms will be slow.", file=sys.stderr,
        )
    return 0


def cmd_sources(_: argparse.Namespace) -> int:
    import importlib
    for name, module_path in sorted(SOURCE_MODULES.items()):
        try:
            module = importlib.import_module(module_path)
            capabilities = module.provides()
            role = "labels" if capabilities.supplies_labels else "features only"
            print(f"{name:<16} {role:<14} licence={capabilities.licence}")
            if capabilities.feature_columns:
                print(f"{'':<16} columns: {', '.join(capabilities.feature_columns)}")
            if capabilities.notes:
                print(f"{'':<16} note: {capabilities.notes}")
        except Exception as error:                      # noqa: BLE001
            print(f"{name:<16} UNAVAILABLE ({error})")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    import importlib
    import pandas as pd
    from vpdl.assemble import assemble
    from vpdl.provenance import write_manifest
    from vpdl.sources.uniprot import MMR_ACCESSIONS, load_sequences

    requested = (sorted(SOURCE_MODULES) if args.sources == ["all"] else args.sources)
    cache = Path(args.cache_dir)
    sequences = load_sequences(cache / "uniprot")

    frames: dict[str, pd.DataFrame] = {}
    for name in requested:
        if name == "uniprot":
            continue                                  # coordinate authority, not a row source
        module = importlib.import_module(SOURCE_MODULES[name])

        if name in API_SOURCES:
            frames[name] = module.load(
                cache / name,
                genes=list(MMR_ACCESSIONS),
                uniprot_by_gene={g: a for g, (a, _) in MMR_ACCESSIONS.items()},
            )
            continue

        filename = (args.files or {}).get(name) or DEFAULT_FILES.get(name)
        path = Path(args.input_dir) / filename if filename else None
        if path is None or not path.exists():
            # A requested source that cannot be read is a failed build. Skipping
            # it would silently produce a table missing a whole source — which
            # is the pooled-vs-individual experiment's independent variable.
            print(f"ERROR: --sources includes '{name}' but {path} does not exist.",
                  file=sys.stderr)
            return 2

        uniprot_by_gene = {g: a for g, (a, _) in MMR_ACCESSIONS.items()}

        # Each source's load() takes the arguments it actually needs; a generic
        # call would silently mismatch, which is how alphamissense was
        # unreachable from this command until the 2026-09-20 review.
        if name == "clinvar":
            frames[name] = module.load(
                path, genes=list(MMR_ACCESSIONS),
                uniprot_by_gene=uniprot_by_gene,
                pms2_policy={"codon_range": (382, 862)},
            )
        elif name == "alphamissense":
            frames[name] = module.load(
                path,
                accessions=[a for a, _ in MMR_ACCESSIONS.values()],
                gene_by_accession={a: g for g, (a, _) in MMR_ACCESSIONS.items()},
            )
        else:
            frames[name] = module.load(path, uniprot_by_gene=uniprot_by_gene)

    out_path = Path(args.out)
    table, report = assemble(
        frames, sequences=sequences,
        expected_genes=list(MMR_ACCESSIONS) if args.assert_genes else None,
        out_path=out_path,
    )
    write_manifest(
        out_path.with_suffix(".manifest.json"), artefacts=[out_path],
        meta=report.as_dict(),
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def _parse_caps(items: Sequence[str] | None) -> tuple[tuple[str, int], ...]:
    """``["pg_dms=300"]`` -> ``(("pg_dms", 300),)``."""
    caps = []
    for item in items or ():
        source, sep, value = item.partition("=")
        if not sep or not source or not value.isdigit():
            raise ValueError(f"--train-cap expects SOURCE=N, got {item!r}")
        caps.append((source, int(value)))
    return tuple(caps)


def cmd_paired(args: argparse.Namespace) -> int:
    import pandas as pd
    from vpdl.analysis import feature_predictions, load_predictions, paired_table

    predictions = load_predictions(args.runs)
    if not args.cross_model:
        # One model per table by default. Comparing an MLP arm against the GBM
        # reference would mix a model effect into what should be a data effect.
        predictions = predictions[predictions["model"] == args.reference_model]

    extra = []
    if args.feature_baseline:
        if not args.data:
            print("ERROR: --feature-baseline needs --data (the assembled table).",
                  file=sys.stderr)
            return 2
        table = pd.read_csv(args.data, low_memory=False)
        extra = [feature_predictions(table, feature) for feature in args.feature_baseline]

    result = paired_table(predictions, (args.reference, args.reference_model),
                          n_bootstrap=args.n_bootstrap, extra_arms=extra)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Paired delta-AUC against {args.reference}:{args.reference_model} "
              f"({args.n_bootstrap} paired resamples; negative = worse than reference)\n")
        print(result.to_string(index=False))

    out = Path(args.runs) / f"paired_vs_{args.reference}_{args.reference_model}.csv"
    result.to_csv(out, index=False)
    print(f"\nwritten: {out}", file=sys.stderr)
    return 0


def cmd_pg_reproduce(args: argparse.Namespace) -> int:
    from vpdl.proteingym.bench import reproduce

    for path in (args.folds, args.published):
        if not Path(path).exists():
            print(f"ERROR: {path} not found — see docs/v2/DGX_RUNBOOK.md, "
                  "ProteinGym section, for the download commands.", file=sys.stderr)
            return 2

    table, summary = reproduce(args.folds, args.published, scheme=args.scheme,
                               model=args.model, alpha=args.alpha,
                               out_dir=args.out, only=args.only or None)

    compared = table.dropna(subset=["diff"])
    worst = compared.loc[compared["diff"].abs().sort_values(ascending=False).index].head(5)
    print(json.dumps(summary, indent=2))
    print("\nlargest disagreements with the published number:")
    print(worst[["DMS_id", "n", "ours", "published", "diff"]].to_string(index=False))
    return 0


def cmd_pg_combined(args: argparse.Namespace) -> int:
    from vpdl.device import log_summary
    from vpdl.proteingym.combined import run_comparison

    log_summary()

    for path in (args.scores, args.folds):
        if not Path(path).exists():
            print(f"ERROR: {path} not found — see docs/v2/DGX_RUNBOOK.md, "
                  "ProteinGym section, for the download commands.", file=sys.stderr)
            return 2
    for label, path in (("--reference", args.reference),
                        ("--published", args.published)):
        if not Path(path).exists():
            print(f"WARNING: {label} {path} not found; "
                  + ("sibling assays grouped by ID heuristic."
                     if label == "--reference" else "reading check skipped."),
                  file=sys.stderr)

    try:
        result, summary, reading = run_comparison(
            args.scores, args.folds,
            reference_csv=args.reference, published_zero_shot_csv=args.published,
            scheme=args.scheme, model=args.model, min_coverage=args.min_coverage,
            seed=args.seed, out_dir=args.out, only=args.only or None,
            device=args.device,
        )
    except RuntimeError as error:                 # a refused --device cuda
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if reading is not None and len(reading):
        print("reading check (our raw zero-shot Spearman vs ProteinGym's), worst 5:")
        print(reading.tail(5).to_string(index=False))
    print(json.dumps(summary, indent=2))
    ranked = result.dropna(subset=["delta"]).sort_values("delta")
    columns = ["DMS_id", "n", "has_siblings", "individual", "combined", "delta"]
    print("\ncombined helps most:")
    print(ranked.tail(5).iloc[::-1][columns].to_string(index=False))
    print("\ncombined hurts most:")
    print(ranked.head(5)[columns].to_string(index=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd
    from vpdl.device import log_summary
    from vpdl.experiment import SEQUENCE_WINDOW_MODELS, CellConfig, run_cell

    log_summary()
    table = pd.read_csv(args.data, low_memory=False)
    feature_columns = [c for c in table.columns if c.startswith("feature_")]
    if not feature_columns:
        print("No feature_* columns in the table.", file=sys.stderr)
        return 2

    sequences = None
    if args.model in SEQUENCE_WINDOW_MODELS:
        from vpdl.sources.uniprot import load_sequences
        sequences = load_sequences(Path(args.cache_dir) / "uniprot")

    try:
        caps = _parse_caps(args.train_caps)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    for seed in args.seeds:
        config = CellConfig(
            sources=tuple(args.sources),
            train_sources=tuple(args.train_sources),
            train_caps=caps,
            eval_source=args.eval_source,
            model=args.model, seed=seed,
            drop_groups=tuple(args.drop_groups or ()),
            n_bootstrap=args.n_bootstrap,
        )
        result = run_cell(table, config, feature_columns, args.data, args.out,
                          sequences=sequences)
        print(json.dumps(result.summary(), indent=2, default=str))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd
    from vpdl.provenance import assert_comparable

    summaries = sorted(Path(args.runs).glob("summary_*.json"))
    if not summaries:
        print(f"No summaries under {args.runs}", file=sys.stderr)
        return 2

    records = [json.loads(path.read_text()) for path in summaries]

    def refuse(error: Exception) -> int | None:
        print(f"REFUSED: {error}", file=sys.stderr)
        if not args.force:
            return 3
        print("Continuing anyway (--force).", file=sys.stderr)
        return None

    # Across arms: same table, same held-out test set. Within an arm: the seeds
    # must additionally share a feature schema — they are replicates.
    try:
        assert_comparable([r.get("provenance", {}) for r in records])
    except ValueError as error:
        if (code := refuse(error)) is not None:
            return code

    frame = pd.DataFrame([{
        "arm": r.get("arm", "+".join(r["sources"])),
        "model": r["model"],
        "seed": r["seed"],
        "provenance": r.get("provenance", {}),
        "mean_roc_auc_scoreable": r["mean_roc_auc_scoreable"],
    } for r in records])

    for (arm, model), group in frame.groupby(["arm", "model"]):
        try:
            assert_comparable(list(group["provenance"]), as_replicates=True)
        except ValueError as error:
            print(f"arm {arm}/{model}: seeds are not replicates.", file=sys.stderr)
            if (code := refuse(error)) is not None:
                return code
        if len(group) < 3:
            print(f"WARNING: {arm}/{model} has {len(group)} seed(s); three is the "
                  "floor for reading a difference.", file=sys.stderr)

    pooled = frame.groupby(["arm", "model"])["mean_roc_auc_scoreable"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    print(pooled.to_string(index=False))
    pooled.to_csv(Path(args.runs) / "comparison.csv", index=False)
    return 0


# -- knowledge base (docs/kb/) --------------------------------------------------

def _utf8_stdout() -> None:
    # Attributions carry the (c) and (R) signs GeneReviews' terms ask for.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_kb_build(args: argparse.Namespace) -> int:
    from vpdl.kb.genereviews import load_genereviews
    from vpdl.kb.ollama import LocalOllama
    from vpdl.kb.search import KnowledgeIndex
    from vpdl.kb.variants import build_variant_db

    try:
        return _kb_build(args, load_genereviews, LocalOllama, KnowledgeIndex, build_variant_db)
    except RuntimeError as error:                 # Ollama down, model missing, or on CPU
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def _kb_build(args, load_genereviews, LocalOllama, KnowledgeIndex, build_variant_db) -> int:
    kb = Path(args.kb)
    did_something = False
    if args.genereviews:
        for path in (args.genereviews, args.chapter_ids):
            if not Path(path).exists():
                print(f"ERROR: {path} not found - see docs/kb/RUNBOOK.md.", file=sys.stderr)
                return 2
        chunks = load_genereviews(args.genereviews, args.chapter_ids)
        embeddings = None
        if not args.no_embed:
            client = LocalOllama(require_gpu=not args.allow_cpu)
            embeddings = KnowledgeIndex.embed_chunks(
                chunks, lambda texts: client.embed(texts, args.embed_model))
        KnowledgeIndex(chunks, embeddings, None if args.no_embed else args.embed_model).save(kb)
        print(f"passages: {len(chunks)} -> {kb / 'chunks.jsonl'}"
              + ("" if args.no_embed else f"; embeddings: {args.embed_model}"))
        did_something = True
    if args.clinvar:
        if not Path(args.clinvar).exists():
            print(f"ERROR: {args.clinvar} not found.", file=sys.stderr)
            return 2
        genes = None if args.genes == ["all"] else args.genes
        rows = build_variant_db(args.clinvar, kb / "clinvar.sqlite", genes)
        print(f"ClinVar lookup: {rows} records -> {kb / 'clinvar.sqlite'}")
        did_something = True
    if not did_something:
        print("Nothing to build: pass --genereviews and/or --clinvar.", file=sys.stderr)
        return 2
    return 0


def _open_kb(args: argparse.Namespace):
    from vpdl.kb.search import KnowledgeIndex
    from vpdl.kb.variants import EvidenceTable, VariantDB

    index = KnowledgeIndex.load(args.kb)
    variant_path = Path(args.kb) / "clinvar.sqlite"
    variant_db = VariantDB(variant_path) if variant_path.exists() else None
    evidence = None
    if getattr(args, "evidence", None) and Path(args.evidence).exists():
        evidence = EvidenceTable(args.evidence)
    return index, variant_db, evidence


def cmd_kb_ask(args: argparse.Namespace) -> int:
    from vpdl.kb.answer import ask, render
    from vpdl.kb.ollama import LocalOllama

    _utf8_stdout()
    try:
        index, variant_db, evidence = _open_kb(args)
        answer = ask(" ".join(args.question), index,
                     LocalOllama(require_gpu=not args.allow_cpu), args.model,
                     k=args.k, variant_db=variant_db, evidence=evidence)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(render(answer, show_retrieved=args.show_retrieved))
    return 0


def cmd_kb_eval(args: argparse.Namespace) -> int:
    from vpdl.kb.evaluate import evaluate, load_questions
    from vpdl.kb.ollama import LocalOllama

    _utf8_stdout()
    try:
        index, variant_db, _ = _open_kb(args)
        summaries = evaluate(load_questions(args.questions), index,
                             LocalOllama(require_gpu=not args.allow_cpu),
                             args.models, args.out, k=args.k, variant_db=variant_db)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summaries, indent=2))
    print(f"\nPer-question answers for review: {args.out}/answers_<model>.jsonl",
          file=sys.stderr)
    return 0


# -- small language model from scratch (docs/slm/) --------------------------------

def cmd_slm_corpus(args: argparse.Namespace) -> int:
    from vpdl.slm.corpus import build_corpus

    if args.pubmed is None and args.kb is None:
        print("ERROR: pass --pubmed and/or --kb.", file=sys.stderr)
        return 2
    try:
        summary = build_corpus(args.out, pubmed_dir=args.pubmed, kb_dir=args.kb,
                               limit_files=args.limit_files, workers=args.workers)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


def cmd_slm_tokenizer(args: argparse.Namespace) -> int:
    from vpdl.slm.tokenizer import train_tokenizer

    _utf8_stdout()
    meta = train_tokenizer(args.corpus, args.out, vocab_size=args.vocab,
                           sample_bytes=args.sample_gb * 1e9)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def cmd_slm_pack(args: argparse.Namespace) -> int:
    from vpdl.slm.pack import pack

    meta = pack(args.corpus, args.tokenizer, args.out)
    print(json.dumps(meta["splits"], indent=2))
    return 0


def cmd_slm_train(args: argparse.Namespace) -> int:
    from vpdl.slm.model import DEFAULTS
    from vpdl.slm.train import TrainConfig, train

    defaults = DEFAULTS[args.size]
    config = TrainConfig(size=args.size, context=args.context, micro_batch=args.micro_batch,
                         total_tokens=int(args.tokens or defaults["tokens"]),
                         lr=args.lr or defaults["lr"], seed=args.seed, compile=args.compile,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_minutes=args.checkpoint_minutes,
                         keep_last=args.keep_last, milestone_every=args.milestone_every)
    try:
        result = train(args.data, args.out, config, benchmark_steps=args.benchmark,
                       resume_from=args.resume_from)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vpdl", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("device", help="report detected hardware").set_defaults(
        func=cmd_device)
    sub.add_parser("sources", help="list data sources").set_defaults(
        func=cmd_sources)

    build = sub.add_parser("build", help="assemble sources into a table")
    build.add_argument("--sources", nargs="+", required=True)
    build.add_argument("--input-dir", default="data/raw")
    build.add_argument("--cache-dir", default="data/cache")
    build.add_argument("--out", required=True)
    build.add_argument("--assert-genes", action="store_true", default=True)
    build.add_argument("--files", type=json.loads, default=None,
                       help='JSON map of source -> filename within --input-dir')
    build.set_defaults(func=cmd_build)

    train = sub.add_parser("train", help="leave-one-gene-out over a built table")
    train.add_argument("--data", required=True)
    train.add_argument("--sources", nargs="+", required=True,
                       help="what --data was BUILT from; recorded in provenance")
    train.add_argument("--train-sources", nargs="+", default=["clinvar"],
                       dest="train_sources",
                       help="label sources this arm TRAINS on — the experiment's "
                            "independent variable (e.g. clinvar | clinvar pg_dms | pg_dms)")
    train.add_argument("--eval-source", default="clinvar", dest="eval_source",
                       help="labels every arm is SCORED against; hold this fixed")
    train.add_argument("--train-cap", nargs="*", dest="train_caps", default=[],
                       metavar="SOURCE=N",
                       help="subsample rows labelled only by SOURCE to N, e.g. "
                            "pg_dms=300 — separates a source's labels from its volume")
    train.add_argument("--model", default="gbm", choices=["gbm", "mlp", "bilstm"])
    train.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train.add_argument("--drop-groups", nargs="*", dest="drop_groups")
    train.add_argument("--n-bootstrap", type=int, default=10_000)
    train.add_argument("--cache-dir", default="data/cache",
                       help="where UniProt sequences are cached (bilstm needs them)")
    train.add_argument("--out", default="runs")
    train.set_defaults(func=cmd_train)

    paired = sub.add_parser(
        "paired", help="delta-AUC vs a reference arm, paired on identical variants")
    paired.add_argument("--runs", default="runs")
    paired.add_argument("--reference", default="train-clinvar")
    paired.add_argument("--reference-model", default="gbm", dest="reference_model")
    paired.add_argument("--data", help="assembled table; needed for --feature-baseline")
    paired.add_argument("--feature-baseline", nargs="*", default=[],
                        dest="feature_baseline",
                        help="zero-training arms ranked by one raw feature, "
                             "e.g. feature_alphamissense_score")
    paired.add_argument("--n-bootstrap", type=int, default=10_000, dest="n_bootstrap")
    paired.add_argument("--cross-model", action="store_true", dest="cross_model",
                        help="also compare arms of OTHER models against the "
                             "reference (default: one model per table)")
    paired.set_defaults(func=cmd_paired)

    pg = sub.add_parser(
        "pg-reproduce",
        help="reproduce ProteinGym's published per-assay supervised results")
    pg.add_argument("--folds", default="data/raw/cv_folds_singles_substitutions.zip")
    pg.add_argument("--published",
                    default="data/raw/pg_supervised_spearman_fold_random_5.csv",
                    help="ProteinGym's per-assay Spearman table for the same scheme")
    pg.add_argument("--scheme", default="fold_random_5",
                    choices=["fold_random_5", "fold_modulo_5", "fold_contiguous_5"])
    pg.add_argument("--model", default="ohe", choices=["ohe"])
    pg.add_argument("--alpha", type=float, default=1.0)
    pg.add_argument("--only", nargs="*", default=[], metavar="DMS_ID",
                    help="restrict to these assays, e.g. MSH2_HUMAN_Jia_2020")
    pg.add_argument("--out", default="runs/pg")
    pg.set_defaults(func=cmd_pg_reproduce)

    pgc = sub.add_parser(
        "pg-combined",
        help="one model per assay vs one model on all assays, per ProteinGym assay")
    pgc.add_argument("--scores", default="data/raw/zero_shot_substitutions_scores.zip",
                     help="ProteinGym's zero-shot scores zip (the features)")
    pgc.add_argument("--folds", default="data/raw/cv_folds_singles_substitutions.zip")
    pgc.add_argument("--reference", default="data/raw/DMS_substitutions.csv",
                     help="ProteinGym reference file; maps each assay to its protein "
                          "so sibling assays are kept out of each other's training")
    pgc.add_argument("--published",
                     default="data/raw/DMS_substitutions_Spearman_DMS_level.csv",
                     help="published zero-shot Spearman, for the reading check")
    pgc.add_argument("--scheme", default="fold_random_5",
                     choices=["fold_random_5", "fold_modulo_5", "fold_contiguous_5"])
    pgc.add_argument("--model", default="ridge", choices=["ridge", "gbm"])
    pgc.add_argument("--min-coverage", type=float, default=0.9, dest="min_coverage",
                     help="use a zero-shot model as a feature only if it scores at "
                          "least this fraction of all variants")
    pgc.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                     help="auto: GPU if the model demonstrably trains there, else CPU "
                          "with a warning; cuda: GPU or stop")
    pgc.add_argument("--seed", type=int, default=0)
    pgc.add_argument("--only", nargs="*", default=[], metavar="DMS_ID")
    pgc.add_argument("--out", default="runs/pg")
    pgc.set_defaults(func=cmd_pg_combined)

    compare = sub.add_parser("compare", help="pool cells, provenance-gated")
    compare.add_argument("--runs", default="runs")
    compare.add_argument("--force", action="store_true",
                         help="pool despite a provenance mismatch (records it)")
    compare.set_defaults(func=cmd_compare)

    kb_build = sub.add_parser("kb-build", help="knowledge base: passages, embeddings, "
                                               "ClinVar lookup table")
    kb_build.add_argument("--kb", default="data/kb")
    kb_build.add_argument("--genereviews", default=None,
                          help="gene_NBK1116.tar.gz (or an unpacked folder)")
    kb_build.add_argument("--chapter-ids", default="data/raw/GRtitle_shortname_NBKid.txt",
                          dest="chapter_ids")
    kb_build.add_argument("--embed-model", default="bge-m3", dest="embed_model")
    kb_build.add_argument("--no-embed", action="store_true", dest="no_embed",
                          help="exact-word search only (no Ollama needed)")
    kb_build.add_argument("--clinvar", default=None, help="variant_summary.txt.gz")
    kb_build.add_argument("--genes", nargs="+", default=["MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"],
                          help="genes for the ClinVar lookup; 'all' for every gene")
    kb_build.add_argument("--allow-cpu", action="store_true", dest="allow_cpu",
                        help="run even if Ollama puts the model on the CPU "
                             "(default: stop, so a slow CPU run is never silent)")
    kb_build.set_defaults(func=cmd_kb_build)

    kb_ask = sub.add_parser("kb-ask", help="ask the knowledge base a question")
    kb_ask.add_argument("question", nargs="+")
    kb_ask.add_argument("--kb", default="data/kb")
    # Chosen by kb-eval (docs/RUNLOG.md, 2026-09-22): as accurate as qwen3:32b on
    # the development set, never withheld, 6x faster. llama3.1:8b is disqualified.
    kb_ask.add_argument("--model", default="medgemma:27b")
    kb_ask.add_argument("--k", type=int, default=6, help="passages given to the model")
    kb_ask.add_argument("--evidence", default="data/built/mmr.csv",
                        help="vpdl table with AlphaMissense/gnomAD values (optional)")
    kb_ask.add_argument("--show-retrieved", action="store_true", dest="show_retrieved")
    kb_ask.add_argument("--allow-cpu", action="store_true", dest="allow_cpu",
                        help="run even if Ollama puts the model on the CPU "
                             "(default: stop, so a slow CPU run is never silent)")
    kb_ask.set_defaults(func=cmd_kb_ask)

    kb_eval = sub.add_parser("kb-eval", help="score models on the evaluation questions")
    kb_eval.add_argument("--kb", default="data/kb")
    kb_eval.add_argument("--questions", default="docs/kb/eval_questions.jsonl")
    kb_eval.add_argument("--models", nargs="+", default=["medgemma:27b", "qwen3:32b"])
    kb_eval.add_argument("--k", type=int, default=6)
    kb_eval.add_argument("--out", default="runs/kb")
    kb_eval.add_argument("--allow-cpu", action="store_true", dest="allow_cpu",
                        help="run even if Ollama puts the model on the CPU "
                             "(default: stop, so a slow CPU run is never silent)")
    kb_eval.set_defaults(func=cmd_kb_eval)

    slm_corpus = sub.add_parser("slm-corpus", help="small LM: build the pretraining text")
    slm_corpus.add_argument("--pubmed", default=None, help="folder of pubmed*.xml.gz files")
    slm_corpus.add_argument("--kb", default=None, help="knowledge base folder (GeneReviews passages)")
    slm_corpus.add_argument("--out", default="data/slm/corpus")
    slm_corpus.add_argument("--limit-files", type=int, default=None, dest="limit_files",
                            help="read only the first N PubMed files (a quick trial run)")
    slm_corpus.add_argument("--workers", type=int, default=None,
                            help="processes reading PubMed files (default: all cores)")
    slm_corpus.set_defaults(func=cmd_slm_corpus)

    slm_tok = sub.add_parser("slm-tokenizer", help="small LM: train the vocabulary")
    slm_tok.add_argument("--corpus", default="data/slm/corpus")
    slm_tok.add_argument("--out", default="data/slm/tokenizer.json")
    slm_tok.add_argument("--vocab", type=int, default=32_000)
    slm_tok.add_argument("--sample-gb", type=float, default=2.0, dest="sample_gb",
                         help="train on an even sample of about this much text")
    slm_tok.set_defaults(func=cmd_slm_tokenizer)

    slm_pack = sub.add_parser("slm-pack", help="small LM: text -> token files")
    slm_pack.add_argument("--corpus", default="data/slm/corpus")
    slm_pack.add_argument("--tokenizer", default="data/slm/tokenizer.json")
    slm_pack.add_argument("--out", default="data/slm")
    slm_pack.set_defaults(func=cmd_slm_pack)

    slm_train = sub.add_parser("slm-train", help="small LM: pretrain from random weights")
    slm_train.add_argument("--data", default="data/slm")
    slm_train.add_argument("--out", default=None, help="default: runs/slm/<size>")
    slm_train.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    slm_train.add_argument("--tokens", type=float, default=None,
                           help="training tokens (default ~20 per parameter: small 2.5e9, medium 7e9)")
    slm_train.add_argument("--context", type=int, default=2048)
    slm_train.add_argument("--micro-batch", type=int, default=16, dest="micro_batch")
    slm_train.add_argument("--lr", type=float, default=None,
                           help="peak learning rate (default: small 6e-4, medium 3e-4)")
    slm_train.add_argument("--seed", type=int, default=0)
    slm_train.add_argument("--no-compile", action="store_false", dest="compile",
                           help="skip torch.compile (on by default: 1.5x faster on the DGX, "
                                "identical losses, 2026-09-22)")
    slm_train.add_argument("--checkpoint-every", type=int, default=250, dest="checkpoint_every",
                           help="save a full checkpoint every N steps...")
    slm_train.add_argument("--checkpoint-minutes", type=float, default=30.0,
                           dest="checkpoint_minutes",
                           help="...and at least this often, whichever comes first")
    slm_train.add_argument("--keep-last", type=int, default=3, dest="keep_last",
                           help="full checkpoints kept for rolling back")
    slm_train.add_argument("--milestone-every", type=int, default=1000, dest="milestone_every",
                           help="save weights for evaluation every N steps (0 = off)")
    slm_train.add_argument("--resume-from", default=None, dest="resume_from", metavar="CHECKPOINT",
                           help="roll back to this checkpoint (default: resume from the newest)")
    slm_train.add_argument("--benchmark", type=int, default=0, metavar="STEPS",
                           help="run this many steps, report speed and projected time, save nothing")
    slm_train.set_defaults(func=cmd_slm_train)

    args = parser.parse_args(argv)
    if getattr(args, "func", None) is cmd_slm_train and args.out is None:
        args.out = f"runs/slm/{args.size}"
    _configure_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
