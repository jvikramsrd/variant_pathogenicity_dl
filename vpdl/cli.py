"""One entry point. v1 had 23 separate scripts; this replaces them.

    vpdl device                          what hardware am I actually on
    vpdl sources                         what can be loaded, and what it gives
    vpdl build --sources clinvar         assemble one source into a table
    vpdl build --sources all             assemble the pooled table
    vpdl train --data T --model gbm      leave-one-gene-out over a built table
    vpdl compare --runs runs/            pool cells, provenance-gated
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
    paired.set_defaults(func=cmd_paired)

    compare = sub.add_parser("compare", help="pool cells, provenance-gated")
    compare.add_argument("--runs", default="runs")
    compare.add_argument("--force", action="store_true",
                         help="pool despite a provenance mismatch (records it)")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
