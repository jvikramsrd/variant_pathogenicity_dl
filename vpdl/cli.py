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
    if info.kind == "cpu":
        print(
            "\nNo accelerator detected. Tabular arms (gbm) run fine here; "
            "the PLM and bilstm arms will be slow.", file=sys.stderr,
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

        filename = (args.files or {}).get(name)
        path = Path(args.input_dir) / filename if filename else None
        if path is None or not path.exists():
            logger.warning(
                "No input file for %s (pass --files '{\"%s\": \"<filename>\"}') "
                "— skipping.", name, name,
            )
            continue

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


def cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd
    from vpdl.experiment import CellConfig, run_cell

    table = pd.read_csv(args.data, low_memory=False)
    feature_columns = [c for c in table.columns if c.startswith("feature_")]
    if not feature_columns:
        print("No feature_* columns in the table.", file=sys.stderr)
        return 2

    for seed in args.seeds:
        config = CellConfig(
            sources=tuple(args.sources), model=args.model, seed=seed,
            drop_groups=tuple(args.drop_groups or ()),
            n_bootstrap=args.n_bootstrap,
        )
        result = run_cell(table, config, feature_columns, args.data, args.out)
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
    provenances = [record.get("provenance", {}) for record in records]

    try:
        assert_comparable(provenances)
    except ValueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        if not args.force:
            return 3
        print("Continuing anyway (--force).", file=sys.stderr)

    frame = pd.DataFrame([{
        "cell": r["cell"],
        "sources": "+".join(r["sources"]),
        "model": r["model"],
        "seed": r["seed"],
        "mean_roc_auc_scoreable": r["mean_roc_auc_scoreable"],
    } for r in records])

    pooled = frame.groupby(["sources", "model"])["mean_roc_auc_scoreable"].agg(
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
                       help="recorded in provenance; must describe --data")
    train.add_argument("--model", default="gbm", choices=["gbm", "mlp", "bilstm"])
    train.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train.add_argument("--drop-groups", nargs="*", dest="drop_groups")
    train.add_argument("--n-bootstrap", type=int, default=10_000)
    train.add_argument("--out", default="runs")
    train.set_defaults(func=cmd_train)

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
