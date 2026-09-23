"""``vpdl-dl`` — the DL branch's commands (``python -m vpdl.dl`` works too).

Kept separate from ``vpdl/cli.py`` on purpose: that file also hosts the LLM
branch's commands, which this work does not touch.

    hardware     what the machine offers (scripts/check_hardware.py)
    canonical    assembled table -> canonical variant table + QC report
    leakage      leakage report for a split scheme; exit 3 on critical findings
    structure    WT (+ precomputed VT) structural feature columns
    genomic      exon-structure feature columns (+ PMS2 range cross-check)
    embed        PLM WT/VT embeddings -> feature store
    zeroshot     masked-marginal / wild-type-marginal scores -> feature store
    corpus       MMR pretraining corpus (strict per-fold or transductive)
    pretrain     continued / variant-aware pretraining arm P0-P4
    train        one DL cell through vpdl.experiment.run_cell (same protocol)
    calibrate    fit calibrators on inner-validation predictions, apply, report
    functional   independent functional validation of held-out-gene scores
    failure      failure-analysis report
    arms         exact arm/model names in a runs directory (for vpdl paired)
    results      every run -> paper tables (CSV/Markdown/LaTeX) + checks + manifest
    export       DL output records (frozen schema) for later integration

Most options can come from a TOML file (``--config``); explicit flags win.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("vpdl.dl")

GENES = ("MLH1", "MSH2", "MSH6", "PMS2")


# -- shared helpers --------------------------------------------------------------

def _table(path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)


def _sequences(args) -> dict[str, str]:
    """Canonical sequences: the canonical table's sidecar, else the UniProt cache,
    else (offline) AlphaFold metadata. Lengths are always checked."""
    from vpdl.sources.uniprot import MMR_ACCESSIONS, load_sequences

    data = getattr(args, "data", None)
    if data:
        sidecar = Path(data).with_suffix("").as_posix() + ".sequences.json"
        if Path(sidecar).exists():
            payload = json.loads(Path(sidecar).read_text())
            return {acc: entry["sequence"] for acc, entry in payload.items()}
    alphafold = getattr(args, "sequences_from", None)
    if alphafold:
        out = {}
        for gene, (acc, length) in MMR_ACCESSIONS.items():
            meta = json.loads((Path(alphafold) / f"{acc}_metadata.json").read_text())
            sequence = meta.get("uniprotSequence") or meta["sequence"]
            if len(sequence) != length:
                raise ValueError(f"{acc}: {len(sequence)} residues, pinned {length}")
            out[acc] = sequence
        return out
    return load_sequences(Path(args.cache_dir) / "uniprot")


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataset_version(path) -> str:
    from vpdl.provenance import file_sha256
    return file_sha256(path)


def _track(kind, config, **kwargs):
    from vpdl.dl.tracking import append_registry, run_record
    record = run_record(kind, config, **kwargs)
    append_registry(record)
    logger.info("registry: %s %s", record["experiment_id"], record["run_id"])
    return record


def _load_toml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import tomllib
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _explicit(parser, argv: Sequence[str]) -> set[str]:
    """Destinations of options actually typed on the command line."""
    typed = {token.split("=", 1)[0] for token in argv if token.startswith("-")}
    return {action.dest for action in parser._actions                 # noqa: SLF001
            if typed & set(action.option_strings)}


def _apply_config(args, parser, argv: Sequence[str] = ()) -> None:
    """TOML values fill every option not typed on the command line."""
    values = _load_toml(getattr(args, "config", None))
    explicit = _explicit(parser, argv)
    for key, value in values.items():
        name = key.replace("-", "_")
        if not hasattr(args, name):
            raise SystemExit(f"--config: unknown option {key!r}")
        if name not in explicit:
            setattr(args, name, value)


def _write_sidecar(out_path, sequences) -> None:
    """Derived tables carry the canonical sequences forward (same format)."""
    from vpdl.dl.homology import sequence_sha
    sidecar = Path(Path(out_path).with_suffix("").as_posix() + ".sequences.json")
    sidecar.write_text(json.dumps({acc: {"sequence": seq, "sha256_12": sequence_sha(seq)}
                                   for acc, seq in sorted(sequences.items())}, indent=1))


def _register_fold(blocks_file, fold, location) -> None:
    """Record ``fold -> family/model_tag/key`` for ``perfold=`` embedding specs."""
    if not blocks_file:
        return
    if not fold:
        raise SystemExit("--blocks-file needs --fold (the held-out gene this backbone excludes)")
    path = Path(blocks_file)
    mapping = json.loads(path.read_text()) if path.exists() else {}
    mapping[fold] = location
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"per-fold mapping {path}: {fold} -> {location}  "
          f"(use --embedding-blocks 'perfold={path}:<level>.<pairing>')")


def _point_latest(store, name: str, location: str) -> Path:
    """``features/latest/<name>.txt`` holds the newest entry's location, so shell
    scripts can write ``$(cat features/latest/esm2_650m@P0-full.txt)``."""
    pointer = Path(store) / "latest" / f"{name}.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(location)
    print(f"pointer: {pointer} -> {location}")
    return pointer


def _rows(table, which: str):
    import pandas as pd

    if which == "all":
        return table
    labelled = pd.Series(False, index=table.index)
    for column in (c for c in table.columns if c.startswith("label__")):
        labelled |= table[column].notna()
    if which == "labelled":
        return table[labelled]
    functional = table.get("functional_assay_available", pd.Series(False, index=table.index))
    if which == "labelled+functional":
        return table[labelled | functional.astype(bool)]
    raise SystemExit(f"--rows must be labelled | labelled+functional | all, got {which}")


# -- commands -----------------------------------------------------------------------

def cmd_hardware(args) -> int:
    from vpdl.dl.hardware import hardware_report
    report = hardware_report(smoke=args.smoke)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_canonical(args) -> int:
    import pandas as pd

    from vpdl.dl.canonical import (build_canonical, load_confirmations, load_expert_labels,
                                   validate_canonical, write_canonical)
    from vpdl.dl import functional as fn
    from vpdl.sources.uniprot import MMR_ACCESSIONS

    table = _table(args.table)
    sequences = _sequences(args)
    uniprot_by_gene = {g: a for g, (a, _) in MMR_ACCESSIONS.items()}
    clinvar = gnomad = None
    if args.clinvar:
        from vpdl.sources.clinvar import iter_variant_summary
        clinvar = iter_variant_summary(args.clinvar, GENES, min_stars=0)
    if args.gnomad_cache:
        from vpdl.sources.gnomad import iter_gnomad_records
        gnomad = iter_gnomad_records(args.gnomad_cache, GENES)
    parts = []
    if args.cimra:
        parts.append(fn.load_cimra(args.cimra, uniprot_by_gene))
    if args.pg_dms:
        parts.append(fn.load_proteingym_continuous(args.pg_dms, uniprot_by_gene, GENES))
    for spec in args.mavedb or ():
        path, gene, urn, direction = spec.split(",")
        parts.append(fn.load_mavedb_scores(path, gene, uniprot_by_gene[gene], urn,
                                           higher_is_damaging=direction == "damaging"))
    functional = pd.concat(parts, ignore_index=True) if parts else None
    structures = None
    if args.structures:
        from vpdl.dl.structure import structure_index
        structures = structure_index(args.structures, sequences)
    canonical, report, flagged = build_canonical(
        table, sequences, clinvar_records=clinvar, gnomad_records=gnomad,
        functional=functional, structure_index=structures,
        expert=load_expert_labels(args.expert) if args.expert else None,
        confirmations=load_confirmations(args.confirmations) if args.confirmations else None,
        min_stars=args.min_stars)
    problems = validate_canonical(canonical)
    if problems:
        print("REFUSED — canonical table failed validation:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        return 3
    extra = []
    if functional is not None:
        from vpdl.splits import variant_keys
        key = ["uniprot_id", "position", "wt_aa", "mut_aa"]
        if "feature_alphamissense_score" in table.columns:
            anchor = functional[key].merge(
                table[key + ["feature_alphamissense_score"]].drop_duplicates(key),
                on=key, how="left")["feature_alphamissense_score"]
            orientation = fn.assert_functional_orientation(functional.reset_index(drop=True),
                                                           anchor)
            print(f"functional orientation vs AlphaMissense (Spearman): {orientation}")
        else:
            print("WARNING: no AlphaMissense column — functional orientation unverified",
                  file=sys.stderr)
        path = Path(args.out).with_suffix("").as_posix() + ".functional.csv"
        functional.assign(variant_key=variant_keys(functional)).to_csv(path, index=False)
        extra.append(Path(path))
    paths = write_canonical(canonical, report, flagged, sequences, args.out,
                            meta={"assembled_table": str(args.table),
                                  "assembled_sha256": _dataset_version(args.table)},
                            extra_artefacts=extra)
    print(json.dumps(report.as_dict(), indent=2, default=str))
    print(f"\nwritten: {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
    return 0


def _work(table, train_sources, eval_source):
    from vpdl.assemble import resolve_labels
    train = resolve_labels(table, train_sources)
    work = table.assign(_train=train, _eval=table[f"label__{eval_source}"])
    return work[work["_train"].notna() | work["_eval"].notna()].reset_index(drop=True)


def _columns(table, drop_groups, allow_proxy_leak=False):
    from vpdl.features import drop_gene_constant, resolve_ablation
    features = [c for c in table.columns if c.startswith("feature_")]
    return drop_gene_constant(table, resolve_ablation(features, list(drop_groups),
                                                      allow_proxy_leak=allow_proxy_leak))


def cmd_leakage(args) -> int:
    from vpdl.dl.leakage import run_checks
    from vpdl.dl.splits import make_folds

    table = _table(args.data)
    sequences = _sequences(args)
    work = _work(table, args.train_sources, args.eval_source)
    columns = _columns(table, _drop_for(args.modalities, args.drop_groups))
    folds = make_folds(work, args.split)
    functional_keys = (set(table.loc[table["functional_assay_available"], "variant_id"])
                       if args.functional and "functional_assay_available" in table else set())
    report = run_checks(work, folds, args.split, columns, args.train_sources,
                        args.eval_source, sequences=sequences,
                        validating_functional=args.functional, functional_keys=functional_keys)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"leakage_{args.split}.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / f"leakage_{args.split}.json").write_text(report.to_json(), encoding="utf-8")
    print(report.to_markdown())
    return 3 if report.critical else 0


def _drop_for(modalities: str | None, drop_groups: Sequence[str] | None) -> list[str]:
    """``--modalities population,structure`` -> drop every other feature group."""
    from vpdl.dl.runner import MODALITY_OF_GROUP
    from vpdl.features import PRIOR_GROUPS

    drops = set(drop_groups or ())
    if modalities is not None:
        wanted = {m.strip() for m in modalities.split(",") if m.strip() and m.strip() != "none"}
        known = {MODALITY_OF_GROUP[g] for g in PRIOR_GROUPS}
        if unknown := wanted - known:
            raise SystemExit(f"unknown modalities {sorted(unknown)}; tabular modalities are "
                             f"{sorted(known)} (embeddings come from --embedding-blocks)")
        drops |= {g for g in PRIOR_GROUPS if MODALITY_OF_GROUP[g] not in wanted}
    return sorted(drops)


def cmd_structure(args) -> int:
    from vpdl.dl.structure import (PrecomputedVariantStructures, WT_STRUCTURE_COLUMNS,
                                   wt_structure_table)
    from vpdl.provenance import write_manifest

    table = _table(args.data)
    sequences = _sequences(args)
    residues, index = wt_structure_table(args.models, sequences, with_sasa=not args.no_sasa)
    out = table.drop(columns=[c for c in WT_STRUCTURE_COLUMNS if c in table.columns])
    out = out.merge(residues.drop(columns=["residue"]), on=["uniprot_id", "position"],
                    how="left")
    out["feature_struct_available"] = out["feature_struct_available"].fillna(0.0)
    if args.ddg or args.vt_models:
        provider = PrecomputedVariantStructures(args.models, args.ddg, args.vt_models)
        deltas = provider.deltas(out)
        for column in deltas.columns:
            out[column] = deltas[column].to_numpy()
    out.to_csv(args.out, index=False)
    _write_sidecar(args.out, sequences)
    per_residue = Path(args.out).with_suffix("").as_posix() + ".residues.csv"
    residues.to_csv(per_residue, index=False)
    write_manifest(Path(args.out).with_suffix("").as_posix() + ".manifest.json",
                   artefacts=[args.out, per_residue],
                   meta={"kind": "structure", "source_table": str(args.data),
                         "structures": {k: {kk: vv for kk, vv in v.items() if kk != "residues"}
                                        for k, v in index.items()}})
    print(json.dumps({acc: {k: v for k, v in entry.items() if k != "residues"}
                      for acc, entry in index.items()}, indent=2, default=str))
    return 0


def cmd_genomic(args) -> int:
    from vpdl.dl.genomic import (MANE_ENSEMBL, exon_codon_table, fetch_transcript,
                                 genomic_features, homology_codon_range)
    from vpdl.sources.clinvar import PMS2_HOMOLOGY_CODONS
    from vpdl.sources.uniprot import MMR_ACCESSIONS

    table = _table(args.data)
    lengths = {g: n for g, (_, n) in MMR_ACCESSIONS.items()}
    exon_tables = {}
    for gene, transcript_id in MANE_ENSEMBL.items():
        cache = Path(args.ensembl_cache) / f"{transcript_id}.json"
        if not cache.exists() and not args.fetch:
            raise SystemExit(f"{cache} missing; rerun with --fetch (network) first")
        exon_tables[gene] = exon_codon_table(fetch_transcript(transcript_id, args.ensembl_cache),
                                             lengths[gene])
    derived = homology_codon_range(exon_tables["PMS2"], lengths["PMS2"])
    print(f"PMS2CL homology codons from the exon table: {derived}; "
          f"constant in vpdl.sources.clinvar: {PMS2_HOMOLOGY_CODONS}")
    if tuple(derived) != tuple(PMS2_HOMOLOGY_CODONS):
        print("REFUSED: the derived range disagrees with the pinned constant.", file=sys.stderr)
        return 3
    features = genomic_features(table, exon_tables, lengths,
                                include_nucleotide=args.include_nucleotide)
    out = table.drop(columns=[c for c in features.columns if c in table.columns])
    out = out.join(features)
    out.to_csv(args.out, index=False)
    _write_sidecar(args.out, _sequences(args))
    from vpdl.provenance import write_manifest
    write_manifest(Path(args.out).with_suffix("").as_posix() + ".manifest.json",
                   artefacts=[args.out],
                   meta={"kind": "genomic", "source_table": str(args.data),
                         "transcripts": MANE_ENSEMBL, "pms2_homology_codons": list(derived),
                         "include_nucleotide": args.include_nucleotide})
    return 0


def cmd_embed(args) -> int:
    from vpdl.dl.feature_store import FeatureStore
    from vpdl.dl.homology import sequence_sha
    from vpdl.dl.plm.backbones import load_backbone
    from vpdl.dl.plm.embed import LEVELS, PAIRINGS, EmbeddingExtractor
    from vpdl.splits import variant_keys

    import numpy as np

    started = time.time()
    table = _table(args.data)
    sequences = _sequences(args)
    rows = _rows(table, args.rows)
    backbone = load_backbone(args.backbone, device=_device())
    adaptation = None
    if args.adapted_from:
        from vpdl.dl.plm.finetune import load_adapted_backbone
        adaptation = load_adapted_backbone(backbone, args.adapted_from)
    extractor = EmbeddingExtractor(backbone, args.policy, args.radius, args.batch_size,
                                   args.layer, args.chunk)
    ids, blocks = [], {}
    for accession, group in rows.groupby("uniprot_id"):
        variants = list(zip(group["position"].astype(int), group["wt_aa"], group["mut_aa"]))
        out = extractor.extract(sequences[accession], variants)
        ids += list(variant_keys(group))
        for name, array in out.items():
            blocks.setdefault(name, []).append(array)
    blocks = {name: np.concatenate(parts) for name, parts in blocks.items()}
    identity = {"model": backbone.spec.hf_id, "model_version": backbone.version,
                "sequence_version": {acc: sequence_sha(s) for acc, s in sequences.items()},
                "dataset_version": _dataset_version(args.data), "policy": args.policy,
                "local_radius": args.radius, "layer": args.layer, "rows": args.rows,
                "pretrain_arm": args.arm, "adaptation": adaptation}
    family = "esm1b" if backbone.spec.name == "esm1b" else "esm2"
    tag = f"{backbone.spec.name}@{args.arm}" + (f"-{args.fold}" if args.fold else "")
    entry = FeatureStore(args.store).write(
        family, tag, identity, ids, blocks, dtype=args.dtype,
        extra={"context": {acc: extractor.describe(len(s)) for acc, s in sequences.items()}})
    location = f"{family}/{tag}/{entry.meta['key']}"
    _register_fold(args.blocks_file, args.fold, location)
    _point_latest(args.store, f"{tag}-{args.policy}", location)
    print(f"feature entry: {entry.path}")
    print("embedding blocks for `vpdl-dl train --embedding-blocks`:")
    for level in LEVELS:
        print("  " + "  ".join(f"{location}:{level}.{p}" for p in PAIRINGS))
    _track("embed", identity | {"store": args.store}, dataset_path=args.data,
           feature_version=location, model_version=backbone.version, started=started,
           artefacts={"entry": str(entry.path)})
    return 0


def cmd_zeroshot(args) -> int:
    import numpy as np

    from vpdl.dl.feature_store import FeatureStore
    from vpdl.dl.homology import sequence_sha
    from vpdl.dl.plm.backbones import load_backbone
    from vpdl.dl.plm.zeroshot import zeroshot_frame

    started = time.time()
    table = _table(args.data)
    sequences = _sequences(args)
    rows = _rows(table, args.rows)
    backbone = load_backbone(args.backbone, device=_device())
    adaptation = None
    if args.adapted_from:
        from vpdl.dl.plm.finetune import load_adapted_backbone
        adaptation = load_adapted_backbone(backbone, args.adapted_from)
    scores = zeroshot_frame(backbone, rows, sequences, args.method, args.policy,
                            args.batch_size)
    identity = {"model": backbone.spec.hf_id, "model_version": backbone.version,
                "sequence_version": {acc: sequence_sha(s) for acc, s in sequences.items()},
                "dataset_version": _dataset_version(args.data), "method": args.method,
                "policy": args.policy, "rows": args.rows, "pretrain_arm": args.arm,
                "adaptation": adaptation}
    tag = f"{backbone.spec.name}@{args.arm}-{args.method}" + (f"-{args.fold}" if args.fold else "")
    entry = FeatureStore(args.store).write(
        "zeroshot", tag, identity, list(scores["variant_key"]),
        {"pathogenicity": scores[["pathogenicity"]].to_numpy(np.float32),
         "raw_llr": scores[["raw_llr"]].to_numpy(np.float32)}, dtype="float32")
    column = f"zeroshot_{backbone.spec.name}_{args.arm}_{args.method}"
    scores.to_csv(Path(entry.path) / "scores.csv", index=False)
    if args.table_out:
        from vpdl.splits import variant_keys
        # Accumulate: a second zero-shot run adds its column to the same copy.
        base = _table(args.table_out) if Path(args.table_out).exists() else table
        if set(variant_keys(base)) != set(variant_keys(table)):
            raise SystemExit(f"{args.table_out} holds different variants than {args.data}")
        lookup = scores.set_index("variant_key")["pathogenicity"]
        base[column] = lookup.reindex(variant_keys(base)).to_numpy()
        table = base
        table.to_csv(args.table_out, index=False)
        _write_sidecar(args.table_out, sequences)
        print(f"column {column} added (not a feature_ column: it never trains); "
              f"use `vpdl paired --feature-baseline {column} --data {args.table_out}`")
    _register_fold(args.blocks_file, args.fold, f"zeroshot/{tag}/{entry.meta['key']}")
    _point_latest(args.store, f"zeroshot-{tag}", f"zeroshot/{tag}/{entry.meta['key']}")
    print(f"feature entry: zeroshot/{tag}/{entry.meta['key']}:pathogenicity")
    _track("zeroshot", identity, dataset_path=args.data,
           feature_version=f"zeroshot/{tag}/{entry.meta['key']}",
           model_version=backbone.version, started=started)
    return 0


def cmd_corpus(args) -> int:
    from vpdl.dl.pretrain.corpus import CORPUS_QUERIES, build_corpus, fetch_uniprot_fasta
    from vpdl.sources.uniprot import MMR_ACCESSIONS

    raw = Path(args.raw_dir)
    fastas = list(args.fasta or [])
    if args.fetch:
        fastas += [fetch_uniprot_fasta(q, raw / f"{name}.fasta")
                   for name, q in CORPUS_QUERIES.items()]
    if not fastas:
        raise SystemExit("no input: pass --fasta files or --fetch")
    panel = _sequences(args)
    holdout = MMR_ACCESSIONS[args.holdout][0] if args.holdout else None
    report = build_corpus(fastas, panel, args.out, mode=args.mode, holdout=holdout,
                          max_identity=args.max_identity, seed=args.seed)
    print(json.dumps({k: v for k, v in report.__dict__.items() if k != "excluded"}, indent=2))
    return 0


def cmd_pretrain(args) -> int:
    from dataclasses import asdict

    from vpdl.dl.pretrain.run import PretrainConfig, pretrain

    started = time.time()
    config = PretrainConfig(arm=args.arm, backbone=args.backbone, corpus_dir=args.corpus,
                            objectives=json.loads(args.objectives) if args.objectives else {},
                            crop=args.crop, epochs=args.epochs, patience=args.patience,
                            batch_size=args.batch_size, grad_accum=args.grad_accum,
                            lr=args.lr, precision=args.precision, compile=args.compile,
                            seed=args.seed)
    if args.strategy:
        config.strategy = json.loads(args.strategy)
    summary = pretrain(config, args.out, resume=args.resume)
    _track("pretrain", asdict(config), seed=args.seed,
           model_version={"backbone": args.backbone, "arm": args.arm},
           metrics={k: summary.get(k) for k in ("best_val_mlm_loss", "val_mlm_perplexity")},
           checkpoint=summary.get("delta"), started=started)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("corpus", "fit")},
                     indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    from dataclasses import asdict

    from vpdl.dl.runner import DLContext
    from vpdl.experiment import CellConfig, run_cell

    table = _table(args.data)
    sequences = _sequences(args)
    functional_keys = (frozenset(table.loc[table["functional_assay_available"].astype(bool),
                                           "variant_id"])
                       if "functional_assay_available" in table else frozenset())
    context = DLContext(store_root=args.store, functional_keys=functional_keys,
                        export_dir=args.export_dir, sequences=sequences)
    feature_columns = [c for c in table.columns if c.startswith("feature_")]
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    tag = args.tag or ""
    if model_kwargs and not tag:
        import hashlib
        tag = hashlib.sha256(json.dumps(model_kwargs, sort_keys=True).encode()).hexdigest()[:6]
    for seed in args.seeds:
        started = time.time()
        config = CellConfig(
            sources=tuple(args.sources), train_sources=tuple(args.train_sources),
            eval_source=args.eval_source, model=args.model, seed=seed,
            drop_groups=tuple(_drop_for(args.modalities, args.drop_groups)),
            allow_proxy_leak=args.allow_proxy_leak, n_bootstrap=args.n_bootstrap,
            model_kwargs=model_kwargs, split=args.split,
            embedding_blocks=tuple(args.embedding_blocks or ()), score_rows=args.score_rows,
            tag=tag)
        _archive_superseded(args.out, config.slug)
        result = run_cell(table, config, feature_columns, args.data, args.out,
                          sequences=sequences, dl_context=context)
        summary = result.summary()
        _track("train", asdict(config), seed=seed, dataset_path=args.data,
               feature_version=summary["provenance"].get("feature_version"),
               model_version={"model": args.model, "tag": tag, "kwargs": model_kwargs},
               metrics={k: summary[k] for k in ("mean_roc_auc_all", "mean_roc_auc_scoreable")}
               | {"per_gene": {r["gene"]: {m: r[m] for m in ("roc_auc", "mcc", "pr_auc")}
                               for r in result.per_gene}},
               artefacts={"summary": str(Path(args.out) / f"summary_{config.slug}.json")},
               started=started)
        print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_calibrate(args) -> int:
    import numpy as np
    import pandas as pd

    from vpdl.dl.calibration import CALIBRATORS, calibration_report, plot_reliability, \
        reliability_bins

    runs = Path(args.runs)
    reports, calibrated_all = [], []
    for path in sorted(runs.glob("predictions_*.csv")):
        slug = path.stem[len("predictions_"):]
        valpath = runs / f"valpreds_{slug}.csv"
        if not valpath.exists():
            continue
        test, val = pd.read_csv(path), pd.read_csv(valpath)
        key = "fold" if "fold" in test.columns and "fold" in val.columns else "gene"
        bins = {}
        for fold, rows in test.groupby(key):
            fit_rows = val[val[key] == fold]
            if fit_rows["label"].nunique() < 2:
                continue
            calibrator = CALIBRATORS[args.method]().fit(fit_rows["label"], fit_rows["score"])
            rows = rows.assign(calibrated=calibrator.transform(rows["score"]))
            calibrated_all.append(rows.assign(cell=slug, method=args.method))
            for gene, sub in rows.groupby("gene"):
                before = calibration_report(sub["label"], sub["score"])
                after = calibration_report(sub["label"], sub["calibrated"])
                reports.append({"cell": slug, "gene": gene, "method": args.method,
                                **{f"raw_{k}": v for k, v in before.items()},
                                **{f"cal_{k}": v for k, v in after.items() if k != "n"}})
                bins[gene] = reliability_bins(sub["label"], sub["calibrated"], args.bins)
        plot_reliability(bins, runs / f"reliability_{slug}_{args.method}.png",
                         title=f"{slug} ({args.method})")
    if not reports:
        print("no predictions/valpreds pairs found", file=sys.stderr)
        return 2
    table = pd.DataFrame(reports)
    table.to_csv(runs / f"calibration_{args.method}.csv", index=False)
    pd.concat(calibrated_all).to_csv(runs / f"calibrated_predictions_{args.method}.csv",
                                     index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    return 0


def cmd_functional(args) -> int:
    import pandas as pd

    from vpdl.dl.functional import functional_validation

    functional = pd.read_csv(args.functional)
    rows = []
    for path in sorted(Path(args.runs).glob("scores_*.csv")):
        scores = pd.read_csv(path)
        if scores.empty:
            continue
        result = functional_validation(scores, functional, n_bootstrap=args.n_bootstrap)
        rows.append(result.assign(cell=path.stem[len("scores_"):]))
    if not rows:
        print("no scores_*.csv — train with --score-rows functional", file=sys.stderr)
        return 2
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(Path(args.runs) / "functional_validation.csv", index=False)
    print(table.to_string(index=False))
    return 0


def cmd_failure(args) -> int:
    import pandas as pd

    from vpdl.dl import failure

    table = _table(args.data)
    predictions = pd.read_csv(Path(args.runs) / f"predictions_{args.cell}.csv")
    valpreds = pd.read_csv(Path(args.runs) / f"valpreds_{args.cell}.csv")
    columns = _columns(table, [])
    sections = {
        "gene shortcut": failure.gene_shortcut(table, columns),
        "population shortcut": failure.population_shortcut(predictions, table)
        if "feature_gnomad_observed" in table else pd.DataFrame(),
        "overfitting": failure.overfitting(predictions, valpreds),
        "MSH6 context": failure.msh6_context(predictions),
        "class imbalance": failure.class_imbalance(predictions),
        "calibration by gene": failure.calibration_by_gene(predictions),
    }
    if args.paired:
        sections["pretraining regression"] = failure.pretraining_regression(
            pd.read_csv(args.paired))
    text = failure.failure_report(sections)
    out = Path(args.runs) / f"failure_{args.cell}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


def _archive_superseded(out_dir, slug: str) -> Path | None:
    """Move a cell's previous artefacts aside instead of overwriting them.

    Results only exist once; a re-run at different code or data must not delete
    the numbers a draft may already cite. The old files go to
    ``<out>/superseded/<UTC timestamp>/`` and are ignored by `vpdl-dl results`.
    """
    import shutil
    from datetime import datetime, timezone

    out_dir = Path(out_dir)
    existing = [p for prefix in ("summary_", "results_", "predictions_", "valpreds_", "scores_")
                for p in out_dir.glob(f"{prefix}{slug}.*")]
    if not (out_dir / f"summary_{slug}.json").exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out_dir / "superseded" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    print(f"previous run of {slug} moved to {archive}", file=sys.stderr)
    return archive


def cmd_results(args) -> int:
    from vpdl.dl.report import write_paper_bundle

    comparisons = None
    if args.compare:
        comparisons = []
        for spec in args.compare:
            try:
                runs_dir, rest = spec.split("=", 1)
                arm, model = rest.rsplit(":", 1)
            except ValueError:
                raise SystemExit(f"--compare expects DIR=ARM:MODEL, got {spec!r}")
            comparisons.append(((runs_dir, arm, model), None))
    bundle = write_paper_bundle(args.runs, args.out, comparisons=comparisons,
                               canonical_report=args.canonical_report,
                               leakage_dir=args.leakage, n_bootstrap=args.n_bootstrap,
                               plots=not args.no_plots)
    print(json.dumps({k: str(v) for k, v in bundle.files.items()}, indent=2))
    checks = bundle.checks
    print(f"\n{checks['n_cells']} cells, {checks['n_arms']} arms.", file=sys.stderr)
    for note in checks["notes"]:
        print(f"NOTE: {note}", file=sys.stderr)
    for problem in checks["problems"]:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    print(("READY TO CITE: no problems found." if checks["citable"] else
           "NOT READY TO CITE: fix the problems above (they are in checks.json)."),
          file=sys.stderr)
    return 0


def cmd_arms(args) -> int:
    """Exact (arm, model) names in a runs directory — for `vpdl paired --reference`."""
    import pandas as pd

    from vpdl.analysis import parse_cell

    rows = []
    for path in sorted(Path(args.runs).glob("summary_*.json")):
        arm, model, seed = parse_cell(json.loads(path.read_text())["cell"])
        rows.append({"arm": arm, "model": model, "seed": seed})
    if not rows:
        print(f"no summaries under {args.runs}", file=sys.stderr)
        return 2
    table = (pd.DataFrame(rows).groupby(["arm", "model"])["seed"]
             .apply(lambda s: " ".join(map(str, sorted(s)))).reset_index())
    with pd.option_context("display.width", 250, "display.max_colwidth", 200):
        print(table.to_string(index=False))
    return 0


def cmd_export(args) -> int:
    import numpy as np

    from vpdl.dl.interface import DLRepresentation, write_dl_outputs

    runs = Path(args.export_dir)
    summary = json.loads((Path(args.runs) / f"summary_{args.cell}.json").read_text())
    provenance = summary.get("provenance", {})
    representations = []
    for path in sorted(runs.glob(f"representations_{args.cell}__*.npz")):
        blob = np.load(path, allow_pickle=True)
        fold = path.stem.split("__")[-1]
        for i, variant in enumerate(blob["variant_keys"]):
            uncertainty = float(blob["uncertainty"][i])
            representations.append(DLRepresentation(
                str(variant), blob["representation"][i], float(blob["score"][i]),
                None if not np.isfinite(uncertainty) else uncertainty,
                {"gene": str(blob["gene"][i]), "fold": fold,
                 "uncertainty_method": "mc_dropout" if np.isfinite(uncertainty) else None,
                 "model_version": f"{args.cell}@{(provenance.get('git') or {}).get('commit')}",
                 "feature_version": provenance.get("feature_version") or "tabular-only",
                 "dataset_version": provenance.get("dataset_sha256")}))
    if not representations:
        print("no representation files — train with --export-dir", file=sys.stderr)
        return 2
    paths = write_dl_outputs(representations, args.out, name=f"dl_outputs_{args.cell}")
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


# -- parser -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vpdl-dl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    parser.subcommands = {}

    def command(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        parser.subcommands[name] = p
        p.set_defaults(func=func)
        p.add_argument("--config", default=None, help="TOML file of option defaults")
        p.add_argument("--cache-dir", default="data/cache", dest="cache_dir")
        p.add_argument("--sequences-from", default=None, dest="sequences_from",
                       help="AlphaFold metadata dir to read sequences from (offline)")
        return p

    p = command("hardware", cmd_hardware, "what this machine offers the DL branch")
    p.add_argument("--smoke", action="store_true")

    p = command("canonical", cmd_canonical, "canonical variant table + QC")
    p.add_argument("--table", required=True, help="assembled table from `vpdl build`")
    p.add_argument("--clinvar", help="variant_summary.txt.gz (all records, for QC)")
    p.add_argument("--gnomad-cache", dest="gnomad_cache", help="data/cache/gnomad")
    p.add_argument("--structures", help="AlphaFold model dir (<acc>.pdb + metadata)")
    p.add_argument("--cimra", help="CIMRA OddsPath CSV (validation only)")
    p.add_argument("--pg-dms", dest="pg_dms", help="ProteinGym zip (continuous score)")
    p.add_argument("--mavedb", nargs="*", help="path,GENE,urn,damaging|functional")
    p.add_argument("--expert", help="InSiGHT / ClinGen expert classification CSV")
    p.add_argument("--confirmations", help="PMS2 orthogonal confirmation CSV")
    p.add_argument("--min-stars", type=int, default=2, dest="min_stars")
    p.add_argument("--out", default="data/built/canonical.csv")

    def split_options(p):
        p.add_argument("--data", required=True, help="canonical table")
        p.add_argument("--split", default="logo",
                       choices=["logo", "logo_purged", "family", "random_debug"])
        p.add_argument("--train-sources", nargs="+", default=["clinvar"], dest="train_sources")
        p.add_argument("--eval-source", default="clinvar", dest="eval_source")
        p.add_argument("--modalities", default=None,
                       help="comma-separated tabular modalities to KEEP (population, structure, "
                            "genomic, annotation, external_priors, other; 'none' = no tabular)")
        p.add_argument("--drop-groups", nargs="*", dest="drop_groups", default=[])

    p = command("leakage", cmd_leakage, "leakage report; exit 3 on critical findings")
    split_options(p)
    p.add_argument("--functional", action="store_true",
                   help="check as if validating on functional-assay variants")
    p.add_argument("--out", default="runs/dl/leakage")

    p = command("structure", cmd_structure, "structural feature columns")
    p.add_argument("--data", required=True)
    p.add_argument("--models", default="data/mmr/raw/alphafold")
    p.add_argument("--ddg", default=None, help="precomputed ddG CSV (FoldX etc.)")
    p.add_argument("--vt-models", default=None, dest="vt_models",
                   help="precomputed mutant models <acc>_<wt><pos><mut>.pdb")
    p.add_argument("--no-sasa", action="store_true", dest="no_sasa")
    p.add_argument("--out", required=True)

    p = command("genomic", cmd_genomic, "exon-structure feature columns")
    p.add_argument("--data", required=True)
    p.add_argument("--ensembl-cache", default="data/cache/ensembl", dest="ensembl_cache")
    p.add_argument("--fetch", action="store_true", help="download missing exon tables")
    p.add_argument("--include-nucleotide", action="store_true", dest="include_nucleotide")
    p.add_argument("--out", required=True)

    def plm_options(p):
        p.add_argument("--data", required=True)
        p.add_argument("--backbone", default="esm2_650m",
                       help="esm1b | esm2_8m | esm2_35m | esm2_150m | esm2_650m | esm2_3b | existing")
        p.add_argument("--policy", default="full",
                       choices=["full", "sliding", "centered", "asymmetric", "hierarchical"])
        p.add_argument("--rows", default="labelled+functional",
                       help="labelled | labelled+functional | all")
        p.add_argument("--arm", default="P0", help="pretraining arm this backbone represents")
        p.add_argument("--adapted-from", default=None, dest="adapted_from",
                       help="pretraining output dir holding backbone_delta.pt")
        p.add_argument("--batch-size", type=int, default=8, dest="batch_size")
        p.add_argument("--store", default="features")
        p.add_argument("--fold", default=None, choices=list(GENES),
                       help="held-out gene this (strict-mode pretrained) backbone excludes")
        p.add_argument("--blocks-file", default=None, dest="blocks_file",
                       help="per-fold mapping JSON to register this entry in")

    p = command("embed", cmd_embed, "PLM WT/VT embeddings -> feature store")
    plm_options(p)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])

    p = command("zeroshot", cmd_zeroshot, "zero-shot PLM scores -> feature store")
    plm_options(p)
    p.add_argument("--method", default="masked_marginal",
                   choices=["masked_marginal", "wt_marginal"])
    p.add_argument("--table-out", default=None, dest="table_out",
                   help="write a copy of the table with the score as a baseline column")

    p = command("corpus", cmd_corpus, "MMR pretraining corpus")
    p.add_argument("--fasta", nargs="*", default=[])
    p.add_argument("--fetch", action="store_true", help="download the pinned UniProt queries")
    p.add_argument("--raw-dir", default="data/dl/corpus/raw", dest="raw_dir")
    p.add_argument("--mode", default="strict", choices=["strict", "transductive"])
    p.add_argument("--holdout", default=None, choices=list(GENES))
    p.add_argument("--max-identity", type=float, default=0.5, dest="max_identity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None)
    p.add_argument("--out", required=True)

    p = command("pretrain", cmd_pretrain, "continued / variant-aware pretraining arm")
    p.add_argument("--arm", default="P1", choices=["P0", "P1", "P2", "P3", "P4"])
    p.add_argument("--backbone", default="esm2_650m")
    p.add_argument("--corpus", required=True)
    p.add_argument("--objectives", default=None, help='JSON overrides, e.g. {"contrastive": 1}')
    p.add_argument("--strategy", default=None, help="JSON FinetuneStrategy for pretraining")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--grad-accum", type=int, default=2, dest="grad_accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--precision", default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", required=True)

    p = command("train", cmd_train, "one DL cell through the shared protocol")
    split_options(p)
    p.add_argument("--sources", nargs="+", required=True, help="what --data was built from")
    p.add_argument("--model", required=True,
                   choices=["gbm", "mlp", "bilstm", "aa_mlp", "cnn", "bilstm_attn",
                            "transformer", "fusion", "plm_finetune"])
    p.add_argument("--model-kwargs", default=None, dest="model_kwargs", help="JSON")
    p.add_argument("--tag", default=None, help="name for this model configuration")
    p.add_argument("--embedding-blocks", nargs="*", dest="embedding_blocks", default=[])
    p.add_argument("--score-rows", default="none", dest="score_rows",
                   choices=["none", "functional", "all"])
    p.add_argument("--allow-proxy-leak", action="store_true", dest="allow_proxy_leak")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--n-bootstrap", type=int, default=10_000, dest="n_bootstrap")
    p.add_argument("--store", default="features")
    p.add_argument("--export-dir", default=None, dest="export_dir")
    p.add_argument("--out", default="runs/dl")

    p = command("calibrate", cmd_calibrate, "calibrate on inner-validation predictions")
    p.add_argument("--runs", required=True)
    p.add_argument("--method", default="temperature", choices=["platt", "temperature", "isotonic"])
    p.add_argument("--bins", type=int, default=10)

    p = command("functional", cmd_functional, "independent functional validation")
    p.add_argument("--runs", required=True)
    p.add_argument("--functional", required=True, help="<canonical>.functional.csv")
    p.add_argument("--n-bootstrap", type=int, default=2000, dest="n_bootstrap")

    p = command("failure", cmd_failure, "failure-analysis report for one cell")
    p.add_argument("--data", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--cell", required=True, help="cell slug")
    p.add_argument("--paired", default=None, help="paired table vs the P0 arm (optional)")

    p = command("results", cmd_results, "collect every run into paper-ready tables")
    p.add_argument("--runs", default="runs/dl", help="root directory, searched recursively")
    p.add_argument("--out", default="results/dl")
    p.add_argument("--compare", nargs="*", default=[], metavar="DIR=ARM:MODEL",
                   help="reference arm(s) for the paired tables; omit to infer them")
    p.add_argument("--canonical-report", default="data/built/canonical.report.json",
                   dest="canonical_report")
    p.add_argument("--leakage", default="runs/dl/leakage")
    p.add_argument("--n-bootstrap", type=int, default=10_000, dest="n_bootstrap")
    p.add_argument("--no-plots", action="store_true", dest="no_plots")

    p = command("arms", cmd_arms, "list (arm, model) names in a runs directory")
    p.add_argument("--runs", required=True)

    p = command("export", cmd_export, "DL output records for integration")
    p.add_argument("--runs", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--export-dir", required=True, dest="export_dir")
    p.add_argument("--out", default="runs/dl/export")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    _apply_config(args, parser.subcommands[args.command], argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
