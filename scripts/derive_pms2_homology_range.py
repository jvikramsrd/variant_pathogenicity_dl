#!/usr/bin/env python3
"""Derive the PMS2 exon 11-15 (PMS2CL homology) span in protein coordinates.

PROJECT_PLAN.md Phase 1 requires PMS2 variants in the PMS2CL pseudogene
homology region (exons 11-15) to be gated before anything downstream touches
them, and :func:`src.mmr_dataset.apply_pms2_homology_gate` deliberately
refuses to invent the exon->codon mapping. This script *derives* it, from
Ensembl's own exon table for the MANE Select transcript, so the number that
goes into the gate is reproducible rather than remembered.

Method
------
1. Fetch ``ENST00000265849`` (PMS2 MANE Select / NM_000535.7, 15 exons) with
   its exon structure and translation bounds from the Ensembl REST API.
2. Clip each exon to the CDS, walk them in transcript order, and accumulate
   coding length to get each exon's ``c.`` interval.
3. Convert to codons via ``codon = ceil(c_pos / 3)``.
4. The homology span is the first codon of exon 11 through the last codon of
   exon 15.

The run is self-validating: total CDS length must imply the 862 aa recorded
for UniProt P54278 in :data:`src.mmr_dataset.MMR_UNIPROT`, otherwise the
transcript or the pinned reference moved and the script fails loudly instead
of emitting a plausible-looking wrong range.

Usage
-----
    python scripts/derive_pms2_homology_range.py
    python scripts/build_mmr_dataset.py --pms2_codon_range 382 862
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mmr_dataset import MMR_UNIPROT  # noqa: E402

logger = logging.getLogger("derive_pms2")

ENSEMBL_REST = "https://rest.ensembl.org"
#: PMS2 MANE Select transcript (= RefSeq NM_000535.7), 15 exons, minus strand.
PMS2_TRANSCRIPT = "ENST00000265849"
#: Exons whose sequence is shared with the PMS2CL pseudogene.
HOMOLOGY_EXONS = (11, 15)


def fetch_transcript(transcript_id: str, timeout: int = 60) -> dict:
    url = f"{ENSEMBL_REST}/lookup/id/{transcript_id}?expand=1"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def exon_codon_table(transcript: dict) -> list[dict]:
    """Per-exon ``c.`` and codon intervals, in transcript (5'->3') order."""
    strand = int(transcript["strand"])
    cds_lo = int(transcript["Translation"]["start"])
    cds_hi = int(transcript["Translation"]["end"])
    # Genomic coordinates always ascend; transcript order reverses on the
    # minus strand, which PMS2 is on.
    exons = sorted(transcript["Exon"], key=lambda e: int(e["start"]),
                   reverse=(strand == -1))

    rows: list[dict] = []
    cds_consumed = 0
    for number, exon in enumerate(exons, start=1):
        lo = max(int(exon["start"]), cds_lo)
        hi = min(int(exon["end"]), cds_hi)
        coding_bp = max(0, hi - lo + 1)
        if coding_bp == 0:          # wholly untranslated exon
            rows.append({"exon": number, "coding_bp": 0})
            continue
        c_start = cds_consumed + 1
        c_end = cds_consumed + coding_bp
        cds_consumed += coding_bp
        rows.append({
            "exon": number, "coding_bp": coding_bp,
            "c_start": c_start, "c_end": c_end,
            # ceil division: c.1 c.2 c.3 -> codon 1.
            "codon_start": (c_start + 2) // 3,
            "codon_end": (c_end + 2) // 3,
        })
    return rows


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcript", default=PMS2_TRANSCRIPT)
    p.add_argument("--json_out", type=Path,
                   default=ROOT / "data/raw/pms2/pms2_exon_codon_map.json")
    args = p.parse_args(argv)

    tx = fetch_transcript(args.transcript)
    rows = exon_codon_table(tx)
    coding = [r for r in rows if r["coding_bp"]]
    total_bp = sum(r["coding_bp"] for r in coding)

    # Self-validation against the pinned UniProt reference.
    expected_aa = MMR_UNIPROT["PMS2"][1]
    derived_aa = total_bp // 3 - 1          # minus the stop codon
    if derived_aa != expected_aa:
        raise SystemExit(
            f"{args.transcript} implies {derived_aa} aa but "
            f"src.mmr_dataset pins PMS2 at {expected_aa} aa. The transcript or "
            "the pinned reference changed; resolve that before trusting any "
            "codon range derived here.")

    print(f"{'exon':>5} {'coding bp':>10} {'c. interval':>16} {'codons':>14}")
    for r in rows:
        if not r["coding_bp"]:
            print(f"{r['exon']:>5} {0:>10}   (untranslated)")
            continue
        # Built with .format rather than nested same-quote f-strings, which
        # only parse on Python 3.12+; this repo also runs on 3.11 venvs.
        cdna = "c.{}-{}".format(r["c_start"], r["c_end"])
        codons = "{}-{}".format(r["codon_start"], r["codon_end"])
        print("{:>5} {:>10} {:>16} {:>14}".format(
            r["exon"], r["coding_bp"], cdna, codons))

    first, last = HOMOLOGY_EXONS
    by_exon = {r["exon"]: r for r in coding}
    if first not in by_exon or last not in by_exon:
        raise SystemExit(f"Transcript has no coding exon {first}/{last}.")
    start = by_exon[first]["codon_start"]
    end = by_exon[last]["codon_end"]
    # The final codon of the CDS is the stop codon, which has no residue.
    end = min(end, expected_aa)

    print(f"\nCDS {total_bp} bp -> {derived_aa} aa (matches pinned P54278).")
    print(f"PMS2CL homology region = exons {first}-{last} "
          f"= c.{by_exon[first]['c_start']}-{by_exon[last]['c_end']} "
          f"= protein codons {start}-{end}")
    print(f"\n  python scripts/build_mmr_dataset.py --pms2_codon_range {start} {end}\n")
    print(f"Residues OUTSIDE the region (usable without orthogonal "
          f"confirmation): 1-{start - 1} ({start - 1}/{expected_aa} aa).")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({
        "transcript": args.transcript,
        "source": f"{ENSEMBL_REST}/lookup/id/{args.transcript}?expand=1",
        "assembly": tx.get("assembly_name"),
        "cds_bp": total_bp, "protein_aa": derived_aa,
        "homology_exons": list(HOMOLOGY_EXONS),
        "homology_codon_range": [start, end],
        "exons": rows,
    }, indent=2))
    logger.info("Wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
