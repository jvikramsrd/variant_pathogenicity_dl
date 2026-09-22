"""Structural features — WT from AlphaFold, VT through a provider interface.

Wild-type features are computed from an AlphaFold DB model per residue:

    feature_struct_plddt         model confidence (pLDDT, B-factor column)
    feature_struct_low_conf      pLDDT < 50 — typically intrinsically disordered
    feature_struct_rsa           relative solvent accessibility (Shrake-Rupley
                                 SASA / Tien et al. 2013 theoretical maximum)
    feature_struct_helix         backbone phi/psi in the alpha region
    feature_struct_strand        backbone phi/psi in the beta region
    feature_struct_contacts8     C-beta neighbours within 8 A (packing)
    feature_struct_contacts12    within 12 A (environment)
    feature_struct_hse_up        half-sphere exposure along CA->CB (13 A)
    feature_struct_ca_curvature  CA(i-1)-CA(i)-CA(i+1) angle (local geometry)
    feature_struct_available     1 when the residue has coordinates

Secondary structure is a dihedral-region PROXY, labelled as one — no DSSP
binary is assumed. Everything is numpy/scipy; no Biopython, no GPU.

Variant-type structure is an interface, not a computation: nothing here runs
AlphaFold, ESMFold or FoldX. :class:`PrecomputedVariantStructures` reads mutant
models and/or a ddG table produced elsewhere (on the DGX, later) and returns
WT/VT differences (``feature_struct_delta_*``); :class:`NoVariantStructures` is
the default and returns nothing, which the fusion model sees as a missing
modality rather than as zeros.

Column names avoid the prior-score keywords (``_score``, ``eve`` ...) so every
one resolves to the ``structure`` ablation group (tests enforce this).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["WT_STRUCTURE_COLUMNS", "VT_STRUCTURE_COLUMNS", "MAX_ASA", "Residue",
           "parse_pdb", "shrake_rupley", "residue_features", "wt_structure_table",
           "structure_index", "VariantStructureProvider", "NoVariantStructures",
           "PrecomputedVariantStructures"]

WT_STRUCTURE_COLUMNS = (
    "feature_struct_plddt", "feature_struct_low_conf", "feature_struct_rsa",
    "feature_struct_helix", "feature_struct_strand", "feature_struct_contacts8",
    "feature_struct_contacts12", "feature_struct_hse_up", "feature_struct_ca_curvature",
    "feature_struct_available",
)
VT_STRUCTURE_COLUMNS = ("feature_struct_delta_ddg", "feature_struct_delta_rsa",
                        "feature_struct_delta_contacts8", "feature_struct_local_rmsd")

# Tien et al. 2013 (theoretical) maximum accessible surface area, A^2.
MAX_ASA = {"A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0,
           "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0,
           "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0,
           "Y": 263.0, "V": 174.0}
_RADII = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "H": 1.1, "SE": 1.9}
_THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


@dataclass
class Residue:
    number: int
    aa: str
    atoms: dict[str, np.ndarray]
    bfactor: float


def parse_pdb(path: Path | str, chain: str | None = None) -> list[Residue]:
    """ATOM records of the first model, fixed-column PDB format."""
    residues: dict[int, Residue] = {}
    order: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if chain and line[21] != chain:
                continue
            if line[16] not in (" ", "A"):          # keep altloc A only
                continue
            number = int(line[22:26])
            name = line[12:16].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if number not in residues:
                residues[number] = Residue(number, _THREE.get(line[17:20], "X"), {},
                                           float(line[60:66]))
                order.append(number)
            residues[number].atoms[name] = xyz
    return [residues[n] for n in order]


def _sphere(points: int = 96) -> np.ndarray:
    index = np.arange(points) + 0.5
    phi = np.arccos(1 - 2 * index / points)
    theta = np.pi * (1 + 5 ** 0.5) * index
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def shrake_rupley(residues: Sequence[Residue], probe: float = 1.4,
                  points: int = 96) -> np.ndarray:
    """Per-residue SASA (A^2), heavy atoms, Shrake & Rupley (1973)."""
    from scipy.spatial import cKDTree

    coords, radii, owner = [], [], []
    for i, residue in enumerate(residues):
        for name, xyz in residue.atoms.items():
            element = name[0] if name[:2] != "SE" else "SE"
            if element == "H":
                continue
            coords.append(xyz)
            radii.append(_RADII.get(element, 1.8) + probe)
            owner.append(i)
    coords, radii, owner = np.array(coords), np.array(radii), np.array(owner)
    tree = cKDTree(coords)
    sphere = _sphere(points)
    sasa = np.zeros(len(residues))
    reach = radii.max() * 2
    for atom in range(len(coords)):
        surface = coords[atom] + radii[atom] * sphere
        neighbours = [j for j in tree.query_ball_point(coords[atom], radii[atom] + reach)
                      if j != atom]
        if neighbours:
            d = np.linalg.norm(surface[:, None, :] - coords[neighbours][None], axis=-1)
            exposed = ~(d < radii[neighbours][None]).any(axis=1)
        else:
            exposed = np.ones(points, dtype=bool)
        sasa[owner[atom]] += 4 * np.pi * radii[atom] ** 2 * exposed.mean()
    return sasa


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def residue_features(residues: Sequence[Residue], with_sasa: bool = True) -> pd.DataFrame:
    """One row per residue with the WT feature columns (1-based `position`)."""
    from scipy.spatial import cKDTree

    n = len(residues)
    ca = np.array([r.atoms.get("CA", np.full(3, np.nan)) for r in residues])
    # Virtual C-beta for glycine (ideal geometry), real C-beta otherwise.
    cb = []
    for r, ca_i in zip(residues, ca):
        if "CB" in r.atoms:
            cb.append(r.atoms["CB"])
        elif {"N", "C"} <= set(r.atoms):
            b, c = ca_i - r.atoms["N"], r.atoms["C"] - ca_i
            a = np.cross(b, c)
            cb.append(-0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_i)
        else:
            cb.append(ca_i)
    cb = np.array(cb)
    tree = cKDTree(cb)
    contacts8 = np.array([len(tree.query_ball_point(x, 8.0)) - 1 for x in cb])
    contacts12 = np.array([len(tree.query_ball_point(x, 12.0)) - 1 for x in cb])
    ca_tree = cKDTree(ca)
    hse_up = np.zeros(n)
    for i in range(n):
        direction = cb[i] - ca[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        near = [j for j in ca_tree.query_ball_point(ca[i], 13.0) if j != i]
        hse_up[i] = sum(np.dot(ca[j] - ca[i], direction) > 0 for j in near)

    helix = np.zeros(n)
    strand = np.zeros(n)
    curvature = np.full(n, np.nan)
    for i in range(1, n - 1):
        prev, cur, nxt = residues[i - 1].atoms, residues[i].atoms, residues[i + 1].atoms
        if {"C"} <= set(prev) and {"N", "CA", "C"} <= set(cur) and "N" in nxt:
            phi = _dihedral(prev["C"], cur["N"], cur["CA"], cur["C"])
            psi = _dihedral(cur["N"], cur["CA"], cur["C"], nxt["N"])
            helix[i] = float(-160 <= phi <= -20 and -120 <= psi <= 50)
            strand[i] = float(-180 <= phi <= -45 and (psi >= 90 or psi <= -150))
        u, v = ca[i - 1] - ca[i], ca[i + 1] - ca[i]
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        curvature[i] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

    plddt = np.array([r.bfactor for r in residues])
    rsa = np.full(n, np.nan)
    if with_sasa:
        sasa = shrake_rupley(residues)
        rsa = np.array([min(1.5, s / MAX_ASA.get(r.aa, 200.0))
                        for s, r in zip(sasa, residues)])
    return pd.DataFrame({
        "position": [r.number for r in residues], "residue": [r.aa for r in residues],
        "feature_struct_plddt": plddt, "feature_struct_low_conf": (plddt < 50).astype(float),
        "feature_struct_rsa": rsa, "feature_struct_helix": helix,
        "feature_struct_strand": strand, "feature_struct_contacts8": contacts8.astype(float),
        "feature_struct_contacts12": contacts12.astype(float),
        "feature_struct_hse_up": hse_up, "feature_struct_ca_curvature": curvature,
        "feature_struct_available": 1.0,
    })


def wt_structure_table(model_dir: Path | str, sequences: Mapping[str, str],
                       with_sasa: bool = True) -> tuple[pd.DataFrame, dict]:
    """Per-residue WT features for every accession with a model in `model_dir`.

    Expects AlphaFold DB files ``<accession>.pdb`` (+ ``<accession>_metadata.json``).
    The model's residue sequence must equal the pinned UniProt sequence,
    residue for residue — a model of another isoform is refused, never mapped.
    """
    frames, index = [], {}
    for accession, sequence in sequences.items():
        pdb = Path(model_dir) / f"{accession}.pdb"
        if not pdb.exists():
            logger.warning("no structure model for %s in %s", accession, model_dir)
            continue
        residues = parse_pdb(pdb)
        observed = "".join(r.aa for r in residues)
        numbers = [r.number for r in residues]
        mismatched = [n for n, aa in zip(numbers, observed)
                      if not 1 <= n <= len(sequence) or sequence[n - 1] != aa]
        if mismatched:
            raise ValueError(f"{pdb}: {len(mismatched)} residues disagree with the UniProt "
                             f"sequence (first at {mismatched[:5]}). Refusing to map.")
        features = residue_features(residues, with_sasa=with_sasa)
        features.insert(0, "uniprot_id", accession)
        frames.append(features)
        meta = _metadata(Path(model_dir) / f"{accession}_metadata.json")
        version = meta.get("latestVersion")
        index[accession] = {
            "structure_id": f"{meta.get('entryId', 'AF-' + accession + '-F1')}"
                            + (f"-model_v{version}" if version else ""),
            "residues": set(numbers), "source": str(pdb),
            "tool": meta.get("toolUsed"), "model_created": meta.get("modelCreatedDate"),
            "global_plddt": meta.get("globalMetricValue")}
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["uniprot_id", "position", "residue", *WT_STRUCTURE_COLUMNS])
    return table, index


def structure_index(model_dir: Path | str, sequences: Mapping[str, str]) -> dict:
    """``{accession: {structure_id, residues}}`` without computing features."""
    index = {}
    for accession in sequences:
        pdb = Path(model_dir) / f"{accession}.pdb"
        if pdb.exists():
            meta = _metadata(Path(model_dir) / f"{accession}_metadata.json")
            version = meta.get("latestVersion")
            index[accession] = {
                "structure_id": f"{meta.get('entryId', 'AF-' + accession + '-F1')}"
                                + (f"-model_v{version}" if version else ""),
                "residues": {r.number for r in parse_pdb(pdb)}}
    return index


def _metadata(path: Path) -> dict:
    """AlphaFold metadata as a dict; the API returns a one-element list."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return (payload[0] if payload else {}) if isinstance(payload, list) else payload


class VariantStructureProvider(Protocol):
    """Source of variant-type structural differences (WT -> VT)."""

    def deltas(self, variants: pd.DataFrame) -> pd.DataFrame: ...


class NoVariantStructures:
    """Default: no VT structures. Columns come back absent, never zero."""

    def deltas(self, variants: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=variants.index)


class PrecomputedVariantStructures:
    """VT differences from files produced OUTSIDE this code (FoldX, ESMFold...).

    * ``ddg_csv``: ``uniprot_id, position, wt_aa, mut_aa, ddg`` (kcal/mol,
      positive = destabilising, FoldX convention) -> ``feature_struct_delta_ddg``.
    * ``model_dir``: mutant models named ``<accession>_<wt><pos><mut>.pdb``;
      compared with the WT model -> delta RSA, delta packing and the CA RMSD of
      the +/-8-residue window after superposition.

    A variant with no precomputed input gets NaN (missing), never 0 — "no
    change" and "not computed" must stay distinguishable.
    """

    def __init__(self, wt_model_dir: Path | str, ddg_csv: Path | str | None = None,
                 model_dir: Path | str | None = None, window: int = 8) -> None:
        self.wt_model_dir = Path(wt_model_dir)
        self.ddg = pd.read_csv(ddg_csv) if ddg_csv else None
        self.model_dir = Path(model_dir) if model_dir else None
        self.window = window
        self._wt_cache: dict[str, tuple[list[Residue], pd.DataFrame]] = {}

    def _wt(self, accession: str):
        if accession not in self._wt_cache:
            residues = parse_pdb(self.wt_model_dir / f"{accession}.pdb")
            self._wt_cache[accession] = (residues, residue_features(residues))
        return self._wt_cache[accession]

    def deltas(self, variants: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=variants.index, columns=list(VT_STRUCTURE_COLUMNS))
        if self.ddg is not None:
            key = ["uniprot_id", "position", "wt_aa", "mut_aa"]
            merged = variants[key].merge(self.ddg.drop_duplicates(key), on=key, how="left")
            out["feature_struct_delta_ddg"] = merged["ddg"].to_numpy()
        if self.model_dir is None:
            return out
        for index, row in variants.iterrows():
            path = self.model_dir / f"{row.uniprot_id}_{row.wt_aa}{int(row.position)}{row.mut_aa}.pdb"
            if not path.exists():
                continue
            wt_residues, wt_features = self._wt(row.uniprot_id)
            vt_residues = parse_pdb(path)
            vt_features = residue_features(vt_residues)
            at = wt_features["position"] == int(row.position)
            vt_at = vt_features["position"] == int(row.position)
            for column, name in (("feature_struct_rsa", "feature_struct_delta_rsa"),
                                 ("feature_struct_contacts8", "feature_struct_delta_contacts8")):
                out.loc[index, name] = float(vt_features.loc[vt_at, column].iloc[0]
                                             - wt_features.loc[at, column].iloc[0])
            out.loc[index, "feature_struct_local_rmsd"] = _local_rmsd(
                wt_residues, vt_residues, int(row.position), self.window)
        return out


def _local_rmsd(a: Sequence[Residue], b: Sequence[Residue], position: int, window: int) -> float:
    """Kabsch-superposed CA RMSD over position +/- window."""
    pick = lambda residues: {r.number: r.atoms["CA"] for r in residues if "CA" in r.atoms}
    ca_a, ca_b = pick(a), pick(b)
    shared = [n for n in range(position - window, position + window + 1)
              if n in ca_a and n in ca_b]
    if len(shared) < 3:
        return float("nan")
    p = np.array([ca_a[n] for n in shared])
    q = np.array([ca_b[n] for n in shared])
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ rotation - q) ** 2).sum(1).mean()))
