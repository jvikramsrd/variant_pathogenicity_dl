#!/usr/bin/env python3
"""Cross-platform end-to-end pipeline runner.

This is the canonical implementation used by the Bash and PowerShell wrappers.
It keeps the existing environment-variable controls while avoiding shell-specific
path handling, virtualenv layout assumptions, and argument array syntax.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
FEATURE_CHOICES = ("priors", "esm+priors")


def env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in {"", "0", "false", "no", "off"}


def banner(text: str) -> None:
    print(f"\n==== {text} ====", flush=True)


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_pip(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def current_venv_python() -> Path | None:
    active = os.environ.get("VIRTUAL_ENV")
    if not active:
        return None

    candidate = venv_python(Path(active))
    if candidate.exists():
        return candidate
    return None


def nvidia_gpu_available() -> bool:
    """True only if ``nvidia-smi`` exists *and* can reach a driver.

    The binary alone is not evidence of a usable GPU. Distributions ship it
    with driver packages that end up installed on machines with no NVIDIA
    hardware, where it exits non-zero with "NVIDIA-SMI has failed because it
    couldn't communicate with the NVIDIA driver". Selecting the requirements
    file on ``shutil.which`` alone therefore pulled the multi-GB CUDA wheel set
    (cuDNN alone is ~707 MB) onto a CPU-only box, then left it with a torch
    build whose ``cuda.is_available()`` is still False.
    """

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def ensure_environment(env: dict[str, str]) -> Path:
    """Create/use a virtualenv and return the Python executable for stages.

    Readiness is gated on a stamp file written only after ``pip install``
    returns successfully, not on the interpreter existing. ``python -m venv``
    finishes in a second while the install takes minutes, so an install that
    fails or is interrupted (Ctrl-C, a dropped connection, a full disk) leaves
    a venv whose interpreter exists but whose dependencies do not. Gating on
    the interpreter alone made every later run silently adopt that broken
    environment and fail deep inside a stage with an ImportError.

    The stamp records which requirements file was installed, so switching
    between the CPU and CUDA requirement sets also forces a reinstall.
    """

    active_python = current_venv_python()
    if active_python is not None:
        return active_python

    venv_dir = ROOT / ".venv"
    python = venv_python(venv_dir)
    req = "requirements-cuda.txt" if nvidia_gpu_available() else "requirements.txt"
    stamp = venv_dir / ".requirements-installed"
    ready = python.exists() and stamp.exists() and stamp.read_text().strip() == req

    if not ready:
        launcher = os.environ.get("PYTHON", sys.executable)
        banner(f"Stage 0 - creating virtualenv + installing {req}")
        if not python.exists():
            run([launcher, "-m", "venv", str(venv_dir)], env)
        elif stamp.exists():
            print(f"Requirements file changed to {req}; reinstalling.")
        else:
            print("Previous dependency install did not complete; reinstalling.")
        run([str(venv_pip(venv_dir)), "install", "-r", req], env)
        stamp.write_text(req + "\n")

    return python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tests, dataset build, audit, and training end-to-end."
    )
    try:
        default_k_folds = int(os.environ.get("K_FOLDS", "5"))
    except ValueError:
        parser.error(f"K_FOLDS must be an integer; got {os.environ['K_FOLDS']!r}")

    parser.add_argument(
        "--features",
        choices=FEATURE_CHOICES,
        default=os.environ.get("FEATURES", "priors"),
        help="Feature mode for scripts/train_extended.py (env: FEATURES).",
    )
    parser.add_argument(
        "--esm-model",
        default=os.environ.get("ESM_MODEL", DEFAULT_ESM_MODEL),
        help="HuggingFace ESM checkpoint for esm+priors mode (env: ESM_MODEL).",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=default_k_folds,
        help="Number of cross-validation folds (env: K_FOLDS).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=env_flag("SKIP_BUILD"),
        help="Reuse data/processed/extended/extended_dataset.csv when present.",
    )
    args = parser.parse_args()
    if args.features not in FEATURE_CHOICES:
        parser.error(
            f"FEATURES must be one of {', '.join(FEATURE_CHOICES)}; got {args.features!r}"
        )
    return args


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    py = ensure_environment(env)

    banner("Stage 1/4 - unit tests")
    run([str(py), "tests/test_datasets.py"], env)
    # This pipeline builds the same master table, so the merge invariants
    # apply here too.
    run([str(py), "tests/test_merge.py"], env)

    panel = ROOT / "data" / "raw" / "uniprot" / "expanded_panel.json"
    if not panel.exists():
        banner("Stage 2a - resolving expanded panel (UniProt REST)")
        run([str(py), "scripts/make_expanded_panel.py"], env)
    else:
        print(f"Panel cache present: {panel.relative_to(ROOT)}")

    dataset = ROOT / "data" / "processed" / "extended" / "extended_dataset.csv"
    if dataset.exists() and args.skip_build:
        print("SKIP_BUILD set -> reusing existing extended_dataset.csv")
    else:
        banner("Stage 2b - building extended dataset (downloads cached)")
        run(
            [
                str(py),
                "scripts/build_extended_dataset.py",
                "--panel_file",
                str(panel),
            ],
            env,
        )

    banner("Stage 3/4 - auditing merge integrity (12 checks) + emitting train CSV")
    run([str(py), "scripts/audit_extended_dataset.py"], env)

    banner(f"Stage 4/4 - training MLP head ({args.features} mode, k={args.k_folds})")
    train_args = [
        str(py),
        "scripts/train_extended.py",
        "--k_folds",
        str(args.k_folds),
        "--no_dms_features",
    ]
    if args.features == "esm+priors":
        train_args.extend(["--features", "esm+priors", "--esm_model", args.esm_model])
    run(train_args, env)

    banner("DONE")
    print("Dataset   : data/processed/extended/extended_dataset.csv (+ _train.csv)")
    print("Audit     : data/processed/extended/audit_report.json")
    print("Model/mets: data/processed/extended_train/")
    print("Run log   : append a dated entry to docs/RUNLOG.md")


if __name__ == "__main__":
    main()
