"""Shared synthetic fixtures for the DL-branch tests. No data files, no network.

The builders themselves are in ``dl_helpers.py`` (import them from there, never
``from conftest import ...`` — see that module's docstring for why).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_helpers import (AA, PANEL, assembled_table, clinvar_record,  # noqa: E402,F401
                        random_sequences)


@pytest.fixture
def sequences():
    return random_sequences()


@pytest.fixture
def table(sequences):
    return assembled_table(sequences)
