"""Tests for the generated-artifact existence guard."""
from __future__ import annotations

import pytest

from src.paths import require_exists


def test_require_exists_returns_path_unchanged_when_present(tmp_path):
    p = tmp_path / "panel.json"
    p.write_text("{}")
    assert require_exists(p, "some generator command") == p


def test_require_exists_accepts_a_string_path(tmp_path):
    p = tmp_path / "panel.json"
    p.write_text("{}")
    assert require_exists(str(p), "some generator command") == p


def test_require_exists_raises_actionable_error_when_missing(tmp_path):
    p = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError) as exc:
        require_exists(p, "python scripts/make_expanded_panel.py")
    msg = str(exc.value)
    assert str(p) in msg
    assert "python scripts/make_expanded_panel.py" in msg
