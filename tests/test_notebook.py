"""Automated tests for validating the competition Jupyter notebook."""

from pathlib import Path
import nbformat
import pytest

from src.config import get_config


def test_notebook_file_exists():
    cfg = get_config()
    nb_path = cfg.notebooks_dir / "grammar_scoring_engine.ipynb"
    assert nb_path.exists(), f"Notebook not found at {nb_path}"
    assert nb_path.stat().st_size > 5000, "Notebook file appears suspiciously small."


def test_notebook_structure_and_cells():
    cfg = get_config()
    nb_path = cfg.notebooks_dir / "grammar_scoring_engine.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)

    assert len(nb.cells) >= 20, f"Expected at least 20 cells, got {len(nb.cells)}"
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]

    assert len(code_cells) >= 10, f"Expected at least 10 code cells, got {len(code_cells)}"
    assert len(markdown_cells) >= 8, f"Expected at least 8 markdown cells, got {len(markdown_cells)}"


def test_notebook_execution_integrity():
    cfg = get_config()
    nb_path = cfg.notebooks_dir / "grammar_scoring_engine.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)

    errors = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    errors.append((idx, out.get("ename"), out.get("evalue")))

    assert len(errors) == 0, f"Notebook contains execution errors: {errors}"
