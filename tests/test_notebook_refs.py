"""Check that every tutor.* attribute the notebook touches actually exists.

Written after moving the settings from tutor.llm into tutor.config broke three
notebook cells with AttributeError. Notebook code is invisible to every linter and
to the rest of the test suite - it only fails when a human runs the cell, which in
this project means it fails in front of the class.

This does not execute the notebook (no API key, no network). It parses the cells
and resolves attribute access against the real modules, which is enough to catch a
rename or a move.

Run:  python tests/test_notebook_refs.py
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "notebooks" / "tutor.ipynb"
WATCHED = {"config", "llm", "embeddings", "embedding_cache", "progress", "vectorstore"}


def notebook_code_cells() -> list[tuple[int, str]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Drop IPython magics (%pip, %autoreload): not valid Python syntax.
        source = "\n".join(line for line in source.split("\n") if not line.lstrip().startswith("%"))
        cells.append((index, source))
    return cells


def test_every_cell_parses():
    for index, source in notebook_code_cells():
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"cell {index} does not parse: {error}") from error
    print("cells parse           OK")


def test_attributes_exist():
    from tutor import config, embedding_cache, embeddings, llm, progress, vectorstore  # noqa: F401

    modules = {name: sys.modules[f"tutor.{name}"] for name in WATCHED}
    missing = []
    for index, source in notebook_code_cells():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            module_name = node.value.id
            if module_name not in modules:
                continue
            if not hasattr(modules[module_name], node.attr):
                missing.append(f"cell {index}: {module_name}.{node.attr}")
    assert not missing, "notebook references attributes that do not exist:\n  " + "\n  ".join(missing)
    print(f"attributes resolve    OK ({len(WATCHED)} modules checked)")


if __name__ == "__main__":
    test_every_cell_parses()
    test_attributes_exist()
    print("\nnotebook reference tests passed")
