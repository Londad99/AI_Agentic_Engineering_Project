"""Check that every tutor.* attribute the front ends touch actually exists.

Written after moving the settings from tutor.llm into tutor.config broke three
notebook cells with AttributeError. Notebook code is invisible to every linter and
to the rest of the test suite - it only fails when a human runs the cell, which in
this project means it fails in front of the class.

Covers the notebook, app.py and scripts/chat.py: none of them is reachable by the rest
of the test suite, and all three break the same way when something in tutor/ is renamed.

Nothing is executed (no API key, no network, no Streamlit): the files are parsed and
attribute access is resolved against the real modules, which is enough to catch a
rename or a move.

Run:  python tests/test_notebook_refs.py
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
WATCHED = {"config", "llm", "embeddings", "embedding_cache", "progress", "vectorstore",
           "agents", "orchestrator", "prompts", "scoring", "grounding", "session"}


def notebook_code_cells() -> list[tuple[str, str]]:
    """Every code cell of every notebook, labelled so a failure names the file."""
    cells = []
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Drop IPython magics (%pip, %autoreload): not valid Python syntax.
            source = "\n".join(
                line for line in source.split("\n") if not line.lstrip().startswith("%")
            )
            cells.append((f"{path.name}:{index}", source))
    return cells


def source_files() -> list[tuple[str, str]]:
    """The front ends that import tutor/ but no test ever runs."""
    out = []
    for path in [ROOT / "app.py", ROOT / "scripts" / "chat.py"]:
        if path.exists():
            out.append((path.name, path.read_text(encoding="utf-8")))
    return out


def test_every_cell_parses():
    for index, source in notebook_code_cells():
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"{index} does not parse: {error}") from error
    print(f"cells parse           OK ({len(NOTEBOOKS)} notebooks)")


def test_attributes_exist():
    from tutor import (  # noqa: F401
        agents, config, embedding_cache, embeddings, grounding, llm, orchestrator,
        progress, prompts, scoring, session, vectorstore,
    )

    modules = {name: sys.modules[f"tutor.{name}"] for name in WATCHED}

    # A module name can be shadowed by an ordinary variable - the notebook binds
    # `session = StudySession(...)`, which is not tutor.session. Anything the notebook
    # assigns is not the module, so it must not be checked against it.
    shadowed = set()
    for _, source in notebook_code_cells() + source_files():
        for node in ast.walk(ast.parse(source)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            elif isinstance(node, ast.For):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    shadowed.add(target.id)
    for name in shadowed:
        modules.pop(name, None)

    missing = []
    for index, source in notebook_code_cells() + source_files():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            module_name = node.value.id
            if module_name not in modules:
                continue
            if not hasattr(modules[module_name], node.attr):
                missing.append(f"{index}: {module_name}.{node.attr}")
    assert not missing, "notebook references attributes that do not exist:\n  " + "\n  ".join(missing)
    print(f"attributes resolve    OK ({len(modules)} modules checked)")


def test_modules_are_imported_before_they_are_used():
    """Catch a cell that uses `agents.x` when nothing in that notebook imported agents.

    Moving cells between notebooks breaks exactly this way: the code is fine, the import
    was three cells up in the file it came from, and the failure only shows when a human
    runs the cell.
    """
    by_notebook: dict[str, list[tuple[str, str]]] = {}
    for label, source in notebook_code_cells():
        by_notebook.setdefault(label.split(":")[0], []).append((label, source))

    problems = []
    for notebook, cells in by_notebook.items():
        imported: set[str] = set()
        for label, source in cells:
            tree = ast.parse(source)

            # Record this cell's own imports first: importing and using a module in the
            # same cell is normal, and flagging it would make the test useless.
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.update(alias.asname or alias.name for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported.update(
                        (alias.asname or alias.name).split(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.Assign):
                    imported.update(t.id for t in node.targets if isinstance(t, ast.Name))

            used = {
                node.value.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            }
            for name in sorted(used & WATCHED):
                if name not in imported:
                    problems.append(f"{label}: uses {name}.* but it was never imported")
    assert not problems, "notebook cells use modules they never import:\n  " + "\n  ".join(problems)
    print("imports before use    OK")


def test_front_ends_parse():
    for name, source in source_files():
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"{name} does not parse: {error}") from error
    print(f"front ends parse      OK ({len(source_files())} files)")


if __name__ == "__main__":
    test_every_cell_parses()
    test_modules_are_imported_before_they_are_used()
    test_front_ends_parse()
    test_attributes_exist()
    print("\nnotebook reference tests passed")
