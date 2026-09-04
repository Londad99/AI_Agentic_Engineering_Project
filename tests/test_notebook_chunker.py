"""The chunker written out in the notebook must behave like the one in tutor/.

Two copies of an algorithm drift. Here the risk is specific: the notebook version is
what the reader learns from, so if it diverges the notebook teaches something the
project does not do. It already happened once - the inline version forgot to count the
newline that '\\n'.join adds back, produced 15 chunks where the module produced 19, and
let chunks run over the size budget.

Run:  python tests/test_notebook_chunker.py
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tutor.ingest import _chunk_text  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "tutor.ipynb"


def inline_chunker():
    """Pull chunk_text out of the notebook without running the rest of its cell.

    Parsed with ast rather than cut at a marker string: the cell also loads the PDF and
    builds the index, and exec-ing that here would need an API key and would rewrite the
    store. Splitting on a magic substring broke the first time the cell was edited.
    """
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "def chunk_text" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == "chunk_text":
                namespace: dict = {}
                exec(ast.get_source_segment(source, node), namespace)
                return namespace["chunk_text"]
    raise AssertionError("the notebook no longer defines chunk_text")


TEXT = "\n".join(
    f"Parrafo numero {i} con contenido suficiente para ocupar espacio real en el chunk."
    for i in range(60)
)


def test_same_chunks_as_the_module():
    inline = inline_chunker()
    for size, overlap in [(400, 80), (1200, 200), (250, 50)]:
        theirs = _chunk_text(TEXT, size, overlap)
        ours = inline(TEXT, size, overlap)
        assert len(ours) == len(theirs), (
            f"size={size}: notebook produced {len(ours)} chunks, module {len(theirs)}"
        )
        assert [c.strip() for c in ours] == [c.strip() for c in theirs], f"size={size}: content differs"
    print("notebook == module    OK")


def test_neither_exceeds_the_budget():
    inline = inline_chunker()
    for size, overlap in [(400, 80), (1200, 200)]:
        for chunk in inline(TEXT, size, overlap):
            assert len(chunk) <= size, f"chunk of {len(chunk)} chars exceeds size={size}"
    print("size budget respected OK")


def notebook_ids(page_source: str, page_number: int, text: str) -> list[str]:
    """Reproduce the id the notebook builds for each chunk."""
    import hashlib

    inline = inline_chunker()
    ids = []
    for index, piece in enumerate(inline(text, 400, 80)):
        key = f"{page_source}|{page_number}|{index}|{piece}"
        ids.append(hashlib.sha256(key.encode()).hexdigest()[:16])
    return ids


def test_ids_match_the_module():
    """The notebook writes into the same ChromaDB collection as the CLI and the app.

    If the two built different ids for the same chunk, running the notebook and then
    `scripts/ingest.py` would store every chunk twice - same text, two rows - and
    retrieval would start returning duplicates. Nothing would raise.
    """
    from tutor.ingest import Page, chunk_pages

    page = Page(source="notes.pdf", page_number=3, text=TEXT)
    module_ids = [chunk.id for chunk in chunk_pages([page], size=400, overlap=80)]
    assert notebook_ids("notes.pdf", 3, TEXT) == module_ids, (
        "the notebook and the module would write different rows for the same chunk"
    )
    print("ids match             OK")


if __name__ == "__main__":
    test_same_chunks_as_the_module()
    test_neither_exceeds_the_budget()
    test_ids_match_the_module()
    print("\nall notebook chunker tests passed")
