"""Offline tests for the ingestion layer. No API key and no network needed:
cleaning and chunking are pure functions, which is exactly why they are plain
code and not agents.

Run:  python tests/test_ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.ingest import chunk_pages  # noqa: E402
from tutor.ingest import clean_pages  # noqa: E402
from tutor.ingest import Page  # noqa: E402

HEADER = "Universidad de Santander - Facultad de Ingenierias"
FOOTER = "Algoritmos y Estructuras de Datos - 2026"
BULLET = "Objetivo de la clase: comparar algoritmos por costo."
BODY = [
    "Un algoritmo de ordenamiento reorganiza una secuencia segun un criterio de orden.",
    "El costo se mide en comparaciones e intercambios, no en segundos de reloj.",
    "QuickSort divide el arreglo alrededor de un pivote elegi-\ndo y ordena cada mitad.",
    "Su complejidad promedio es O(n log n), pero degrada a O(n^2) con un pivote pesimo.",
    "MergeSort garantiza O(n log n) en el peor caso a cambio de memoria adicional.",
    "La estabilidad de un ordenamiento importa cuando se ordena por varias claves.",
]


def fake_pages() -> list[Page]:
    return [
        Page(
            source="notas.pdf",
            page_number=i,
            text=f"{HEADER}\n{BULLET}\n" + "\n".join(BODY[: 1 + i % 6]) + f"\n{FOOTER}\nPage {i} of 8\n",
        )
        for i in range(1, 9)
    ]


def test_cleaner():
    cleaned = clean_pages(fake_pages())
    text = "\n".join(p.text for p in cleaned)
    assert HEADER not in text, "running header was not removed"
    assert FOOTER not in text, "running footer was not removed"
    assert "Page 1 of 8" not in text, "page numbering was not removed"
    assert BULLET in text, "a repeated BODY line was wrongly deleted (false positive)"
    assert "elegido" in text, "hyphen split across lines was not repaired"
    print("cleaner  OK")


def test_chunker():
    cleaned = clean_pages(fake_pages())
    chunks = chunk_pages(cleaned, size=300, overlap=80)
    assert chunks, "no chunks produced"
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids collide"
    assert chunk_pages(cleaned, size=300, overlap=80)[0].id == chunks[0].id, "ids are not deterministic"
    assert max(len(c.text) for c in chunks) <= 380, "chunk exceeded size + overlap budget"
    assert all({"source", "page", "chunk_index"} <= c.metadata.keys() for c in chunks), "missing metadata"
    print(f"chunker  OK ({len(chunks)} chunks)")


if __name__ == "__main__":
    test_cleaner()
    test_chunker()
    print("\nall ingestion tests passed")
