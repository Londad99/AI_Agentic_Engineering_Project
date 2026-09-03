"""Is the local model actually usable? Answers in one command.

Run:  python scripts/check_ollama.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import config, llm  # noqa: E402

print(f"host            {config.OLLAMA_HOST}")
print(f"OLLAMA_MODEL    {config.OLLAMA_MODEL}")

installed = llm.ollama_models()
if installed is None:
    print("\nNothing is answering there. Start the server with:  ollama serve")
    raise SystemExit(1)

print(f"installed tags  {', '.join(installed) or '(none)'}")

if config.OLLAMA_MODEL not in installed:
    print(f"\nOllama is running but has no tag '{config.OLLAMA_MODEL}'.")
    print("Tags must match exactly, including the part after the colon.")
    near_matches = [m for m in installed if m.split(":")[0] == config.OLLAMA_MODEL.split(":")[0]]
    if near_matches:
        print(f"Did you mean: {', '.join(near_matches)}  -> put it in OLLAMA_MODEL in .env")
    else:
        print(f"Pull it with:  ollama pull {config.OLLAMA_MODEL}")
    raise SystemExit(1)

print("\nSending a test message ...")
try:
    response = llm._call_ollama("Responde solo con la palabra: listo.", None, None, 0.0)
    print(f"reply           {response.strip()[:120]}")
    print("\nThe local model works. LLM_PROVIDER=ollama will run entirely offline.")
except Exception as error:  # noqa: BLE001
    print(f"\nThe call failed: {type(error).__name__}: {error}")
    raise SystemExit(1)
