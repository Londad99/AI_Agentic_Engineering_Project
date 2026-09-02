"""Progress output that actually shows up.

Jupyter buffers stdout, so a plain print() during a long call is only flushed when
the cell finishes - precisely when the user no longer needs it. Everything that
takes time in this project reports through here, with flush=True.

Silence during a slow operation is a bug in its own right: the user cannot tell a
working call from a hung one, and ends up interrupting the kernel halfway through
an ingestion.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

# Set to False to silence progress output (e.g. when running the tests).
VERBOSE = True


def status(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


HEARTBEAT_SECONDS = 5


@contextmanager
def step(label: str, heartbeat: bool = True):
    """Announce a slow operation before it starts, tick while it runs, and time it.

    Announcing *before* is the whole point: a message printed after the call is a
    report, not feedback.

    The heartbeat exists because "->" followed by two minutes of nothing is barely
    better than no message at all - the user still cannot tell a working call from
    a hung one. A ticking counter answers that. It runs on a daemon thread so an
    interrupted cell can never leave it running.
    """
    status(f"  -> {label} ...")
    started = time.monotonic()
    stop = threading.Event()

    def tick():
        while not stop.wait(HEARTBEAT_SECONDS):
            status(f"     still waiting on {label} ... {time.monotonic() - started:.0f}s")

    thread: threading.Thread | None = None
    if heartbeat and VERBOSE:
        thread = threading.Thread(target=tick, daemon=True)
        thread.start()

    try:
        yield
    except Exception:
        status(f"  <- {label} failed after {time.monotonic() - started:.1f}s")
        raise
    else:
        status(f"  <- {label} done in {time.monotonic() - started:.1f}s")
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1)
