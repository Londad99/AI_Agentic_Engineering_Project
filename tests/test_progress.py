"""Tests for progress reporting.

This became load-bearing when the Streamlit app started rendering these lines on the
page: a sink that leaks past its block would keep writing into a container from a
previous interaction, and a sink that raises would take down the work it is reporting.

Run:  python tests/test_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import progress  # noqa: E402

# Other test files silence progress output globally. Set it explicitly here rather than
# inheriting whatever ran before: a test that passes alone and fails in the suite is
# worse than no test.
progress.VERBOSE = True


def test_sink_receives_lines_and_is_removed_after_the_block():
    lines = []
    with progress.capture(lines.append):
        progress.status("inside")
    progress.status("outside")
    assert lines == ["inside"], lines
    assert not progress._sinks, "the sink outlived its block"
    print("sink lifecycle        OK")


def test_step_reports_start_and_end():
    lines = []
    with progress.capture(lines.append):
        with progress.step("thing", heartbeat=False):
            pass
    assert lines[0].startswith("  -> thing"), lines
    assert "done in" in lines[-1], lines
    print("step start and end    OK")


def test_failure_is_reported_then_re_raised():
    lines = []
    try:
        with progress.capture(lines.append):
            with progress.step("thing", heartbeat=False):
                raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("the exception was swallowed")
    assert any("failed after" in line for line in lines), lines
    print("failure reported      OK")


def test_a_broken_sink_does_not_break_the_work():
    """A display problem must never become a work problem."""
    def explode(_line):
        raise RuntimeError("the UI is gone")

    done = []
    with progress.capture(explode):
        progress.status("still runs")
        done.append(True)
    assert done == [True]
    assert not progress._sinks
    print("broken sink tolerated OK")


def test_verbose_false_silences_sinks_too():
    lines = []
    progress.VERBOSE = False
    try:
        with progress.capture(lines.append):
            progress.status("quiet")
    finally:
        progress.VERBOSE = True
    assert lines == [], lines
    print("VERBOSE respected     OK")


if __name__ == "__main__":
    test_sink_receives_lines_and_is_removed_after_the_block()
    test_step_reports_start_and_end()
    test_failure_is_reported_then_re_raised()
    test_a_broken_sink_does_not_break_the_work()
    test_verbose_false_silences_sinks_too()
    print("\nall progress tests passed")
