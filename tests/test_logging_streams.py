from __future__ import annotations

import io

from main import _FilteredTeeStream


def test_filtered_tee_writes_normal_lines_to_both_streams():
    console = io.StringIO()
    log = io.StringIO()
    stream = _FilteredTeeStream(console, log)

    stream.write("hello world\n")
    stream.flush()

    assert console.getvalue() == "hello world\n"
    assert log.getvalue() == "hello world\n"


def test_filtered_tee_suppresses_noisy_lines_from_both_streams():
    console = io.StringIO()
    log = io.StringIO()
    stream = _FilteredTeeStream(console, log)

    stream.write("No supported GPU was found.\n")
    stream.flush()

    assert console.getvalue() == ""
    assert log.getvalue() == ""


def test_filtered_tee_flushes_both_streams():
    class FlushProbe(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    console = FlushProbe()
    log = FlushProbe()
    stream = _FilteredTeeStream(console, log)

    stream.write("x")
    stream.flush()

    assert console.flush_count >= 1
    assert log.flush_count >= 1
