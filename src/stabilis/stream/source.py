"""Where live samples come from. Generators, not classes.

A source is a generator. That is the whole abstraction: no base class, no
`start`/`stop`/`read` protocol, no object to keep alive. `for line in
tail_file(path):` is the interface, `break` is the shutdown, and a
generator that has been closed has released its socket or file handle
because `finally` ran. Anything a class would have carried -- partial-line
buffering, reconnect, idle timeout -- is local state in exactly one
function.

Two shapes live here and they are deliberately different.

* **`replay` yields `(t, value)` pairs**, because it already knows both.
  It is the source the test suite runs on, and the one that makes the
  streaming path checkable at all: replaying a record and analysing the
  same record in batch must agree, which is an identity rather than a
  comparison.
* **`tail_file`, `tcp_lines` and `udp_lines` yield text lines**, because
  a byte stream does not know what a sample is. Turning a line into a
  `(t, value)` pair is `collect.parse_lines`, which is where the column
  layout and the missing-timestamp question belong.

`serial` is not here. It needs `pyserial` and, more to the point, hardware
to test against; shipping an untested source and listing it in the feature
table is how a library acquires a function nobody has ever run. It arrives
with the first instrument.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np

__all__ = ["replay", "tail_file", "tcp_lines", "udp_lines", "READ_BYTES"]

# One read from a file or socket. Small enough that a slow trickle is not
# held back waiting for a buffer to fill, large enough that a fast one is
# not a syscall per sample.
READ_BYTES = 1 << 16


def replay(values, rate_hz: float, epoch: float = 0.0, skip_gaps: bool = False):
    """Replay an array as a timestamped stream on a perfect grid.

        t[i] = epoch + i / rate_hz

    `skip_gaps` is the interesting flag: with it set, a non-finite entry is
    **not emitted at all**, which is what a counter does when it fails to
    produce a reading. The timestamps of everything after it are unchanged,
    so the only evidence a sample went missing is the jump in `t` -- and
    reconstructing it is `collect.on_grid`'s job. Turning that flag on and
    getting the same answer back is the test that the ingest path does not
    silently renumber.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    for i, v in enumerate(arr):
        if skip_gaps and not np.isfinite(v):
            continue
        yield epoch + i / rate_hz, float(v)


def tail_file(
    path,
    poll_s: float = 0.25,
    timeout_s: float | None = None,
    from_start: bool = True,
    encoding: str = "utf-8",
):
    """Yield complete lines from a file that another process is appending to.

    `from_start` reads what is already there before following; set it False
    to take only what arrives from now on. `timeout_s` is an **idle**
    timeout -- give up after that long with no new bytes -- and `None`
    follows forever, so a caller wanting a bounded run should pass a number
    or break out of the loop.

    The load-bearing detail is that a read can land in the middle of a
    line, and a half-line parsed as a sample is a wrong number rather than
    an error. So bytes accumulate in `buf` and only text up to the last
    newline is ever yielded; a trailing fragment waits for the rest of
    itself. The final fragment of a file that stops without a newline is
    discarded rather than guessed at.
    """
    path = Path(path)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with open(path, "rb") as fh:
        if not from_start:
            fh.seek(0, 2)
        buf = b""
        while True:
            chunk = fh.read(READ_BYTES)
            if chunk:
                deadline = None if timeout_s is None else time.monotonic() + timeout_s
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines.pop()
                for line in lines:
                    yield line.decode(encoding).rstrip("\r")
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(poll_s)


def tcp_lines(
    host: str,
    port: int,
    timeout_s: float | None = 30.0,
    connect_timeout_s: float = 10.0,
    encoding: str = "utf-8",
):
    """Connect to `host:port` and yield complete lines until the peer closes.

    `timeout_s` is the idle timeout on the socket; `None` blocks forever.
    Same partial-line rule as `tail_file`, for the same reason -- TCP is a
    byte stream and has no opinion about where a sample ends.

    Reconnection is deliberately **not** here. A reconnect loses an unknown
    number of samples, and a source that quietly papers over that gap hands
    the estimator a record whose timebase is wrong in a way nothing
    downstream can see. Catch the exception, note the outage, and start a
    new stream on a new epoch.
    """
    sock = socket.create_connection((host, port), timeout=connect_timeout_s)
    try:
        sock.settimeout(timeout_s)
        yield from _lines(lambda: sock.recv(READ_BYTES), encoding)
    finally:
        sock.close()


def udp_lines(
    port: int,
    host: str = "127.0.0.1",
    timeout_s: float | None = 30.0,
    encoding: str = "utf-8",
):
    """Bind a UDP port and yield lines from the datagrams arriving on it.

    One datagram may hold several lines and, in principle, part of one --
    so the same buffering applies. What UDP does *not* give you is order or
    delivery, and neither is recoverable here: a lost datagram is a lost
    sample, and a reordered one is refused downstream by `on_grid` rather
    than sorted into place. That refusal is the right behaviour. Silently
    sorting would mean the record's timebase came from the network's
    scheduling rather than from the instrument.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
        sock.settimeout(timeout_s)
        yield from _lines(lambda: sock.recv(READ_BYTES), encoding)
    finally:
        sock.close()


def _lines(read, encoding: str):
    """Complete lines from a callable returning bytes; b"" means end.

    The one place partial-line buffering is implemented for the two socket
    sources. A `socket.timeout` ends the stream, which is what the idle
    timeout is for.
    """
    buf = b""
    while True:
        try:
            chunk = read()
        except (TimeoutError, socket.timeout):
            return
        if not chunk:
            return
        buf += chunk
        parts = buf.split(b"\n")
        buf = parts.pop()
        for line in parts:
            yield line.decode(encoding).rstrip("\r")
