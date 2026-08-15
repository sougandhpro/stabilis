"""Readers and writers. Formats register themselves through `registry`.

Reading and writing are deliberately symmetric -- `read_text`/`write_text`
are the low-level column movers, `load_*`/`save_*` the pair that carry a
whole series and its provenance. What is *not* here is result output: a
sigma-tau table and a report bundle live in `stabilis.export`, on the same
grounds that `validate/stable32.py` gives for sitting outside this package
-- it handles results, not data.
"""

from .text import load_frequency, load_phase, read_header_meta, read_text
from .write import WrittenFile, save_frequency, save_phase, write_text

__all__ = ["load_phase", "load_frequency", "read_text", "read_header_meta",
           "save_phase", "save_frequency", "write_text", "WrittenFile"]
