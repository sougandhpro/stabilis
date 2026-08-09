"""A local page for the routine work, so it needs no CLI flags.

    stabilis gui                    # opens http://127.0.0.1:8765
    stabilis gui --root ~/data      # confine the file browser

**This is a client of the library, not part of it.** Everything under
`gui/` imports from `stabilis` and nothing in `stabilis` imports from
`gui/`, so the whole package could be lifted out and shipped separately
without the engine noticing. That direction is deliberate and load-bearing:
the API is not permitted to grow a shape because a page found it
convenient, and the page is free to be rewritten because no answer depends
on it.

**It computes nothing a user would quote.** It reads files, builds derived
columns, chooses which stretches of a record to keep, and draws pictures.
Every variance, every degree of freedom and every confidence bound comes
from `stats/`, through the same calls the CLI makes, so the page and the
command line cannot disagree.

The work is arranged as three stages, which is the order the questions
actually arrive in:

1. **Data** -- pick a file, confirm what its columns are called, look at
   them, and mark the stretches worth analysing. Nothing is converted or
   assumed here; the point is to see the record before deciding anything.
2. **Prepare** -- say what the numbers *are*: phase or frequency, in which
   units, at what rate, and what arithmetic turns the file's columns into
   the quantity of interest.
3. **Analyse** -- run an estimator over what stage 2 produced, and read the
   result off an interactive curve with the table and the equivalent script
   beside it.

Modules: `files` reaches the disk and caches what it read, `expr` parses
column arithmetic without an evaluator, `prepare` turns columns into series
the library accepts, `request` reads the settings every endpoint is sent,
`api` serves the reading and drawing endpoints, `analysis` is the one that
runs an estimator, `server` is the HTTP and the bind, and `assets/` is the
page itself.
"""

from .api import estimators
from .files import cache_clear, safe_path
from .prepare import FREQ_UNITS, PHASE_UNITS
from .server import DEFAULT_PORT, Handler, serve

__all__ = [
    "serve",
    "DEFAULT_PORT",
    "estimators",
    "Handler",
    "PHASE_UNITS",
    "FREQ_UNITS",
    "safe_path",
    "cache_clear",
]
