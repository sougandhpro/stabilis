"""Can this bench separate its oscillators at all? And if not, what fixes it?

Under independence a measurement of sources `i` and `j` through channel `c`
has, at each tau,

    sigma2_meas = sigma2_i + sigma2_j + sigma2_c

so the source block of the design matrix is the **unsigned** incidence
matrix of the graph whose vertices are sources and whose edges are
measurements. Its rank is a graph property:

    rank = n_sources - (number of bipartite connected components)

which gives the result this whole module exists for:

> **A connected set of measurements determines its sources' individual
> variances if and only if it contains an odd cycle.**

That is not a curiosity. It is *why the technique is called
three-cornered hat*: a triangle is the smallest odd cycle. Four oscillators
measured round a ring A-B, B-C, C-D, D-A form a 4-cycle, which is
bipartite, so **that setup determines nothing about any individual
oscillator however long it runs** -- and one diagonal fixes it. A chain of
measurements is a tree, also bipartite, also hopeless.

Users get this wrong constantly and no tool tells them; they return a
number from an underdetermined system, or silently regularize. This one
refuses, says which combination is unidentifiable, and names the single
extra measurement that would fix it -- **before any data is collected**,
which is when the answer is still cheap to act on.

**A back-to-back run pins its own channel and contributes nothing to
separating sources.** So a bench made only of noise-floor runs is perfectly
well-conditioned in its channel block and completely useless in its source
block. That is a real mistake people make, and `check` says so in as many
words.

Why the parity argument works: an unsigned incidence row is `e_i + e_j`.
Assign each vertex `+1` or `-1` by a 2-colouring; then for every edge of a
bipartite component the two endpoints differ in colour, so the vector `v`
with `v_k` = the colour is orthogonal to every row -- a null direction, and
exactly the "shift all of one side up and the other down" ambiguity. An odd
cycle makes the 2-colouring impossible, and the null direction with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Bench

__all__ = ["Identifiability", "design_matrix", "check", "suggest_edge"]


@dataclass(frozen=True)
class Identifiability:
    """Whether a bench's unknowns are determined, and what to do if not."""

    sources: tuple[str, ...]
    channels: tuple[str, ...]
    rank: int
    n_unknowns: int
    components: tuple[tuple[str, ...], ...]
    bipartite: tuple[bool, ...]
    null_space: np.ndarray
    suggestions: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]

    @property
    def identifiable(self) -> bool:
        return self.rank == self.n_unknowns

    @property
    def deficiency(self) -> int:
        return self.n_unknowns - self.rank

    def report(self) -> str:
        """Human-readable verdict, for the CLI and for exception messages."""
        lines = []
        if self.identifiable:
            lines.append(
                f"IDENTIFIABLE: {self.n_unknowns} unknowns "
                f"({len(self.sources)} sources + {len(self.channels)} channels), "
                f"rank {self.rank}."
            )
        else:
            lines.append(
                f"NOT IDENTIFIABLE: rank {self.rank} of {self.n_unknowns} "
                f"unknowns -- {self.deficiency} independent combination(s) "
                "cannot be determined by any amount of averaging."
            )
        for comp, bip in zip(self.components, self.bipartite):
            state = "bipartite (no odd cycle)" if bip else "contains an odd cycle"
            lines.append(f"  component {{{', '.join(comp)}}}: {state}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        if self.suggestions:
            pairs = ", ".join(f"{a} vs {b}" for a, b in self.suggestions)
            lines.append(f"  add any one of these measurements: {pairs}")
        elif not self.identifiable:
            lines.append(
                "  no single extra measurement fixes this; more than one "
                "component needs joining."
            )
        return "\n".join(lines)


def design_matrix(bench: Bench):
    """`(A, sources, free_channels, known)` for `A @ v = b - known`.

    One row per measurement: `1` in each of its two source columns, `1` in
    its channel column. Two conventions carry real weight:

    * **A back-to-back row is `0` across every source.** Writing
      `A[row, a] += 1` twice would be wrong -- both ends are the same
      oscillator, so its noise is common to both and cancels in the
      difference rather than doubling.
    * **A channel whose `floor` is declared gets no column.** It is a known
      constant, so it moves to the right-hand side in `known` instead of
      consuming a degree of freedom. This is what makes the classical
      three-cornered hat work: three sources through one channel is four
      unknowns against three rows and hopeless, but declaring
      `Channel(floor=0.0)` -- the assumption the classical method makes
      silently -- leaves three unknowns and three rows, which is exactly
      determined. The assumption becomes something the user writes down
      rather than something the algorithm hides.

    Built without touching a single data file, which is what lets `check`
    run while the experiment is still being planned.
    """
    sources = bench.measured_sources
    free = tuple(c for c in bench.used_channels if bench.channel(c).floor is None)
    si = {s: i for i, s in enumerate(sources)}
    ci = {c: len(sources) + i for i, c in enumerate(free)}

    a = np.zeros((len(bench.measurements), len(sources) + len(free)))
    known = np.zeros(len(bench.measurements))
    for row, m in enumerate(bench.measurements):
        if not m.is_back_to_back:
            a[row, si[m.source_a]] = 1.0
            a[row, si[m.source_b]] = 1.0
        floor = bench.channel(m.channel_id).floor
        if floor is None:
            a[row, ci[m.channel_id]] = 1.0
        else:
            known[row] = floor
    return a, sources, free, known


def _components(bench: Bench, sources: tuple[str, ...]):
    """Connected components of the source graph, and whether each is bipartite.

    Two-colouring by breadth-first search: an edge joining two vertices of
    the same colour is an odd cycle. Back-to-back measurements are self
    loops and are skipped -- a self loop is trivially an odd cycle in
    general graph theory, but here it carries no information about *pairs*
    of sources, so counting it would make an unidentifiable bench look
    identifiable.
    """
    adjacency: dict[str, set[str]] = {s: set() for s in sources}
    for m in bench.measurements:
        if m.is_back_to_back:
            continue
        adjacency[m.source_a].add(m.source_b)
        adjacency[m.source_b].add(m.source_a)

    colour: dict[str, int] = {}
    comps, bips = [], []
    for start in sources:
        if start in colour:
            continue
        colour[start], queue, members, bipartite = 0, [start], [start], True
        while queue:
            node = queue.pop()
            for nxt in adjacency[node]:
                if nxt not in colour:
                    colour[nxt] = 1 - colour[node]
                    members.append(nxt)
                    queue.append(nxt)
                elif colour[nxt] == colour[node]:
                    bipartite = False
        comps.append(tuple(sorted(members)))
        bips.append(bipartite)
    return tuple(comps), tuple(bips), colour


def check(bench: Bench) -> Identifiability:
    """Full identifiability verdict. Reads no data files.

    Runs before an experiment, not after it -- which is the whole value.
    """
    a, sources, channels, _known = design_matrix(bench)
    comps, bips, _ = _components(bench, sources)
    n_unknowns = a.shape[1]

    if a.size == 0:
        rank, null = 0, np.zeros((0, n_unknowns))
    else:
        rank = int(np.linalg.matrix_rank(a))
        # Right singular vectors past the rank span the null space: the
        # combinations of unknowns the measurements cannot see at all.
        _, _, vt = np.linalg.svd(a)
        null = vt[rank:]

    notes = []
    b2b = [m for m in bench.measurements if m.is_back_to_back]
    if b2b and len(b2b) == len(bench.measurements):
        notes.append(
            "every measurement is back-to-back. The channel floors are "
            "well determined and no source variance is determined at all -- "
            "this bench measures the instrument, not the oscillators."
        )
    for comp, bip in zip(comps, bips):
        if bip and len(comp) > 1:
            notes.append(
                f"component {{{', '.join(comp)}}} is bipartite, so only sums "
                "across the split are determined, never individual sources. "
                "A cycle of even length -- a ring of four, or any chain -- "
                "does this."
            )
        elif bip and len(comp) == 1:
            notes.append(
                f"source {comp[0]} appears only in back-to-back runs, so "
                "nothing determines it."
            )
    if len(channels) and len(bench.measurements) < n_unknowns:
        notes.append(
            f"{len(bench.measurements)} measurements for {n_unknowns} "
            "unknowns: the system is short of rows before conditioning is "
            "even considered."
        )

    suggestions = suggest_edge(bench) if rank < n_unknowns else ()
    return Identifiability(
        sources=sources,
        channels=channels,
        rank=rank,
        n_unknowns=n_unknowns,
        components=comps,
        bipartite=bips,
        null_space=null,
        suggestions=suggestions,
        notes=tuple(notes),
    )


def suggest_edge(bench: Bench) -> tuple[tuple[str, str], ...]:
    """Pairs whose measurement would most improve identifiability.

    Every pair not already measured is tried, the rank recomputed, and the
    ones giving the largest increase are returned. Directly actionable --
    "measure B against D" -- and cheap, because the search is over pairs of
    sources rather than anything involving data.

    Ties are common and are all returned: on a 4-cycle either diagonal
    works, and saying so is more useful than picking one arbitrarily.
    """
    a, sources, channels, _known = design_matrix(bench)
    if len(sources) < 2:
        return ()
    base = int(np.linalg.matrix_rank(a)) if a.size else 0
    existing = {
        frozenset((m.source_a, m.source_b))
        for m in bench.measurements
        if not m.is_back_to_back
    }

    scored: dict[int, list[tuple[str, str]]] = {}
    for i, first in enumerate(sources):
        for second in sources[i + 1:]:
            if frozenset((first, second)) in existing:
                continue
            row = np.zeros((1, a.shape[1]))
            row[0, i] = 1.0
            row[0, sources.index(second)] = 1.0
            # Through an existing channel, so the suggestion needs no new
            # hardware -- the point is which *pair* to measure.
            if channels:
                row[0, len(sources)] = 1.0
            gain = int(np.linalg.matrix_rank(np.vstack([a, row]))) - base
            if gain > 0:
                scored.setdefault(gain, []).append((first, second))
    if not scored:
        return ()
    return tuple(scored[max(scored)])
