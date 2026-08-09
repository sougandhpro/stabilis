"""Which technique this bench supports, read off the graph.

`check` answers "can the hat separate these oscillators". That is the
wrong question to ask on its own, because the hat is not the only thing a
bench can do and it is not always the thing that should be done. There are
two techniques and **they are not alternatives** -- they remove different
unknowns:

|                        | separates the sources | removes the channel term |
|------------------------|-----------------------|--------------------------|
| N-cornered hat         | yes, given an odd cycle | no: declare the floor, or measure it back-to-back |
| cross-correlation      | no                    | yes, in expectation, declaring nothing |

The design matrix has a column per source and a column per undeclared
channel. The odd cycle is what makes the source columns solvable; a
repeated edge is what makes a channel column *unnecessary*. A bench that
has both needs neither assumption.

That composition is the useful part, and it is concrete: measure A against
B **twice, through two independent channels**, and the cross-variance of
those two records estimates `sigma2_A + sigma2_B` with the channel noise
rejected. That is a hat row with no channel term. For the classical
three-cornered hat -- three sources through one channel, four unknowns
against three rows, hopeless -- one such repeated edge closes the system
without anyone having to assert that the channel is silent.
`would_identify` computes exactly that, by adding the derived row to a
hypothetical bench and re-running `check`.

**The correction this module exists to make.** It is tempting, and it was
written down here in an earlier draft, that two channels of one instrument
are the *good* case for cross-correlation because the shared instrument
noise is what you want to reject. That is backwards. Cross-correlation
rejects the part of the channel noise that is **independent** between the
two channels; the part they share -- one timebase, one LO, one supply, one
oven -- passes straight through:

    x_1 = x_ab + n_common + n_1'
    x_2 = x_ab + n_common + n_2'
    E[d_1 d_2] = Var(x_ab) + Var(n_common)

So two channels of one device is *better than one channel* and *worse than
two devices*, and the residual is not something more data fixes -- it is a
floor. **Independent instruments are better for both techniques.** What is
true of the same-device case is only that it is a configuration
cross-correlation can use at all and the hat cannot, since both records
name the same source pair.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import check
from .model import Bench, Channel, Device, Measurement

__all__ = ["CrossOpportunity", "Recommendation", "recommend",
           "cross_opportunities"]

# A derived cross-correlation row is attributed to this synthetic channel,
# whose floor is zero because rejecting it is what the technique does. The
# name is deliberately unusable as a real channel id so it cannot collide
# with anything a user wrote.
DERIVED_CHANNEL = "__cross__"


@dataclass(frozen=True)
class CrossOpportunity:
    """One source pair measured through more than one channel."""

    source_a: str
    source_b: str
    measurements: tuple[str, ...]
    channels: tuple[str, ...]
    devices: tuple[str, ...]
    # "clean" when no two of the channels share a device, so the only
    # correlation left is the environment they sit in. "common-mode" when
    # they do, which caps the rejection at a floor more data cannot lower.
    quality: str

    @property
    def clean(self) -> bool:
        return self.quality == "clean"


@dataclass(frozen=True)
class Recommendation:
    """What this bench can do, and which assumption each answer rests on."""

    opportunities: tuple[CrossOpportunity, ...]
    # Whether the hat alone determines the sources -- `check(bench).identifiable`.
    hat_identifiable: bool
    hat_deficiency: int
    # Whether adding the derived channel-free rows would close a bench the
    # hat cannot close on its own. False when there is nothing to close.
    would_identify: bool
    # Channels that become *determined* once the derived rows are in --
    # not merely unnecessary. The derived row carries no channel term, so
    # subtracting it from a real row leaves that row's channel alone.
    channels_determined: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def has_cross(self) -> bool:
        return bool(self.opportunities)

    def report(self) -> str:
        lines = ["technique:"]
        if self.hat_identifiable:
            lines.append("  N-cornered hat: yes -- the graph determines every "
                         "source it measures.")
        else:
            lines.append(
                f"  N-cornered hat: no -- {self.hat_deficiency} unknown"
                f"{'' if self.hat_deficiency == 1 else 's'} undetermined. "
                "See `bench check` for the edge that would fix it.")
        if not self.opportunities:
            lines.append("  cross-correlation: no -- no source pair is "
                         "measured through two channels.")
        for o in self.opportunities:
            how = ("independent instruments, so the channel noise cancels"
                   if o.clean else
                   f"channels of one instrument ({o.devices[0]}), so only the "
                   "part they do NOT share cancels")
            lines.append(
                f"  cross-correlation: {o.source_a} vs {o.source_b} via "
                f"{', '.join(o.measurements)} -- {how}")
        if self.would_identify:
            lines.append(
                "  the two compose: cross-correlating the repeated edge gives "
                "a row with no channel term, which closes this bench without "
                "declaring a floor. Subtracting it from a real row then "
                f"measures that row's channel: {', '.join(self.channels_determined)}.")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def cross_opportunities(bench: Bench) -> tuple[CrossOpportunity, ...]:
    """Source pairs measured through two or more distinct channels.

    Back-to-back rows are included: two channels measuring the *same*
    source against itself cross-correlate to the source's own variance
    with both channels rejected, which is the purest form of the technique
    and the one a phase-noise test set implements in hardware.

    Two records through *one* channel are not an opportunity. One channel
    cannot make two simultaneous measurements, so either they are the same
    record described twice or they ran at different times -- and neither
    gives two independent views of one interval.
    """
    groups: dict[tuple[str, str], list[Measurement]] = {}
    for m in bench.measurements:
        key = (m.source_a, m.source_b) if m.source_a <= m.source_b else (
            m.source_b, m.source_a)
        groups.setdefault(key, []).append(m)

    owner = bench.channel_index()
    out = []
    for (a, b), rows in sorted(groups.items()):
        by_channel: dict[str, Measurement] = {}
        for m in rows:
            by_channel.setdefault(m.channel_id, m)
        if len(by_channel) < 2:
            continue
        channels = tuple(sorted(by_channel))
        devices = tuple(sorted({owner[c] for c in channels}))
        out.append(CrossOpportunity(
            source_a=a, source_b=b,
            measurements=tuple(by_channel[c].id for c in channels),
            channels=channels, devices=devices,
            quality="clean" if len(devices) == len(channels) else "common-mode",
        ))
    return tuple(out)


def recommend(bench: Bench) -> Recommendation:
    """Which technique this bench supports, and whether they compose.

    Reads no data files, for the same reason `check` does not: the answer
    is worth most while the experiment is still being planned. "Run that
    pair again on the other counter" is cheap advice on a Tuesday.
    """
    opportunities = cross_opportunities(bench)
    ident = check(bench)
    clean = [o for o in opportunities if o.clean]

    freed: tuple[str, ...] = ()
    would = False
    if clean and not ident.identifiable:
        hypothetical = _with_derived_rows(bench, clean)
        after = check(hypothetical)
        would = after.identifiable
        freed = tuple(sorted({c for o in clean for c in o.channels}))

    return Recommendation(
        opportunities=opportunities,
        hat_identifiable=ident.identifiable,
        hat_deficiency=ident.deficiency,
        would_identify=would,
        channels_determined=freed if would else (),
        notes=_notes(bench, opportunities, ident, would),
    )


def _with_derived_rows(bench: Bench, clean) -> Bench:
    """The bench plus one channel-free row per clean opportunity.

    Hypothetical and never returned to the caller. The derived rows go on a
    synthetic channel whose floor is declared zero, which is what
    `design_matrix` needs in order to give them no column -- and which is
    not an assumption here but a description of what the cross-variance
    did.
    """
    device = Device(id=DERIVED_CHANNEL,
                    channels=(Channel(id=DERIVED_CHANNEL, floor=0.0),))
    rows = tuple(
        Measurement(id=f"{DERIVED_CHANNEL}_{i}", channel_id=DERIVED_CHANNEL,
                    source_a=o.source_a, source_b=o.source_b,
                    reason="derived by cross-correlating "
                           f"{' and '.join(o.measurements)}")
        for i, o in enumerate(clean)
    )
    return Bench(sources=bench.sources,
                 devices=bench.devices + (device,),
                 measurements=bench.measurements + rows,
                 schema=bench.schema)


def _notes(bench, opportunities, ident, would) -> tuple[str, ...]:
    """What the verdict alone does not say."""
    notes = []
    common = [o for o in opportunities if not o.clean]
    for o in common:
        notes.append(
            f"{o.source_a} vs {o.source_b} is measured through "
            f"{len(o.channels)} channels of one instrument "
            f"({o.devices[0]}). Cross-correlating them rejects the part of "
            "the channel noise those channels do not share, and nothing of "
            "the part they do -- one timebase, one LO, one supply, one oven. "
            "The residual is a floor, not something more data lowers. Two "
            "instruments would reject all of it.")
    if not opportunities and not ident.identifiable:
        free = [c for c in bench.used_channels
                if bench.channel(c).floor is None]
        if free:
            notes.append(
                "no pair is measured twice, so every channel here is an "
                f"unknown ({', '.join(free)}) that something must pin. The "
                "options are: declare the floor, add a back-to-back run to "
                "measure it, or run one existing pair again through an "
                "independent channel and cross-correlate -- which assumes "
                "nothing at all.")
    if would:
        notes.append(
            "this is the classical three-cornered hat's missing row. The "
            "textbook method needs sigma2_channel = 0 asserted and unproven; "
            "a repeated edge measures the same thing instead.")
    if opportunities and ident.identifiable:
        notes.append(
            "both techniques apply. They answer different questions -- the "
            "hat splits the sources, the cross-variance removes the channel "
            "-- so run both and compare: a disagreement is the channel floor "
            "the hat had to assume.")
    return tuple(notes)
