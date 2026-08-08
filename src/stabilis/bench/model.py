"""The bench: which oscillator, against which, through which instrument.

Every other tool treats a measurement as an anonymous column of numbers.
But `x_ab(t)` is never just a column -- it is *this* source against *that*
one, through *this* channel of *that* device. Once the program knows that,
four things become mechanical rather than manual: the N-cornered hat,
instrument noise-floor subtraction, common-mode detection, and the honest
refusal that says a setup cannot separate its oscillators no matter how
long it runs.

Data only. No behaviour beyond validation, and **`stats/` must never import
`bench/`** -- an AST test enforces it, the same independence discipline the
certified oracle has. The estimators must stay usable, and testable, with
no notion of where their data came from.

Two conventions that look like mistakes and are not:

* **`source_a == source_b` is legal.** That is a back-to-back run, both
  ends of the measurement on one source, and it is how a channel's noise
  floor gets *measured* rather than assumed. It contributes nothing to
  separating oscillators and everything to pinning its channel.
* **`spec_adev` is a vendor's claim, not a measurement.** It is carried so
  a result can be plotted against the specification the part was bought to
  meet, and it is never mixed into a fit. The validation-integrity rule
  applies to datasheet points exactly as it does to published tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SCHEMA_VERSION",
    "Source",
    "Channel",
    "Device",
    "Measurement",
    "Bench",
]

# Bumped when the JSON layout changes in a way an older reader would
# misread. `store.load` refuses a version it does not know rather than
# guessing at the difference.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Source:
    """An oscillator, maser, GPSDO -- anything with its own phase.

    `spec_adev` is `((tau_s, adev), ...)` straight off a datasheet, and
    `spec_floor` the vendor's claimed noise floor. Both are claims, marked
    as such wherever they appear.
    """

    id: str
    name: str = ""
    kind: str = ""
    nominal_hz: float | None = None
    spec_adev: tuple[tuple[float, float], ...] = ()
    spec_floor: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.id, "Source")
        for tau, adev in self.spec_adev:
            if tau <= 0.0 or adev < 0.0:
                raise ValueError(
                    f"Source {self.id!r}: spec_adev needs tau > 0 and adev >= 0, "
                    f"got ({tau}, {adev})"
                )


@dataclass(frozen=True)
class Channel:
    """One measurement path through a device -- "A-1", say.

    `floor` is a *measured* channel noise floor if one is known ahead of
    time. Leaving it `None` is the better habit: a back-to-back
    `Measurement` lets the solver determine it from data instead.
    """

    id: str
    label: str = ""
    floor: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.id, "Channel")
        if self.floor is not None and self.floor < 0.0:
            raise ValueError(
                f"Channel {self.id!r}: floor is a variance and cannot be "
                f"negative, got {self.floor}"
            )


@dataclass(frozen=True)
class Device:
    """A measurement instrument, owning its channels.

    Channels belong to a device rather than floating free because that
    ownership is what makes the common-mode check possible: two channels of
    one device share a timebase, an LO, a power supply and a thermal
    environment, so their noise is common and does not cancel.
    """

    id: str
    model: str = ""
    serial: str = ""
    channels: tuple[Channel, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        _require_id(self.id, "Device")
        seen = set()
        for ch in self.channels:
            if ch.id in seen:
                raise ValueError(f"Device {self.id!r}: duplicate channel {ch.id!r}")
            seen.add(ch.id)


@dataclass(frozen=True)
class Measurement:
    """One record: two sources, one channel, one file.

    The file is referenced by path and SHA-256, never contained. `store`
    verifies the hash on load and refuses to analyse a bench whose data has
    changed underneath it -- a measurement whose bytes moved is a different
    measurement.
    """

    id: str
    channel_id: str
    source_a: str
    source_b: str
    path: str = ""
    sha256: str = ""
    kind: str = "phase"
    tau0_s: float | None = None
    epoch: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        _require_id(self.id, "Measurement")
        if self.kind not in ("phase", "frequency"):
            raise ValueError(
                f"Measurement {self.id!r}: kind must be 'phase' or "
                f"'frequency', got {self.kind!r}"
            )
        if self.tau0_s is not None and self.tau0_s <= 0.0:
            raise ValueError(
                f"Measurement {self.id!r}: tau0_s must be positive, "
                f"got {self.tau0_s}"
            )

    @property
    def is_back_to_back(self) -> bool:
        """Both ends on one source: this row measures its channel, nothing else."""
        return self.source_a == self.source_b


@dataclass(frozen=True)
class Bench:
    """A whole setup. Referentially checked on construction.

    The checking is the point. A bench that names a channel no device owns,
    or a source no measurement can reach, is a typo that would otherwise
    surface as a confusing rank deficiency three steps later.
    """

    sources: tuple[Source, ...] = ()
    devices: tuple[Device, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    schema: int = SCHEMA_VERSION
    created: str = ""
    notes: str = ""
    path: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        _no_duplicates([s.id for s in self.sources], "source")
        _no_duplicates([d.id for d in self.devices], "device")
        _no_duplicates([m.id for m in self.measurements], "measurement")
        _no_duplicates([c.id for d in self.devices for c in d.channels], "channel")

        sources = {s.id for s in self.sources}
        channels = set(self.channel_index())
        for m in self.measurements:
            for ref, what in ((m.source_a, "source_a"), (m.source_b, "source_b")):
                if ref not in sources:
                    raise ValueError(
                        f"Measurement {m.id!r}: {what}={ref!r} is not a source "
                        f"in this bench. Known: {sorted(sources)}"
                    )
            if m.channel_id not in channels:
                raise ValueError(
                    f"Measurement {m.id!r}: channel {m.channel_id!r} is not a "
                    f"channel of any device here. Known: {sorted(channels)}"
                )

    def channel_index(self) -> dict[str, str]:
        """`channel id -> owning device id`, for the common-mode checks."""
        return {c.id: d.id for d in self.devices for c in d.channels}

    def channel(self, channel_id: str) -> Channel:
        for d in self.devices:
            for c in d.channels:
                if c.id == channel_id:
                    return c
        raise KeyError(f"no channel {channel_id!r} in this bench")

    def source(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"no source {source_id!r} in this bench")

    @property
    def measured_sources(self) -> tuple[str, ...]:
        """Sources some measurement actually touches, in bench order.

        A source declared but never measured has no column in the solve. It
        is not an error -- benches get written before data is taken -- but
        it must not silently become an unidentifiable unknown.
        """
        used = {m.source_a for m in self.measurements}
        used |= {m.source_b for m in self.measurements}
        return tuple(s.id for s in self.sources if s.id in used)

    @property
    def used_channels(self) -> tuple[str, ...]:
        seen, out = set(), []
        for m in self.measurements:
            if m.channel_id not in seen:
                seen.add(m.channel_id)
                out.append(m.channel_id)
        return tuple(out)


def _require_id(value: str, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what}: id must be a non-empty string, got {value!r}")


def _no_duplicates(ids: list[str], what: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {what} id {i!r}")
        seen.add(i)
