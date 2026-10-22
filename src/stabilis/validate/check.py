"""The record of one validation claim, and of a whole run.

Shared by every tier module; nothing here computes evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..stats import adev, mdev, oadev, ohdev, tdev
from .certified import (
    exact_avar,
    exact_mvar,
    exact_ohvar,
    exact_oavar,
    exact_tvar,
)

__all__ = ["TIERS", "ESTIMATORS", "SEEDS", "MIN_COVERAGE_TRIALS", "Check",
           "Report"]


TIERS = ("identity", "certified", "coverage", "interop", "self-test")


ESTIMATORS = [
    ("adev", adev, exact_avar),
    ("oadev", oadev, exact_oavar),
    ("mdev", mdev, exact_mvar),
    ("tdev", tdev, exact_tvar),
    ("ohdev", ohdev, exact_ohvar),
]


# Published so the Monte-Carlo tiers can be repeated by a third party.
SEEDS = {"coverage": 22, "gaps": 7, "stream": 3}


# Coverage cannot be measured meaningfully with fewer than this many
# realizations -- the estimate's own standard error would swamp the effect
# being checked. A lower request is raised to this rather than producing a
# number that looks measured and is not.
MIN_COVERAGE_TRIALS = 300


@dataclass(frozen=True)
class Check:
    """One assertion, with the tolerance it was made at and why."""

    tier: str
    name: str
    claim: str
    tolerance: str
    measured: str
    passed: bool

    def __post_init__(self) -> None:
        # numpy comparisons yield np.bool_, which json refuses. The report
        # is serialized and hashed, so coerce once here rather than at
        # every call site and hope none is missed.
        object.__setattr__(self, "passed", bool(self.passed))
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")


@dataclass(frozen=True)
class Report:
    """The whole evidence run: every check, the environment, the digest."""

    checks: tuple[Check, ...]
    environment: dict
    seeds: dict
    digest: str
    generated: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def by_tier(self, tier: str) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.tier == tier)

