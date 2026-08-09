"""The bench model: sources, devices, channels, and what they determine."""

from .graph import (Identifiability, check, derivable_edges, design_matrix,
                    suggest_edge)
from .recommend import (CrossOpportunity, Recommendation, cross_opportunities,
                        recommend)
from .model import SCHEMA_VERSION, Bench, Channel, Device, Measurement, Source
# Re-exported because `from stabilis.bench import load` is the obvious way to
# reach for it, and two call sites in the CLI already wrote it that way and
# would have raised ImportError the first time a user passed `report --bench`.
from .store import HashMismatch, from_dict, load, save, to_dict, verify

__all__ = [
    "SCHEMA_VERSION",
    "Bench",
    "Source",
    "Device",
    "Channel",
    "Measurement",
    "Identifiability",
    "check",
    "design_matrix",
    "suggest_edge",
    "derivable_edges",
    "CrossOpportunity",
    "Recommendation",
    "cross_opportunities",
    "recommend",
    "load",
    "save",
    "to_dict",
    "from_dict",
    "verify",
    "HashMismatch",
]
