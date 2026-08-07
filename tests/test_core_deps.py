"""The core is numpy-only.

This is a selling point -- customers embed the engine in production test
rigs, where a heavy dependency tree is a sales objection -- and it is exactly
the kind of property that regresses silently the first time someone reaches
for `scipy.special`. So it is asserted, in a subprocess, on every run.
"""

import subprocess
import sys

_PROBE = """
import sys
import stabilis
import stabilis.io
import stabilis.preprocess
import stabilis.reference
import stabilis.registry
import stabilis.synth
import stabilis.tau
import stabilis.types

import numpy as np
from stabilis.synth import power_law_phase
from stabilis.types import to_frequency

ps = power_law_phase(256, -1, 1.0, np.random.default_rng(0))
assert to_frequency(ps).n == 256

leaked = sorted(m for m in ("scipy", "matplotlib", "pandas") if m in sys.modules)
print("LEAKED:" + ",".join(leaked))
"""


def test_no_heavy_imports_and_core_computes():
    """Importing and computing must not pull in scipy, matplotlib or pandas."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip().removeprefix("LEAKED:")
    assert leaked == "", f"core imported optional dependencies: {leaked}"
