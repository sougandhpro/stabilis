"""Stability estimators."""

from .bias import deviation_bias_factor, unbias_deviation
from .ci import ONE_SIGMA, TWO_SIGMA, Confidence, confidence
from .classical import adev, mdev, oadev, ohdev, tdev
from .core import Shape, StabilityResult
from .edof import edof, eta1, eta3
from .greenhall import greenhall_edof, greenhall_edof_pooled
from .mtie import mtie, tierms
from .theo1 import theo1
from .theoh import theobr, theoh
from .wavelet import wavelet_variance, wdev
from .total import totdev
from .mtotal import htotdev, mtotdev, total_subsequence_terms, ttotdev
from .noiseid import NoiseID, identify, identify_curve
from .powerlaw import PowerLawFit, decompose
from .psd import Spectrum, script_l, welch
from .spur import Spur, SpurTable, detect_spurs, notch_spurs, sinusoid_dev
from .taucov import TauCovariance, tau_covariance

__all__ = [
    "Shape",
    "StabilityResult",
    "adev",
    "oadev",
    "mdev",
    "tdev",
    "ohdev",
    "mtie",
    "tierms",
    "totdev",
    "mtotdev",
    "ttotdev",
    "htotdev",
    "total_subsequence_terms",
    "theo1",
    "theobr",
    "theoh",
    "wdev",
    "wavelet_variance",
    "Confidence",
    "confidence",
    "ONE_SIGMA",
    "TWO_SIGMA",
    "edof",
    "eta1",
    "eta3",
    "greenhall_edof",
    "greenhall_edof_pooled",
    "NoiseID",
    "identify",
    "identify_curve",
    "deviation_bias_factor",
    "unbias_deviation",
    "PowerLawFit",
    "decompose",
    "Spectrum",
    "welch",
    "script_l",
    "Spur",
    "SpurTable",
    "detect_spurs",
    "notch_spurs",
    "sinusoid_dev",
    "TauCovariance",
    "tau_covariance",
]
