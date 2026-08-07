# References and reference data (verified 2026-08-07)

Status labels follow the same convention as `gpu-power-metrology`:
"Confirmed" means the URL was fetched and its content inspected during the
session noted above. Anything not directly checked says so.

## Standards and primary literature

| Citation | Status | Note |
|---|---|---|
| NIST SP 1065 (Riley), *Handbook of Frequency Stability Analysis*, 2008 | Reported, not fetched this session | The field's handbook. Table 31 (p. 108) carries the NBS14 1000-point expected deviations. Those values were produced by Stable32 — see the epistemic caveat below. |
| IEEE Std 1139 | Not fetched | Definitions of the variance family. |
| Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite Differences", PTTI 2003 | Reported, not fetched | The EDOF algorithm Stable32 uses for its Min/Max Sigma columns. Needed for CI parity at M3. |
| Riley & Greenhall, "Power Law Noise Identification Using the Lag 1 Autocorrelation", EFTF 2004 | Reported, not fetched | The lag-1 ACF noise-ID method. |
| Percival & Walden, *Wavelet Methods for Time Series Analysis*, CUP 2000 | Not fetched | MODWT, the unbiased wavelet variance estimator, and the EDOF estimators η₁/η₃. |
| Percival & Guttorp, "Long-memory processes, the Allan variance and wavelets", 1994 | Not fetched | The Haar-wavelet-variance = Allan-variance equivalence this library is built around. |
| Galleani & Tavella, "The Dynamic Allan Variance" | Not fetched | DAVAR definition and normalization. Needed at M5. |
| Howe & Percival, TOTVAR bias | Not fetched | TOTDEV is biased low for FM noise; the correction needs a noise type, which couples TOTVAR to noise ID. |

## Reference data

| Source | Status | Note |
|---|---|---|
| [`wriley.com/nbs.dat`](http://www.wriley.com/nbs.dat) | **Confirmed** | Fetched and inspected. Exactly nine values: 892, 809, 823, 798, 671, 644, 883, 903, 677. The classic set from NBS Monograph 140 Annex 8.E. Small enough to hand-check. |
| [`wriley.com/tst_suit.dat`](http://www.wriley.com/tst_suit.dat) | **Confirmed** | Fetched. 1000 lines, single column, `%.15e`, values in (0,1). The Stable32 1000-point pseudo-random frequency test suite. |
| [allantools `tests/phasedat/`](https://github.com/aewallin/allantools/tree/master/tests/phasedat) | **Confirmed** | Directory listing fetched: `PHASE.DAT` plus ~28 Stable32 reference outputs covering adev, oadev, mdev, tdev, hdev, ohdev, totdev, ttotdev, mtotdev, htotdev, theo1, mtie, tierms at all/octave/decade τ. |
| `phase_dat_adev_octave.txt` | **Confirmed** | Fetched and read. Literal **Stable32 1.60** output including the header, and crucially the `Alpha`, `Min Sigma`, `Max Sigma` columns — Greenhall-EDOF 68.3% bounds. Reproducing those is the strongest parity claim available. |
| [allantools `tests/`](https://github.com/aewallin/allantools/tree/master/tests) | **Confirmed** | Listing fetched. Contains `Cs5071A`, `ocxo`, `gps`, `pink_frequency`, `nbs14`, `realtime`, and **`Keysight53230A_ti_noise_floor`** — a real noise-floor set from the same instrument family this project targets. |
| NBS14 1000-point Lehmer/MINSTD recursion | Reported, **not yet verified in code** | `n₀ = 1234567890; nᵢ₊₁ = 16807·nᵢ mod 2147483647; y = n/2147483647`, with `n₁ = 395529916` as the check value. If it holds, the fixture needs no data file and carries no licence question. **M0's test verifies this.** |
| NBS14 expected deviations | Reported, not independently checked | ADEV(1) = 2.922319e-01, ADEV(10) = 9.965736e-02 for the 1000-point set; ADEV(1) = 91.22945, HDEV(1) = 70.80608 for the 9-point set. Seven significant figures, so assert at `rtol=1e-6`. |
| [Stable32 datasheet](http://www.wriley.com/datasht4.htm) | **Confirmed** | Fetched. Source of the parity matrix — the authoritative list of what Stable32 actually does. |
| [Stable32 program, IEEE UFFC](https://ieee-uffc.org/software) | **Confirmed reachable** (HTTP 200) | Free since ownership transferred to IEEE UFFC in Dec 2017. Mirror at `wriley.com/162Stable32.exe`. Needed to generate our own reference outputs. |
| allantools licence | **Unverified** — the raw `LICENSE` URL returned 404 | Believed LGPL-3.0. Until confirmed, treat its fixtures as un-vendorable and use it only as a test-only differential-testing dependency. Resolve before committing anything derived from that repo. |

## The epistemic caveat

Every published reference ADEV table traces back to Stable32. NIST SP 1065
Table 31 is Riley's own tool's output; allantools documents its expected
values as "Stable32 calculations". Validating against them demonstrates
**interoperability, not correctness** — which is commercially necessary and
epistemically weak.

Correctness therefore rests on analytic identities asserted at machine
precision, and on measured Monte-Carlo coverage of the confidence
intervals. Reference fixtures are labelled as interop assertions in the
test docstrings, deliberately.

## Open

- Confirm the allantools licence before any fixture is copied.
- Generate our own Stable32 reference outputs natively on Windows, over
  `PHASE.DAT`, the NBS suites, and our own WR/OCXO phase records, and
  commit them with a `PROVENANCE.md` recording version, date, input
  SHA-256 and menu settings.
