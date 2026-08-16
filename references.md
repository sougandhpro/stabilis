# References and reference data (verified 2026-08-07)

Status labels follow the same convention as `gpu-power-metrology`:
"Confirmed" means the URL was fetched and its content inspected during the
session noted above. Anything not directly checked says so.

## Standards and primary literature

| Citation | Status | Note |
|---|---|---|
| [NIST SP 1065 (Riley), *Handbook of Frequency Stability Analysis*, 2008](https://tf.nist.gov/general/pdf/2220.pdf) | **Confirmed — read 2026-08-08** | The field's handbook, 136 pages. **Use the `tf.nist.gov` copy, not the `nvlpubs.nist.gov` one**: they are the same document, but nvlpubs serves an OCR'd scan that renders `1.00000e+00` as `l.OOOOOe+00` and mangles every superscript, while tf.nist.gov has the real text layer from the original Word document. Transcribing digits from the scan would be exactly the failure mode `CLAUDE.md` forbids. Taken from it: **section 7.1 (p. 73–74)**, eq. (63)–(67) and the conversion table, into `powerlaw_coeffs.py` and `theory/powerlaw.md` — with a **contradiction in the source recorded, not silently corrected**: the flicker-PM row is printed one way in eq. (67) and another in the p. 74 table, the two differ by a factor of two in the denominator and by `tau` vs `tau0` inside the logarithm, and the handbook's own eq. (65) supports neither printed form exactly but the third combination of them. Also **Table 30 (p. 107) and Table 31 (p. 108)**, the NBS14 expected deviations, transcribed directly into `reference.py` — which corrected two relayed values, `hdev(1) = 70.80607` (not …08) and 1000-point `hdev(100) = 3.910861e-02` (not …860e-02). Section 7.1 gives **no** modified-Allan conversions, so those are derived from eq. (66) rather than transcribed. Those values were produced by Stable32 — see the epistemic caveat below. |
| IEEE Std 1139 | Not fetched | Definitions of the variance family. |
| [ITU-T Recommendation G.811 (09/97), *Timing characteristics of primary reference clocks*](https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-G.811-199709-I!!PDF-E&type=items) | **Confirmed — fetched and text-extracted 2026-08-16** | Clause 6.1, transcribed into `mask.CATALOG`. **MTIE:** `0.275 × 10⁻³ τ + 0.025 µs` for `0.1 < τ ≤ 1000 s`, then `10⁻⁵ τ + 0.29 µs` for `τ > 1000 s`. **TDEV:** `3 ns` for `0.1 < τ ≤ 100 s`, `0.03 τ ns` for `100 < τ ≤ 1000 s`, `30 ns` for `1000 < τ < 10 000 s`. Both are continuous at every breakpoint, which is the arithmetic check that the digits went in correctly, and both are stored converted to SI seconds. Fetching this mattered: the widely repeated form of the second MTIE branch is a flat 300 ns ceiling, and the Recommendation says `10⁻⁵ τ + 0.29 µs`, which *rises* — the two agree only at τ = 1000 s. Clause 6 also states measurement conditions that are part of the limit and are enforced as such: an equivalent 10 Hz first-order low-pass filter (recorded, not checkable here), a sampling time `τ₀ ≤ 1/30 s`, and for TDEV a record of at least `12τ`. The TDEV mask **ends** at 10 000 s; beyond it the Recommendation states no limit. |
| [Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite Differences", 35th PTTI, 2003](http://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf) | **Confirmed — read in full 2026-08-08** | The EDOF algorithm Stable32 uses for its Min/Max Sigma columns. The copy at that URL is the preprint dated 3 October 2003, 9 pages; §5 "Examples" and §6 "Conclusions" are **empty headings**, so the paper supplies no worked edf values and this project has no primary-source fixture table for it. Everything else is complete: the GACV table (eq. 7), `sx` (8), `sz` (9), `BasicSum` (10), `M` and `J` (11–12), the simplified procedure (13), the four-case full procedure, and Tables 1–3 of the large-`r` coefficients. Transcribed into `theory/greenhall.md`. Two things to carry forward — α is the *frequency* exponent (`S_y(f) ~ f^α`), the same convention as `noiseid.ALPHA_NAMES`, so no conversion is needed anywhere; and the Table 3 caption reads "FLFM (α = 1)" where §4.6.3 and the input table both say α = 1 is **FLPM**, a caption slip in the source that is recorded here rather than silently corrected. |
| Riley & Greenhall, "Power Law Noise Identification Using the Lag 1 Autocorrelation", EFTF 2004 | Reported, not fetched | The lag-1 ACF noise-ID method. |
| Percival & Walden, *Wavelet Methods for Time Series Analysis*, CUP 2000 | Not fetched | MODWT, the unbiased wavelet variance estimator, and the EDOF estimators η₁/η₃. |
| Percival & Guttorp, "Long-memory processes, the Allan variance and wavelets", 1994 | Not fetched | The Haar-wavelet-variance = Allan-variance equivalence this library is built around. |
| [Galleani & Tavella, "Fast Computation of the Dynamic Allan Variance", *Proc. 2009 IEEE Int. Frequency Control Symposium*, 685–687](https://engineering.purdue.edu/oxidemems/conferences/fcs2009/PDFs/Papers/140_7132.pdf) | **Confirmed — read in full 2026-08-16** | The DAVAR definition used by M6. Eq. (9): `σ²_y[n,k] = 1/(2k²τ₀²) · 1/(N_w−2k) · Σ_{m=n−N_w/2}^{n+N_w/2−2k−1} (x[m+2k] − 2x[m+k] + x[m])²` — i.e. **exactly the overlapping Allan variance of the `N_w` phase samples centred on `n`**, normalized by the nominal term count, with no separate normalization to reconstruct. Transcribed into `theory/davar.md`. The paper's own contribution is the O(1) recursion, eq. (13), which adds the entering `Δ²_k` and subtracts the expiring one; **this library deliberately does not use it**, and the reason is measured rather than assumed — see `theory/davar.md`. Worth recording about the source itself: it presents that recursion purely for speed and **contains no discussion of its numerical behaviour** — "accuracy", "numeric", "rounding" and "error" do not occur in the text at all. Also note eq. (7), which stops the estimator at `k = N/2 − 1`, the same algebraic limit this library enforces. |
| Galleani & Tavella, "The Dynamic Allan Variance", *IEEE Trans. UFFC* **56**(3), 450–464, March 2009 | Not fetched (paywalled) | The full treatment; reference [2] of the FCS paper above, which restates its definition. Nothing in M6 depends on anything beyond what the FCS paper states. |
| Howe & Percival, TOTVAR bias | Not fetched | TOTDEV is biased low for FM noise; the correction needs a noise type, which couples TOTVAR to noise ID. |
| [Howe & Schlossberger, "Characterizing Frequency Stability Measurements Having Multiple Data Gaps", *IEEE Trans. UFFC* **69**(2) 468–472, 2022](https://tf.nist.gov/general/pdf/3150.pdf) | **Confirmed — read in full 2026-08-08** | The gap-filling method for M7, "Total Imputer". Fills a gap of `n` points with the preceding `n` points reversed and inverted, sloped to join continuously, with endpoints matched on a low-pass-filtered copy (their eq. 3) rather than pointwise. Multi-gap order: start from the longest run, impute rightward, reverse, repeat. Two things to carry forward — imputation *raises* EDOF by up to 6× for white noise, so filled samples must be excluded from the degrees-of-freedom count; and the authors themselves state imputation is not a substitute for a measurement deficiency and that gaps must be declared with results. |
| Mondal & Percival, "Wavelet variance analysis for gappy time series", *Ann. Inst. Stat. Math.* **62**(5) 943–966, 2010 | Reported (cited as [7] by Howe & Schlossberger), not fetched | The imputation-*free* alternative: an unbiased wavelet variance for gapped series. Since Haar wavelet variance is the Allan variance, this drops into the M3 wavelet module and gives a gap-tolerant ADEV that invents no samples and inflates no EDOF. Shipping it alongside the imputer, and reporting where the two disagree, is a differentiator. |

## Third-party code we do not use

| Source | Licence | Decision |
|---|---|---|
| [`nkschlos/time-series-imputation`](https://github.com/nkschlos/time-series-imputation) — the authors' reference implementation of Total Imputer | **None.** GitHub API reports `license: null`, and there is no `LICENSE` file. All rights reserved. | **Do not vendor, port, or copy.** The published paper describes the algorithm completely enough to implement independently, which is what M7 does. Cite the paper, not the repository. |
| `allantools` fixtures | LGPL-3.0 (unconfirmed, see below) | Test-only differential-testing dependency. No fixture is copied into this MPL repo. |
| `allantools.edf_greenhall` | LGPL-3.0 (unconfirmed, see below) | It implements the same paper. **Not read and not ported** — `stats/greenhall.py` was written from the PTTI preprint alone, before that function was looked at. It may be called as a *numeric* differential-test oracle only, under `pytest.importorskip`, which is comparing two independent implementations of a published algorithm and copies nothing. |

## Reference data

| Source | Status | Note |
|---|---|---|
| [`wriley.com/nbs.dat`](http://www.wriley.com/nbs.dat) | **Confirmed** | Fetched and inspected. Exactly nine values: 892, 809, 823, 798, 671, 644, 883, 903, 677. The classic set from NBS Monograph 140 Annex 8.E. Small enough to hand-check. |
| [`wriley.com/tst_suit.dat`](http://www.wriley.com/tst_suit.dat) | **Confirmed** | Fetched. 1000 lines, single column, `%.15e`, values in (0,1). The Stable32 1000-point pseudo-random frequency test suite. |
| [allantools `tests/phasedat/`](https://github.com/aewallin/allantools/tree/master/tests/phasedat) | **Confirmed** | Directory listing fetched: `PHASE.DAT` plus ~28 Stable32 reference outputs covering adev, oadev, mdev, tdev, hdev, ohdev, totdev, ttotdev, mtotdev, htotdev, theo1, mtie, tierms at all/octave/decade τ. |
| `phase_dat_adev_octave.txt` | **Confirmed** | Fetched and read. Literal **Stable32 1.60** output including the header, and crucially the `Alpha`, `Min Sigma`, `Max Sigma` columns — Greenhall-EDOF 68.3% bounds. Reproducing those is the strongest parity claim available. |
| [allantools `tests/`](https://github.com/aewallin/allantools/tree/master/tests) | **Confirmed** | Listing fetched. Contains `Cs5071A`, `ocxo`, `gps`, `pink_frequency`, `nbs14`, `realtime`, and **`Keysight53230A_ti_noise_floor`** — a real noise-floor set from the same instrument family this project targets. |
| NBS14 1000-point Lehmer/MINSTD recursion | Reported, **not yet verified in code** | `n₀ = 1234567890; nᵢ₊₁ = 16807·nᵢ mod 2147483647; y = n/2147483647`, with `n₁ = 395529916` as the check value. If it holds, the fixture needs no data file and carries no licence question. **M0's test verifies this.** |
| NBS14 expected deviations | **Confirmed — transcribed from SP 1065 Tables 30 and 31, 2026-08-08** | ADEV(1) = 2.922319e-01, ADEV(10) = 9.965736e-02 for the 1000-point set; ADEV(1) = 91.22945, HDEV(1) = **70.80607** for the 9-point set. The relayed HDEV(1) had been 70.80608, and the exact-rational oracle had already flagged it as one unit high in the seventh figure; the handbook agrees with the oracle, so the disagreement belonged to the intermediate source and `KNOWN_DISCREPANCIES` is now empty. Seven significant figures, so assert at `rtol=1e-6`. |
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
