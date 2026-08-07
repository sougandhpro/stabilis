# Provenance and the step log

## Why it is in the core

The commercial layer prints signed uncertainty statements. IEEE Std
1139-2022 clause 4.2 makes it normative that *any pretreatment on measured
data shall be stated together with the results*, and ISO/IEC 17025:2017
clause 7.11.6 requires that calculations and data transfers be checked in
an appropriate and systematic manner.

Both of those are questions about what the tool did, not about what it
computed. So provenance is a core concern, present from the first commit,
rather than something the reporting layer bolts on. Retrofitting it after
the API has users is painful, and it is the only place where the paid layer
imposes a hard requirement on the free core.

## What is recorded

`Provenance` carries the library version, the numpy version, a UTC
creation timestamp, the source description, the SHA-256 of the input file
where there was one, and an ordered tuple of `Step` records.

Each `Step` has a timestamp, the module, the action, the parameters that
went in, and the outcome. The outcome field is the interesting one: it
records *what was chosen*, including "fell back to X because Y".

## The rule for what gets logged

Log every decision the library takes on the caller's behalf, and especially
the ones taken by default. The tau grid and why. The gap policy. Which EDOF
method actually ran, as opposed to which was requested. Whether `f_h` was
supplied or assumed. Whether a precision warning fired. Any estimator that
refused to report a point, and why.

Decisions the caller made explicitly are the least important to record,
because the caller already knows about them. Decisions taken silently by
default are the ones an auditor asks about, and they are the ones that are
invisible in the output otherwise.

## Immutability

`Provenance` is a frozen dataclass and `add_step` returns a new one rather
than mutating. A result object therefore cannot have its history edited
after the fact. Arrays inside `PhaseSeries` and `FreqSeries` are likewise
set read-only at construction, so "immutable" is enforced rather than
merely documented.

## One subsystem, two outputs

`log.record` writes a line to the standard-library logger for whoever is
watching, and appends a `Step` for whoever has to defend the number later.
The human-readable log and the machine-readable audit trail carry the same
information, so they come from one call and cannot drift apart.

`log.configure` is opt-in and is called by the CLI. A library must not
configure logging on import.

## Validation status

**Validated.** Provenance threading is asserted in
`tests/test_types.py::test_precision_warning_is_recorded_in_provenance`,
`tests/test_io_text.py::test_provenance_records_source_hash` and
`test_timetag_column_and_rate_inference`, and
`tests/test_synth.py::test_series_wrappers_carry_provenance`.

## References

* ISO/IEC 17025:2017 clauses 6.4.1, 7.2.2.4, 7.11.2, 7.11.6.
* IEEE Std 1139-2022 clauses 4.1 and 4.2.
