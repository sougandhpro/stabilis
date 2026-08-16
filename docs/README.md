# Documentation map

Four kinds of document live here, and they answer different questions.
Two of them are generated from the source and must never be hand-edited.

| Want to know | Read |
| --- | --- |
| What this is and why it exists | [`../README.md`](../README.md) |
| Why an estimator is defined the way it is | the theory pages, `stabilis docs --list` |
| What a function takes and returns | [`api/index.md`](api/index.md) |
| What a command does | [`cli-reference.md`](cli-reference.md) |
| What is proven, and how | `stabilis validate`, [`validation-report.html`](validation-report.html) |
| What is *not* yet proven | [`../VALIDATION-DEBT.md`](../VALIDATION-DEBT.md) |
| What has been built so far, with measured numbers | [`stabilis-progress.pdf`](stabilis-progress.pdf) |

## The theory pages

The 36 theory pages ship *inside* the package, in `src/stabilis/theory/`,
rather than here — so they survive a `pip install` and can be read on a
machine that never saw the source tree:

```bash
stabilis docs --list          # every topic, with its title
stabilis docs greenhall       # one page
```

They carry the derivations, the conventions, the published sources, and a
validation-status line saying what each claim rests on. They are written by
hand and reviewed like code.

## The generated pages

```bash
python3 docs/make_api.py      # -> docs/api/*.md      from src/stabilis/**.py
python3 docs/make_cli.py      # -> docs/cli-reference.md  from the argparse tree
python3 docs/make_report.py   # -> docs/stabilis-progress.html (--pdf for PDF)
```

`make_api.py` parses the source with `ast` and never imports the package,
so it emits the same bytes whether or not matplotlib and scipy happen to be
installed — and so the reference cannot describe an argument that no longer
exists. The public surface it documents is each module's `__all__`; anything
else is internal. It warns on any public name with no docstring.

`make_cli.py` captures `--help` from the live parser, with `COLUMNS` pinned
so the wrapping does not depend on the terminal it ran in.

`tests/test_generated_docs.py` runs both with `--check` and fails if either
page is stale, so a signature change that is not regenerated is a red test
rather than a document that quietly lies. Regenerate with the two commands
above; there is nothing to edit by hand in `api/` or `cli-reference.md`.
