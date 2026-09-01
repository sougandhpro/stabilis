"""`python3 -m stabilis.cli`.

The `__name__` guard is not decoration. `pkgutil.iter_modules` lists
`__main__` among a package's modules, and `tests/test_gui_bench.py` imports
every module it walks to read the AST -- so without the guard, collecting
the test suite would run the command line with pytest's own `sys.argv`.
`runpy` sets `__name__` to `"__main__"`, so the guard costs nothing here.
"""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
