"""`stabilis gui` -- the one module on the boundary with the browser page.

`DEFAULT_PORT` is imported at module scope because argparse captures a
default when the parser is built, not when the command runs. `serve` is
imported inside the handler, so nothing starts an HTTP server or touches
`gui/assets` until someone asks for the page.

That split is why `tests/test_gui_bench.py` exempts this module by name:
`gui/` is a client of `stabilis` and the arrow points one way, so no other
module in the package may reach it.
"""

from __future__ import annotations

from ..gui import DEFAULT_PORT as GUI_DEFAULT_PORT

__all__ = ["add_gui"]


def add_gui(sub) -> None:
    p_gui = sub.add_parser(
        "gui", help="a small local page in your browser, instead of flags")
    p_gui.add_argument("--port", type=int, default=GUI_DEFAULT_PORT)
    p_gui.add_argument(
        "--root", help="confine the file browser to this directory "
                       "(default: the current one)")
    p_gui.add_argument("--no-browser", action="store_true",
                       help="just print the URL")
    p_gui.set_defaults(func=_cmd_gui)


def _cmd_gui(args) -> int:
    """Serve the local page. Blocks until interrupted."""
    from ..gui import serve

    serve(port=args.port, root=args.root, open_browser=not args.no_browser)
    return 0
