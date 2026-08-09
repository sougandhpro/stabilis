"""Column arithmetic for the page, without an evaluator behind the box.

`edit.parse_expression` in the library covers a *sum of whole columns* --
`NCS - VSL`, `-VSL` -- and stops there deliberately. The page needs a
little more than that: a channel scaled into different units, a divider
ratio backed out, a mean removed. So this module implements exactly that
little more, and nothing else.

**There is still no Python evaluator anywhere near a user string.** This is
a hand-written tokenizer and recursive-descent parser over a closed grammar
-- numbers, column references, `+ - * /`, parentheses, unary sign. There is
no name lookup that can reach a builtin, no attribute access, no call
syntax, no indexing, no comprehension. An input the grammar does not
describe is refused with a message, never passed on to something that might
run it. The same reasoning is why this library refuses pickle: a file, or
an expression, arriving from someone else's rig is untrusted input.

Two rules that matter to the answer rather than to safety:

**Column names win over numbers.** A label is matched before a numeric
literal is tried, longest first, so a file whose header really is `2` still
works and a label containing a space or a minus sign is usable. Where the
label form is not wanted, `#2` always means the third column by position.

**An expression that names no column is refused.** `2 - 1` reads as
arithmetic here, not as "column 2 minus column 1" the way it does in
`edit.parse_expression`. Silently returning the scalar 1.0 would be a wrong
answer that looks plausible, so the parser insists that at least one column
took part and names the `#N` syntax in the error.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["evaluate", "tokenize", "ExpressionError"]

_NUMBER = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
_INDEX = re.compile(r"#(\d+)")
_OPERATORS = frozenset("+-*/")


class ExpressionError(ValueError):
    """The expression is not in the grammar, or names something absent."""


def tokenize(expression: str, labels) -> list[tuple[str, object]]:
    """`"NCS - VSL/2"` into `[("col", 1), ("op", "-"), ...]`.

    Longest label first, so `VSL_B` is not read as `VSL` followed by
    rubbish. Matching is case-sensitive first and case-insensitive second,
    because a header is a name and typing it in the wrong case is a typo
    rather than a different column.
    """
    names = [(str(name), i) for i, name in enumerate(labels)]
    names.sort(key=lambda pair: -len(pair[0]))
    lowered = [(name.lower(), i) for name, i in names]

    tokens: list[tuple[str, object]] = []
    at, end = 0, len(expression)
    while at < end:
        if expression[at].isspace():
            at += 1
            continue
        column = _match_column(expression, at, names, lowered)
        if column is not None:
            index, at = column
            tokens.append(("col", index))
            continue
        hit = _INDEX.match(expression, at)
        if hit:
            tokens.append(("col", int(hit.group(1))))
            at = hit.end()
            continue
        hit = _NUMBER.match(expression, at)
        if hit:
            tokens.append(("num", float(hit.group())))
            at = hit.end()
            continue
        char = expression[at]
        if char in _OPERATORS:
            tokens.append(("op", char))
        elif char in "()":
            tokens.append((char, char))
        else:
            raise ExpressionError(
                f"cannot read {char!r} at position {at} of {expression!r}. "
                f"Columns in this file: {[str(n) for n in labels]}; or write "
                "#0, #1 to name them by position."
            )
        at += 1
    if not tokens:
        raise ExpressionError("empty expression")
    return tokens


def _match_column(expression: str, at: int, names, lowered):
    """`(index, end)` if a column label starts here, else None."""
    for name, index in names:
        if name and expression.startswith(name, at):
            return index, at + len(name)
    tail = expression[at:].lower()
    for name, index in lowered:
        if name and tail.startswith(name):
            return index, at + len(name)
    return None


def evaluate(expression: str, labels, rows) -> tuple[np.ndarray, int]:
    """The column `expression` describes, and how many divisions blew up.

    `rows` is 2-D, one column per label. Returns `(values, n_infinite)`.

    A NaN anywhere in a term makes the result NaN there, which is what
    should happen -- a difference is unknown wherever either side is. That
    needs no counting, because those points are already gaps and the record
    summary reports them.

    An **infinity** is different: it means a divisor was exactly zero, and
    it is a fact about the arithmetic rather than about the input. It
    becomes a gap, because NaN is this library's gap representation and
    every estimator already knows what to do with one, and the count comes
    back so the page can say it happened instead of it being absorbed
    silently.
    """
    tokens = tokenize(expression, labels)
    state = {"at": 0, "used": False}
    value = _expr(tokens, state, expression, labels, rows)
    if state["at"] != len(tokens):
        kind, text = tokens[state["at"]]
        raise ExpressionError(
            f"unexpected {text!r} in {expression!r}; the grammar is columns, "
            "numbers, + - * / and parentheses")
    if not state["used"]:
        raise ExpressionError(
            f"{expression!r} names no column, so it is just arithmetic on "
            "constants. Column 2 is written `#2` here, or by its header name.")

    out = np.asarray(value, dtype=np.float64)
    blown = np.isinf(out)
    n_blown = int(blown.sum())
    if n_blown:
        out = np.where(blown, np.nan, out)
    return out, n_blown


def _expr(tokens, state, text, labels, rows):
    """term (('+'|'-') term)*"""
    value = _term(tokens, state, text, labels, rows)
    while _peek(tokens, state) in ("+", "-"):
        op = tokens[state["at"]][1]
        state["at"] += 1
        right = _term(tokens, state, text, labels, rows)
        value = value + right if op == "+" else value - right
    return value


def _term(tokens, state, text, labels, rows):
    """factor (('*'|'/') factor)*"""
    value = _factor(tokens, state, text, labels, rows)
    while _peek(tokens, state) in ("*", "/"):
        op = tokens[state["at"]][1]
        state["at"] += 1
        right = _factor(tokens, state, text, labels, rows)
        if op == "*":
            value = value * right
        else:
            # errstate, not a guard: a zero divisor is a property of the
            # data, not a bug, and `evaluate` turns the result into a gap
            # and reports the count.
            with np.errstate(divide="ignore", invalid="ignore"):
                value = value / right
    return value


def _factor(tokens, state, text, labels, rows):
    """('+'|'-') factor | primary"""
    if _peek(tokens, state) in ("+", "-"):
        op = tokens[state["at"]][1]
        state["at"] += 1
        value = _factor(tokens, state, text, labels, rows)
        return value if op == "+" else -value
    return _primary(tokens, state, text, labels, rows)


def _primary(tokens, state, text, labels, rows):
    """NUMBER | COLUMN | '(' expr ')'"""
    if state["at"] >= len(tokens):
        raise ExpressionError(f"{text!r} ends early; something is missing "
                              "after the last operator")
    kind, value = tokens[state["at"]]
    state["at"] += 1
    if kind == "num":
        return float(value)
    if kind == "col":
        index = int(value)
        if not 0 <= index < len(labels):
            raise ExpressionError(
                f"column {index} is out of range in {text!r}; this file has "
                f"{len(labels)} columns")
        state["used"] = True
        return np.asarray(rows[:, index], dtype=np.float64)
    if kind == "(":
        inner = _expr(tokens, state, text, labels, rows)
        if _peek(tokens, state) != ")":
            raise ExpressionError(f"unclosed '(' in {text!r}")
        state["at"] += 1
        return inner
    raise ExpressionError(f"unexpected {value!r} in {text!r}")


def _peek(tokens, state):
    """The next operator or bracket character, or None at the end."""
    if state["at"] >= len(tokens):
        return None
    kind, value = tokens[state["at"]]
    return value if kind in ("op", "(", ")") else None
