"""Comment removal for the query text drivers read themselves.

A comment either reaches the engine untouched (SQL, Cypher, PromQL and ES|QL
all have their own) or is part of a query file's grammar only (Lucene's
``--``, the ``//`` of Extended JSON, S3's ``#``) and must never reach it. Either
way a driver that parses the text itself — a client-side command, a
leading-keyword check, a splice — must not mistake a comment for part of the
statement, nor miss the statement behind a leading one.
"""

_BLOCK_OPEN, _BLOCK_CLOSE = "/*", "*/"


def blank_comments(
    query: str,
    *,
    line: tuple[str, ...] = (),
    block: bool = False,
    quotes: str = '"',
    escape: str | None = "\\",
    after: str | None = None,
) -> str:
    """Return *query* with every comment overwritten by spaces.

    Newlines inside a comment are kept, so every offset, line and column of the
    result is that of *query* — an engine's error position still points into
    the text the user submitted.

    Args:
        query: The query text.
        line: Markers opening a comment that runs to the end of the line.
        block: Whether ``/* */`` comments are recognised. An unterminated one
            runs to the end of the query.
        quotes: Characters each opening a literal that runs to the next one
            (``''`` doubling reads as two adjacent literals, the same text).
            Comment markers inside a literal are left alone.
        escape: Character making the next one literal, inside a literal or
            out; None for languages without one (SQL).
        after: When set, a line marker only opens a comment at the start of
            the query or after whitespace or one of these characters — for
            languages where the marker is literal inside a word (``a--b`` in
            Lucene, ``s3://b/a#b`` in a shell-like command).
    """
    out = list(query)
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch == escape:
            i += 2
        elif ch in quotes:
            i += 1
            while i < n and query[i] != ch:
                i += 2 if query[i] == escape else 1
            i += 1
        elif block and query.startswith(_BLOCK_OPEN, i):
            close = query.find(_BLOCK_CLOSE, i + 2)
            end = n if close < 0 else close + 2
            _blank(out, i, end)
            i = end
        elif any(query.startswith(m, i) for m in line) and _at_boundary(
            query, i, after
        ):
            eol = query.find("\n", i)
            end = n if eol < 0 else eol
            _blank(out, i, end)
            i = end
        else:
            i += 1
    return "".join(out)


def _at_boundary(query: str, i: int, after: str | None) -> bool:
    if after is None or i == 0:
        return True
    prev = query[i - 1]
    return prev.isspace() or prev in after


def _blank(out: list[str], start: int, end: int) -> None:
    for k in range(start, end):
        if out[k] != "\n":
            out[k] = " "
