from grannos.drivers.comments import blank_comments


def _sql(query: str) -> str:
    return blank_comments(query, line=("--",), block=True, quotes="'\"", escape=None)


class TestBlankComments:
    def test_line_comment_blanked_to_end_of_line(self) -> None:
        assert _sql("SELECT 1 -- one\nFROM t") == "SELECT 1       \nFROM t"

    def test_block_comment_keeps_its_newlines(self) -> None:
        query = "/* a\nb */SELECT 1"
        assert _sql(query) == "    \n    SELECT 1"
        assert len(_sql(query)) == len(query)

    def test_unterminated_block_runs_to_end(self) -> None:
        assert _sql("SELECT 1 /* open") == "SELECT 1        "

    def test_markers_inside_literals_are_kept(self) -> None:
        query = "SELECT '--', \"/*x*/\" FROM t"
        assert _sql(query) == query

    def test_doubled_quote_stays_inside_the_literal(self) -> None:
        query = "SELECT 'it''s -- not a comment'"
        assert _sql(query) == query

    def test_backslash_does_not_escape_without_escape_char(self) -> None:
        assert _sql("SELECT 'C:\\' -- c") == "SELECT 'C:\\'     "

    def test_escaped_quote_stays_inside_the_literal(self) -> None:
        query = '"a\\"//b" // c'
        assert blank_comments(query, line=("//",)) == '"a\\"//b"     '

    def test_after_requires_a_boundary(self) -> None:
        def shell(q: str) -> str:
            return blank_comments(q, line=("#",), quotes="'\"", after="")

        assert shell("ls s3://b/a#b") == "ls s3://b/a#b"
        assert shell("ls s3://b # all") == "ls s3://b      "
        assert shell("# first\nls") == "       \nls"

    def test_after_accepts_listed_characters(self) -> None:
        def lucene(q: str) -> str:
            return blank_comments(q, line=("--",), quotes='"/', after="(")

        assert lucene("a--b") == "a--b"
        assert lucene("(--x\na)") == "(   \na)"
