"""Unit tests for OracleDriver — no live database required."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from grannos.drivers.base import (
    ConnectionLostError,
    DriverError,
    DriverSettings,
    FindNotSupported,
)
from grannos.drivers.oracle.driver import (
    OracleDriver,
    _alter_session_property,
    _format_db_error,
    _format_type,
    _is_explain_plan,
    _offset_to_line_col,
    _created_object,
    _reject_sqlplus_terminator,
    _replace_undecodable_text,
    _statement_start_line,
)
from grannos.drivers.oracle.load import (
    LoadCommand,
    LoadOptions,
    build_insert_statement,
    parse_load,
    read_rows,
)
from grannos.drivers.oracle.queries import (
    _PRE12_SYSTEM_SCHEMAS_SQL,
    ColumnDetail,
    render_lob,
)
from grannos.protocol import (
    ExploreItem,
    IndexDescription,
    LobPlaceholder,
    NodeType,
    ReadResult,
    SearchScope,
    WriteResult,
)


def _make_lob(
    type_name: str, size: int, content: bytes | str | None = None
) -> MagicMock:
    lob = MagicMock()
    lob.read = AsyncMock(return_value=content)
    lob.type.name = type_name
    lob.size = AsyncMock(return_value=size)
    return lob


def _make_index_driver(index_row: tuple | None, col_rows: list) -> OracleDriver:
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchone = AsyncMock(side_effect=[index_row, None])  # meta, then DDL
    cur.fetchall = AsyncMock(side_effect=[col_rows, []])  # fields, then join tables
    conn = MagicMock(spec=oracledb.AsyncConnection)
    conn.cursor.return_value = cur
    return OracleDriver({}, conn, True, DriverSettings())


def _make_driver(rows: list, has_oracle_maintained: bool) -> OracleDriver:
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows)
    cur.__aiter__.return_value = iter(rows)
    conn = MagicMock(spec=oracledb.AsyncConnection)
    conn.cursor.return_value = cur
    return OracleDriver({}, conn, has_oracle_maintained, DriverSettings())


def _make_db_error(message: str, offset: int = 0) -> oracledb.DatabaseError:
    err = MagicMock()
    err.offset = offset
    err.is_session_dead = False
    err.__str__ = lambda self: message
    return oracledb.DatabaseError(err)


class TestRootListing:
    def test_12c_uses_oracle_maintained_filter(self) -> None:
        driver = _make_driver([("ALICE",), ("BOB",)], has_oracle_maintained=True)
        asyncio.run(driver.explore_list([]))
        sql = driver._conn.cursor().execute.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "ORACLE_MAINTAINED" in sql
        assert "NOT IN" not in sql

    def test_pre12c_uses_exclusion_list(self) -> None:
        driver = _make_driver([("ALICE",), ("BOB",)], has_oracle_maintained=False)
        asyncio.run(driver.explore_list([]))
        sql = driver._conn.cursor().execute.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "ORACLE_MAINTAINED" not in sql
        assert "NOT IN" in sql

    def test_pre12c_exclusion_list_contains_known_system_schemas(self) -> None:
        for schema in ("SYS", "SYSTEM", "DBSNMP", "XDB", "OUTLN", "MDSYS"):
            assert f"'{schema}'" in _PRE12_SYSTEM_SCHEMAS_SQL

    def test_returns_schema_items(self) -> None:
        driver = _make_driver([("ALICE",), ("BOB",)], has_oracle_maintained=True)
        items = asyncio.run(driver.explore_list([]))
        assert items == [
            ExploreItem(name="ALICE", type="schema", expandable=True),
            ExploreItem(name="BOB", type="schema", expandable=True),
        ]


class TestOffsetToLineCol:
    def test_single_line_gives_line_1(self) -> None:
        assert _offset_to_line_col("SELECT FROM t", 7) == (1, 8)

    def test_second_line(self) -> None:
        query = "SELECT\nFROM t"
        assert _offset_to_line_col(query, 7) == (2, 1)

    def test_middle_of_second_line(self) -> None:
        query = "SELECT\nFROM t"
        assert _offset_to_line_col(query, 10) == (2, 4)


class TestFormatDbError:
    def test_no_offset_returns_plain_message(self) -> None:
        exc = _make_db_error("ORA-00936: missing expression", offset=0)
        assert _format_db_error(exc, "SELECT FROM t") == "ORA-00936: missing expression"

    def test_with_offset_appends_line_col_and_excerpt(self) -> None:
        exc = _make_db_error("ORA-00936: missing expression", offset=7)
        result = _format_db_error(exc, "SELECT FROM t")
        assert result == (
            "ORA-00936: missing expression (line 1, col 8)\n"
            "1 | SELECT FROM t\n"
            "           ^"
        )

    def test_multiline_query_offset_shows_correct_line(self) -> None:
        query = "SELECT\nFROM t"
        exc = _make_db_error("ORA-00936: missing expression", offset=7)
        result = _format_db_error(exc, query)
        assert result == (
            "ORA-00936: missing expression (line 2, col 1)\n2 | FROM t\n    ^"
        )

    def test_position_in_message_quotes_the_line(self) -> None:
        query = "BEGIN\n  bogus\nEND;"
        exc = _make_db_error(
            "ORA-06550: line 2, column 3:\nPLS-00103: Encountered the symbol", offset=0
        )
        result = _format_db_error(exc, query)
        assert result.endswith("\n2 |   bogus\n      ^")

    def test_position_in_message_skips_leading_comments(self) -> None:
        query = "-- a note\n\nBEGIN\n  bogus\nEND;"
        exc = _make_db_error("ORA-06550: line 2, column 3:", offset=0)
        result = _format_db_error(exc, query)
        assert result.endswith("\n4 |   bogus\n      ^")

    def test_offset_wins_over_a_position_in_the_message(self) -> None:
        exc = _make_db_error("ORA-06550: line 9, column 9:", offset=7)
        result = _format_db_error(exc, "SELECT FROM t")
        assert result.startswith("ORA-06550: line 9, column 9: (line 1, col 8)\n1 | ")

    def test_position_outside_the_query_is_dropped(self) -> None:
        exc = _make_db_error("ORA-06550: line 40, column 3:", offset=0)
        result = _format_db_error(exc, "BEGIN\n  bogus\nEND;")
        assert result == "ORA-06550: line 40, column 3:"

    def test_column_past_the_end_of_the_line_is_dropped(self) -> None:
        exc = _make_db_error("ORA-06550: line 1, column 80:", offset=0)
        result = _format_db_error(exc, "BEGIN\n  bogus\nEND;")
        assert result == "ORA-06550: line 1, column 80:"

    def test_tabs_are_narrowed_so_the_caret_lines_up(self) -> None:
        exc = _make_db_error("ORA-00936: missing expression", offset=8)
        result = _format_db_error(exc, "SELECT\t*\tFROM")
        assert result.endswith("\n1 | SELECT\t*\tFROM\n            ^".expandtabs(1))


def _col(
    type_: str, char_length: int | None = None, byte_length: int | None = None
) -> ColumnDetail:
    return ColumnDetail(
        name="c",
        type=type_,
        nullable=True,
        default=None,
        char_length=char_length,
        byte_length=byte_length,
    )


class TestFormatType:
    def test_varchar2_shows_char_length(self) -> None:
        col = _col("VARCHAR2", char_length=50, byte_length=50)
        assert _format_type(col) == "VARCHAR2(50)"

    def test_varchar_shows_char_length(self) -> None:
        col = _col("VARCHAR", char_length=20, byte_length=20)
        assert _format_type(col) == "VARCHAR(20)"

    def test_shows_byte_length_when_it_differs(self) -> None:
        col = _col("VARCHAR2", char_length=50, byte_length=200)
        assert _format_type(col) == "VARCHAR2(50 CHAR, 200 BYTE)"

    def test_same_char_and_byte_length_omits_byte_length(self) -> None:
        col = _col("VARCHAR2", char_length=50, byte_length=50)
        assert "BYTE" not in _format_type(col)

    def test_non_varchar_type_is_unchanged(self) -> None:
        col = _col("NUMBER", char_length=0, byte_length=22)
        assert _format_type(col) == "NUMBER"

    def test_missing_char_length_leaves_type_unchanged(self) -> None:
        col = _col("VARCHAR2", char_length=None, byte_length=50)
        assert _format_type(col) == "VARCHAR2"


class TestRenderLob:
    def test_passes_through_non_lob_values(self) -> None:
        assert asyncio.run(render_lob("hello")) == "hello"
        assert asyncio.run(render_lob(42)) == 42

    def test_renders_clob_as_char_count(self) -> None:
        lob = _make_lob("DB_TYPE_CLOB", 3423)
        assert asyncio.run(render_lob(lob)) == LobPlaceholder(text="CLOB (3423 chars)")

    def test_renders_blob_as_byte_count(self) -> None:
        lob = _make_lob("DB_TYPE_BLOB", 128)
        assert asyncio.run(render_lob(lob)) == LobPlaceholder(text="BLOB (128 bytes)")

    def test_undecodable_clob_becomes_a_placeholder(self) -> None:
        lob = _make_lob("DB_TYPE_CLOB", 3)
        lob.read = AsyncMock(side_effect=_bad_bytes_error())

        result = asyncio.run(render_lob(lob, lambda value, text: pytest.fail(text)))
        assert result == LobPlaceholder(text="CLOB (undecodable text)")

    def test_with_register_lob_reads_content_and_caches_it(self) -> None:
        lob = _make_lob("DB_TYPE_CLOB", 4, content="text")
        cache: dict[str, bytes | str] = {}

        def register_lob(value: bytes | str, text: str) -> LobPlaceholder:
            ref = "some-ref"
            cache[ref] = value
            return LobPlaceholder(text=text, ref=ref)

        result = asyncio.run(render_lob(lob, register_lob))
        assert result == LobPlaceholder(text="CLOB (4 chars)", ref="some-ref")
        assert cache["some-ref"] == "text"


class TestExecuteRendersLobs:
    def test_replaces_lob_values_with_downloadable_placeholders(self) -> None:
        content = "x" * 3423
        lob = _make_lob("DB_TYPE_CLOB", 3423, content=content)
        cur = MagicMock()
        cur.execute = AsyncMock()
        cur.description = [("ID",), ("NOTES",)]
        cur.fetchall = AsyncMock(return_value=[(1, lob)])
        cur.__aiter__.return_value = iter([(1, lob)])
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        result = asyncio.run(driver.execute("SELECT id, notes FROM t", []))
        assert isinstance(result, ReadResult)
        assert result.rows[0][0] == 1
        placeholder = result.rows[0][1]
        assert isinstance(placeholder, LobPlaceholder)
        assert placeholder.text == "CLOB (3423 chars)"
        assert placeholder.ref is not None
        assert driver._lob_cache[placeholder.ref] == content


class TestExecuteErrorPropagation:
    def test_db_error_without_offset_raises_driver_error(self) -> None:
        exc = _make_db_error("ORA-00936: missing expression", offset=0)
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(DriverError, match="ORA-00936"):
            asyncio.run(driver.execute("SELECT FROM t", []))

    def test_db_error_with_offset_includes_position(self) -> None:
        exc = _make_db_error("ORA-00936: missing expression", offset=7)
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(DriverError, match=r"line 1, col 8"):
            asyncio.run(driver.execute("SELECT FROM t", []))

    def test_dead_session_raises_connection_lost(self) -> None:
        error = MagicMock()
        error.is_session_dead = True
        exc = oracledb.DatabaseError(error)
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("SELECT 1 FROM DUAL", []))

    def test_explore_list_not_connected_raises_connection_lost(self) -> None:
        exc = oracledb.InterfaceError("DPY-1001: not connected to the database")
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_list([]))

    def test_explore_describe_not_connected_raises_connection_lost(self) -> None:
        exc = oracledb.InterfaceError("DPY-1001: not connected to the database")
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_describe(["MYSCHEMA", "MYTABLE"]))


class TestAlterSessionProperty:
    def test_extracts_property_name(self) -> None:
        assert (
            _alter_session_property("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'")
            == "NLS_DATE_FORMAT"
        )

    def test_case_insensitive(self) -> None:
        assert (
            _alter_session_property("alter session set nls_date_format = 'YYYY-MM-DD'")
            == "NLS_DATE_FORMAT"
        )

    def test_leading_whitespace(self) -> None:
        assert _alter_session_property("   ALTER SESSION SET FOO = 1") == "FOO"

    def test_non_alter_session_returns_none(self) -> None:
        assert _alter_session_property("SELECT 1 FROM DUAL") is None

    def test_alter_table_returns_none(self) -> None:
        assert _alter_session_property("ALTER TABLE t ADD (c NUMBER)") is None


class TestExecuteRecordsSessionStatements:
    def test_records_on_success(self) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        stmt = "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"
        asyncio.run(driver.execute(stmt, []))
        assert driver._session_statements == {"NLS_DATE_FORMAT": stmt}

    def test_later_value_replaces_earlier(self) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        first = "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"
        second = "ALTER SESSION SET NLS_DATE_FORMAT = 'DD-MM-YYYY'"
        asyncio.run(driver.execute(first, []))
        asyncio.run(driver.execute(second, []))
        assert driver._session_statements == {"NLS_DATE_FORMAT": second}

    def test_does_not_record_plain_query(self) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        asyncio.run(driver.execute("SELECT 1 FROM DUAL", []))
        assert driver._session_statements == {}

    def test_failed_statement_not_recorded(self) -> None:
        exc = _make_db_error("ORA-00922: missing or invalid option")
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=exc)
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(DriverError):
            asyncio.run(driver.execute("ALTER SESSION SET BOGUS = 1", []))
        assert driver._session_statements == {}


class TestReconnectReplaysSessionStatements:
    def test_replays_recorded_statements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        stmt = "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"
        asyncio.run(driver.execute(stmt, []))

        new_cur = MagicMock()
        new_cur.execute = AsyncMock()
        new_conn = MagicMock(spec=oracledb.AsyncConnection)
        new_conn.cursor.return_value = new_cur

        async def fake_open(params: dict) -> tuple:
            return new_conn, True

        monkeypatch.setattr(OracleDriver, "_open", staticmethod(fake_open))
        asyncio.run(driver.reconnect())

        new_cur.execute.assert_awaited_with(stmt)

    def test_no_statements_means_no_replay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver = _make_driver([], has_oracle_maintained=True)

        new_conn = MagicMock(spec=oracledb.AsyncConnection)

        async def fake_open(params: dict) -> tuple:
            return new_conn, True

        monkeypatch.setattr(OracleDriver, "_open", staticmethod(fake_open))
        asyncio.run(driver.reconnect())

        new_conn.cursor.assert_not_called()

    def test_replay_failure_raises_driver_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        asyncio.run(driver.execute("ALTER SESSION SET FOO = 1", []))

        exc = _make_db_error("ORA-02248: invalid option for ALTER SESSION")
        new_cur = MagicMock()
        new_cur.execute = AsyncMock(side_effect=exc)
        new_conn = MagicMock(spec=oracledb.AsyncConnection)
        new_conn.cursor.return_value = new_cur

        async def fake_open(params: dict) -> tuple:
            return new_conn, True

        monkeypatch.setattr(OracleDriver, "_open", staticmethod(fake_open))
        with pytest.raises(DriverError, match="ORA-02248"):
            asyncio.run(driver.reconnect())


class TestDisconnect:
    def test_closes_connection(self) -> None:
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.close = AsyncMock()
        driver = OracleDriver({}, conn, True, DriverSettings())
        asyncio.run(driver.disconnect())
        conn.close.assert_awaited_once()

    def test_swallows_not_connected_error(self) -> None:
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.close = AsyncMock(
            side_effect=oracledb.InterfaceError(
                "DPY-1001: not connected to the database"
            )
        )
        driver = OracleDriver({}, conn, True, DriverSettings())
        asyncio.run(driver.disconnect())  # must not raise


class TestExploreDescribeIndex:
    def test_returns_index_description(self) -> None:
        driver = _make_index_driver(
            index_row=("NORMAL", "UNIQUE", "VISIBLE", "N"),
            col_rows=[("ID", "ASC"), ("NAME", "ASC")],
        )
        result = asyncio.run(
            driver.explore_describe(["MYSCHEMA", "MYTABLE", "indexes", "MY_IDX"])
        )
        assert isinstance(result, IndexDescription)
        assert result.name == "MY_IDX"
        assert result.unique is True
        assert result.tables == ["MYTABLE"]
        assert [f.name for f in result.fields] == ["ID", "NAME"]
        assert all(f.direction == "asc" for f in result.fields)

    def test_desc_direction(self) -> None:
        driver = _make_index_driver(
            index_row=("NORMAL", "NONUNIQUE", "VISIBLE", "N"),
            col_rows=[("CREATED_AT", "DESC")],
        )
        result = asyncio.run(
            driver.explore_describe(["MYSCHEMA", "MYTABLE", "indexes", "MY_IDX"])
        )
        assert isinstance(result, IndexDescription)
        assert result.fields[0].direction == "desc"

    def test_non_unique_index(self) -> None:
        driver = _make_index_driver(
            index_row=("NORMAL", "NONUNIQUE", "VISIBLE", "N"),
            col_rows=[("COL", "ASC")],
        )
        result = asyncio.run(
            driver.explore_describe(["MYSCHEMA", "MYTABLE", "indexes", "MY_IDX"])
        )
        assert isinstance(result, IndexDescription)
        assert result.unique is False

    def test_returns_none_when_index_not_found(self) -> None:
        driver = _make_index_driver(index_row=None, col_rows=[])
        result = asyncio.run(
            driver.explore_describe(["MYSCHEMA", "MYTABLE", "indexes", "NO_SUCH"])
        )
        assert result is None


class TestIsExplainPlan:
    def test_uppercase_matches(self) -> None:
        assert _is_explain_plan("EXPLAIN PLAN FOR SELECT 1 FROM DUAL") is True

    def test_lowercase_matches(self) -> None:
        assert _is_explain_plan("explain plan for SELECT 1 FROM DUAL") is True

    def test_mixed_case_matches(self) -> None:
        assert _is_explain_plan("Explain Plan For SELECT 1 FROM DUAL") is True

    def test_leading_whitespace_matches(self) -> None:
        assert _is_explain_plan("  EXPLAIN PLAN FOR SELECT 1 FROM DUAL") is True

    def test_leading_comment_line_matches(self) -> None:
        assert (
            _is_explain_plan("-- a comment\nEXPLAIN PLAN FOR SELECT 1 FROM DUAL")
            is True
        )

    def test_multiple_leading_comment_lines_match(self) -> None:
        assert _is_explain_plan("-- c1\n-- c2\nEXPLAIN PLAN FOR SELECT 1") is True

    def test_regular_select_does_not_match(self) -> None:
        assert _is_explain_plan("SELECT 1 FROM DUAL") is False

    def test_insert_does_not_match(self) -> None:
        assert _is_explain_plan("INSERT INTO t VALUES (1)") is False


_PLSQL_BLOCK = "BEGIN\n    NULL;\nEND;"


class TestRejectSqlplusTerminator:
    def test_trailing_slash_is_rejected(self) -> None:
        with pytest.raises(DriverError, match="SQL\\*Plus terminator"):
            _reject_sqlplus_terminator(f"{_PLSQL_BLOCK}\n/")

    def test_error_names_the_offending_line(self) -> None:
        with pytest.raises(DriverError, match="^line 4: "):
            _reject_sqlplus_terminator(f"{_PLSQL_BLOCK}\n/")

    def test_trailing_whitespace_after_slash_is_rejected(self) -> None:
        with pytest.raises(DriverError, match="SQL\\*Plus terminator"):
            _reject_sqlplus_terminator(f"{_PLSQL_BLOCK}\n  /  \n\n")

    def test_block_without_slash_is_accepted(self) -> None:
        _reject_sqlplus_terminator(_PLSQL_BLOCK)

    def test_plain_select_is_accepted(self) -> None:
        _reject_sqlplus_terminator("SELECT 1 FROM DUAL")

    def test_line_broken_division_is_accepted(self) -> None:
        _reject_sqlplus_terminator("SELECT a\n/\nb FROM t")

    def test_slash_inside_expression_is_accepted(self) -> None:
        _reject_sqlplus_terminator("SELECT a / b FROM t")


class TestExplainPlanExecute:
    @pytest.fixture()
    def explain_cur(self) -> MagicMock:
        cur = MagicMock()
        cur.execute = AsyncMock()
        cur.fetchall = AsyncMock(
            return_value=[
                ("Plan hash value: 12345",),
                ("| Id | Operation |",),
                ("|  0 | SELECT STATEMENT |",),
            ]
        )
        return cur

    @pytest.fixture()
    def explain_driver(self, explain_cur: MagicMock) -> OracleDriver:
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = explain_cur
        return OracleDriver({}, conn, True, DriverSettings())

    def test_returns_read_result(
        self, explain_driver: OracleDriver, explain_cur: MagicMock
    ) -> None:
        result = asyncio.run(
            explain_driver.execute("EXPLAIN PLAN FOR SELECT 1 FROM DUAL", [])
        )
        assert isinstance(result, ReadResult)

    def test_columns_are_plan_table_output(
        self, explain_driver: OracleDriver, explain_cur: MagicMock
    ) -> None:
        result = asyncio.run(
            explain_driver.execute("EXPLAIN PLAN FOR SELECT 1 FROM DUAL", [])
        )
        assert isinstance(result, ReadResult)
        assert result.columns == ["PLAN_TABLE_OUTPUT"]

    def test_rows_contain_plan_lines(
        self, explain_driver: OracleDriver, explain_cur: MagicMock
    ) -> None:
        result = asyncio.run(
            explain_driver.execute("EXPLAIN PLAN FOR SELECT 1 FROM DUAL", [])
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [
            ["Plan hash value: 12345"],
            ["| Id | Operation |"],
            ["|  0 | SELECT STATEMENT |"],
        ]

    def test_calls_dbms_xplan_display(
        self, explain_driver: OracleDriver, explain_cur: MagicMock
    ) -> None:
        asyncio.run(explain_driver.execute("EXPLAIN PLAN FOR SELECT 1 FROM DUAL", []))
        second_call_sql = explain_cur.execute.call_args_list[1][0][0]
        assert "DBMS_XPLAN.DISPLAY" in second_call_sql


class TestCreatedObject:
    def test_procedure(self) -> None:
        assert _created_object("CREATE PROCEDURE p AS BEGIN NULL; END;") == (
            "P",
            "PROCEDURE",
        )

    def test_or_replace_and_lowercase(self) -> None:
        assert _created_object("create or replace function f return number as") == (
            "F",
            "FUNCTION",
        )

    def test_package_body_type_has_a_single_space(self) -> None:
        assert _created_object("CREATE OR REPLACE PACKAGE\n  BODY pkg AS") == (
            "PKG",
            "PACKAGE BODY",
        )

    def test_editionable_is_skipped(self) -> None:
        assert _created_object("CREATE NONEDITIONABLE TRIGGER trg BEFORE INSERT") == (
            "TRG",
            "TRIGGER",
        )

    def test_schema_qualified_name_uses_the_object_name(self) -> None:
        assert _created_object("CREATE PROCEDURE myschema.p AS") == ("P", "PROCEDURE")

    def test_quoted_name_keeps_its_case(self) -> None:
        assert _created_object('CREATE PROCEDURE "keepCase" AS') == (
            "keepCase",
            "PROCEDURE",
        )

    def test_leading_comment_is_skipped(self) -> None:
        assert _created_object("-- note\nCREATE PROCEDURE p AS") == ("P", "PROCEDURE")

    def test_non_create_returns_none(self) -> None:
        assert _created_object("SELECT 1 FROM DUAL") is None

    def test_create_table_returns_none(self) -> None:
        assert _created_object("CREATE TABLE t (id NUMBER)") is None


class TestStatementStartLine:
    def test_first_line(self) -> None:
        assert _statement_start_line("SELECT 1 FROM DUAL") == 1

    def test_skips_leading_comments(self) -> None:
        assert _statement_start_line("-- a\n-- b\nSELECT 1") == 3

    def test_skips_blank_lines(self) -> None:
        assert _statement_start_line("\n\nSELECT 1") == 3

    def test_all_comments_falls_back_to_one(self) -> None:
        assert _statement_start_line("-- nothing here\n") == 1


def _bad_bytes_error() -> UnicodeDecodeError:
    return UnicodeDecodeError("utf-8", b"A\xffB", 1, 2, "invalid start byte")


class TestReplaceUndecodableText:
    def test_char_columns_are_decoded_with_replacement(self) -> None:
        cursor = MagicMock()
        cursor.arraysize = 100
        metadata = MagicMock(type_code=oracledb.DB_TYPE_VARCHAR, display_size=50)

        assert _replace_undecodable_text(cursor, metadata) is cursor.var.return_value
        cursor.var.assert_called_once_with(
            oracledb.DB_TYPE_VARCHAR,
            size=50,
            arraysize=100,
            encoding_errors="replace",
        )

    def test_unsized_column_uses_the_default_size(self) -> None:
        cursor = MagicMock()
        cursor.arraysize = 100
        metadata = MagicMock(type_code=oracledb.DB_TYPE_LONG, display_size=None)

        _replace_undecodable_text(cursor, metadata)
        assert cursor.var.call_args.kwargs["size"] == 0

    def test_non_char_columns_keep_the_default_var(self) -> None:
        cursor = MagicMock()
        metadata = MagicMock(type_code=oracledb.DB_TYPE_NUMBER, display_size=None)

        assert _replace_undecodable_text(cursor, metadata) is None
        cursor.var.assert_not_called()


class TestDecodeErrorPropagation:
    def test_execute_reports_the_offending_byte(self) -> None:
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=_bad_bytes_error())
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(DriverError, match="not valid utf-8 at byte 1"):
            asyncio.run(driver.execute("SELECT txt FROM t", []))

    def test_explore_describe_reports_the_offending_byte(self) -> None:
        cur = MagicMock()
        cur.execute = AsyncMock(side_effect=_bad_bytes_error())
        conn = MagicMock(spec=oracledb.AsyncConnection)
        conn.cursor.return_value = cur
        driver = OracleDriver({}, conn, True, DriverSettings())
        with pytest.raises(DriverError, match="not valid utf-8"):
            asyncio.run(driver.explore_describe(["MYSCHEMA", "MYTABLE"]))


class TestExploreFind:
    """One dictionary query resolves a symbol across every schema, in place of
    the generic walker's schema-by-schema descent."""

    @staticmethod
    def _find(rows: list, node_type: str, name: str, scopes: list) -> tuple:
        """Run a find against a driver returning *rows*, and hand back the
        resulting paths with the SQL and binds the driver submitted."""
        driver = _make_driver(rows, has_oracle_maintained=True)
        paths = asyncio.run(driver.explore_find(node_type, name, scopes))
        call = driver._conn.cursor().execute.call_args  # ty: ignore[unresolved-attribute]
        return paths, call[0][0], call[0][1]

    def test_table_resolves_across_schemas_in_one_query(self) -> None:
        paths, sql, _ = self._find(
            [("HR", "EMPLOYEES"), ("PAYROLL", "EMPLOYEES")],
            NodeType.TABLE,
            "employees",
            [],
        )
        assert paths == [["HR", "EMPLOYEES"], ["PAYROLL", "EMPLOYEES"]]
        assert "OWNER IN (:" not in sql  # no scope, so no bound owner predicate

    def test_ambiguous_table_returns_every_candidate(self) -> None:
        """Two schemas holding the same name is a picker, not an error."""
        paths, _, _ = self._find(
            [("HR", "EMPLOYEES"), ("PAYROLL", "EMPLOYEES")],
            NodeType.TABLE,
            "employees",
            [],
        )
        assert len(paths) == 2

    def test_view_search_also_matches_tables(self) -> None:
        """The tree holds tables and views at one level, and a client naming a
        symbol from surrounding syntax cannot tell them apart."""
        paths, sql, _ = self._find([("HR", "EMP_V")], NodeType.VIEW, "emp_v", [])
        assert paths == [["HR", "EMP_V"]]
        assert "ALL_TABLES" in sql
        assert "ALL_VIEWS" in sql

    def test_column_path_carries_the_group_segment(self) -> None:
        paths, _, _ = self._find(
            [("HR", "EMPLOYEES", "SALARY")], NodeType.COLUMN, "salary", []
        )
        assert paths == [["HR", "EMPLOYEES", "columns", "SALARY"]]

    def test_index_is_keyed_on_the_table_it_indexes(self) -> None:
        """The tree hangs an index off its table, so the path needs TABLE_OWNER
        rather than the index's own owner."""
        paths, sql, _ = self._find(
            [("HR", "EMPLOYEES", "EMP_PK")], NodeType.INDEX, "emp_pk", []
        )
        assert paths == [["HR", "EMPLOYEES", "indexes", "EMP_PK"]]
        assert "TABLE_OWNER" in sql

    def test_table_scope_narrows_a_column_search(self) -> None:
        _, sql, binds = self._find(
            [("HR", "EMPLOYEES", "ID")],
            NodeType.COLUMN,
            "id",
            [SearchScope(name="employees", type=NodeType.TABLE)],
        )
        assert "TABLE_NAME IN" in sql
        assert "EMPLOYEES" in binds

    def test_schema_scope_narrows_the_owner(self) -> None:
        _, sql, binds = self._find(
            [("HR", "EMPLOYEES")],
            NodeType.TABLE,
            "employees",
            [SearchScope(name="hr", type=NodeType.SCHEMA)],
        )
        assert "OWNER IN" in sql
        assert "HR" in binds

    def test_unquoted_name_is_matched_upper_cased(self) -> None:
        """Oracle stores an unquoted identifier folded up, so the symbol as
        typed rarely matches the dictionary verbatim."""
        _, _, binds = self._find([("HR", "EMPLOYEES")], NodeType.TABLE, "employees", [])
        assert "EMPLOYEES" in binds
        assert "employees" in binds

    def test_already_upper_name_needs_no_second_form(self) -> None:
        """Tested on the single-statement column query, so the count reflects
        the name forms alone and not the union's per-branch binds."""
        _, _, binds = self._find([("HR", "EMPLOYEES", "ID")], NodeType.COLUMN, "ID", [])
        assert binds == ["ID"]

    def test_union_binds_each_branch_separately(self) -> None:
        """oracledb counts a repeated ``:1`` as a second positional bind, so the
        branches cannot share one placeholder list (DPY-4009)."""
        _, sql, binds = self._find(
            [("HR", "EMPLOYEES")], NodeType.TABLE, "EMPLOYEES", []
        )
        assert binds == ["EMPLOYEES", "EMPLOYEES", "EMPLOYEES"]
        assert ":1" in sql and ":2" in sql and ":3" in sql

    def test_find_excludes_oracle_maintained_schemas(self) -> None:
        """A path the object tree does not contain would 404 on describe."""
        _, sql, _ = self._find([("HR", "EMPLOYEES")], NodeType.TABLE, "employees", [])
        assert "ORACLE_MAINTAINED" in sql

    def test_pre12c_find_uses_the_exclusion_list(self) -> None:
        driver = _make_driver([("HR", "EMPLOYEES")], has_oracle_maintained=False)
        asyncio.run(driver.explore_find(NodeType.TABLE, "employees", []))
        sql = driver._conn.cursor().execute.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "ORACLE_MAINTAINED" not in sql
        assert "NOT IN" in sql

    def test_find_stays_one_round_trip(self) -> None:
        """The union's branches are SQL fragments built in Python, not queries
        of their own — a driver that resolved synonyms with a second execute
        would double the cost of every table find."""
        driver = _make_driver([], has_oracle_maintained=True)
        asyncio.run(driver.explore_find(NodeType.TABLE, "emp", []))
        assert driver._conn.cursor().execute.call_count == 1  # ty: ignore[unresolved-attribute]

    def test_synonym_resolves_to_its_base_object(self) -> None:
        """The tree holds no synonyms, so a find returns the path of what the
        synonym points at — which is also what makes every synonym of one table
        share that table's describe, the explore cache being keyed by path."""
        paths, sql, _ = self._find([("HR", "EMPLOYEES")], NodeType.TABLE, "emp", [])
        assert paths == [["HR", "EMPLOYEES"]]
        assert "ALL_SYNONYMS" in sql
        assert "S.TABLE_OWNER, S.TABLE_NAME" in sql

    def test_synonym_over_a_database_link_is_excluded(self) -> None:
        """A remote object is in no schema of this tree."""
        _, sql, _ = self._find([], NodeType.TABLE, "emp", [])
        assert "DB_LINK IS NULL" in sql

    def test_synonym_resolves_only_to_a_listed_table_or_view(self) -> None:
        """Repeats what fetch_tables_and_views lists, so a dangling synonym —
        or a chained one, whose base is another synonym — resolves to nothing
        rather than to a path describe would 404 on."""
        _, sql, _ = self._find([], NodeType.TABLE, "emp", [])
        assert "EXISTS (SELECT 1 FROM ALL_TABLES T" in sql
        assert "EXISTS (SELECT 1 FROM ALL_VIEWS V" in sql

    def test_synonym_base_owner_is_filtered_like_every_other_path(self) -> None:
        """A synonym onto a system schema would leave the tree."""
        _, sql, _ = self._find([], NodeType.TABLE, "emp", [])
        assert "S.TABLE_OWNER IN" in sql

    def test_schema_scope_narrows_the_synonym_not_its_base(self) -> None:
        """A client scoping ``SALES.EMP`` is naming where the synonym lives;
        the path it resolves to is the base object's own owner."""
        paths, sql, binds = self._find(
            [("HR", "EMPLOYEES")],
            NodeType.TABLE,
            "emp",
            [SearchScope(name="sales", type=NodeType.SCHEMA)],
        )
        assert paths == [["HR", "EMPLOYEES"]]
        assert "S.OWNER IN (:" in sql
        assert "SALES" in binds

    def test_public_synonyms_answer_only_an_unqualified_name(self) -> None:
        """PUBLIC is no schema in the tree, so it survives only while the
        branch leaves the synonym's owner unconstrained."""
        _, sql, _ = self._find([], NodeType.TABLE, "emp", [])
        assert "S.OWNER IN (:" not in sql

    def test_schema_search_is_handed_to_the_walker(self) -> None:
        """The root listing is one cheap call the explore cache already serves."""
        driver = _make_driver([], has_oracle_maintained=True)
        with pytest.raises(FindNotSupported):
            asyncio.run(driver.explore_find(NodeType.SCHEMA, "hr", []))

    def test_unknown_node_type_is_handed_to_the_walker(self) -> None:
        driver = _make_driver([], has_oracle_maintained=True)
        with pytest.raises(FindNotSupported):
            asyncio.run(driver.explore_find("relationship_type", "ACTED_IN", []))


def _make_load_driver(
    error: Exception | None = None, rowcount: int = 0
) -> tuple[OracleDriver, MagicMock]:
    cur = MagicMock()
    cur.execute = AsyncMock()
    cur.executemany = AsyncMock(side_effect=error)
    cur.fetchall = AsyncMock(
        return_value=[
            ("NLS_DATE_FORMAT", "DD-MON-RR"),
            ("NLS_TIMESTAMP_FORMAT", "DD-MON-RR HH.MI.SSXFF AM"),
            ("NLS_TIMESTAMP_TZ_FORMAT", "DD-MON-RR HH.MI.SSXFF AM TZR"),
        ]
    )
    cur.rowcount = rowcount
    cur.warning = None
    conn = MagicMock(spec=oracledb.AsyncConnection)
    conn.cursor.return_value = cur
    return OracleDriver({}, conn, True, DriverSettings()), cur


def _alter_session_calls(cur: MagicMock) -> list[str]:
    return [
        call[0][0]
        for call in cur.execute.call_args_list
        if str(call[0][0]).startswith("ALTER SESSION")
    ]


class TestParseLoad:
    def test_returns_none_for_non_load_query(self) -> None:
        assert parse_load("SELECT 1 FROM dual") is None

    def test_parses_table_and_path(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.csv'")
        assert cmd == LoadCommand(
            table="employees", columns=None, path="/tmp/e.csv", options=LoadOptions()
        )

    def test_from_is_optional(self) -> None:
        """SQLcl writes the path positionally, with no FROM."""
        cmd = parse_load("LOAD employees '/tmp/e.csv'")
        assert cmd is not None
        assert cmd.path == "/tmp/e.csv"

    def test_accepts_table_keyword(self) -> None:
        cmd = parse_load("LOAD TABLE employees FROM '/tmp/e.csv'")
        assert cmd is not None
        assert cmd.table == "employees"

    def test_parses_schema_qualified_table(self) -> None:
        cmd = parse_load("LOAD hr.employees FROM '/tmp/e.csv'")
        assert cmd is not None
        assert cmd.table == "hr.employees"

    def test_parses_column_list(self) -> None:
        cmd = parse_load("LOAD employees (id, name) FROM '/tmp/e.csv'")
        assert cmd is not None
        assert cmd.columns == ["id", "name"]

    def test_case_insensitive(self) -> None:
        assert parse_load("load employees from '/tmp/e.csv'") is not None

    def test_unescapes_doubled_quotes_in_path(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/o''brien.csv'")
        assert cmd is not None
        assert cmd.path == "/tmp/o'brien.csv"

    def test_unquoted_path_names_the_missing_quotes(self) -> None:
        with pytest.raises(
            DriverError, match="must be in single quotes: FROM '/tmp/e.csv'"
        ):
            parse_load("LOAD employees FROM /tmp/e.csv")

    def test_unquoted_path_is_caught_without_the_from_keyword(self) -> None:
        with pytest.raises(DriverError, match="must be in single quotes"):
            parse_load("LOAD employees /tmp/e.csv (HEADER)")

    def test_unterminated_quote_is_named_as_such(self) -> None:
        with pytest.raises(DriverError, match="unterminated quote"):
            parse_load("LOAD employees FROM '/tmp/e.csv")

    def test_load_without_a_path_reports_the_syntax(self) -> None:
        with pytest.raises(DriverError, match="malformed LOAD — syntax is"):
            parse_load("LOAD employees")

    def test_rejects_non_identifier_table(self) -> None:
        with pytest.raises(DriverError, match="not a valid Oracle identifier"):
            parse_load("LOAD emp;DROP FROM '/tmp/e.csv'")

    def test_rejects_non_identifier_column(self) -> None:
        with pytest.raises(DriverError, match="not a valid Oracle identifier"):
            parse_load("LOAD employees (id, 1 + 1) FROM '/tmp/e.csv'")


class TestParseLoadOptions:
    def test_defaults(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.csv'")
        assert cmd is not None
        assert cmd.options == LoadOptions()

    def test_header_and_format(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.csv' (FORMAT csv, HEADER)")
        assert cmd is not None
        assert cmd.options.header is True

    def test_quoted_values(self) -> None:
        cmd = parse_load(
            "LOAD employees FROM '/tmp/e.csv' "
            "(DELIMITER '|', QUOTE '#', NULL 'NA', ENCODING 'latin-1')"
        )
        assert cmd is not None
        assert cmd.options.delimiter == "|"
        assert cmd.options.quote == "#"
        assert cmd.options.null == "NA"
        assert cmd.options.encoding == "latin-1"

    def test_tab_escape(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.tsv' (DELIMITER '\\t')")
        assert cmd is not None
        assert cmd.options.delimiter == "\t"

    def test_numeric_values(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.csv' (SKIP 2, BATCH 50)")
        assert cmd is not None
        assert cmd.options.skip == 2
        assert cmd.options.batch == 50

    def test_comma_inside_quoted_value(self) -> None:
        cmd = parse_load("LOAD employees FROM '/tmp/e.csv' (NULL 'a,b', HEADER)")
        assert cmd is not None
        assert cmd.options.null == "a,b"
        assert cmd.options.header is True

    def test_date_and_timestamp_formats(self) -> None:
        cmd = parse_load(
            "LOAD employees FROM '/tmp/e.csv' "
            "(DATEFORMAT 'YYYY-MM-DD', TIMESTAMPFORMAT 'YYYY-MM-DD HH24:MI:SS')"
        )
        assert cmd is not None
        assert cmd.options.date_format == "YYYY-MM-DD"
        assert cmd.options.timestamp_format == "YYYY-MM-DD HH24:MI:SS"

    def test_date_format_needs_a_value(self) -> None:
        with pytest.raises(DriverError, match="DATEFORMAT needs a value"):
            parse_load("LOAD employees FROM '/tmp/e.csv' (DATEFORMAT)")

    def test_rejects_unknown_option(self) -> None:
        with pytest.raises(DriverError, match="unknown LOAD option"):
            parse_load("LOAD employees FROM '/tmp/e.csv' (TRUNCATE)")

    def test_rejects_non_csv_format(self) -> None:
        with pytest.raises(DriverError, match="only FORMAT csv"):
            parse_load("LOAD employees FROM '/tmp/e.csv' (FORMAT binary)")

    def test_rejects_multi_character_delimiter(self) -> None:
        with pytest.raises(DriverError, match="single character"):
            parse_load("LOAD employees FROM '/tmp/e.csv' (DELIMITER '::')")

    def test_rejects_zero_batch(self) -> None:
        with pytest.raises(DriverError, match="positive whole number"):
            parse_load("LOAD employees FROM '/tmp/e.csv' (BATCH 0)")

    def test_rejects_unparenthesised_options(self) -> None:
        with pytest.raises(DriverError, match="parenthesised"):
            parse_load("LOAD employees FROM '/tmp/e.csv' HEADER")


class TestBuildInsertStatement:
    def test_without_columns_uses_table_order(self) -> None:
        assert (
            build_insert_statement("employees", None, 2)
            == "INSERT INTO employees VALUES (:1, :2)"
        )

    def test_with_columns(self) -> None:
        assert (
            build_insert_statement("hr.employees", ["id", "name"], 2)
            == "INSERT INTO hr.employees (id, name) VALUES (:1, :2)"
        )


class TestReadRows:
    def test_reads_plain_rows(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1,alice\n2,bob\n")
        with open(path, newline="") as f:
            header, rows = read_rows(f, LoadOptions())
            assert header is None
            assert list(rows) == [(1, ["1", "alice"]), (2, ["2", "bob"])]

    def test_reads_header_and_skips_lines(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("# dump\nid,name\n1,alice\n")
        with open(path, newline="") as f:
            header, rows = read_rows(f, LoadOptions(header=True, skip=1))
            assert header == ["id", "name"]
            # line 1 was skipped and line 2 was the header, so the data starts at 3
            assert list(rows) == [(3, ["1", "alice"])]

    def test_empty_field_is_read_as_an_empty_string(self, tmp_path) -> None:
        """Quoted or not, an empty field lands in Oracle as NULL — it stores a
        zero-length string that way whatever the column type — so the reader
        need not tell the two apart (and could not before Python 3.13)."""
        path = tmp_path / "e.csv"
        path.write_text('1,,""\n')
        with open(path, newline="") as f:
            _, rows = read_rows(f, LoadOptions())
            assert list(rows) == [(1, ["1", "", ""])]

    def test_null_option_maps_sentinel(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1,NA\n")
        with open(path, newline="") as f:
            _, rows = read_rows(f, LoadOptions(null="NA"))
            assert list(rows) == [(1, ["1", None])]

    def test_blank_lines_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1,alice\n\n2,bob\n")
        with open(path, newline="") as f:
            _, rows = read_rows(f, LoadOptions())
            assert list(rows) == [(1, ["1", "alice"]), (3, ["2", "bob"])]


class TestExecuteLoad:
    def test_loads_file_in_table_column_order(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1,alice\n2,bob\n")
        driver, cur = _make_load_driver()
        result = asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2
        statement, rows = cur.executemany.call_args[0]
        assert statement == "INSERT INTO employees VALUES (:1, :2)"
        assert rows == [["1", "alice"], ["2", "bob"]]

    def test_header_supplies_column_names(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("id,name\n1,alice\n")
        driver, cur = _make_load_driver()
        asyncio.run(driver.execute(f"LOAD employees FROM '{path}' (HEADER)"))
        statement, rows = cur.executemany.call_args[0]
        assert statement == "INSERT INTO employees (id, name) VALUES (:1, :2)"
        assert rows == [["1", "alice"]]

    def test_explicit_column_list_wins_over_header(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("a,b\n1,alice\n")
        driver, cur = _make_load_driver()
        asyncio.run(driver.execute(f"LOAD employees (id, name) FROM '{path}' (HEADER)"))
        statement, _ = cur.executemany.call_args[0]
        assert statement == "INSERT INTO employees (id, name) VALUES (:1, :2)"

    def test_sends_one_round_trip_per_batch(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1\n2\n3\n")
        driver, cur = _make_load_driver()
        result = asyncio.run(driver.execute(f"LOAD employees FROM '{path}' (BATCH 2)"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 3
        assert [call[0][1] for call in cur.executemany.call_args_list] == [
            [["1"], ["2"]],
            [["3"]],
        ]

    def test_empty_file_loads_nothing(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("")
        driver, cur = _make_load_driver()
        result = asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 0
        cur.executemany.assert_not_called()

    def test_missing_file_raises_driver_error(self, tmp_path) -> None:
        driver, _ = _make_load_driver()
        with pytest.raises(DriverError, match="could not read"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{tmp_path}/nope.csv'"))

    def test_ragged_row_raises_driver_error(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1,alice\n2\n")
        driver, _ = _make_load_driver()
        with pytest.raises(DriverError, match="line 2 .* has 1 fields, expected 2"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))

    def test_non_identifier_header_raises_driver_error(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("id,unit price\n1,3\n")
        driver, _ = _make_load_driver()
        with pytest.raises(DriverError, match="not a valid Oracle identifier"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}' (HEADER)"))

    def test_undecodable_file_raises_driver_error(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_bytes(b"1,caf\xe9\n")
        driver, _ = _make_load_driver()
        with pytest.raises(DriverError, match="not valid utf-8"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))

    def test_rejected_row_is_named_by_its_file_line(self, tmp_path) -> None:
        """The line, not the row ordinal: the header shifts the two apart."""
        path = tmp_path / "e.csv"
        path.write_text("ID\n1\n2\nx\n")
        error = _make_db_error("ORA-01722: invalid number")
        driver, _ = _make_load_driver(error=error, rowcount=2)
        with pytest.raises(DriverError, match=r"invalid number \(line 4 of "):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}' (HEADER)"))

    def test_dead_session_raises_connection_lost(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1\n")
        error = oracledb.OperationalError("connection closed")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))


class TestLoadNlsFormats:
    def test_no_format_option_leaves_the_session_alone(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("1\n")
        driver, cur = _make_load_driver()
        asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))
        assert _alter_session_calls(cur) == []

    def test_date_format_is_applied_then_restored(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        driver, cur = _make_load_driver()
        asyncio.run(
            driver.execute(f"LOAD employees FROM '{path}' (DATEFORMAT 'YYYY-MM-DD')")
        )
        applied, restored = _alter_session_calls(cur)[:3], _alter_session_calls(cur)[3:]
        assert applied == [
            "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'",
            "ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD'",
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = 'YYYY-MM-DD'",
        ]
        assert restored == [
            "ALTER SESSION SET NLS_DATE_FORMAT = 'DD-MON-RR'",
            "ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'DD-MON-RR HH.MI.SSXFF AM'",
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = "
            "'DD-MON-RR HH.MI.SSXFF AM TZR'",
        ]

    def test_both_formats_are_applied(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        driver, cur = _make_load_driver()
        asyncio.run(
            driver.execute(
                f"LOAD employees FROM '{path}' "
                "(DATEFORMAT 'YYYY-MM-DD', TIMESTAMPFORMAT 'YYYY-MM-DD HH24:MI:SS')"
            )
        )
        assert _alter_session_calls(cur)[:3] == [
            "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'",
            "ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS'",
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = 'YYYY-MM-DD HH24:MI:SS'",
        ]

    def test_format_is_restored_after_a_failed_load(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        error = _make_db_error("ORA-01843: not a valid month")
        driver, cur = _make_load_driver(error=error)
        with pytest.raises(DriverError):
            asyncio.run(
                driver.execute(
                    f"LOAD employees FROM '{path}' (DATEFORMAT 'DD/MM/YYYY')"
                )
            )
        assert _alter_session_calls(cur)[3] == (
            "ALTER SESSION SET NLS_DATE_FORMAT = 'DD-MON-RR'"
        )

    def test_timestamp_format_overrides_the_date_format(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        driver, cur = _make_load_driver()
        asyncio.run(
            driver.execute(
                f"LOAD employees FROM '{path}' "
                "(DATEFORMAT 'YYYY-MM-DD', TIMESTAMPFORMAT 'YYYY-MM-DD HH24:MI:SS')"
            )
        )
        applied = _alter_session_calls(cur)[:3]
        assert "NLS_DATE_FORMAT = 'YYYY-MM-DD'" in applied[0]
        assert "NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS'" in applied[1]

    def test_timestamp_format_alone_leaves_the_date_format(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        driver, cur = _make_load_driver()
        asyncio.run(
            driver.execute(
                f"LOAD employees FROM '{path}' (TIMESTAMPFORMAT 'YYYY-MM-DD')"
            )
        )
        applied = _alter_session_calls(cur)[:2]
        assert applied == [
            "ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD'",
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = 'YYYY-MM-DD'",
        ]

    def test_quotes_in_a_format_model_are_escaped(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2024-03-01\n")
        driver, cur = _make_load_driver()
        asyncio.run(
            driver.execute(f"LOAD employees FROM '{path}' (DATEFORMAT 'YY''MM')")
        )
        assert _alter_session_calls(cur)[0] == (
            "ALTER SESSION SET NLS_DATE_FORMAT = 'YY''MM'"
        )


class TestLoadConversionHint:
    def test_date_error_names_the_session_format(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2026-09-04\n")
        error = _make_db_error("ORA-01843: not a valid month")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(DriverError, match="NLS_DATE_FORMAT 'DD-MON-RR'"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))

    def test_date_error_suggests_the_option(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2026-09-04\n")
        error = _make_db_error("ORA-01861: literal does not match format string")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(DriverError, match="DATEFORMAT 'YYYY-MM-DD'"):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))

    def test_hint_points_at_the_given_format_when_one_was_set(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("2026-09-04\n")
        error = _make_db_error("ORA-01843: not a valid month")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(DriverError, match="neither format the load set"):
            asyncio.run(
                driver.execute(
                    f"LOAD employees FROM '{path}' (DATEFORMAT 'DD/MM/YYYY')"
                )
            )

    def test_hint_names_the_timestamp_model_too(self, tmp_path) -> None:
        """A TIMESTAMP column converts through its own model, not the date one."""
        path = tmp_path / "e.csv"
        path.write_text("2026-09-04\n")
        error = _make_db_error("ORA-01843: not a valid month")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(
            DriverError, match="NLS_TIMESTAMP_FORMAT 'DD-MON-RR HH.MI.SSXFF AM'"
        ):
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))

    def test_unrelated_error_gets_no_hint(self, tmp_path) -> None:
        path = tmp_path / "e.csv"
        path.write_text("x\n")
        error = _make_db_error("ORA-01722: invalid number")
        driver, _ = _make_load_driver(error=error)
        with pytest.raises(DriverError) as excinfo:
            asyncio.run(driver.execute(f"LOAD employees FROM '{path}'"))
        assert "NLS_DATE_FORMAT" not in str(excinfo.value)
