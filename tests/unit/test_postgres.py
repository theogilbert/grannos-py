"""Unit tests for PostgresDriver — no live database required."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg import sql

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.postgres.copy import (
    CopyFromCommand,
    CopyToCommand,
    build_copy_from_statement,
    build_copy_to_statement,
    parse_copy_from,
    parse_copy_to,
)
from grannos.drivers.postgres.driver import (
    PostgresDriver,
    _maybe_raise_connection_lost,
    _set_property,
)
from grannos.drivers.postgres.queries import render_lob
from grannos.protocol import ExploreItem, LobPlaceholder, ReadResult, WriteResult


def _make_driver(
    rows: list | None = None,
    description: list | None = None,
    rowcount: int = 0,
    execute_side_effect: Exception | None = None,
) -> tuple[PostgresDriver, MagicMock]:
    cur = MagicMock()
    cur.execute = AsyncMock(side_effect=execute_side_effect)
    cur.fetchall = AsyncMock(return_value=rows or [])
    cur.description = description
    cur.rowcount = rowcount
    conn = MagicMock(spec=psycopg.AsyncConnection)
    conn.cursor.return_value = cur
    return PostgresDriver({}, conn, DriverSettings()), cur


class _FakeAsyncCopy:
    """Mimics psycopg's Copy object: an async context manager and async iterator."""

    def __init__(
        self, chunks: list[bytes] | None = None, enter_exc: Exception | None = None
    ) -> None:
        self._chunks = list(chunks or [])
        self._enter_exc = enter_exc

    async def __aenter__(self) -> "_FakeAsyncCopy":
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def __aiter__(self) -> "_FakeAsyncCopy":
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _make_copy_driver(
    chunks: list[bytes] | None = None,
    rowcount: int = 0,
    enter_exc: Exception | None = None,
) -> tuple[PostgresDriver, MagicMock]:
    cur = MagicMock()
    cur.rowcount = rowcount
    cur.copy = MagicMock(return_value=_FakeAsyncCopy(chunks, enter_exc))
    conn = MagicMock(spec=psycopg.AsyncConnection)
    conn.cursor.return_value = cur
    return PostgresDriver({}, conn, DriverSettings()), cur


class _FakeAsyncCopyIn:
    """Mimics psycopg's Copy object for ``COPY ... FROM STDIN``: an async
    context manager that records what's written to it."""

    def __init__(self, enter_exc: Exception | None = None) -> None:
        self.written: list[bytes] = []
        self._enter_exc = enter_exc

    async def __aenter__(self) -> "_FakeAsyncCopyIn":
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def write(self, buffer: bytes) -> None:
        self.written.append(bytes(buffer))


def _make_copy_from_driver(
    rowcount: int = 0,
    enter_exc: Exception | None = None,
) -> tuple[PostgresDriver, MagicMock, _FakeAsyncCopyIn]:
    cur = MagicMock()
    cur.rowcount = rowcount
    fake_copy = _FakeAsyncCopyIn(enter_exc)
    cur.copy = MagicMock(return_value=fake_copy)
    conn = MagicMock(spec=psycopg.AsyncConnection)
    conn.cursor.return_value = cur
    return PostgresDriver({}, conn, DriverSettings()), cur, fake_copy


class TestExecuteParams:
    def test_no_binds_passes_none_as_params(self) -> None:
        driver, cur = _make_driver(description=None, rowcount=0)
        asyncio.run(driver.execute("DELETE FROM t WHERE x LIKE '%foo%'"))
        assert cur.execute.call_args[0][1] is None

    def test_empty_binds_list_passes_none_as_params(self) -> None:
        driver, cur = _make_driver(description=None, rowcount=0)
        asyncio.run(driver.execute("DELETE FROM t", []))
        assert cur.execute.call_args[0][1] is None

    def test_binds_are_forwarded(self) -> None:
        driver, cur = _make_driver(
            description=[SimpleNamespace(name="val")], rows=[(42,)]
        )
        asyncio.run(driver.execute("SELECT %s AS val", [42]))
        assert cur.execute.call_args[0][1] == [42]


class TestExecuteResults:
    def test_returns_read_result_for_select(self) -> None:
        driver, _ = _make_driver(
            description=[SimpleNamespace(name="n"), SimpleNamespace(name="s")],
            rows=[(1, "hello")],
        )
        result = asyncio.run(driver.execute("SELECT 1 AS n, 'hello' AS s"))
        assert isinstance(result, ReadResult)
        assert result.columns == ["n", "s"]
        assert result.rows == [[1, "hello"]]

    def test_returns_write_result_for_dml(self) -> None:
        driver, _ = _make_driver(description=None, rowcount=3)
        result = asyncio.run(driver.execute("DELETE FROM t"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 3

    def test_negative_rowcount_defaults_to_zero(self) -> None:
        driver, _ = _make_driver(description=None, rowcount=-1)
        result = asyncio.run(driver.execute("CREATE TABLE t (id integer)"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 0

    def test_replaces_bytea_values_with_placeholders(self) -> None:
        driver, _ = _make_driver(
            description=[SimpleNamespace(name="id"), SimpleNamespace(name="data")],
            rows=[(1, b"\x00\x01")],
        )
        result = asyncio.run(driver.execute("SELECT id, data FROM t"))
        assert isinstance(result, ReadResult)
        [[_, lob]] = result.rows
        assert isinstance(lob, LobPlaceholder)
        assert lob.text == "BYTEA (2 bytes)"
        assert lob.ref is not None


def _null_register_lob(value: object, text: str) -> LobPlaceholder:
    return LobPlaceholder(text=text)


class TestRenderLob:
    def test_passes_through_non_binary_values(self) -> None:
        assert render_lob(_null_register_lob, "hello") == "hello"
        assert render_lob(_null_register_lob, 42) == 42
        assert render_lob(_null_register_lob, None) is None

    def test_renders_bytes_as_byte_count(self) -> None:
        assert render_lob(_null_register_lob, b"\x01\x02\x03") == LobPlaceholder(
            text="BYTEA (3 bytes)"
        )


class TestExecuteErrorPropagation:
    def test_database_error_raises_driver_error(self) -> None:
        exc = psycopg.errors.SyntaxError("syntax error at or near")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(DriverError, match="syntax error"):
            asyncio.run(driver.execute("SELEC 1"))

    def test_operational_error_raises_connection_lost(self) -> None:
        exc = psycopg.OperationalError("server closed the connection unexpectedly")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("SELECT 1"))

    def test_interface_error_raises_connection_lost(self) -> None:
        exc = psycopg.InterfaceError("the connection is closed")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("SELECT 1"))

    def test_explore_list_not_connected_raises_connection_lost(self) -> None:
        exc = psycopg.OperationalError("connection lost")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_list([]))

    def test_explore_describe_not_connected_raises_connection_lost(self) -> None:
        exc = psycopg.OperationalError("connection lost")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_describe(["myschema", "mytable"]))


class TestMaybeRaiseConnectionLost:
    def test_operational_error_raises(self) -> None:
        with pytest.raises(ConnectionLostError):
            _maybe_raise_connection_lost(psycopg.OperationalError("down"))

    def test_interface_error_raises(self) -> None:
        with pytest.raises(ConnectionLostError):
            _maybe_raise_connection_lost(psycopg.InterfaceError("closed"))

    def test_other_error_does_not_raise(self) -> None:
        _maybe_raise_connection_lost(psycopg.errors.SyntaxError("bad syntax"))


class TestSetProperty:
    def test_extracts_property_name_with_equals(self) -> None:
        assert _set_property("SET search_path = 'public'") == "search_path"

    def test_extracts_property_name_with_to(self) -> None:
        assert _set_property("SET search_path TO public") == "search_path"

    def test_case_insensitive(self) -> None:
        assert _set_property("set SEARCH_PATH = public") == "search_path"

    def test_leading_whitespace(self) -> None:
        assert _set_property("   SET foo = 1") == "foo"

    def test_session_keyword(self) -> None:
        assert _set_property("SET SESSION search_path = public") == "search_path"

    def test_set_local_returns_none(self) -> None:
        assert _set_property("SET LOCAL statement_timeout = 5000") is None

    def test_non_set_returns_none(self) -> None:
        assert _set_property("SELECT 1") is None

    def test_set_time_zone_returns_none(self) -> None:
        assert _set_property("SET TIME ZONE 'UTC'") is None


class TestExecuteRecordsSessionStatements:
    def test_records_on_success(self) -> None:
        driver, _ = _make_driver()
        stmt = "SET search_path = 'public'"
        asyncio.run(driver.execute(stmt))
        assert driver._session_statements == {"search_path": stmt}

    def test_records_after_a_leading_comment(self) -> None:
        driver, _ = _make_driver()
        stmt = "-- schema\nSET search_path = 'public'"
        asyncio.run(driver.execute(stmt))
        assert driver._session_statements == {"search_path": stmt}

    def test_later_value_replaces_earlier(self) -> None:
        driver, _ = _make_driver()
        first = "SET search_path = 'public'"
        second = "SET search_path = 'other'"
        asyncio.run(driver.execute(first))
        asyncio.run(driver.execute(second))
        assert driver._session_statements == {"search_path": second}

    def test_does_not_record_plain_query(self) -> None:
        driver, _ = _make_driver()
        asyncio.run(driver.execute("SELECT 1"))
        assert driver._session_statements == {}

    def test_failed_statement_not_recorded(self) -> None:
        exc = psycopg.errors.SyntaxError("syntax error at or near")
        driver, _ = _make_driver(execute_side_effect=exc)
        with pytest.raises(DriverError):
            asyncio.run(driver.execute("SET bogus = 1"))
        assert driver._session_statements == {}


class TestReconnectReplaysSessionStatements:
    def test_replays_recorded_statements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = _make_driver()
        stmt = "SET search_path = 'public'"
        asyncio.run(driver.execute(stmt))

        new_cur = MagicMock()
        new_cur.execute = AsyncMock()
        new_conn = MagicMock(spec=psycopg.AsyncConnection)
        new_conn.cursor.return_value = new_cur

        async def fake_open(params: dict) -> MagicMock:
            return new_conn

        monkeypatch.setattr(PostgresDriver, "_open", staticmethod(fake_open))
        asyncio.run(driver.reconnect())

        # params is an explicit None now that every statement routes through
        # _exec; psycopg treats that the same as omitting the argument.
        new_cur.execute.assert_awaited_with(sql.SQL(stmt), None)

    def test_no_statements_means_no_replay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, _ = _make_driver()

        new_conn = MagicMock(spec=psycopg.AsyncConnection)

        async def fake_open(params: dict) -> MagicMock:
            return new_conn

        monkeypatch.setattr(PostgresDriver, "_open", staticmethod(fake_open))
        asyncio.run(driver.reconnect())

        new_conn.cursor.assert_not_called()

    def test_replay_failure_raises_driver_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, _ = _make_driver()
        asyncio.run(driver.execute("SET foo = 1"))

        exc = psycopg.errors.SyntaxError("syntax error at or near")
        new_cur = MagicMock()
        new_cur.execute = AsyncMock(side_effect=exc)
        new_conn = MagicMock(spec=psycopg.AsyncConnection)
        new_conn.cursor.return_value = new_cur

        async def fake_open(params: dict) -> MagicMock:
            return new_conn

        monkeypatch.setattr(PostgresDriver, "_open", staticmethod(fake_open))
        with pytest.raises(DriverError, match="syntax error"):
            asyncio.run(driver.reconnect())


class TestRootListing:
    def test_returns_schema_items(self) -> None:
        driver, _ = _make_driver(rows=[("alice",), ("bob",)])
        items = asyncio.run(driver.explore_list([]))
        assert items == [
            ExploreItem(name="alice", type="schema", expandable=True),
            ExploreItem(name="bob", type="schema", expandable=True),
        ]


class TestParseCopyTo:
    def test_keeps_comments_of_the_source_query(self) -> None:
        query = "-- export\n\\copy (SELECT /* all */ 1) TO '/tmp/o.csv' -- done"
        code = "         \n\\copy (SELECT           1) TO '/tmp/o.csv'        "
        cmd = parse_copy_to(query, code)
        assert cmd is not None
        assert cmd.source == "(SELECT /* all */ 1)"
        assert cmd.path == "/tmp/o.csv"

    def test_returns_none_for_non_copy_query(self) -> None:
        assert parse_copy_to("SELECT 1") is None

    def test_parses_table_source(self) -> None:
        cmd = parse_copy_to("\\copy orders TO '/tmp/orders.csv'")
        assert cmd == CopyToCommand(source="orders", path="/tmp/orders.csv", options="")

    def test_parses_schema_qualified_table_source(self) -> None:
        cmd = parse_copy_to("\\copy public.orders TO '/tmp/orders.csv'")
        assert cmd is not None
        assert cmd.source == "public.orders"

    def test_parses_query_source(self) -> None:
        cmd = parse_copy_to(
            "\\copy (SELECT * FROM orders WHERE status = 'open') TO '/tmp/o.csv'"
        )
        assert cmd is not None
        assert cmd.source == "(SELECT * FROM orders WHERE status = 'open')"
        assert cmd.path == "/tmp/o.csv"

    def test_parses_trailing_options(self) -> None:
        cmd = parse_copy_to(
            "\\copy orders TO '/tmp/orders.csv' WITH (FORMAT csv, HEADER)"
        )
        assert cmd is not None
        assert cmd.options == "WITH (FORMAT csv, HEADER)"

    def test_unescapes_doubled_quotes_in_path(self) -> None:
        cmd = parse_copy_to("\\copy orders TO '/tmp/o''brien.csv'")
        assert cmd is not None
        assert cmd.path == "/tmp/o'brien.csv"

    def test_case_insensitive(self) -> None:
        assert parse_copy_to("\\COPY orders TO '/tmp/orders.csv'") is not None

    def test_ignores_from_direction(self) -> None:
        assert parse_copy_to("\\copy orders FROM '/tmp/orders.csv'") is None


class TestBuildCopyToStatement:
    def test_without_options(self) -> None:
        cmd = CopyToCommand(source="orders", path="/tmp/o.csv", options="")
        assert build_copy_to_statement(cmd) == "COPY orders TO STDOUT"

    def test_with_options(self) -> None:
        cmd = CopyToCommand(
            source="orders", path="/tmp/o.csv", options="WITH (FORMAT csv, HEADER)"
        )
        assert (
            build_copy_to_statement(cmd)
            == "COPY orders TO STDOUT WITH (FORMAT csv, HEADER)"
        )


class TestParseCopyFrom:
    def test_returns_none_for_non_copy_query(self) -> None:
        assert parse_copy_from("SELECT 1") is None

    def test_parses_table_target(self) -> None:
        cmd = parse_copy_from("\\copy orders FROM '/tmp/orders.csv'")
        assert cmd == CopyFromCommand(
            target="orders", path="/tmp/orders.csv", options=""
        )

    def test_parses_schema_qualified_table_target(self) -> None:
        cmd = parse_copy_from("\\copy public.orders FROM '/tmp/orders.csv'")
        assert cmd is not None
        assert cmd.target == "public.orders"

    def test_parses_column_list_target(self) -> None:
        cmd = parse_copy_from("\\copy orders (id, status) FROM '/tmp/orders.csv'")
        assert cmd is not None
        assert cmd.target == "orders (id, status)"

    def test_parses_trailing_options(self) -> None:
        cmd = parse_copy_from(
            "\\copy orders FROM '/tmp/orders.csv' WITH (FORMAT csv, HEADER)"
        )
        assert cmd is not None
        assert cmd.options == "WITH (FORMAT csv, HEADER)"

    def test_unescapes_doubled_quotes_in_path(self) -> None:
        cmd = parse_copy_from("\\copy orders FROM '/tmp/o''brien.csv'")
        assert cmd is not None
        assert cmd.path == "/tmp/o'brien.csv"

    def test_case_insensitive(self) -> None:
        assert parse_copy_from("\\COPY orders FROM '/tmp/orders.csv'") is not None

    def test_ignores_to_direction(self) -> None:
        assert parse_copy_from("\\copy orders TO '/tmp/orders.csv'") is None


class TestBuildCopyFromStatement:
    def test_without_options(self) -> None:
        cmd = CopyFromCommand(target="orders", path="/tmp/o.csv", options="")
        assert build_copy_from_statement(cmd) == "COPY orders FROM STDIN"

    def test_with_options(self) -> None:
        cmd = CopyFromCommand(
            target="orders", path="/tmp/o.csv", options="WITH (FORMAT csv, HEADER)"
        )
        assert (
            build_copy_from_statement(cmd)
            == "COPY orders FROM STDIN WITH (FORMAT csv, HEADER)"
        )


class TestExecuteCopyTo:
    def test_streams_result_to_local_file(self, tmp_path) -> None:
        dest = tmp_path / "orders.csv"
        driver, cur = _make_copy_driver(
            chunks=[b"id,status\n", b"1,open\n"], rowcount=1
        )
        result = asyncio.run(
            driver.execute(f"\\copy orders TO '{dest}' (FORMAT csv, HEADER)")
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        assert dest.read_bytes() == b"id,status\n1,open\n"
        cur.copy.assert_called_once()

        assert cur.copy.call_args[0][0] == sql.SQL(
            "COPY orders TO STDOUT (FORMAT csv, HEADER)"
        )

    def test_negative_rowcount_defaults_to_zero(self, tmp_path) -> None:
        dest = tmp_path / "orders.csv"
        driver, _ = _make_copy_driver(chunks=[], rowcount=-1)
        result = asyncio.run(driver.execute(f"\\copy orders TO '{dest}'"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 0

    def test_write_failure_raises_driver_error(self) -> None:
        driver, _ = _make_copy_driver(chunks=[b"data"])
        with pytest.raises(DriverError, match="could not write"):
            asyncio.run(
                driver.execute("\\copy orders TO '/nonexistent-dir/orders.csv'")
            )

    def test_database_error_raises_driver_error(self) -> None:
        exc = psycopg.errors.SyntaxError('relation "orders" does not exist')
        driver, _ = _make_copy_driver(enter_exc=exc)
        with pytest.raises(DriverError, match="does not exist"):
            asyncio.run(driver.execute("\\copy orders TO '/tmp/orders.csv'"))

    def test_operational_error_raises_connection_lost(self) -> None:
        exc = psycopg.OperationalError("server closed the connection unexpectedly")
        driver, _ = _make_copy_driver(enter_exc=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("\\copy orders TO '/tmp/orders.csv'"))


class TestExecuteCopyFrom:
    def test_streams_local_file_into_table(self, tmp_path) -> None:
        src = tmp_path / "orders.csv"
        src.write_bytes(b"id,status\n1,open\n")
        driver, cur, fake_copy = _make_copy_from_driver(rowcount=1)
        result = asyncio.run(
            driver.execute(f"\\copy orders FROM '{src}' (FORMAT csv, HEADER)")
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        assert b"".join(fake_copy.written) == b"id,status\n1,open\n"
        cur.copy.assert_called_once()

        assert cur.copy.call_args[0][0] == sql.SQL(
            "COPY orders FROM STDIN (FORMAT csv, HEADER)"
        )

    def test_negative_rowcount_defaults_to_zero(self, tmp_path) -> None:
        src = tmp_path / "orders.csv"
        src.write_bytes(b"")
        driver, _, _ = _make_copy_from_driver(rowcount=-1)
        result = asyncio.run(driver.execute(f"\\copy orders FROM '{src}'"))
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 0

    def test_read_failure_raises_driver_error(self) -> None:
        driver, _, _ = _make_copy_from_driver()
        with pytest.raises(DriverError, match="could not read"):
            asyncio.run(
                driver.execute("\\copy orders FROM '/nonexistent-dir/orders.csv'")
            )

    def test_database_error_raises_driver_error(self, tmp_path) -> None:
        src = tmp_path / "orders.csv"
        src.write_bytes(b"1,open\n")
        exc = psycopg.errors.SyntaxError('relation "orders" does not exist')
        driver, _, _ = _make_copy_from_driver(enter_exc=exc)
        with pytest.raises(DriverError, match="does not exist"):
            asyncio.run(driver.execute(f"\\copy orders FROM '{src}'"))

    def test_operational_error_raises_connection_lost(self, tmp_path) -> None:
        src = tmp_path / "orders.csv"
        src.write_bytes(b"1,open\n")
        exc = psycopg.OperationalError("server closed the connection unexpectedly")
        driver, _, _ = _make_copy_from_driver(enter_exc=exc)
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute(f"\\copy orders FROM '{src}'"))


class TestTableGroupListing:
    def test_returns_fixed_groups(self) -> None:
        driver, _ = _make_driver()
        items = asyncio.run(driver.explore_list(["public", "orders"]))
        assert {i.name for i in items} == {"columns", "indexes"}
        assert all(i.type == "group" and i.expandable for i in items)
