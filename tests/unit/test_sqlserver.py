"""Unit tests for SQLServerDriver — no live database required."""

import asyncio
import os
import socket
from unittest.mock import MagicMock

import mssql_python
import pytest

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.sqlserver import SQLServerDriver, _Link, _render_lob
from grannos.protocol import LobPlaceholder, ReadResult


def _closed_connection_error() -> mssql_python.InterfaceError:
    return mssql_python.InterfaceError(
        driver_error="Cannot create cursor on closed connection",
        ddbc_error="Cannot create cursor on closed connection",
    )


def _make_driver(cur: MagicMock) -> SQLServerDriver:
    conn = MagicMock()
    conn.execute.return_value = cur
    return SQLServerDriver({}, conn, DriverSettings())


def _null_register_lob(value: object, text: str) -> LobPlaceholder:
    return LobPlaceholder(text=text)


class TestRenderLob:
    def test_passes_through_non_binary_values(self) -> None:
        assert _render_lob(_null_register_lob, "hello") == "hello"
        assert _render_lob(_null_register_lob, 42) == 42
        assert _render_lob(_null_register_lob, None) is None

    def test_renders_bytes_as_byte_count(self) -> None:
        assert _render_lob(_null_register_lob, b"\x01\x02\x03") == LobPlaceholder(
            text="VARBINARY (3 bytes)"
        )

    def test_renders_bytearray_as_byte_count(self) -> None:
        assert _render_lob(_null_register_lob, bytearray(5)) == LobPlaceholder(
            text="VARBINARY (5 bytes)"
        )


class TestExecuteRendersLobs:
    def test_replaces_binary_values_with_placeholders(self) -> None:
        cur = MagicMock()
        cur.description = [("ID",), ("DATA",)]
        cur.fetchall.return_value = [(1, b"\x00\x01")]
        driver = _make_driver(cur)
        result = asyncio.run(driver.execute("SELECT id, data FROM t", []))
        assert isinstance(result, ReadResult)
        [[_, lob]] = result.rows
        assert isinstance(lob, LobPlaceholder)
        assert lob.text == "VARBINARY (2 bytes)"
        assert lob.ref is not None


class TestConnectionLostTranslation:
    def test_execute_raises_connection_lost(self) -> None:
        conn = MagicMock()
        conn.execute.side_effect = _closed_connection_error()
        driver = SQLServerDriver({}, conn, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("SELECT 1", []))

    def test_communication_link_failure_raises_connection_lost(self) -> None:
        conn = MagicMock()
        conn.execute.side_effect = mssql_python.OperationalError(
            driver_error="Communication link failure", ddbc_error="TCP reset"
        )
        driver = SQLServerDriver({}, conn, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("SELECT 1", []))

    def test_database_access_denied_is_driver_error(self) -> None:
        # Error 916 arrives as SQLSTATE 08004 on a live connection: a
        # reconnect cannot fix it, so it must not be treated as a lost link.
        conn = MagicMock()
        conn.execute.side_effect = mssql_python.OperationalError(
            driver_error="Server rejected the connection",
            ddbc_error="The server principal is not able to access the database",
        )
        driver = SQLServerDriver({}, conn, DriverSettings())
        with pytest.raises(DriverError, match="not able to access"):
            asyncio.run(driver.execute("SELECT 1", []))

    def test_explore_list_raises_connection_lost(self) -> None:
        conn = MagicMock()
        conn.cursor.side_effect = _closed_connection_error()
        driver = SQLServerDriver({}, conn, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_list([]))

    def test_explore_describe_raises_connection_lost(self) -> None:
        conn = MagicMock()
        conn.cursor.side_effect = _closed_connection_error()
        driver = SQLServerDriver({}, conn, DriverSettings())
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_describe(["dbo", "orders"]))


def _identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


class TestLinkSever:
    def test_should_shut_down_socket(self) -> None:
        ours, peer = socket.socketpair()
        with ours, peer:
            _Link(ours.fileno(), _identity(ours.fileno())).sever()
            assert peer.recv(1) == b""  # EOF: the link is cut
            assert ours.fileno() != -1  # the driver's fd itself stays open

    def test_should_leave_reused_fd_alone(self) -> None:
        ours, peer = socket.socketpair()
        with ours, peer:
            _Link(ours.fileno(), (-1, -1)).sever()
            ours.sendall(b"x")
            assert peer.recv(1) == b"x"

    def test_should_ignore_closed_fd(self) -> None:
        ours, peer = socket.socketpair()
        fd, identity = ours.fileno(), _identity(ours.fileno())
        ours.close()
        peer.close()
        _Link(fd, identity).sever()

    async def test_disconnect_should_tolerate_severed_link(self) -> None:
        conn = MagicMock()
        conn.close.side_effect = mssql_python.OperationalError(
            driver_error="Communication link failure", ddbc_error=""
        )
        await SQLServerDriver({}, conn, DriverSettings()).disconnect()
