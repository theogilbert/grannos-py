"""Unit tests for SQLiteDriver — no live database required."""

import asyncio
import sqlite3

from grannos.drivers.base import DriverSettings
from grannos.drivers.sqlite.driver import SQLiteDriver, _render_lob
from grannos.protocol import LobPlaceholder, ReadResult


def _make_driver() -> SQLiteDriver:
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    return SQLiteDriver({}, conn, DriverSettings())


def _null_register_lob(value: object, text: str) -> LobPlaceholder:
    return LobPlaceholder(text=text)


class TestRenderLob:
    def test_passes_through_non_binary_values(self) -> None:
        assert _render_lob(_null_register_lob, "hello") == "hello"
        assert _render_lob(_null_register_lob, 42) == 42
        assert _render_lob(_null_register_lob, None) is None

    def test_renders_bytes_as_byte_count(self) -> None:
        assert _render_lob(_null_register_lob, b"\x01\x02\x03") == LobPlaceholder(
            text="BLOB (3 bytes)"
        )

    def test_renders_bytearray_as_byte_count(self) -> None:
        assert _render_lob(_null_register_lob, bytearray(5)) == LobPlaceholder(
            text="BLOB (5 bytes)"
        )


class TestExecuteRendersLobs:
    def test_replaces_blob_values_with_placeholders(self) -> None:
        driver = _make_driver()
        asyncio.run(driver.execute("CREATE TABLE t (id INTEGER, data BLOB)", []))
        asyncio.run(driver.execute("INSERT INTO t VALUES (?, ?)", [1, b"\x00\x01"]))
        result = asyncio.run(driver.execute("SELECT id, data FROM t", []))
        assert isinstance(result, ReadResult)
        [[_, lob]] = result.rows
        assert isinstance(lob, LobPlaceholder)
        assert lob.text == "BLOB (2 bytes)"
        assert lob.ref is not None
