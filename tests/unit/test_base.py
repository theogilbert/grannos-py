"""Unit tests for shared driver helpers in grannos.drivers.base."""

import base64
from pathlib import Path
from typing import Any

import pytest

from grannos.drivers.base import (
    BaseDriver,
    DriverError,
    DriverSettings,
    build_column_samples,
)
from grannos.protocol import (
    DownloadResult,
    ExploreItem,
    LobPlaceholder,
    ReadResult,
    WriteResult,
)


class _StubDriver(BaseDriver):
    """Minimal concrete BaseDriver for exercising base-class-only behavior
    (LOB ref cache, explore_download_ref) without a real database."""

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "_StubDriver":
        return cls(params, settings)

    async def reconnect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        return WriteResult(rows_affected=0)

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        return []

    async def explore_describe(self, path: list[str]) -> None:
        return None


def _make_driver() -> _StubDriver:
    return _StubDriver({}, DriverSettings())


def _ref(placeholder: LobPlaceholder) -> str:
    """Return a placeholder's download ref, asserting the driver cached one.

    ``ref`` is optional on the wire — None when the driver chose not to cache
    the value — but every placeholder these tests register has one.
    """
    assert placeholder.ref is not None
    return placeholder.ref


class TestRegisterLobAndDownloadRef:
    async def test_register_then_download_ref_returns_base64(self) -> None:
        driver = _make_driver()
        placeholder = driver._register_lob(b"hello", "text (5 bytes)")
        assert placeholder.text == "text (5 bytes)"
        assert placeholder.ref is not None
        result = await driver.explore_download_ref(placeholder.ref, None)
        assert isinstance(result, DownloadResult)
        assert result.content_base64 == "aGVsbG8="
        assert result.written_to is None
        assert result.size == 5

    async def test_register_then_download_ref_writes_to_dest_path(
        self, tmp_path: Path
    ) -> None:
        driver = _make_driver()
        placeholder = driver._register_lob(b"hello", "text")
        dest = tmp_path / "out.bin"
        result = await driver.explore_download_ref(_ref(placeholder), str(dest))
        assert result.written_to == str(dest)
        assert result.content_base64 is None
        assert dest.read_bytes() == b"hello"

    async def test_str_value_is_utf8_encoded(self) -> None:
        driver = _make_driver()
        placeholder = driver._register_lob("héllo", "text")
        result = await driver.explore_download_ref(_ref(placeholder), None)
        assert result.content_base64 is not None
        assert base64.b64decode(result.content_base64) == "héllo".encode()
        assert result.content_type == "text/plain"

    async def test_bytes_value_gets_octet_stream_content_type(self) -> None:
        driver = _make_driver()
        placeholder = driver._register_lob(b"\x00\x01", "bin")
        result = await driver.explore_download_ref(_ref(placeholder), None)
        assert result.content_type == "application/octet-stream"

    async def test_unknown_ref_raises_driver_error(self) -> None:
        driver = _make_driver()
        with pytest.raises(DriverError):
            await driver.explore_download_ref("nonexistent", None)

    async def test_cache_evicts_oldest_beyond_max(self) -> None:
        driver = _make_driver()
        refs = [
            _ref(driver._register_lob(f"v{i}".encode(), "t"))
            for i in range(driver._LOB_CACHE_MAX + 1)
        ]
        with pytest.raises(DriverError):
            await driver.explore_download_ref(refs[0], None)
        result = await driver.explore_download_ref(refs[-1], None)
        assert isinstance(result, DownloadResult)


class TestExploreDownloadDefault:
    async def test_raises_driver_error_when_not_overridden(self) -> None:
        driver = _make_driver()
        with pytest.raises(DriverError):
            await driver.explore_download(["a"], None)


class TestHistogramDefault:
    async def test_raises_driver_error(self) -> None:
        driver = _make_driver()
        assert _StubDriver.SUPPORTS_HISTOGRAM is False
        with pytest.raises(DriverError, match="does not support histograms"):
            await driver.histogram("anything", 50)


class TestBuildColumnSamples:
    def test_dedupes_repeated_values(self) -> None:
        rows = [("x",), ("y",), ("x",)]
        assert build_column_samples(["VAL"], rows, 3) == {"VAL": ["x", "y"]}

    def test_skips_nulls(self) -> None:
        rows = [(None,), ("a",), (None,)]
        assert build_column_samples(["VAL"], rows, 3) == {"VAL": ["a"]}

    def test_caps_at_n_values(self) -> None:
        rows = [("a",), ("b",), ("c",), ("d",)]
        assert build_column_samples(["VAL"], rows, 2) == {"VAL": ["a", "b"]}

    def test_skips_unserialisable_types(self) -> None:
        rows = [(b"\x00",), ("ok",)]
        assert build_column_samples(["VAL"], rows, 3) == {"VAL": ["ok"]}

    def test_all_null_column_yields_empty_list(self) -> None:
        rows = [(1, None), (2, None)]
        result = build_column_samples(["ID", "VAL"], rows, 3)
        assert result == {"ID": [1, 2], "VAL": []}

    def test_no_rows_yields_empty_lists(self) -> None:
        assert build_column_samples(["ID", "VAL"], [], 3) == {"ID": [], "VAL": []}
