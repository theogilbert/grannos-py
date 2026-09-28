"""Unit tests for Neo4jDriver — no live database required."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import neo4j.exceptions
import pytest

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.neo4j import (
    Neo4jDriver,
    _maybe_raise_connection_lost,
    _plan_keyword,
    _plan_to_result,
    _serialize,
)
from grannos.protocol import LobPlaceholder, ReadResult


def _make_driver(summary: MagicMock) -> Neo4jDriver:
    result = MagicMock()
    result.keys = MagicMock(return_value=[])
    result.consume = AsyncMock(return_value=summary)

    session = MagicMock()
    session.run = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _session(**_):
        yield session

    driver_mock = MagicMock()
    driver_mock.session = _session
    return Neo4jDriver({}, driver_mock, DriverSettings())


def _make_erroring_driver(exc: Exception) -> Neo4jDriver:
    def _session(**_):
        raise exc

    driver_mock = MagicMock()
    driver_mock.session = _session
    return Neo4jDriver({}, driver_mock, DriverSettings())


def _make_summary(plan: dict | None = None, profile: dict | None = None) -> MagicMock:
    summary = MagicMock()
    summary.plan = plan
    summary.profile = profile
    return summary


_EXPLAIN_PLAN = {
    "operatorType": "ProduceResults",
    "args": {"EstimatedRows": 100.0},
    "identifiers": ["n"],
    "children": [
        {
            "operatorType": "NodeByLabelScan",
            "args": {"EstimatedRows": 100.0},
            "identifiers": ["n"],
            "children": [],
        }
    ],
}

_PROFILE_PLAN = {
    "operatorType": "ProduceResults",
    "args": {},
    "identifiers": ["n"],
    "rows": 5,
    "dbHits": 0,
    "children": [
        {
            "operatorType": "NodeByLabelScan",
            "args": {},
            "identifiers": ["n"],
            "rows": 5,
            "dbHits": 25,
            "children": [],
        }
    ],
}


class TestPlanKeyword:
    def test_explain_uppercase(self) -> None:
        assert _plan_keyword("EXPLAIN MATCH (n) RETURN n") == "explain"

    def test_profile_uppercase(self) -> None:
        assert _plan_keyword("PROFILE MATCH (n) RETURN n") == "profile"

    def test_explain_lowercase(self) -> None:
        assert _plan_keyword("explain match (n) return n") == "explain"

    def test_profile_lowercase(self) -> None:
        assert _plan_keyword("profile match (n) return n") == "profile"

    def test_leading_comment_skipped(self) -> None:
        assert _plan_keyword("// find all\nEXPLAIN MATCH (n) RETURN n") == "explain"

    def test_leading_block_comment_skipped(self) -> None:
        assert _plan_keyword("/* find\nall */ PROFILE MATCH (n) RETURN n") == "profile"

    def test_regular_query_returns_none(self) -> None:
        assert _plan_keyword("MATCH (n) RETURN n") is None

    def test_empty_query_returns_none(self) -> None:
        assert _plan_keyword("") is None


class TestPlanToResult:
    def test_explain_columns(self) -> None:
        result = _plan_to_result(_EXPLAIN_PLAN, is_profile=False)
        assert result.columns == ["operator", "estimated_rows", "identifiers"]

    def test_profile_columns(self) -> None:
        result = _plan_to_result(_PROFILE_PLAN, is_profile=True)
        assert result.columns == ["operator", "rows", "db_hits", "identifiers"]

    def test_explain_flattens_tree_with_indentation(self) -> None:
        result = _plan_to_result(_EXPLAIN_PLAN, is_profile=False)
        assert result.rows[0][0] == "ProduceResults"
        assert result.rows[1][0] == "  NodeByLabelScan"

    def test_explain_converts_whole_float_to_int(self) -> None:
        result = _plan_to_result(_EXPLAIN_PLAN, is_profile=False)
        assert result.rows[0][1] == 100

    def test_profile_includes_db_hits(self) -> None:
        result = _plan_to_result(_PROFILE_PLAN, is_profile=True)
        assert result.rows[1] == ["  NodeByLabelScan", 5, 25, "n"]

    def test_rows_total_matches_row_count(self) -> None:
        result = _plan_to_result(_EXPLAIN_PLAN, is_profile=False)
        assert result.rows_total == len(result.rows) == 2


class TestExecuteExplain:
    def test_explain_returns_read_result(self) -> None:
        driver = _make_driver(_make_summary(plan=_EXPLAIN_PLAN))
        result = asyncio.run(driver.execute("EXPLAIN MATCH (n) RETURN n", []))
        assert isinstance(result, ReadResult)

    def test_explain_columns(self) -> None:
        driver = _make_driver(_make_summary(plan=_EXPLAIN_PLAN))
        result = asyncio.run(driver.execute("EXPLAIN MATCH (n) RETURN n", []))
        assert isinstance(result, ReadResult)
        assert result.columns == ["operator", "estimated_rows", "identifiers"]

    def test_explain_missing_plan_returns_write_result(self) -> None:
        from grannos.protocol import WriteResult

        driver = _make_driver(_make_summary(plan=None))
        result = asyncio.run(driver.execute("EXPLAIN MATCH (n) RETURN n", []))
        assert isinstance(result, WriteResult)


class TestExecuteProfile:
    def test_profile_returns_read_result(self) -> None:
        driver = _make_driver(_make_summary(profile=_PROFILE_PLAN))
        result = asyncio.run(driver.execute("PROFILE MATCH (n) RETURN n", []))
        assert isinstance(result, ReadResult)

    def test_profile_columns(self) -> None:
        driver = _make_driver(_make_summary(profile=_PROFILE_PLAN))
        result = asyncio.run(driver.execute("PROFILE MATCH (n) RETURN n", []))
        assert isinstance(result, ReadResult)
        assert result.columns == ["operator", "rows", "db_hits", "identifiers"]


class TestMaybeRaiseConnectionLost:
    def test_service_unavailable_raises(self) -> None:
        with pytest.raises(ConnectionLostError):
            _maybe_raise_connection_lost(neo4j.exceptions.ServiceUnavailable("down"))

    def test_session_expired_raises(self) -> None:
        with pytest.raises(ConnectionLostError):
            _maybe_raise_connection_lost(neo4j.exceptions.SessionExpired("expired"))

    def test_driver_closed_raises(self) -> None:
        with pytest.raises(ConnectionLostError):
            _maybe_raise_connection_lost(neo4j.exceptions.DriverError("Driver closed"))

    def test_other_driver_error_does_not_raise(self) -> None:
        _maybe_raise_connection_lost(neo4j.exceptions.CypherSyntaxError("bad syntax"))

    def test_other_error_does_not_raise(self) -> None:
        _maybe_raise_connection_lost(ValueError("unrelated"))


class TestExecuteErrorPropagation:
    def test_driver_closed_raises_connection_lost(self) -> None:
        driver = _make_erroring_driver(neo4j.exceptions.DriverError("Driver closed"))
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("MATCH (n) RETURN n", []))

    def test_service_unavailable_raises_connection_lost(self) -> None:
        driver = _make_erroring_driver(neo4j.exceptions.ServiceUnavailable("down"))
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.execute("MATCH (n) RETURN n", []))

    def test_other_error_raises_driver_error(self) -> None:
        driver = _make_erroring_driver(neo4j.exceptions.CypherSyntaxError("bad"))
        with pytest.raises(DriverError):
            asyncio.run(driver.execute("MATCH (n) RETURN n", []))

    def test_explore_list_driver_closed_raises_connection_lost(self) -> None:
        driver = _make_erroring_driver(neo4j.exceptions.DriverError("Driver closed"))
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_list(["indexes"]))

    def test_explore_describe_driver_closed_raises_connection_lost(self) -> None:
        driver = _make_erroring_driver(neo4j.exceptions.DriverError("Driver closed"))
        with pytest.raises(ConnectionLostError):
            asyncio.run(driver.explore_describe(["indexes"]))


def _null_register_lob(value: object, text: str) -> LobPlaceholder:
    return LobPlaceholder(text=text)


class TestSerialize:
    def test_passes_through_plain_values(self) -> None:
        assert _serialize(_null_register_lob, "hello") == "hello"
        assert _serialize(_null_register_lob, 42) == 42
        assert _serialize(_null_register_lob, None) is None

    def test_renders_byte_array_as_byte_count(self) -> None:
        assert _serialize(
            _null_register_lob, bytearray(b"\x01\x02\x03")
        ) == LobPlaceholder(text="ByteArray (3 bytes)")

    def test_renders_byte_array_nested_in_list(self) -> None:
        result = _serialize(_null_register_lob, [bytearray(b"\x00\x01")])
        assert result == [LobPlaceholder(text="ByteArray (2 bytes)")]
