import asyncio
import pathlib
from collections.abc import Generator
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from grannos.dispatcher import Connection, Dispatcher, DispatchError, IdleTimer
from grannos.drivers.base import ConnectionLostError, DriverSettings
from grannos.protocol import (
    PROTOCOL_VERSION,
    DriverParam,
    CapabilitiesResult,
    ConnectResult,
    DriverHelpResult,
    ExploreDiagramResult,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    Method,
    MessageLevel,
    ParamType,
    ExecuteMessage,
    ExecuteHistogramResult,
    ExecuteReadResult,
    ExecuteWriteResult,
    ExploreDescribeResult,
    ExploreFindResult,
    ExploreListResult,
    HistogramBucket,
    HistogramResult,
    OkResult,
    ReadResult,
    SearchScope,
    WriteResult,
)


async def noop_progress(status: str, message: str) -> None:
    pass


async def connect(dispatcher: Dispatcher, params: dict[str, Any] | None = None) -> str:
    """Open a connection and return its id, narrowing the dispatch result."""
    result = await dispatcher.dispatch(
        Method.CONNECT, params or {"driver": "mock"}, noop_progress
    )
    assert isinstance(result, ConnectResult)
    return result.connection_id


def _make_mock_driver() -> AsyncMock:
    d = AsyncMock()
    d.DEFAULT_IDLE_TIMEOUT = 600
    d.FIND_PATHS = {}
    d.execute.return_value = ReadResult(columns=[], rows=[], rows_total=0)
    d.explore_list.return_value = []
    d.explore_describe.return_value = None
    return d


def _driver_class(
    mock_driver: AsyncMock, params: list[DriverParam] | None = None
) -> AsyncMock:
    cls = AsyncMock()
    cls.create = AsyncMock(return_value=mock_driver)
    if params is not None:
        cls.PARAMS = params
    return cls


@pytest.fixture
def dispatcher(tmp_path: pathlib.Path) -> Dispatcher:
    return Dispatcher(driver_settings=DriverSettings(), cache_dir=tmp_path)


@pytest.fixture
def mock_driver() -> Generator[AsyncMock, None, None]:
    """Yields a mock driver and patches get_driver to return it for the test's duration."""
    d = _make_mock_driver()
    with patch("grannos.dispatcher.get_driver", return_value=_driver_class(d)):
        yield d


@pytest.fixture
def mock_get_driver():
    """Patches get_driver and yields the mock so tests can set its return_value."""
    with patch("grannos.dispatcher.get_driver") as m:
        yield m


@pytest.fixture
def two_drivers():
    """Patches get_driver to return driver_a then driver_b on successive CONNECT calls."""
    driver_a = _make_mock_driver()
    driver_b = _make_mock_driver()
    drivers = iter([_driver_class(driver_a), _driver_class(driver_b)])
    with patch("grannos.dispatcher.get_driver", side_effect=lambda _: next(drivers)):
        yield driver_a, driver_b


@pytest.fixture
async def connected(
    dispatcher: Dispatcher, mock_driver: AsyncMock
) -> tuple[Dispatcher, str, AsyncMock]:
    result = await dispatcher.dispatch(
        Method.CONNECT, {"driver": "mock"}, noop_progress
    )
    assert isinstance(result, ConnectResult)
    return dispatcher, result.connection_id, mock_driver


class TestConnection:
    async def test_context_manager_grants_access(self) -> None:
        conn = Connection("1", AsyncMock(), max_concurrency=1, timeout=0)
        async with conn as c:
            assert c is conn

    async def test_limits_concurrency(self) -> None:
        conn = Connection("1", AsyncMock(), max_concurrency=1, timeout=0)
        order: list[str] = []
        gate = asyncio.Event()
        t1 = asyncio.create_task(self._acquire_conn_and_wait(conn, order, gate, "a"))
        t2 = asyncio.create_task(self._acquire_conn_and_wait(conn, order, gate, "b"))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(t1, t2)
        assert order == ["start:a", "end:a", "start:b", "end:b"]

    @staticmethod
    async def _acquire_conn_and_wait(
        conn: Connection, order: list[str], gate: asyncio.Event, label: str
    ) -> None:
        async with conn:
            order.append(f"start:{label}")
            await gate.wait()
            order.append(f"end:{label}")


class TestIdleTimer:
    async def test_fires_after_timeout(self) -> None:
        on_expire = AsyncMock()
        IdleTimer(0.05, on_expire)
        await asyncio.sleep(0.15)
        on_expire.assert_awaited_once_with(0.05)

    async def test_reset_restarts_countdown(self) -> None:
        on_expire = AsyncMock()
        timer = IdleTimer(0.1, on_expire)
        await asyncio.sleep(0.07)
        timer.reset()
        await asyncio.sleep(0.07)
        on_expire.assert_not_awaited()
        await asyncio.sleep(0.1)
        on_expire.assert_awaited_once()

    async def test_cancel_prevents_expiry(self) -> None:
        on_expire = AsyncMock()
        timer = IdleTimer(0.05, on_expire)
        timer.cancel()
        await asyncio.sleep(0.15)
        on_expire.assert_not_awaited()


class TestCapabilities:
    async def test_should_return_server_name(self, dispatcher: Dispatcher) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        assert result.server == "grannos"

    async def test_should_return_protocol_version(self, dispatcher: Dispatcher) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        assert result.protocol_version == PROTOCOL_VERSION

    async def test_should_always_include_sqlite(self, dispatcher: Dispatcher) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        assert "sqlite" in [t.driver for t in result.drivers]

    async def test_all_drivers_have_at_least_one_param(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        assert all(len(tech.params) > 0 for tech in result.drivers)

    async def test_all_params_have_required_fields(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        assert all(
            p.key and p.type and p.label for tech in result.drivers for p in tech.params
        )

    async def test_sqlite_supports_writes(self, dispatcher: Dispatcher) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        sqlite = next(t for t in result.drivers if t.driver == "sqlite")
        assert sqlite.supports_writes is True

    async def test_prometheus_does_not_support_writes(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        drivers_by_name = {t.driver: t for t in result.drivers}
        if "prometheus" not in drivers_by_name:
            pytest.skip("prometheus driver not installed")
        assert drivers_by_name["prometheus"].supports_writes is False

    async def test_sqlite_does_not_support_histograms(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        sqlite = next(t for t in result.drivers if t.driver == "sqlite")
        assert sqlite.supports_histogram is False

    async def test_elasticsearch_supports_histograms(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(Method.CAPABILITIES, {}, noop_progress)
        assert isinstance(result, CapabilitiesResult)
        drivers_by_name = {t.driver: t for t in result.drivers}
        if "elasticsearch" not in drivers_by_name:
            pytest.skip("elasticsearch driver not installed")
        assert drivers_by_name["elasticsearch"].supports_histogram is True


class TestDriverHelp:
    async def test_should_return_markdown_content(self, dispatcher: Dispatcher) -> None:
        result = await dispatcher.dispatch(
            Method.DRIVER_HELP, {"driver": "sqlite"}, noop_progress
        )
        assert isinstance(result, DriverHelpResult)
        assert "SQLite" in result.content

    async def test_should_raise_for_unknown_driver(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError, match="Unknown driver"):
            await dispatcher.dispatch(
                Method.DRIVER_HELP, {"driver": "no_such"}, noop_progress
            )

    async def test_should_raise_when_driver_param_missing(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError, match="Missing required param"):
            await dispatcher.dispatch(Method.DRIVER_HELP, {}, noop_progress)


class TestConnect:
    async def test_raises_when_required_param_is_missing(
        self, dispatcher: Dispatcher, mock_get_driver: AsyncMock
    ) -> None:
        mock_get_driver.return_value = _driver_class(
            _make_mock_driver(),
            [DriverParam(key="host", type=ParamType.STRING, label="Host")],
        )
        with pytest.raises(DispatchError, match="Host"):
            await dispatcher.dispatch(Method.CONNECT, {"driver": "mock"}, noop_progress)

    async def test_raises_when_required_param_is_empty_string(
        self, dispatcher: Dispatcher, mock_get_driver: AsyncMock
    ) -> None:
        mock_get_driver.return_value = _driver_class(
            _make_mock_driver(),
            [DriverParam(key="host", type=ParamType.STRING, label="Host")],
        )
        with pytest.raises(DispatchError, match="Host"):
            await dispatcher.dispatch(
                Method.CONNECT, {"driver": "mock", "host": ""}, noop_progress
            )

    async def test_succeeds_when_all_required_params_are_provided(
        self, dispatcher: Dispatcher, mock_get_driver: AsyncMock
    ) -> None:
        mock_get_driver.return_value = _driver_class(
            _make_mock_driver(),
            [DriverParam(key="host", type=ParamType.STRING, label="Host")],
        )
        result = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "host": "localhost"}, noop_progress
        )
        assert isinstance(result, ConnectResult)
        assert result.connection_id

    async def test_optional_params_may_be_absent(
        self, dispatcher: Dispatcher, mock_get_driver: AsyncMock
    ) -> None:
        mock_get_driver.return_value = _driver_class(
            _make_mock_driver(),
            [
                DriverParam(
                    key="user", type=ParamType.STRING, label="User", required=False
                )
            ],
        )
        result = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock"}, noop_progress
        )
        assert isinstance(result, ConnectResult)
        assert result.connection_id


class TestDispatch:
    async def test_should_raise_when_method_is_unknown(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError, match="Unknown method"):
            await dispatcher.dispatch("no_such", {}, noop_progress)  # type: ignore


class TestExecuteHistogram:
    async def test_should_return_buckets_with_duration(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        buckets = [
            HistogramBucket(time=0, count=3),
            HistogramBucket(time=60000, count=0),
        ]
        driver.histogram.return_value = HistogramResult(
            field="@timestamp", interval="1m", buckets=buckets
        )
        result = await disp.dispatch(
            Method.EXECUTE_HISTOGRAM,
            {"connection_id": conn_id, "query": "logs | *"},
            noop_progress,
        )
        assert result == ExecuteHistogramResult(
            field="@timestamp", interval="1m", buckets=buckets, duration_ms=ANY
        )
        driver.histogram.assert_awaited_once_with("logs | *", 50)

    async def test_passes_bucket_count_through_capped(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.histogram.return_value = HistogramResult(
            field="@timestamp", interval="", buckets=[]
        )
        await disp.dispatch(
            Method.EXECUTE_HISTOGRAM,
            {"connection_id": conn_id, "query": "logs | *", "buckets": 120},
            noop_progress,
        )
        driver.histogram.assert_awaited_with("logs | *", 120)
        await disp.dispatch(
            Method.EXECUTE_HISTOGRAM,
            {"connection_id": conn_id, "query": "logs | *", "buckets": 9999},
            noop_progress,
        )
        driver.histogram.assert_awaited_with("logs | *", 500)

    async def test_rejects_bad_bucket_count(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, _ = connected
        for bad in (0, -1, "10", True):
            with pytest.raises(DispatchError, match="buckets"):
                await disp.dispatch(
                    Method.EXECUTE_HISTOGRAM,
                    {"connection_id": conn_id, "query": "logs | *", "buckets": bad},
                    noop_progress,
                )

    async def test_requires_query(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, _ = connected
        with pytest.raises(DispatchError, match="query"):
            await disp.dispatch(
                Method.EXECUTE_HISTOGRAM, {"connection_id": conn_id}, noop_progress
            )


class TestExecute:
    async def test_should_return_columns_and_rows(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.return_value = ReadResult(
            columns=["id"], rows=[[1], [2]], rows_total=2
        )
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "SELECT 1"},
            noop_progress,
        )
        assert result == ExecuteReadResult(
            columns=["id"], rows=[[1], [2]], rows_total=2, duration_ms=ANY
        )

    async def test_should_return_rows_affected_for_dml(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.return_value = WriteResult(rows_affected=3)
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "DELETE FROM t"},
            noop_progress,
        )
        assert result == ExecuteWriteResult(rows_affected=3, duration_ms=ANY)

    async def test_duration_ms_is_non_negative_number(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, _ = connected
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "SELECT 1"},
            noop_progress,
        )
        assert isinstance(result, ExecuteReadResult)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0

    async def test_should_pass_driver_messages_through_for_reads(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        messages = [ExecuteMessage(level=MessageLevel.INFO, text="hello")]
        driver.execute.return_value = ReadResult(
            columns=["id"], rows=[[1]], rows_total=1, messages=messages
        )
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "SELECT 1"},
            noop_progress,
        )
        assert isinstance(result, ExecuteReadResult)
        assert result.messages == messages

    async def test_should_pass_driver_messages_through_for_writes(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        messages = [
            ExecuteMessage(level=MessageLevel.WARNING, text="PLS-00201", line=4, col=5)
        ]
        driver.execute.return_value = WriteResult(rows_affected=0, messages=messages)
        result = await disp.dispatch(
            Method.EXECUTE,
            {
                "connection_id": conn_id,
                "query": "CREATE PROCEDURE p AS BEGIN x(); END;",
            },
            noop_progress,
        )
        assert isinstance(result, ExecuteWriteResult)
        assert result.messages == messages

    async def test_messages_default_to_empty(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.return_value = WriteResult(rows_affected=1)
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "DELETE FROM t"},
            noop_progress,
        )
        assert isinstance(result, ExecuteWriteResult)
        assert result.messages == []

    async def test_should_raise_when_connection_id_is_unknown(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError):
            await dispatcher.dispatch(
                Method.EXECUTE,
                {"connection_id": "x", "query": "SELECT 1"},
                noop_progress,
            )

    async def test_returns_result_after_reconnect_on_connection_lost(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.side_effect = [
            ConnectionLostError(),
            ReadResult(columns=["n"], rows=[[42]], rows_total=1),
        ]
        result = await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "SELECT 1"},
            noop_progress,
        )
        assert result == ExecuteReadResult(
            columns=["n"], rows=[[42]], rows_total=1, duration_ms=ANY
        )

    async def test_reconnects_when_connection_is_lost(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.side_effect = [
            ConnectionLostError(),
            ReadResult(columns=[], rows=[], rows_total=0),
        ]
        await disp.dispatch(
            Method.EXECUTE,
            {"connection_id": conn_id, "query": "SELECT 1"},
            noop_progress,
        )
        driver.reconnect.assert_awaited_once()

    async def test_sends_reconnecting_progress_when_connection_is_lost(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.side_effect = [
            ConnectionLostError(),
            ReadResult(columns=[], rows=[], rows_total=0),
        ]
        progress = AsyncMock()
        await disp.dispatch(
            Method.EXECUTE, {"connection_id": conn_id, "query": "SELECT 1"}, progress
        )
        progress.assert_any_await("reconnecting", ANY)

    async def test_sends_executing_progress_after_successful_reconnect(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.execute.side_effect = [
            ConnectionLostError(),
            ReadResult(columns=[], rows=[], rows_total=0),
        ]
        progress = AsyncMock()
        await disp.dispatch(
            Method.EXECUTE, {"connection_id": conn_id, "query": "SELECT 1"}, progress
        )
        progress.assert_any_await("executing", ANY)


class TestSessionSet:
    async def test_forwards_values_to_driver(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        result = await disp.dispatch(
            Method.SESSION_SET,
            {"connection_id": conn_id, "query_mode": "range"},
            noop_progress,
        )
        assert result == OkResult()
        driver.set_session.assert_awaited_once_with({"query_mode": "range"})

    async def test_excludes_connection_id_from_forwarded_values(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.SESSION_SET,
            {"connection_id": conn_id, "query_mode": "range"},
            noop_progress,
        )
        values = driver.set_session.call_args[0][0]
        assert "connection_id" not in values

    async def test_raises_when_connection_id_is_unknown(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError):
            await dispatcher.dispatch(
                Method.SESSION_SET,
                {"connection_id": "x", "query_mode": "range"},
                noop_progress,
            )


class TestSessionGet:
    async def test_returns_driver_session_values(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.get_session = MagicMock(return_value={"query_mode": "range"})
        result = await disp.dispatch(
            Method.SESSION_GET, {"connection_id": conn_id}, noop_progress
        )
        assert result == {"query_mode": "range"}

    async def test_raises_when_connection_id_is_unknown(
        self, dispatcher: Dispatcher
    ) -> None:
        with pytest.raises(DispatchError):
            await dispatcher.dispatch(
                Method.SESSION_GET, {"connection_id": "x"}, noop_progress
            )


class TestDisconnect:
    async def test_should_return_ok(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, _ = connected
        result = await disp.dispatch(
            Method.DISCONNECT, {"connection_id": conn_id}, noop_progress
        )
        assert result == OkResult()

    async def test_should_succeed_when_connection_id_is_unknown(
        self, dispatcher: Dispatcher
    ) -> None:
        result = await dispatcher.dispatch(
            Method.DISCONNECT, {"connection_id": "x"}, noop_progress
        )
        assert result == OkResult()


class TestExploreList:
    async def test_should_return_items_from_driver(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_list.return_value = [
            ExploreItem(name="t", type="table", expandable=True)
        ]
        result = await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        assert result == ExploreListResult(
            items=[ExploreItem(name="t", type="table", expandable=True)]
        )

    async def test_should_cache_result_on_repeated_calls(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        driver.explore_list.assert_awaited_once()

    async def test_should_cache_separately_per_path(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        await disp.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_id, "path": ["dbo"]},
            noop_progress,
        )
        assert driver.explore_list.await_count == 2

    async def test_should_refresh_cache_when_reset_cache_is_true(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        await disp.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_id, "path": [], "reset_cache": True},
            noop_progress,
        )
        assert driver.explore_list.await_count == 2

    async def test_reset_cache_clears_all_paths(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        await disp.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_id, "path": ["dbo"]},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_id, "path": [], "reset_cache": True},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_id, "path": ["dbo"]},
            noop_progress,
        )
        assert (
            driver.explore_list.await_count == 4
        )  # 2 initial + re-fetch [] + re-fetch ["dbo"]

    async def test_should_keep_separate_caches_per_connection(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        r1 = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock"}, noop_progress
        )
        assert isinstance(r1, ConnectResult)
        r2 = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock"}, noop_progress
        )
        assert isinstance(r2, ConnectResult)
        conn_a, conn_b = r1.connection_id, r2.connection_id
        await dispatcher.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_a, "path": []}, noop_progress
        )
        await dispatcher.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_b, "path": []}, noop_progress
        )
        await dispatcher.dispatch(
            Method.EXPLORE_LIST,
            {"connection_id": conn_a, "path": [], "reset_cache": True},
            noop_progress,
        )
        await dispatcher.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_b, "path": []}, noop_progress
        )
        assert (
            mock_driver.explore_list.await_count == 3
        )  # a, b, a-reset; b still cached


class TestExploreFind:
    async def test_should_return_paths_from_driver(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_find.return_value = [["public", "users", "columns", "id"]]
        result = await disp.dispatch(
            Method.EXPLORE_FIND,
            {
                "connection_id": conn_id,
                "type": "column",
                "name": "id",
                "scope": [{"name": "users", "type": "table"}],
            },
            noop_progress,
        )
        assert result == ExploreFindResult(paths=[["public", "users", "columns", "id"]])
        driver.explore_find.assert_awaited_once_with(
            "column", "id", [SearchScope(name="users", type="table")]
        )

    async def test_should_default_to_an_unscoped_search(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_find.return_value = []
        await disp.dispatch(
            Method.EXPLORE_FIND,
            {"connection_id": conn_id, "type": "table", "name": "users"},
            noop_progress,
        )
        driver.explore_find.assert_awaited_once_with("table", "users", [])

    @pytest.mark.parametrize("missing", ["type", "name"])
    async def test_should_reject_a_request_missing_a_required_param(
        self, connected: tuple[Dispatcher, str, AsyncMock], missing: str
    ) -> None:
        disp, conn_id, _ = connected
        params = {"connection_id": conn_id, "type": "column", "name": "id"}
        del params[missing]
        with pytest.raises(DispatchError, match=missing):
            await disp.dispatch(Method.EXPLORE_FIND, params, noop_progress)

    @pytest.mark.parametrize(
        "scope", ["users", [{"name": "users"}], [{"type": "table"}], ["users"]]
    )
    async def test_should_reject_a_malformed_scope(
        self, connected: tuple[Dispatcher, str, AsyncMock], scope: object
    ) -> None:
        disp, conn_id, _ = connected
        with pytest.raises(DispatchError, match="scope"):
            await disp.dispatch(
                Method.EXPLORE_FIND,
                {
                    "connection_id": conn_id,
                    "type": "column",
                    "name": "id",
                    "scope": scope,
                },
                noop_progress,
            )

    async def test_should_reconnect_and_retry_when_connection_lost(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_find.side_effect = [ConnectionLostError(), [["users"]]]
        result = await disp.dispatch(
            Method.EXPLORE_FIND,
            {"connection_id": conn_id, "type": "table", "name": "users"},
            noop_progress,
        )
        assert result == ExploreFindResult(paths=[["users"]])
        driver.reconnect.assert_awaited_once()


class TestExploreDescribe:
    async def test_should_return_details_from_driver(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        td = EntityDescription(
            name="t",
            kind="table",
            properties=[FieldDescription(name="id", types=["INTEGER"])],
        )
        driver.explore_describe.return_value = td
        result = await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        assert result == ExploreDescribeResult(details=td)

    async def test_should_cache_result_on_repeated_calls(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.return_value = EntityDescription(
            name="t",
            kind="table",
            properties=[FieldDescription(name="id", types=["INTEGER"])],
        )
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        driver.explore_describe.assert_awaited_once()

    async def test_should_refresh_cache_when_reset_cache_is_true(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"], "reset_cache": True},
            noop_progress,
        )
        assert driver.explore_describe.await_count == 2

    async def test_reset_cache_is_shared_with_explore_list(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"], "reset_cache": True},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_LIST, {"connection_id": conn_id, "path": []}, noop_progress
        )
        assert driver.explore_list.await_count == 2

    async def test_should_not_cache_none_result(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.return_value = None
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        assert driver.explore_describe.await_count == 2

    async def test_should_return_null_details_when_driver_returns_none(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.return_value = None
        result = await disp.dispatch(
            Method.EXPLORE_DESCRIBE,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        assert result == ExploreDescribeResult(details=None)


class TestExploreDiagram:
    async def test_should_return_ascii_diagram_from_driver_describe(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.return_value = EntityDescription(
            name="t",
            kind="table",
            properties=[FieldDescription(name="id", types=["INTEGER"], pk=True)],
        )
        result = await disp.dispatch(
            Method.EXPLORE_DIAGRAM,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        assert isinstance(result, ExploreDiagramResult)
        assert "t" in result.diagram
        assert "id" in result.diagram
        assert any(r.path == ["t"] for r in result.regions)
        assert any(r.path == ["t", "columns", "id"] for r in result.regions)

    async def test_should_raise_when_path_does_not_resolve_to_a_table(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.return_value = None
        with pytest.raises(DispatchError):
            await disp.dispatch(
                Method.EXPLORE_DIAGRAM,
                {"connection_id": conn_id, "path": ["t"]},
                noop_progress,
            )

    async def test_reconnects_when_connection_is_lost(
        self, connected: tuple[Dispatcher, str, AsyncMock]
    ) -> None:
        disp, conn_id, driver = connected
        driver.explore_describe.side_effect = [
            ConnectionLostError(),
            EntityDescription(name="t", kind="table", properties=[]),
        ]
        await disp.dispatch(
            Method.EXPLORE_DIAGRAM,
            {"connection_id": conn_id, "path": ["t"]},
            noop_progress,
        )
        driver.reconnect.assert_awaited_once()


class TestConcurrency:
    async def test_should_serialise_concurrent_requests_on_same_connection(
        self, tmp_path: pathlib.Path, mock_driver: AsyncMock
    ) -> None:
        order: list[str] = []
        gate = asyncio.Event()
        mock_driver.execute.side_effect = self._slow_execute(order, gate)
        dispatcher = Dispatcher(
            driver_settings=DriverSettings(), cache_dir=tmp_path, max_concurrency=1
        )
        conn_id = await connect(dispatcher)
        t1 = asyncio.create_task(
            dispatcher.dispatch(
                Method.EXECUTE,
                {"connection_id": conn_id, "query": "SELECT 1"},
                noop_progress,
            )
        )
        t2 = asyncio.create_task(
            dispatcher.dispatch(
                Method.EXECUTE,
                {"connection_id": conn_id, "query": "SELECT 2"},
                noop_progress,
            )
        )
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(t1, t2)
        assert order == ["start", "end", "start", "end"]

    async def test_should_allow_concurrent_requests_on_different_connections(
        self, tmp_path: pathlib.Path, two_drivers: tuple[AsyncMock, AsyncMock]
    ) -> None:
        driver_a, driver_b = two_drivers
        started: list[str] = []
        gate = asyncio.Event()
        driver_a.execute.side_effect = self._slow_execute_labeled(started, gate, "a")
        driver_b.execute.side_effect = self._slow_execute_labeled(started, gate, "b")
        dispatcher = Dispatcher(
            driver_settings=DriverSettings(), cache_dir=tmp_path, max_concurrency=1
        )
        conn_a = await connect(dispatcher)
        conn_b = await connect(dispatcher)
        t1 = asyncio.create_task(
            dispatcher.dispatch(
                Method.EXECUTE, {"connection_id": conn_a, "query": "q"}, noop_progress
            )
        )
        t2 = asyncio.create_task(
            dispatcher.dispatch(
                Method.EXECUTE, {"connection_id": conn_b, "query": "q"}, noop_progress
            )
        )
        await asyncio.sleep(0)
        assert set(started) == {"a", "b"}
        gate.set()
        await asyncio.gather(t1, t2)

    @staticmethod
    def _slow_execute(order: list[str], gate: asyncio.Event):
        async def _fn(*_: object) -> ReadResult:
            order.append("start")
            await gate.wait()
            order.append("end")
            return ReadResult(columns=[], rows=[], rows_total=0)

        return _fn

    @staticmethod
    def _slow_execute_labeled(started: list[str], gate: asyncio.Event, label: str):
        async def _fn(*_: object) -> ReadResult:
            started.append(label)
            await gate.wait()
            return ReadResult(columns=[], rows=[], rows_total=0)

        return _fn


class TestIdleTimeout:
    async def test_connection_closed_after_idle_timeout(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "idle_timeout": 0.05}, noop_progress
        )
        await asyncio.sleep(0.15)
        mock_driver.disconnect.assert_awaited_once()

    async def test_timer_resets_on_request(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        r = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "idle_timeout": 0.1}, noop_progress
        )
        assert isinstance(r, ConnectResult)
        await asyncio.sleep(0.07)
        await dispatcher.dispatch(
            Method.EXECUTE,
            {"connection_id": r.connection_id, "query": "SELECT 1"},
            noop_progress,
        )
        await asyncio.sleep(0.07)
        mock_driver.disconnect.assert_not_awaited()
        await asyncio.sleep(0.1)
        mock_driver.disconnect.assert_awaited_once()

    async def test_default_timeout_does_not_fire_immediately(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        await dispatcher.dispatch(Method.CONNECT, {"driver": "mock"}, noop_progress)
        await asyncio.sleep(0.1)
        mock_driver.disconnect.assert_not_awaited()

    async def test_explicit_disconnect_cancels_timer(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        r = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "idle_timeout": 0.1}, noop_progress
        )
        assert isinstance(r, ConnectResult)
        await dispatcher.dispatch(
            Method.DISCONNECT, {"connection_id": r.connection_id}, noop_progress
        )
        mock_driver.disconnect.assert_awaited_once()
        await asyncio.sleep(0.15)
        assert mock_driver.disconnect.await_count == 1

    async def test_should_reconnect_on_execute_after_timeout_fired(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        r = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "idle_timeout": 0.05}, noop_progress
        )
        assert isinstance(r, ConnectResult)
        await asyncio.sleep(0.15)
        mock_driver.execute.side_effect = [
            ConnectionLostError(),
            ReadResult(columns=[], rows=[], rows_total=0),
        ]
        await dispatcher.dispatch(
            Method.EXECUTE,
            {"connection_id": r.connection_id, "query": "SELECT 1"},
            noop_progress,
        )
        mock_driver.reconnect.assert_awaited_once()

    async def test_no_disconnect_when_driver_default_timeout_is_zero(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        mock_driver.DEFAULT_IDLE_TIMEOUT = 0
        await dispatcher.dispatch(Method.CONNECT, {"driver": "mock"}, noop_progress)
        await asyncio.sleep(0.1)
        mock_driver.disconnect.assert_not_awaited()

    async def test_timer_does_not_fire_while_query_is_running(
        self, dispatcher: Dispatcher, mock_driver: AsyncMock
    ) -> None:
        gate = asyncio.Event()
        mock_driver.execute.side_effect = self._slow_execute(gate)
        r = await dispatcher.dispatch(
            Method.CONNECT, {"driver": "mock", "idle_timeout": 0.05}, noop_progress
        )
        assert isinstance(r, ConnectResult)
        task = asyncio.create_task(
            dispatcher.dispatch(
                Method.EXECUTE,
                {"connection_id": r.connection_id, "query": "SELECT 1"},
                noop_progress,
            )
        )
        await asyncio.sleep(0.15)
        mock_driver.disconnect.assert_not_awaited()
        gate.set()
        await task

    @staticmethod
    def _slow_execute(gate: asyncio.Event):
        async def _fn(*_: object) -> ReadResult:
            await gate.wait()
            return ReadResult(columns=[], rows=[], rows_total=0)

        return _fn
