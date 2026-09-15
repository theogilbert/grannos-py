"""Unit tests for PrometheusDriver — no live server required."""

import re
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.prometheus import (
    PrometheusDriver,
    _data_to_result,
    _escape_label_value,
    _format_duration_ms,
    _format_error,
    _format_health,
    _format_scrape_age,
    _job_selector,
    _parse_range_query,
    _parse_value,
    _resolve_time,
)
from grannos.protocol import (
    EntityDescription,
    FieldDescription,
    GenericRecordDescription,
    RawDocument,
    ReadResult,
    SpecialFloat,
)


class _FakeResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    async def json(self, content_type: str | None = None) -> object:
        return self._body


class _FakeGetCtx:
    def __init__(self, body: object) -> None:
        self._body = body

    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse(self._body)

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _driver_with_response(
    body: object, params: dict | None = None
) -> tuple[PrometheusDriver, MagicMock]:
    session = MagicMock()
    session.closed = False
    session.get = MagicMock(return_value=_FakeGetCtx(body))
    driver = PrometheusDriver(params or {}, session, DriverSettings())
    return driver, session


class TestOpen:
    def test_default_url(self) -> None:
        driver = PrometheusDriver({}, MagicMock(), DriverSettings())
        assert driver._url == "http://localhost:9090"

    def test_custom_url_strips_trailing_slash(self) -> None:
        driver = PrometheusDriver(
            {"url": "http://prom.internal:9090/"}, MagicMock(), DriverSettings()
        )
        assert driver._url == "http://prom.internal:9090"

    async def test_no_auth_without_credentials(self) -> None:
        session = PrometheusDriver._open({})
        try:
            assert "Authorization" not in session.headers
        finally:
            await session.close()

    async def test_basic_auth_with_credentials(self) -> None:
        session = PrometheusDriver._open({"username": "user", "password": "pass"})
        try:
            assert session.headers["Authorization"] == "Basic dXNlcjpwYXNz"
        finally:
            await session.close()


class TestParseRangeQuery:
    def test_valid_range_query(self) -> None:
        start, end, step, promql = _parse_range_query("-1h,now,15s | up")
        assert (start, end, step, promql) == ("-1h", "now", "15s", "up")

    def test_missing_separator_raises(self) -> None:
        with pytest.raises(DriverError, match="format"):
            _parse_range_query("up")

    def test_wrong_field_count_raises(self) -> None:
        with pytest.raises(DriverError, match="3 comma-separated"):
            _parse_range_query("-1h,now | up")

    def test_strips_whitespace(self) -> None:
        start, end, step, promql = _parse_range_query(" -1h , now , 15s  |  up ")
        assert (start, end, step, promql) == ("-1h", "now", "15s", "up")


class TestResolveTime:
    def test_now(self) -> None:
        assert _resolve_time("now", 1000.0) == "1000.0"

    def test_relative_duration_hours(self) -> None:
        assert _resolve_time("-1h", 1000.0) == str(1000.0 - 3600)

    def test_relative_duration_minutes(self) -> None:
        assert _resolve_time("-30m", 1000.0) == str(1000.0 - 30 * 60)

    def test_passthrough_rfc3339(self) -> None:
        assert _resolve_time("2024-01-01T00:00:00Z", 1000.0) == "2024-01-01T00:00:00Z"

    def test_passthrough_unix_timestamp(self) -> None:
        assert _resolve_time("1700000000", 1000.0) == "1700000000"


class TestDataToResult:
    def test_vector_result(self) -> None:
        data = {
            "resultType": "vector",
            "result": [
                {"metric": {"__name__": "up", "job": "a"}, "value": [1700000000, "1"]}
            ],
        }
        result = _data_to_result(data)
        assert result.columns == ["__name__", "job", "timestamp", "value"]
        assert result.rows == [["up", "a", result.rows[0][2], 1.0]]
        assert result.rows_total == 1

    def test_matrix_result_emits_one_row_per_point(self) -> None:
        data = {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"job": "a"},
                    "values": [[1700000000, "1"], [1700000015, "2"]],
                }
            ],
        }
        result = _data_to_result(data)
        assert result.columns == ["job", "timestamp", "value"]
        assert len(result.rows) == 2
        assert [row[2] for row in result.rows] == [1.0, 2.0]
        assert result.rows_total == 2

    def test_scalar_result(self) -> None:
        data = {"resultType": "scalar", "result": [1700000000, "42"]}
        result = _data_to_result(data)
        assert result.columns == ["timestamp", "value"]
        assert result.rows[0][1] == "42"
        assert result.rows_total == 1

    def test_empty_vector_has_no_label_columns(self) -> None:
        data = {"resultType": "vector", "result": []}
        result = _data_to_result(data)
        assert result.columns == ["timestamp", "value"]
        assert result.rows == []

    def test_unsupported_result_type_raises(self) -> None:
        with pytest.raises(DriverError, match="Unsupported"):
            _data_to_result({"resultType": "bogus", "result": []})


class TestParseValue:
    def test_parses_numeric_string(self) -> None:
        assert _parse_value("1.5") == 1.5

    def test_nan_becomes_special_float(self) -> None:
        assert _parse_value("NaN") == SpecialFloat(text="NaN")

    def test_positive_infinity_becomes_special_float(self) -> None:
        assert _parse_value("+Inf") == SpecialFloat(text="+Inf")

    def test_negative_infinity_becomes_special_float(self) -> None:
        assert _parse_value("-Inf") == SpecialFloat(text="-Inf")

    def test_non_numeric_passthrough(self) -> None:
        assert _parse_value("some string") == "some string"


class TestFormatError:
    def test_dict_with_error_and_type(self) -> None:
        msg = _format_error({"error": "bad query", "errorType": "bad_data"})
        assert "bad_data" in msg
        assert "bad query" in msg

    def test_dict_without_error_type(self) -> None:
        msg = _format_error({"error": "bad query"})
        assert "bad query" in msg

    def test_non_dict_body(self) -> None:
        assert "boom" in _format_error("boom")


class TestFormatHealth:
    def test_up_is_checkmark(self) -> None:
        assert _format_health("up") == "✓"

    def test_down_is_cross(self) -> None:
        assert _format_health("down") == "✗"

    def test_unknown_is_question_mark(self) -> None:
        assert _format_health("unknown") == "?"

    def test_other_value_passthrough(self) -> None:
        assert _format_health("bogus") == "bogus"


class TestFormatScrapeAge:
    def test_seconds_ago(self) -> None:
        ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        assert re.fullmatch(r"[0-9]+s ago", _format_scrape_age(ts))

    def test_minutes_ago(self) -> None:
        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        assert re.fullmatch(r"[0-9]+m ago", _format_scrape_age(ts))

    def test_hours_ago(self) -> None:
        ts = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        assert re.fullmatch(r"[0-9]+h ago", _format_scrape_age(ts))

    def test_handles_z_suffix_and_nanosecond_precision(self) -> None:
        assert re.fullmatch(
            r"[0-9]+h ago", _format_scrape_age("2020-01-01T00:00:00.123456789Z")
        )

    def test_no_decimals(self) -> None:
        ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        assert "." not in _format_scrape_age(ts)

    def test_invalid_value_passthrough(self) -> None:
        assert _format_scrape_age("not-a-time") == "not-a-time"


class TestFormatDurationMs:
    def test_converts_seconds_to_whole_milliseconds(self) -> None:
        assert _format_duration_ms(0.012345) == "12ms"

    def test_rounds_to_no_decimals(self) -> None:
        assert _format_duration_ms(0.0126) == "13ms"

    def test_non_numeric_passthrough(self) -> None:
        assert _format_duration_ms("bogus") == "bogus"


class TestJobSelector:
    def test_plain_job_name(self) -> None:
        assert _job_selector("prometheus") == '{job="prometheus"}'

    def test_escapes_quotes_and_backslashes(self) -> None:
        assert _escape_label_value('a"b\\c') == 'a\\"b\\\\c'


class TestExecute:
    async def test_instant_mode_queries_promql_endpoint(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": {"resultType": "vector", "result": []}},
        )
        result = await driver.execute("up", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["timestamp", "value"]
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/query"
        assert kwargs["params"] == {"query": "up"}

    async def test_range_mode_queries_range_endpoint(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": {"resultType": "matrix", "result": []}},
        )
        await driver.set_session({"query_mode": "range"})
        await driver.execute("-1h,now,15s | up", [])
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/query_range"
        assert kwargs["params"]["query"] == "up"
        assert kwargs["params"]["step"] == "15s"

    async def test_unknown_query_mode_rejected_by_set_session(self) -> None:
        driver, _ = _driver_with_response({})
        with pytest.raises(DriverError, match="Unknown query_mode"):
            await driver.set_session({"query_mode": "bogus"})

    async def test_api_error_response_raises_driver_error(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "error", "errorType": "bad_data", "error": "parse error"}
        )
        with pytest.raises(DriverError, match="parse error"):
            await driver.execute("up", [])


class TestSessionParams:
    def test_query_mode_not_in_connect_params(self) -> None:
        assert "query_mode" not in {p.key for p in PrometheusDriver.PARAMS}

    def test_query_mode_declared_as_session_param(self) -> None:
        assert "query_mode" in {p.key for p in PrometheusDriver.SESSION_PARAMS}

    def test_defaults_to_instant(self) -> None:
        driver, _ = _driver_with_response({})
        assert driver.get_session() == {"query_mode": "instant"}

    async def test_set_session_updates_get_session(self) -> None:
        driver, _ = _driver_with_response({})
        await driver.set_session({"query_mode": "range"})
        assert driver.get_session() == {"query_mode": "range"}

    async def test_connect_params_do_not_seed_session(self) -> None:
        driver, _ = _driver_with_response({}, {"query_mode": "range"})
        assert driver.get_session() == {"query_mode": "instant"}


class TestConnectionErrors:
    async def test_connection_error_before_ever_connected_raises_driver_error(
        self,
    ) -> None:
        session = MagicMock()
        session.closed = False
        session.get = MagicMock(side_effect=aiohttp.ClientConnectionError("boom"))
        driver = PrometheusDriver({}, session, DriverSettings())
        with pytest.raises(DriverError):
            await driver.execute("up", [])

    async def test_connection_error_after_ever_connected_raises_connection_lost(
        self,
    ) -> None:
        session = MagicMock()
        session.closed = False
        session.get = MagicMock(side_effect=aiohttp.ClientConnectionError("boom"))
        driver = PrometheusDriver({}, session, DriverSettings())
        driver._ever_connected = True
        with pytest.raises(ConnectionLostError):
            await driver.execute("up", [])

    async def test_closed_session_raises_connection_lost(self) -> None:
        session = MagicMock()
        session.closed = True
        driver = PrometheusDriver({}, session, DriverSettings())
        with pytest.raises(ConnectionLostError):
            await driver.execute("up", [])
        session.get.assert_not_called()


class TestExploreList:
    async def test_root_lists_top_level_sections(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        items = await driver.explore_list([])
        assert [i.name for i in items] == [
            "metrics",
            "jobs",
            "configuration",
            "runtime",
        ]
        by_name = {i.name: i for i in items}
        assert by_name["metrics"].type == "group"
        assert by_name["metrics"].expandable
        assert by_name["configuration"].type == "configuration"
        assert not by_name["configuration"].expandable
        assert by_name["runtime"].type == "settings"
        assert not by_name["runtime"].expandable
        assert by_name["jobs"].type == "group"
        assert by_name["jobs"].expandable

    async def test_metrics_group_lists_metric_names(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "success", "data": ["up", "http_requests_total"]}
        )
        items = await driver.explore_list(["metrics"])
        assert sorted(i.name for i in items) == ["http_requests_total", "up"]
        assert all(i.type == "metric" and i.expandable for i in items)

    async def test_metric_lists_labels_excluding_name(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "success", "data": ["__name__", "job", "instance"]}
        )
        items = await driver.explore_list(["metrics", "up"])
        assert sorted(i.name for i in items) == ["instance", "job"]
        assert all(i.type == "label" and i.expandable for i in items)

    async def test_label_path_lists_its_values_for_the_metric(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": ["node", "prometheus"]}
        )
        items = await driver.explore_list(["metrics", "up", "job"])
        assert [i.name for i in items] == ["node", "prometheus"]
        assert all(i.type == "label_value" and not i.expandable for i in items)
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/label/job/values"
        assert kwargs["params"] == {"match[]": "up", "limit": "1000"}

    async def test_label_path_url_encodes_the_label(self) -> None:
        driver, session = _driver_with_response({"status": "success", "data": []})
        await driver.explore_list(["jobs", "prometheus", "up", "a/b"])
        args, _ = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/label/a%2Fb/values"

    async def test_label_value_path_returns_empty(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        assert await driver.explore_list(["metrics", "up", "job", "prometheus"]) == []

    async def test_jobs_lists_scrape_jobs_as_expandable(self) -> None:
        driver, _ = _driver_with_response(
            {
                "status": "success",
                "data": {
                    "activeTargets": [
                        {
                            "labels": {
                                "job": "prometheus",
                                "instance": "localhost:9090",
                            },
                            "health": "up",
                        },
                        {
                            "labels": {"job": "node", "instance": "host1:9100"},
                            "health": "down",
                        },
                    ]
                },
            }
        )
        items = await driver.explore_list(["jobs"])
        assert sorted(i.name for i in items) == ["node", "prometheus"]
        assert all(i.type == "job" and i.expandable for i in items)

    async def test_job_path_lists_metric_names_scoped_to_job(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": ["up", "http_requests_total"]}
        )
        items = await driver.explore_list(["jobs", "prometheus"])
        assert sorted(i.name for i in items) == ["http_requests_total", "up"]
        assert all(i.type == "metric" and i.expandable for i in items)
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/label/__name__/values"
        assert kwargs["params"] == {"match[]": '{job="prometheus"}'}

    async def test_job_metric_path_lists_labels_same_as_top_level_metric(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": ["__name__", "job", "instance"]}
        )
        items = await driver.explore_list(["jobs", "prometheus", "up"])
        assert sorted(i.name for i in items) == ["instance", "job"]
        assert all(i.type == "label" and i.expandable for i in items)
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/labels"
        assert kwargs["params"] == {"match[]": "up"}


class TestExplorePreview:
    async def test_metric_preview_runs_metric_name_as_query(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": {"resultType": "vector", "result": []}}
        )
        result = await driver.explore_preview(["metrics", "up"])
        assert isinstance(result, ReadResult)
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/query"
        assert kwargs["params"] == {"query": "up"}

    async def test_job_metric_preview_runs_metric_name_as_query(self) -> None:
        driver, session = _driver_with_response(
            {"status": "success", "data": {"resultType": "vector", "result": []}}
        )
        result = await driver.explore_preview(["jobs", "prometheus", "up"])
        assert isinstance(result, ReadResult)
        args, kwargs = session.get.call_args
        assert args[0] == "http://localhost:9090/api/v1/query"
        assert kwargs["params"] == {"query": "up"}

    async def test_job_path_returns_none(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        assert await driver.explore_preview(["jobs", "prometheus"]) is None

    async def test_root_returns_none(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        assert await driver.explore_preview([]) is None


class TestExploreDescribe:
    async def test_returns_none_for_unknown_path(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        assert await driver.explore_describe([]) is None

    async def test_metric_metadata_failure_degrades_gracefully(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": ["job"]})
        with patch.object(
            driver,
            "_get",
            AsyncMock(side_effect=[["job"], DriverError("no metadata"), []]),
        ):
            result = await driver.explore_describe(["metrics", "up"])
        assert isinstance(result, EntityDescription)
        assert result.kind == "metric"
        assert result.comment is None

    async def test_job_metric_path_describes_same_as_top_level_metric(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": ["job"]})
        with patch.object(
            driver,
            "_get",
            AsyncMock(side_effect=[["job"], DriverError("no metadata"), []]),
        ):
            result = await driver.explore_describe(["jobs", "prometheus", "up"])
        assert isinstance(result, EntityDescription)
        assert result.name == "up"
        assert result.kind == "metric"

    async def test_describe_single_label_returns_field_description(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "success", "data": ["a", "b", "c", "d"]}
        )
        result = await driver.explore_describe(["metrics", "up", "job"])
        assert isinstance(result, FieldDescription)
        assert result.name == "job"
        assert result.types == ["label"]
        assert result.sample == ["a", "b", "c"]

    async def test_describe_job_metric_label_matches_top_level_label(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "success", "data": ["a", "b", "c", "d"]}
        )
        result = await driver.explore_describe(["jobs", "prometheus", "up", "job"])
        assert isinstance(result, FieldDescription)
        assert result.name == "job"
        assert result.types == ["label"]
        assert result.sample == ["a", "b", "c"]

    async def test_describe_single_label_returns_empty_sample_on_timeout(
        self,
    ) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        with patch.object(
            driver, "_fetch_label_values_sample", AsyncMock(side_effect=TimeoutError)
        ):
            result = await driver.explore_describe(["metrics", "up", "job"])
        assert isinstance(result, FieldDescription)
        assert result.sample == []

    async def test_configuration_returns_raw_yaml_document(self) -> None:
        driver, _ = _driver_with_response(
            {
                "status": "success",
                "data": {"yaml": "global:\n  scrape_interval: 15s\n"},
            }
        )
        result = await driver.explore_describe(["configuration"])
        assert isinstance(result, RawDocument)
        assert result.filetype == "yaml"
        assert result.content == "global:\n  scrape_interval: 15s\n"

    async def test_runtime_merges_flags_runtime_and_build_info(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        with patch.object(
            driver,
            "_get",
            AsyncMock(
                side_effect=[
                    {"storage.tsdb.retention.time": "15d"},
                    {"storageRetention": "15d", "startTime": "2024-01-01T00:00:00Z"},
                    {"version": "2.53.0"},
                ]
            ),
        ):
            result = await driver.explore_describe(["runtime"])
        assert isinstance(result, GenericRecordDescription)
        assert result.kind == "prometheus.runtime"
        assert result.name == "runtime"
        labels = {f.label: f.value for f in result.fields}
        assert labels["Flag: storage.tsdb.retention.time"] == "15d"
        assert labels["Runtime: storageRetention"] == "15d"
        assert labels["Build: version"] == "2.53.0"

    async def test_describe_job_returns_one_record_per_target(self) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        targets_data = {
            "activeTargets": [
                {
                    "labels": {
                        "job": "prometheus",
                        "instance": "localhost:9090",
                    },
                    "health": "up",
                    "scrapeInterval": "15s",
                    "scrapeTimeout": "10s",
                    "lastScrape": "2024-01-01T00:00:00Z",
                    "lastScrapeDuration": 0.012,
                },
                {
                    "labels": {
                        "job": "prometheus",
                        "instance": "otherhost:9090",
                    },
                    "health": "down",
                },
                {
                    "labels": {"job": "node", "instance": "host1:9100"},
                    "health": "up",
                },
            ]
        }
        metadata = [
            {"target": {"instance": "localhost:9090"}, "metric": "up"},
            {"target": {"instance": "localhost:9090"}, "metric": "go_goroutines"},
        ]
        scrape_samples = {
            "resultType": "vector",
            "result": [
                {"metric": {"instance": "localhost:9090"}, "value": [0, "5"]},
            ],
        }
        with patch.object(
            driver,
            "_get",
            AsyncMock(side_effect=[targets_data, metadata, scrape_samples]),
        ) as get_mock:
            result = await driver.explore_describe(["jobs", "prometheus"])
        # Regression: passing metric="" or limit="0" makes Prometheus match
        # nothing (an empty metric name, or a hard cap of zero results) —
        # only match_target should be sent.
        metadata_call = get_mock.call_args_list[1]
        assert metadata_call.args[0] == "/api/v1/targets/metadata"
        assert metadata_call.args[1] == {"match_target": '{job="prometheus"}'}
        assert isinstance(result, list)
        records = [r for r in result if isinstance(r, GenericRecordDescription)]
        assert len(records) == len(result)
        assert all(r.kind == "prometheus.target" for r in records)
        assert sorted(r.name for r in records) == ["localhost:9090", "otherhost:9090"]

        localhost = next(r for r in records if r.name == "localhost:9090")
        labels = {f.label: f.value for f in localhost.fields}
        assert labels["Status"] == "✓"
        assert labels["Scraped metrics (series)"] == "2 (5)"
        assert not any("Label: " in label for label in labels)
        assert not any("Scrape Pool" in label for label in labels)
        assert [f.label for f in localhost.fields] == [
            "Interval",
            "Timeout",
            "Last Scrape",
            "Status",
            "Last Scrape Duration",
            "Scraped metrics (series)",
        ]

        otherhost = next(r for r in records if r.name == "otherhost:9090")
        otherhost_labels = {f.label: f.value for f in otherhost.fields}
        assert otherhost_labels["Status"] == "✗"
        assert otherhost_labels["Scraped metrics (series)"] == "0 (0)"

    async def test_describe_job_drops_last_error_when_empty_for_all_targets(
        self,
    ) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        targets_data = {
            "activeTargets": [
                {
                    "labels": {"job": "prometheus", "instance": "a:9090"},
                    "health": "up",
                    "lastError": "",
                },
                {
                    "labels": {"job": "prometheus", "instance": "b:9090"},
                    "health": "up",
                    "lastError": "",
                },
            ]
        }
        with patch.object(
            driver, "_get", AsyncMock(side_effect=[targets_data, [], {}])
        ):
            result = await driver.explore_describe(["jobs", "prometheus"])
        assert isinstance(result, list)
        for rec in result:
            assert isinstance(rec, GenericRecordDescription)
            assert not any(f.label == "Last Error" for f in rec.fields)

    async def test_describe_job_keeps_last_error_when_any_target_has_one(
        self,
    ) -> None:
        driver, _ = _driver_with_response({"status": "success", "data": []})
        targets_data = {
            "activeTargets": [
                {
                    "labels": {"job": "prometheus", "instance": "a:9090"},
                    "health": "up",
                    "lastError": "",
                },
                {
                    "labels": {"job": "prometheus", "instance": "b:9090"},
                    "health": "down",
                    "lastError": "connection refused",
                },
            ]
        }
        with patch.object(
            driver, "_get", AsyncMock(side_effect=[targets_data, [], {}])
        ):
            result = await driver.explore_describe(["jobs", "prometheus"])
        assert isinstance(result, list)
        by_name = {}
        for rec in result:
            assert isinstance(rec, GenericRecordDescription)
            by_name[rec.name] = {f.label: f.value for f in rec.fields}
        assert by_name["a:9090"]["Last Error"] == ""
        assert by_name["b:9090"]["Last Error"] == "connection refused"

    async def test_describe_unknown_job_returns_none(self) -> None:
        driver, _ = _driver_with_response(
            {"status": "success", "data": {"activeTargets": []}}
        )
        result = await driver.explore_describe(["jobs", "missing"])
        assert result is None
