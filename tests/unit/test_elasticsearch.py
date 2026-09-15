"""Unit tests for ElasticsearchDriver — no live server required."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import elasticsearch
import pytest

from grannos.drivers.base import DriverError, DriverSettings
from grannos.drivers.elasticsearch import (
    ElasticsearchDriver,
    _duration,
    _es_datetime,
    _fill_gaps,
    _resolve_time,
    _split_commands,
)
from grannos.protocol import HistogramBucket, HistogramResult, ReadResult


async def _hosts(params: dict) -> list[str]:
    with patch("elasticsearch.AsyncElasticsearch") as mock_cls:
        await ElasticsearchDriver.create(params, DriverSettings())
        return mock_cls.call_args.kwargs["hosts"]


def _driver_with_response(response: object) -> ElasticsearchDriver:
    client = MagicMock()
    client.perform_request = AsyncMock(return_value=SimpleNamespace(body=response))
    return ElasticsearchDriver({"query_mode": "dev_tools"}, client, DriverSettings())


class TestOpen:
    async def test_default_protocol_is_https(self) -> None:
        assert await _hosts({"host": "myhost", "port": 9200}) == ["https://myhost:9200"]

    async def test_http_protocol(self) -> None:
        assert await _hosts({"host": "myhost", "port": 9200, "protocol": "http"}) == [
            "http://myhost:9200"
        ]

    async def test_https_protocol_explicit(self) -> None:
        assert await _hosts({"host": "myhost", "port": 9200, "protocol": "https"}) == [
            "https://myhost:9200"
        ]


class TestParseBody:
    def test_empty_body(self) -> None:
        assert ElasticsearchDriver._parse_body("") == (None, None)

    def test_single_json_object(self) -> None:
        body, headers = ElasticsearchDriver._parse_body('{"query": {"match_all": {}}}')
        assert body == {"query": {"match_all": {}}}
        assert headers is not None
        assert headers["Content-Type"] == "application/json"

    def test_multiline_single_json_object(self) -> None:
        body, headers = ElasticsearchDriver._parse_body(
            '{\n  "query": {\n    "match_all": {}\n  }\n}'
        )
        assert body == {"query": {"match_all": {}}}
        assert headers is not None
        assert headers["Content-Type"] == "application/json"

    def test_multiple_json_objects_returns_ndjson(self) -> None:
        body, headers = ElasticsearchDriver._parse_body(
            '{"index": {"_id": "1"}}\n{"name": "Widget"}'
        )
        assert body == b'{"index": {"_id": "1"}}\n{"name": "Widget"}\n'
        assert headers is not None
        assert headers["Content-Type"] == "application/x-ndjson"

    def test_msearch_with_multiline_query_doc(self) -> None:
        payload = '{}\n{\n  "query": {"match_all": {}},\n  "size": 5\n}'
        body, headers = ElasticsearchDriver._parse_body(payload)
        assert body == b'{}\n{"query": {"match_all": {}}, "size": 5}\n'
        assert headers is not None
        assert headers["Content-Type"] == "application/x-ndjson"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(DriverError, match="Invalid request body"):
            ElasticsearchDriver._parse_body("not valid json")


class TestExecuteEsql:
    async def test_returns_columns_and_rows(self) -> None:
        client = MagicMock()
        client.esql.query = AsyncMock(
            return_value={
                "columns": [
                    {"name": "status", "type": "keyword"},
                    {"name": "total", "type": "long"},
                ],
                "values": [["open", 50], ["closed", 30]],
            }
        )
        driver = ElasticsearchDriver({"query_mode": "esql"}, client, DriverSettings())
        result = await driver.execute('FROM orders | WHERE status == "open"', [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["status", "total"]
        assert result.rows == [["open", 50], ["closed", 30]]
        assert result.rows_total == 2
        client.esql.query.assert_awaited_once_with(
            query='FROM orders | WHERE status == "open"', format="json"
        )

    async def test_empty_result(self) -> None:
        client = MagicMock()
        client.esql.query = AsyncMock(
            return_value={
                "columns": [{"name": "status", "type": "keyword"}],
                "values": [],
            }
        )
        driver = ElasticsearchDriver({"query_mode": "esql"}, client, DriverSettings())
        result = await driver.execute("FROM orders | LIMIT 0", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["status"]
        assert result.rows == []
        assert result.rows_total == 0


class TestExecuteDevToolsErrors:
    async def test_raises_on_error_response(self) -> None:
        driver = _driver_with_response(
            {
                "error": {
                    "type": "security_exception",
                    "reason": "missing credentials",
                },
                "status": 401,
            }
        )
        with pytest.raises(DriverError, match="security_exception"):
            await driver.execute("PUT /products\n{}", [])

    async def test_raises_on_string_error(self) -> None:
        driver = _driver_with_response({"error": "index_not_found", "status": 404})
        with pytest.raises(DriverError, match="index_not_found"):
            await driver.execute("GET /missing/_search", [])


def _lucene_driver() -> tuple[ElasticsearchDriver, MagicMock]:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={"hits": {"total": {"value": 0}, "hits": []}}
    )
    return ElasticsearchDriver({}, client, DriverSettings()), client


def _esql_driver() -> ElasticsearchDriver:
    return ElasticsearchDriver({"query_mode": "esql"}, MagicMock(), DriverSettings())


class TestResolveTime:
    def test_now(self) -> None:
        before = datetime.now(UTC)
        assert before <= _resolve_time("now") <= datetime.now(UTC)

    def test_negative_offset(self) -> None:
        delta = datetime.now(UTC) - _resolve_time("-1h")
        assert timedelta(minutes=59) < delta < timedelta(minutes=61)

    def test_now_prefixed_offset(self) -> None:
        delta = datetime.now(UTC) - _resolve_time("now-30m")
        assert timedelta(minutes=29) < delta < timedelta(minutes=31)

    def test_positive_offset(self) -> None:
        assert _resolve_time("+15s") > datetime.now(UTC)

    def test_iso_timestamp(self) -> None:
        assert _resolve_time("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=UTC)

    def test_naive_iso_timestamp_is_utc(self) -> None:
        assert _resolve_time("2024-01-01T00:00:00") == datetime(2024, 1, 1, tzinfo=UTC)

    def test_unix_timestamp(self) -> None:
        assert _resolve_time("1704067200") == datetime(2024, 1, 1, tzinfo=UTC)

    def test_invalid_raises(self) -> None:
        with pytest.raises(DriverError, match="Invalid time value"):
            _resolve_time("last tuesday")

    def test_formats_as_utc_milliseconds(self) -> None:
        when = datetime(2024, 1, 2, 3, 4, 5, 678999, tzinfo=UTC)
        assert _es_datetime(when) == "2024-01-02T03:04:05.678Z"


class TestSessionSettings:
    def test_defaults(self) -> None:
        driver, _ = _lucene_driver()
        assert driver.get_session() == {
            "time_field": "@timestamp",
            "time_from": None,
            "time_to": None,
            "sort_field": None,
            "sort_order": "desc",
        }

    async def test_set_updates_only_given_keys(self) -> None:
        driver, _ = _lucene_driver()
        await driver.set_session({"sort_field": "total"})
        assert driver.get_session()["sort_field"] == "total"
        assert driver.get_session()["sort_order"] == "desc"

    async def test_sort_order_is_normalised(self) -> None:
        driver, _ = _lucene_driver()
        await driver.set_session({"sort_order": "ASC"})
        assert driver.get_session()["sort_order"] == "asc"

    async def test_empty_value_restores_default(self) -> None:
        driver, _ = _lucene_driver()
        await driver.set_session({"time_from": "-1h", "time_field": "created_at"})
        await driver.set_session({"time_from": "", "time_field": ""})
        assert driver.get_session()["time_from"] is None
        assert driver.get_session()["time_field"] == "@timestamp"

    async def test_invalid_time_bound_raises(self) -> None:
        driver, _ = _lucene_driver()
        with pytest.raises(DriverError, match="Invalid time value"):
            await driver.set_session({"time_from": "yesterday"})

    async def test_invalid_sort_order_raises(self) -> None:
        driver, _ = _lucene_driver()
        with pytest.raises(DriverError, match="Unknown sort_order"):
            await driver.set_session({"sort_order": "sideways"})

    async def test_unknown_setting_raises(self) -> None:
        driver, _ = _lucene_driver()
        with pytest.raises(DriverError, match="Unknown session setting: nope"):
            await driver.set_session({"nope": "1"})


class TestLuceneSessionSettings:
    async def test_unset_settings_leave_query_untouched(self) -> None:
        driver, client = _lucene_driver()
        await driver.execute("orders | status:open", [])
        assert client.search.call_args.kwargs == {
            "index": "orders",
            "size": 1000,
            "q": "status:open",
        }

    async def test_time_range_becomes_filtered_bool_query(self) -> None:
        driver, client = _lucene_driver()
        await driver.set_session(
            {"time_from": "2024-01-01T00:00:00Z", "time_to": "2024-01-02T00:00:00Z"}
        )
        await driver.execute("orders | status:open", [])
        kwargs = client.search.call_args.kwargs
        assert "q" not in kwargs
        assert kwargs["query"] == {
            "bool": {
                "must": [{"query_string": {"query": "status:open"}}],
                "filter": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": "2024-01-01T00:00:00.000Z",
                                "lte": "2024-01-02T00:00:00.000Z",
                            }
                        }
                    }
                ],
            }
        }

    async def test_open_ended_range_and_custom_field(self) -> None:
        driver, client = _lucene_driver()
        await driver.set_session(
            {"time_field": "created_at", "time_to": "2024-01-02T00:00:00Z"}
        )
        await driver.execute("orders | *", [])
        assert client.search.call_args.kwargs["query"]["bool"]["filter"] == [
            {"range": {"created_at": {"lte": "2024-01-02T00:00:00.000Z"}}}
        ]

    async def test_sort_field_becomes_sort_clause(self) -> None:
        driver, client = _lucene_driver()
        await driver.set_session({"sort_field": "total", "sort_order": "asc"})
        await driver.execute("orders | *", [])
        kwargs = client.search.call_args.kwargs
        assert kwargs["sort"] == [{"total": {"order": "asc"}}]
        assert kwargs["q"] == "*"


class TestEsqlSessionSettings:
    async def test_time_range_follows_source_command(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"time_from": "2024-01-01T00:00:00Z"})
        assert driver._apply_esql_settings('FROM orders | WHERE status == "open"') == (
            'FROM orders | WHERE @timestamp >= TO_DATETIME("2024-01-01T00:00:00.000Z")'
            ' | WHERE status == "open"'
        )

    async def test_both_bounds(self) -> None:
        driver = _esql_driver()
        await driver.set_session(
            {"time_from": "2024-01-01T00:00:00Z", "time_to": "2024-01-02T00:00:00Z"}
        )
        assert driver._apply_esql_settings("FROM orders") == (
            'FROM orders | WHERE @timestamp >= TO_DATETIME("2024-01-01T00:00:00.000Z")'
            ' AND @timestamp <= TO_DATETIME("2024-01-02T00:00:00.000Z")'
        )

    async def test_sort_is_appended(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"sort_field": "total", "sort_order": "asc"})
        assert (
            driver._apply_esql_settings("FROM orders | KEEP total")
            == "FROM orders | KEEP total | SORT total ASC"
        )

    async def test_sort_goes_before_a_trailing_limit(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"sort_field": "@timestamp"})
        assert (
            driver._apply_esql_settings("FROM orders | LIMIT 10")
            == "FROM orders | SORT @timestamp DESC | LIMIT 10"
        )

    async def test_explicit_sort_wins(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"sort_field": "total"})
        query = "FROM orders | SORT name ASC | LIMIT 10"
        assert driver._apply_esql_settings(query) == query

    async def test_field_needing_quotes_is_backquoted(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"time_field": "event time", "time_from": "0"})
        assert driver._apply_esql_settings("FROM orders") == (
            'FROM orders | WHERE `event time` >= TO_DATETIME("1970-01-01T00:00:00.000Z")'
        )

    async def test_sourceless_query_is_untouched(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"time_from": "-1h", "sort_field": "total"})
        assert driver._apply_esql_settings("ROW a = 1") == "ROW a = 1"

    async def test_pipe_inside_string_literal_does_not_split(self) -> None:
        driver = _esql_driver()
        await driver.set_session({"sort_field": "total"})
        assert (
            driver._apply_esql_settings('FROM logs | WHERE msg == "a | b"')
            == 'FROM logs | WHERE msg == "a | b" | SORT total DESC'
        )

    async def test_settings_reach_the_client(self) -> None:
        client = MagicMock()
        client.esql.query = AsyncMock(return_value={"columns": [], "values": []})
        driver = ElasticsearchDriver({"query_mode": "esql"}, client, DriverSettings())
        await driver.set_session({"sort_field": "total"})
        await driver.execute("FROM orders", [])
        client.esql.query.assert_awaited_once_with(
            query="FROM orders | SORT total DESC", format="json"
        )


class TestSplitCommands:
    def test_single_command(self) -> None:
        assert _split_commands("FROM orders") == [(0, 11)]

    def test_spans_exclude_surrounding_whitespace(self) -> None:
        query = "FROM orders\n| LIMIT 5"
        spans = _split_commands(query)
        assert [query[slice(*span)] for span in spans] == ["FROM orders", "LIMIT 5"]

    def test_pipe_in_backquoted_identifier(self) -> None:
        query = "FROM orders | KEEP `a|b`"
        assert [query[slice(*s)] for s in _split_commands(query)] == [
            "FROM orders",
            "KEEP `a|b`",
        ]

    def test_pipe_in_triple_quoted_string(self) -> None:
        query = 'FROM logs | WHERE m == """a | b"""'
        assert [query[slice(*s)] for s in _split_commands(query)] == [
            "FROM logs",
            'WHERE m == """a | b"""',
        ]

    def test_escaped_quote_inside_string(self) -> None:
        query = 'FROM logs | WHERE m == "a \\" | b" | LIMIT 1'
        assert [query[slice(*s)] for s in _split_commands(query)] == [
            "FROM logs",
            'WHERE m == "a \\" | b"',
            "LIMIT 1",
        ]


class TestEsqlErrors:
    async def test_missing_query_endpoint_is_explained(self) -> None:
        client = MagicMock()
        client.esql.query = AsyncMock(
            side_effect=elasticsearch.BadRequestError(
                "Incorrect HTTP method for uri (/_query?format=json) and method "
                "[POST]. allowed: [HEAD, DELETE, GET, PUT]",
                SimpleNamespace(status=400),  # ty: ignore[invalid-argument-type]
                None,
            )
        )
        driver = ElasticsearchDriver({"query_mode": "esql"}, client, DriverSettings())
        with pytest.raises(DriverError, match="does not support ES|QL"):
            await driver.execute("FROM orders", [])


def _hist_driver(mode: str) -> tuple[ElasticsearchDriver, MagicMock]:
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "aggregations": {
                "histogram": {
                    "interval": "5m",
                    "buckets": [
                        {"key": 1704067200000, "doc_count": 4},
                        {"key": 1704067500000, "doc_count": 0},
                        {"key": 1704067800000, "doc_count": 7},
                    ],
                }
            }
        }
    )
    client.esql.query = AsyncMock()
    return ElasticsearchDriver({"query_mode": mode}, client, DriverSettings()), client


class TestHistogramLucene:
    async def test_aggregates_the_query_with_no_hits(self) -> None:
        driver, client = _hist_driver("lucene")
        result = await driver.histogram("logs | level:error", 40)
        assert client.search.call_args.kwargs == {
            "index": "logs",
            "q": "level:error",
            "size": 0,
            "track_total_hits": False,
            "aggs": {
                "histogram": {
                    "auto_date_histogram": {"field": "@timestamp", "buckets": 40}
                }
            },
        }
        assert result == HistogramResult(
            field="@timestamp",
            interval="5m",
            buckets=[
                HistogramBucket(time=1704067200000, count=4),
                HistogramBucket(time=1704067500000, count=0),
                HistogramBucket(time=1704067800000, count=7),
            ],
        )

    async def test_session_range_and_field_apply(self) -> None:
        driver, client = _hist_driver("lucene")
        await driver.set_session(
            {"time_field": "ts", "time_from": "2024-01-01T00:00:00Z"}
        )
        await driver.histogram("logs | *", 40)
        kwargs = client.search.call_args.kwargs
        assert kwargs["query"]["bool"]["filter"] == [
            {"range": {"ts": {"gte": "2024-01-01T00:00:00.000Z"}}}
        ]
        assert kwargs["aggs"]["histogram"]["auto_date_histogram"]["field"] == "ts"
        assert "sort" not in kwargs

    async def test_no_matches_yields_no_buckets(self) -> None:
        driver, client = _hist_driver("lucene")
        client.search.return_value = {
            "aggregations": {"histogram": {"interval": "1s", "buckets": []}}
        }
        result = await driver.histogram("logs | *", 40)
        assert result.buckets == []

    async def test_bad_query_format_raises(self) -> None:
        driver, _ = _hist_driver("lucene")
        with pytest.raises(DriverError, match="<index> | <query>"):
            await driver.histogram("no pipe here", 40)


class TestHistogramEsql:
    async def test_buckets_between_session_bounds(self) -> None:
        driver, client = _hist_driver("esql")
        await driver.set_session(
            {"time_from": "2024-01-01T00:00:00Z", "time_to": "2024-01-01T01:00:00Z"}
        )
        client.esql.query.return_value = {
            "columns": [{"name": "count"}, {"name": "time"}],
            "values": [
                [3, "2024-01-01T00:00:00.000Z"],
                [2, "2024-01-01T00:15:00.000Z"],
                [5, "2024-01-01T00:45:00.000Z"],
            ],
        }
        result = await driver.histogram(
            'FROM logs | WHERE level == "error" | STATS n = COUNT(*) BY host', 4
        )
        client.esql.query.assert_awaited_once_with(
            query=(
                'FROM logs | WHERE @timestamp >= TO_DATETIME("2024-01-01T00:00:00.000Z")'
                ' AND @timestamp <= TO_DATETIME("2024-01-01T01:00:00.000Z")'
                ' | WHERE level == "error"'
                " | STATS count = COUNT(*) BY time = BUCKET(@timestamp, 4,"
                ' "2024-01-01T00:00:00.000Z", "2024-01-01T01:00:00.000Z") | SORT time | LIMIT 1000'
            ),
            format="json",
        )
        assert result == HistogramResult(
            field="@timestamp",
            interval="15m",
            buckets=[
                HistogramBucket(time=1704067200000, count=3),
                HistogramBucket(time=1704068100000, count=2),
                HistogramBucket(time=1704069000000, count=0),
                HistogramBucket(time=1704069900000, count=5),
            ],
        )

    async def test_open_bounds_are_read_off_the_data(self) -> None:
        driver, client = _hist_driver("esql")
        client.esql.query.side_effect = [
            {
                "columns": [{"name": "lo"}, {"name": "hi"}],
                "values": [["2024-01-01T00:00:00.000Z", "2024-01-01T02:00:00.000Z"]],
            },
            {
                "columns": [{"name": "count"}, {"name": "time"}],
                "values": [[1, "2024-01-01T00:00:00.000Z"]],
            },
        ]
        result = await driver.histogram("FROM logs", 10)
        first, second = [c.kwargs["query"] for c in client.esql.query.await_args_list]
        assert (
            first
            == "FROM logs | STATS lo = MIN(@timestamp), hi = MAX(@timestamp) | LIMIT 1"
        )
        assert second == (
            "FROM logs | STATS count = COUNT(*) BY time = BUCKET(@timestamp, 10,"
            ' "2024-01-01T00:00:00.000Z", "2024-01-01T02:00:00.000Z") | SORT time | LIMIT 1000'
        )
        assert result == HistogramResult(
            field="@timestamp",
            interval="",
            buckets=[HistogramBucket(time=1704067200000, count=1)],
        )

    async def test_no_rows_yields_no_buckets(self) -> None:
        driver, client = _hist_driver("esql")
        client.esql.query.return_value = {
            "columns": [{"name": "lo"}, {"name": "hi"}],
            "values": [[None, None]],
        }
        result = await driver.histogram("FROM logs", 10)
        assert result.buckets == []
        client.esql.query.assert_awaited_once()

    async def test_custom_field_is_quoted(self) -> None:
        driver, client = _hist_driver("esql")
        await driver.set_session(
            {
                "time_field": "event-time",
                "time_from": "2024-01-01T00:00:00Z",
                "time_to": "2024-01-02T00:00:00Z",
            }
        )
        client.esql.query.return_value = {"columns": [], "values": []}
        await driver.histogram("FROM logs | LIMIT 5", 10)
        query = client.esql.query.call_args.kwargs["query"]
        assert "BUCKET(`event-time`, 10," in query
        assert " | LIMIT 5 | STATS count" in query

    async def test_sourceless_query_raises(self) -> None:
        driver, _ = _hist_driver("esql")
        with pytest.raises(DriverError, match="reads an index"):
            await driver.histogram("ROW a = 1", 10)


class TestHistogramDevTools:
    async def test_is_rejected(self) -> None:
        driver, _ = _hist_driver("dev_tools")
        with pytest.raises(DriverError, match="Dev Tools"):
            await driver.histogram("GET /logs/_search", 10)


class TestFillGaps:
    def test_calendar_steps_are_left_alone(self) -> None:
        jan, feb, mar = 1704067200000, 1706745600000, 1709251200000
        raw = [
            HistogramBucket(jan, 1),
            HistogramBucket(feb, 2),
            HistogramBucket(mar, 3),
        ]
        result = _fill_gaps("@timestamp", raw)
        assert result.buckets == raw
        assert result.interval == ""

    def test_fills_multiple_missing_steps(self) -> None:
        raw = [
            HistogramBucket(0, 1),
            HistogramBucket(3000, 1),
            HistogramBucket(4000, 1),
        ]
        result = _fill_gaps("@timestamp", raw)
        assert [b.time for b in result.buckets] == [0, 1000, 2000, 3000, 4000]
        assert [b.count for b in result.buckets] == [1, 0, 0, 1, 1]
        assert result.interval == "1s"

    def test_duration_labels(self) -> None:
        assert _duration(500) == "500ms"
        assert _duration(90_000) == "90s"
        assert _duration(3_600_000) == "1h"
        assert _duration(172_800_000) == "2d"
