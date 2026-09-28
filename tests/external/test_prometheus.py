"""
Integration tests for the Prometheus driver.

Requires a running Prometheus instance whose default config self-scrapes
(job "prometheus", instance "localhost:9090") — the stock behaviour of the
official `prom/prometheus` image. Configure via environment variable:
  PROMETHEUS_URL  (default: http://localhost:9090)

Tests are skipped automatically when aiohttp is not installed or the server
is unreachable.
"""

import os
import re
from collections.abc import AsyncGenerator

import pytest

from grannos.drivers.base import DriverError, DriverSettings
from grannos.drivers.prometheus import PrometheusDriver
from grannos.protocol import (
    EntityDescription,
    FieldDescription,
    GenericRecordDescription,
    RawDocument,
    ReadResult,
)

pytestmark = pytest.mark.external


def _params(**overrides: str) -> dict:
    return {
        "url": os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
        **overrides,
    }


@pytest.fixture
async def driver() -> AsyncGenerator[PrometheusDriver, None]:
    pytest.importorskip("aiohttp")
    d = await PrometheusDriver.create(_params(), DriverSettings())
    yield d
    await d.disconnect()


@pytest.fixture
async def range_driver() -> AsyncGenerator[PrometheusDriver, None]:
    pytest.importorskip("aiohttp")
    d = await PrometheusDriver.create(_params(), DriverSettings())
    await d.set_session({"query_mode": "range"})
    yield d
    await d.disconnect()


class TestExecuteInstant:
    async def test_returns_columns_and_rows(self, driver: PrometheusDriver) -> None:
        result = await driver.execute("up", [])
        assert isinstance(result, ReadResult)
        assert "job" in result.columns
        assert "instance" in result.columns
        assert "timestamp" in result.columns
        assert "value" in result.columns
        assert result.rows_total >= 1

    async def test_no_match_returns_empty(self, driver: PrometheusDriver) -> None:
        result = await driver.execute("nonexistent_metric_xyz", [])
        assert isinstance(result, ReadResult)
        assert result.rows == []

    async def test_invalid_promql_raises(self, driver: PrometheusDriver) -> None:
        with pytest.raises(DriverError):
            await driver.execute("up(((", [])

    async def test_ignores_comments(self, driver: PrometheusDriver) -> None:
        result = await driver.execute("# all targets\nup # scraped", [])
        assert isinstance(result, ReadResult)
        assert result.rows_total >= 1


class TestExecuteRange:
    async def test_returns_rows_across_the_window(
        self, range_driver: PrometheusDriver
    ) -> None:
        result = await range_driver.execute("-1m,now,15s | up", [])
        assert isinstance(result, ReadResult)
        assert "timestamp" in result.columns
        assert result.rows_total >= 1

    async def test_ignores_comments(self, range_driver: PrometheusDriver) -> None:
        result = await range_driver.execute(
            "# last | minute\n-1m,now,15s | up # all", []
        )
        assert isinstance(result, ReadResult)
        assert result.rows_total >= 1

    async def test_missing_separator_raises(
        self, range_driver: PrometheusDriver
    ) -> None:
        with pytest.raises(DriverError, match="format"):
            await range_driver.execute("up", [])


class TestExploreList:
    async def test_root_lists_top_level_sections(
        self, driver: PrometheusDriver
    ) -> None:
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

    async def test_metrics_group_lists_metric_names(
        self, driver: PrometheusDriver
    ) -> None:
        items = await driver.explore_list(["metrics"])
        names = [i.name for i in items]
        assert "up" in names
        assert all(i.type == "metric" and i.expandable for i in items)

    async def test_metric_lists_labels(self, driver: PrometheusDriver) -> None:
        items = await driver.explore_list(["metrics", "up"])
        names = [i.name for i in items]
        assert "job" in names
        assert "instance" in names
        assert "__name__" not in names
        assert all(not i.expandable for i in items)

    async def test_label_path_returns_empty(self, driver: PrometheusDriver) -> None:
        items = await driver.explore_list(["metrics", "up", "job"])
        assert items == []

    async def test_unknown_path_returns_empty(self, driver: PrometheusDriver) -> None:
        items = await driver.explore_list(["metrics", "up", "job", "extra"])
        assert items == []

    async def test_jobs_lists_scrape_jobs_as_leaves(
        self, driver: PrometheusDriver
    ) -> None:
        items = await driver.explore_list(["jobs"])
        assert "prometheus" in [i.name for i in items]
        assert all(i.type == "job" and not i.expandable for i in items)

    async def test_job_path_returns_empty(self, driver: PrometheusDriver) -> None:
        items = await driver.explore_list(["jobs", "prometheus"])
        assert items == []


class TestExploreDescribe:
    async def test_returns_entity_description_for_metric(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["metrics", "up"])
        assert isinstance(result, EntityDescription)
        assert result.name == "up"
        label_names = [f.name for f in result.properties]
        assert "job" in label_names
        assert "instance" in label_names

    async def test_labels_carry_sampled_values(self, driver: PrometheusDriver) -> None:
        result = await driver.explore_describe(["metrics", "up"])
        assert isinstance(result, EntityDescription)
        job_field = next(f for f in result.properties if f.name == "job")
        assert "prometheus" in job_field.sample

    async def test_describe_single_label_returns_field_description(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["metrics", "up", "job"])
        assert isinstance(result, FieldDescription)
        assert result.name == "job"
        assert result.types == ["label"]
        assert "prometheus" in result.sample

    async def test_returns_none_for_unknown_path(
        self, driver: PrometheusDriver
    ) -> None:
        assert await driver.explore_describe([]) is None

    async def test_configuration_returns_raw_yaml_document(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["configuration"])
        assert isinstance(result, RawDocument)
        assert result.filetype == "yaml"
        assert "scrape_configs" in result.content

    async def test_runtime_returns_flag_runtime_and_build_fields(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["runtime"])
        assert isinstance(result, GenericRecordDescription)
        assert result.kind == "prometheus.runtime"
        labels = [f.label for f in result.fields]
        assert any(label.startswith("Flag: ") for label in labels)
        assert any(label.startswith("Runtime: ") for label in labels)
        assert any(label.startswith("Build: ") for label in labels)

    async def test_describe_job_returns_one_record_per_target(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["jobs", "prometheus"])
        assert isinstance(result, list)
        records = [r for r in result if isinstance(r, GenericRecordDescription)]
        assert len(records) == len(result)
        assert records != []
        assert all(r.kind == "prometheus.target" for r in records)

        rec = next(r for r in records if r.name == "localhost:9090")
        labels = {f.label: f.value for f in rec.fields}
        assert labels["Status"] == "✓"
        assert re.fullmatch(r"[0-9]+[smh] ago", labels["Last Scrape"])
        assert re.fullmatch(r"[0-9]+ms", labels["Last Scrape Duration"])
        assert not any("Label: " in label for label in labels)
        assert not any("Scrape Pool" in label for label in labels)
        assert [f.label for f in rec.fields][:4] == [
            "Interval",
            "Timeout",
            "Last Scrape",
            "Status",
        ]
        assert [f.label for f in rec.fields][4] == "Last Scrape Duration"

    async def test_describe_unknown_job_returns_none(
        self, driver: PrometheusDriver
    ) -> None:
        result = await driver.explore_describe(["jobs", "does-not-exist"])
        assert result is None
