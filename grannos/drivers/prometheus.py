"""Prometheus driver — requires: pip install aiohttp"""

import logging
import asyncio
import base64
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import aiohttp

from ..log import log_query
from ..protocol import (
    DescribeResult,
    DriverParam,
    DriverParamChoice,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    GenericRecordDescription,
    Language,
    NodeType,
    ParamType,
    RawDocument,
    RecordField,
    ReadResult,
    SpecialFloat,
    WriteResult,
)
from .base import BaseDriver, ConnectionLostError, DriverError, DriverSettings

_DEFAULT_URL = "http://localhost:9090"

# How many values a label lists under a metric. A high-cardinality label
# (`instance`, request ids) can carry far more, and neither a tree node nor a
# completion popup is the place for them; `limit` is honoured by Prometheus
# from 2.51 and ignored, harmlessly, before.
_LABEL_VALUES_LIMIT = 1000

_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "y": 365 * 86400,
}
_DURATION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)$")


logger = logging.getLogger(__name__)


class PrometheusDriver(BaseDriver):
    """Prometheus driver backed by the HTTP query API.

    Args:
        params: Connect request fields (``url``, ``username``, ``password``). ``query_mode``
            is a SESSION_PARAMS setting, not a connect param — see :attr:`SESSION_PARAMS`.
        session: Open aiohttp session. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "Prometheus"

    LANGUAGES = [Language.PROMQL]

    SUPPORTS_WRITES = False

    FIND_PATHS = {
        NodeType.METRIC: [["metrics", "*"]],
        NodeType.LABEL: [["metrics", "*", "*"]],
        NodeType.JOB: [["jobs", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="url", type=ParamType.STRING, label="URL", default=_DEFAULT_URL
        ),
        DriverParam(
            key="username", type=ParamType.STRING, label="Username", required=False
        ),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
    ]

    SESSION_PARAMS: list[DriverParam] = [
        DriverParam(
            key="query_mode",
            type=ParamType.ENUM,
            label="Query Mode",
            choices=[
                DriverParamChoice(value="instant", label="Instant"),
                DriverParamChoice(value="range", label="Range"),
            ],
            default="instant",
        ),
    ]

    HELP: str = """\
## Prometheus

**Queries:** PromQL, evaluated via the connection's `query_mode` session setting
(instant/range — change it any time via `session.set`, no reconnect needed).

*Instant mode* (default) — a plain PromQL expression, evaluated at the current time:

```
rate(http_requests_total[5m])
```

```
sum by (job) (up)
```

*Range mode* — prefix with `<start>,<end>,<step> | ` giving the evaluation window.
`start`/`end` accept `now`, a relative offset (`-1h`, `-30m`, `-15s`), an RFC3339
timestamp, or a raw Unix timestamp. `step` is a Prometheus duration (`15s`, `1m`).

```
-1h,now,15s | rate(http_requests_total[5m])
```

```
2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,30s | up
```

Vector/matrix results are flattened to one row per series (range queries emit one
row per series per timestamp), with a column per label plus `timestamp` and `value`.
Scalar/string results return a single `timestamp`/`value` row. A `value` of `NaN`,
`+Inf`, or `-Inf` (e.g. from a division in the expression) is returned as a
`SpecialFloat` — plain JSON cannot represent these — rather than a numeric `value`.

**Resources:**

```
(root)
├── metrics
│   └── <metric>
│       └── <label>
│           └── <value>
├── jobs
│   └── <job>
│       └── <metric>
│           └── <label>
│               └── <value>
├── configuration
└── runtime
```

Requires Prometheus >= 2.24 (`/api/v1/labels` with `match[]` support).

A label expands to its values for that metric (`/api/v1/label/<label>/values`
with `match[]=<metric>`), the first 1000 of them in the order Prometheus
returns — the same listing a query bar completes a matcher's value from.

A metric reached by drilling into a job (`["jobs", job, metric]`) behaves
identically to the equivalent top-level `["metrics", metric]` node —
`explore.list` expands it to the same label children (`["metrics", metric,
label]` / `["jobs", job, metric, label]` describe identically too),
`explore.describe` returns the same label metadata (name, up to 3 sampled
values, with `kind`/`comment` from `/api/v1/metadata` when available), and
`explore.preview` runs the metric's bare name as a PromQL query (subject to
the connection's `query_mode`), same as typing it in the query bar.

Describing `configuration` (leaf) returns a `RawDocument` (`filetype: "yaml"`)
holding the running configuration, as reported by `/api/v1/status/config`.

Describing `runtime` (leaf) returns a `GenericRecordDescription`
(`kind: "prometheus.runtime"`) with CLI flags, runtime info (retention, start
time, storage stats, …), and build info as label/value fields, merged from
`/api/v1/status/flags`, `/api/v1/status/runtimeinfo`, and
`/api/v1/status/buildinfo` — each field labeled `Flag: *`, `Runtime: *`, or
`Build: *`.

`jobs` lists scrape jobs (`/api/v1/targets`, grouped by the `job` label).
Expanding a job lists the distinct metric names scraped from it
(`/api/v1/label/__name__/values`, scoped with `match[]={job="<job>"}`) —
each expandable to its labels exactly like a top-level metric node (see
above). Describing a job returns an array of `GenericRecordDescription`
(`kind: "prometheus.target"`), one per target, named after its instance, in
this field order:
`Interval`, `Timeout`, `Last Scrape` (a relative age — `5s ago`, `3m ago`,
`2h ago`), `Status` (`✓`/`✗`/`?` for up/down/unknown), `Last Scrape
Duration` (whole milliseconds, e.g. `12ms`), and `Scraped metrics (series)` —
`"<n> (<m>)"`, where `<n>` is the count of distinct metric names last
scraped from the target (via `/api/v1/targets/metadata`) and `<m>` is its
last `scrape_samples_scraped` value (its series count, assuming the normal
one sample per series per scrape). `Last Error` is appended (on every
target's record) only when at least one target in the job has a non-empty
error. Target labels and the scrape pool name are not exposed (the pool is just the
job name; instance/job already group the record).
"""

    def __init__(
        self,
        params: dict[str, Any],
        session: aiohttp.ClientSession,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._http = session
        self._url = str(params.get("url") or _DEFAULT_URL).rstrip("/")
        self._ever_connected = False
        self._session_values: dict[str, Any] = {
            p.key: p.default for p in self.SESSION_PARAMS
        }
        """Runtime SESSION_PARAMS values, seeded from their declared defaults."""

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "PrometheusDriver":
        return cls(params, cls._open(params), settings)

    @staticmethod
    def _open(params: dict[str, Any]) -> aiohttp.ClientSession:
        headers = {}
        username = params.get("username")
        password = params.get("password")
        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        return aiohttp.ClientSession(headers=headers)

    async def reconnect(self) -> None:
        await self._http.close()
        self._http = self._open(self.params)
        self._ever_connected = False

    async def disconnect(self) -> None:
        await self._http.close()

    async def execute(
        self,
        query: str,
        binds: list[Any],
        diagram_captions: dict[str, str] | None = None,
    ) -> ReadResult | WriteResult:
        """Run a PromQL query.

        Args:
            query: A PromQL expression (instant mode), or a range query prefixed
                with ``<start>,<end>,<step> | `` (range mode).
            binds: Unused for Prometheus.
            diagram_captions: Unused for Prometheus (not a graph driver).
        """
        mode = self._session_values.get("query_mode", "instant")
        if mode == "instant":
            return await self._execute_instant(query)
        if mode == "range":
            return await self._execute_range(query)
        raise DriverError(f"Unknown query_mode: {mode!r}")

    async def set_session(self, values: dict[str, Any]) -> None:
        if "query_mode" in values:
            mode = values["query_mode"]
            if mode not in ("instant", "range"):
                raise DriverError(f"Unknown query_mode: {mode!r}")
            self._session_values["query_mode"] = mode

    def get_session(self) -> dict[str, Any]:
        return dict(self._session_values)

    async def _execute_instant(self, query: str) -> ReadResult:
        data = await self._get("/api/v1/query", {"query": query.strip()})
        return _data_to_result(data)

    async def _execute_range(self, query: str) -> ReadResult:
        start, end, step, promql = _parse_range_query(query)
        now = time.time()
        data = await self._get(
            "/api/v1/query_range",
            {
                "query": promql,
                "start": _resolve_time(start, now),
                "end": _resolve_time(end, now),
                "step": step,
            },
        )
        return _data_to_result(data)

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        # Every request this driver makes goes through here. The PromQL sits in
        # params, so it is folded into the logged line rather than split out.
        log_query(logger, f"GET {path} {params}" if params else f"GET {path}")
        if self._http.closed:
            raise ConnectionLostError("Session is closed")
        try:
            async with self._http.get(f"{self._url}{path}", params=params) as resp:
                body = await resp.json(content_type=None)
        except Exception as exc:
            if isinstance(exc, (aiohttp.ClientConnectionError, TimeoutError)):
                if self._ever_connected:
                    raise ConnectionLostError(str(exc)) from exc
                raise DriverError(str(exc)) from exc
            raise DriverError(str(exc)) from exc
        if not isinstance(body, dict) or body.get("status") != "success":
            raise DriverError(_format_error(body))
        self._ever_connected = True
        return body["data"]

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                return [
                    ExploreItem(name="metrics", type="group", expandable=True),
                    ExploreItem(name="jobs", type="group", expandable=True),
                    ExploreItem(
                        name="configuration", type="configuration", expandable=False
                    ),
                    ExploreItem(name="runtime", type="settings", expandable=False),
                ]
            case ["metrics"]:
                names = await self._get("/api/v1/label/__name__/values", {})
                return [
                    ExploreItem(name=name, type="metric", expandable=True)
                    for name in sorted(names)
                ]
            case ["metrics", metric] | ["jobs", _, metric]:
                labels = await self._get("/api/v1/labels", {"match[]": metric})
                return [
                    ExploreItem(name=label, type="label", expandable=True)
                    for label in sorted(labels)
                    if label != "__name__"
                ]
            case ["metrics", metric, label] | ["jobs", _, metric, label]:
                values = await self._get(
                    f"/api/v1/label/{quote(label, safe='')}/values",
                    {"match[]": metric, "limit": str(_LABEL_VALUES_LIMIT)},
                )
                return [
                    ExploreItem(name=value, type="label_value", expandable=False)
                    for value in values
                ]
            case ["jobs"]:
                data = await self._get("/api/v1/targets", {})
                jobs = sorted({_target_job(t) for t in data.get("activeTargets", [])})
                return [
                    ExploreItem(name=job, type="job", expandable=True) for job in jobs
                ]
            case ["jobs", job]:
                names = await self._get(
                    "/api/v1/label/__name__/values",
                    {"match[]": _job_selector(job)},
                )
                return [
                    ExploreItem(name=name, type="metric", expandable=True)
                    for name in sorted(names)
                ]
            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        # Previewing a metric — reached either via the top-level metrics list or
        # by drilling into a job's scraped metrics — is just running its name as
        # a PromQL instant/range query, same as the query bar would.
        match path:
            case ["metrics", metric] | ["jobs", _, metric]:
                result = await self.execute(metric, [])
                return result if isinstance(result, ReadResult) else None
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        match path:
            case ["metrics", metric] | ["jobs", _, metric]:
                return await self._describe_metric(metric)
            case ["metrics", metric, label] | ["jobs", _, metric, label]:
                return FieldDescription(
                    name=label,
                    types=["label"],
                    sample=await self._label_values_sample(metric, label),
                )
            case ["configuration"]:
                data = await self._get("/api/v1/status/config", {})
                return RawDocument(filetype="yaml", content=data.get("yaml", ""))
            case ["runtime"]:
                return await self._describe_runtime()
            case ["jobs", job]:
                return await self._describe_job(job)
            case _:
                return None

    async def _describe_metric(self, metric: str) -> EntityDescription:
        labels = await self._get("/api/v1/labels", {"match[]": metric})
        label_names = sorted(label for label in labels if label != "__name__")
        metadata = await self._metric_metadata(metric)
        properties = [
            FieldDescription(
                name=label,
                types=["label"],
                sample=await self._label_values_sample(metric, label),
            )
            for label in label_names
        ]
        return EntityDescription(
            name=metric,
            kind=metadata.get("type", "metric"),
            properties=properties,
            comment=metadata.get("help"),
        )

    async def _describe_runtime(self) -> GenericRecordDescription:
        flags = await self._get("/api/v1/status/flags", {})
        runtime_info = await self._get("/api/v1/status/runtimeinfo", {})
        build_info = await self._get("/api/v1/status/buildinfo", {})
        fields = [
            RecordField(label=f"Flag: {key}", value=str(value))
            for key, value in sorted(flags.items())
        ]
        fields += [
            RecordField(label=f"Runtime: {key}", value=str(value))
            for key, value in sorted(runtime_info.items())
        ]
        fields += [
            RecordField(label=f"Build: {key}", value=str(value))
            for key, value in sorted(build_info.items())
        ]
        return GenericRecordDescription(
            kind="prometheus.runtime", name="runtime", fields=fields
        )

    async def _describe_job(self, job: str) -> list[GenericRecordDescription] | None:
        data = await self._get("/api/v1/targets", {})
        targets = [t for t in data.get("activeTargets", []) if _target_job(t) == job]
        if not targets:
            return None
        metric_counts, series_counts = await self._job_scrape_stats(job)
        records = [
            _target_record(
                t,
                _target_instance(t),
                metric_counts.get(_target_instance(t), 0),
                series_counts.get(_target_instance(t), 0),
            )
            for t in sorted(targets, key=_target_instance)
        ]
        any_errors = any(
            f.label == _TARGET_FIELD_LABELS["lastError"] and f.value
            for rec in records
            for f in rec.fields
        )
        if any_errors:
            return records
        return [
            GenericRecordDescription(
                kind=rec.kind,
                name=rec.name,
                fields=[
                    f
                    for f in rec.fields
                    if f.label != _TARGET_FIELD_LABELS["lastError"]
                ],
            )
            for rec in records
        ]

    async def _job_scrape_stats(
        self, job: str
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Per-instance (distinct metric name count, timeseries count) for `job`'s
        targets, in a single round trip each rather than one per target.

        Metric names come from `/api/v1/targets/metadata` (one entry per
        target/metric pair). Timeseries counts come from the `scrape_samples_scraped`
        metric Prometheus attaches to every scrape — the number of samples in a
        target's last scrape, which is its series count assuming (as is normal)
        one sample per series per scrape.
        """
        selector = _job_selector(job)

        metric_names: dict[str, set[str]] = {}
        try:
            metadata = await self._get(
                "/api/v1/targets/metadata",
                {"match_target": selector},
            )
        except DriverError:
            metadata = []
        for entry in metadata:
            instance = entry.get("target", {}).get("instance")
            if not instance:
                continue
            metric_names.setdefault(instance, set()).add(entry.get("metric", ""))
        metric_counts = {
            instance: len(names) for instance, names in metric_names.items()
        }

        series_counts: dict[str, int] = {}
        try:
            data = await self._get(
                "/api/v1/query", {"query": f"scrape_samples_scraped{selector}"}
            )
        except DriverError:
            data = {}
        for s in data.get("result", []):
            instance = s.get("metric", {}).get("instance")
            if not instance:
                continue
            try:
                series_counts[instance] = int(float(s["value"][1]))
            except (TypeError, ValueError, KeyError, IndexError):
                continue

        return metric_counts, series_counts

    async def _metric_metadata(self, metric: str) -> dict[str, str]:
        try:
            data = await self._get("/api/v1/metadata", {"metric": metric})
        except DriverError:
            return {}
        entries = data.get(metric) or []
        if not entries:
            return {}
        entry = entries[0]
        result = {"type": entry.get("type", "metric")}
        if entry.get("help"):
            result["help"] = entry["help"]
        return result

    async def _label_values_sample(self, metric: str, label: str) -> list[Any]:
        try:
            return await asyncio.wait_for(
                self._fetch_label_values_sample(metric, label),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return []

    async def _fetch_label_values_sample(self, metric: str, label: str) -> list[Any]:
        values = await self._get(
            f"/api/v1/label/{quote(label, safe='')}/values", {"match[]": metric}
        )
        return values[: self._settings.column_sample_size]


def _parse_range_query(query: str) -> tuple[str, str, str, str]:
    if " | " not in query:
        raise DriverError(
            "Range query must be in the format: <start>,<end>,<step> | <promql>\n"
            "Example: -1h,now,15s | rate(http_requests_total[5m])"
        )
    header, _, promql = query.partition(" | ")
    parts = [p.strip() for p in header.split(",")]
    if len(parts) != 3:
        raise DriverError(
            "Range header must have 3 comma-separated fields: <start>,<end>,<step>"
        )
    start, end, step = parts
    return start, end, step, promql.strip()


def _resolve_time(value: str, now: float) -> str:
    value = value.strip()
    if value == "now":
        return str(now)
    match = _DURATION_RE.match(value)
    if match:
        amount, unit = match.groups()
        return str(now + float(amount) * _DURATION_UNITS[unit])
    return value


def _format_error(body: Any) -> str:
    if not isinstance(body, dict):
        return f"Prometheus error: {body}"
    error = body.get("error", body)
    error_type = body.get("errorType")
    if error_type:
        return f"Prometheus error ({error_type}): {error}"
    return f"Prometheus error: {error}"


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _data_to_result(data: dict[str, Any]) -> ReadResult:
    result_type = data.get("resultType")
    result = data.get("result", [])
    if result_type == "vector":
        return _series_to_result(result, ranged=False)
    if result_type == "matrix":
        return _series_to_result(result, ranged=True)
    if result_type in ("scalar", "string"):
        ts, value = result
        return ReadResult(
            columns=["timestamp", "value"],
            rows=[[_format_timestamp(float(ts)), value]],
            rows_total=1,
        )
    raise DriverError(f"Unsupported Prometheus result type: {result_type!r}")


def _series_to_result(series: list[dict[str, Any]], ranged: bool) -> ReadResult:
    label_names = sorted(
        {k for s in series for k in s.get("metric", {}) if k != "__name__"}
    )
    has_name = any(s.get("metric", {}).get("__name__") for s in series)
    columns = (["__name__"] if has_name else []) + label_names + ["timestamp", "value"]
    rows: list[list[Any]] = []
    for s in series:
        metric = s.get("metric", {})
        base: list[Any] = [metric.get("__name__")] if has_name else []
        base += [metric.get(name) for name in label_names]
        if ranged:
            for ts, value in s["values"]:
                rows.append([*base, _format_timestamp(float(ts)), _parse_value(value)])
        else:
            ts, value = s["value"]
            rows.append([*base, _format_timestamp(float(ts)), _parse_value(value)])
    return ReadResult(columns=columns, rows=rows, rows_total=len(rows))


def _parse_value(value: str) -> Any:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(parsed) or math.isinf(parsed):
        return SpecialFloat(text=value)
    return parsed


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _job_selector(job: str) -> str:
    return f'{{job="{_escape_label_value(job)}"}}'


def _target_job(target: dict[str, Any]) -> str:
    return target.get("labels", {}).get("job", target.get("scrapePool", "unknown"))


def _target_instance(target: dict[str, Any]) -> str:
    return target.get("labels", {}).get("instance", "unknown")


def _format_health(value: str) -> str:
    if value == "up":
        return "✓"
    if value == "down":
        return "✗"
    if value == "unknown":
        return "?"
    return value


def _format_scrape_age(value: str) -> str:
    try:
        scraped_at = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=UTC)
    seconds = max(0, int((datetime.now(UTC) - scraped_at).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h ago"


def _format_duration_ms(value: Any) -> str:
    try:
        return f"{round(float(value) * 1000)}ms"
    except (TypeError, ValueError):
        return str(value)


_TARGET_FIELD_LABELS: dict[str, str] = {
    "scrapeInterval": "Interval",
    "scrapeTimeout": "Timeout",
    "lastScrape": "Last Scrape",
    "health": "Status",
    "lastScrapeDuration": "Last Scrape Duration",
    "lastError": "Last Error",
}
"""Target dict keys exposed on a target's record, mapped to their display label.
``scrapePool`` (redundant with the job grouping) and raw ``labels`` are
deliberately omitted — see :func:`_target_job`/:func:`_target_instance` for the
only label values actually surfaced (as the record's grouping key)."""

_TARGET_FIELD_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "health": _format_health,
    "lastScrape": _format_scrape_age,
    "lastScrapeDuration": _format_duration_ms,
}


def _target_record(
    target: dict[str, Any], name: str, metric_count: int, series_count: int
) -> GenericRecordDescription:
    fields = [
        RecordField(
            label=label, value=_TARGET_FIELD_FORMATTERS.get(key, str)(target[key])
        )
        for key, label in _TARGET_FIELD_LABELS.items()
        if key in target and key != "lastError"
    ]
    fields.append(
        RecordField(
            label="Scraped metrics (series)",
            value=f"{metric_count} ({series_count})",
        )
    )
    if "lastError" in target:
        fields.append(
            RecordField(
                label=_TARGET_FIELD_LABELS["lastError"],
                value=str(target["lastError"]),
            )
        )
    return GenericRecordDescription(kind="prometheus.target", name=name, fields=fields)
