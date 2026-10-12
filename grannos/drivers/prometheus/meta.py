"""Capabilities of the Prometheus driver, importable without its client package."""

from ...protocol import DriverParam, DriverParamChoice, Language, ParamType
from ..base import DriverMeta

DEFAULT_URL = "http://localhost:9090"


class PrometheusMeta(DriverMeta):
    """What the Prometheus driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("aiohttp",)

    LABEL = "Prometheus"

    LANGUAGES = [Language.PROMQL]

    SUPPORTS_WRITES = False

    PARAMS: list[DriverParam] = [
        DriverParam(key="url", type=ParamType.STRING, label="URL", default=DEFAULT_URL),
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

**Comments:** `#` to the end of the line.

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
