"""Capabilities of the Elasticsearch driver, importable without its client package."""

from ...protocol import DriverParam, DriverParamChoice, Language, ParamType
from ..base import DriverMeta

DEFAULT_TIME_FIELD = "@timestamp"


class ElasticsearchMeta(DriverMeta):
    """What the Elasticsearch driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("elasticsearch", "aiohttp")

    LABEL = "Elasticsearch"

    SUPPORTS_HISTOGRAM = True

    # The default query mode; Dev Tools and ES|QL buffers have no language
    # of their own here.
    LANGUAGES = [Language.LUCENE]

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=9200),
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
        DriverParam(
            key="protocol",
            type=ParamType.ENUM,
            label="Protocol",
            choices=[
                DriverParamChoice(value="https", label="HTTPS"),
                DriverParamChoice(value="http", label="HTTP"),
            ],
            default="https",
        ),
        DriverParam(
            key="query_mode",
            type=ParamType.ENUM,
            label="Query Mode",
            choices=[
                DriverParamChoice(value="lucene", label="Lucene"),
                DriverParamChoice(value="dev_tools", label="Dev Tools"),
                DriverParamChoice(value="esql", label="ES|QL"),
            ],
            default="lucene",
        ),
    ]

    SESSION_PARAMS: list[DriverParam] = [
        DriverParam(
            key="time_field",
            type=ParamType.STRING,
            label="Time Field",
            required=False,
            default=DEFAULT_TIME_FIELD,
        ),
        DriverParam(
            key="time_from", type=ParamType.STRING, label="From", required=False
        ),
        DriverParam(key="time_to", type=ParamType.STRING, label="To", required=False),
        DriverParam(
            key="sort_field", type=ParamType.STRING, label="Sort Field", required=False
        ),
        DriverParam(
            key="sort_order",
            type=ParamType.ENUM,
            label="Sort Order",
            required=False,
            choices=[
                DriverParamChoice(value="desc", label="Descending"),
                DriverParamChoice(value="asc", label="Ascending"),
            ],
            default="desc",
        ),
    ]

    HELP: str = """\
## Elasticsearch

**Queries:** Prefix with the target index name (pattern or alias) and ` | `.
The connection is configured for Lucene, Dev Tools, or ES|QL query syntax.
The index prefix is not used in ES|QL mode, where the source index is named
in the query itself (`FROM <index>`).

*Lucene mode:*

```
orders | status:open AND total:>50
```

```
orders | *
```

*Dev Tools mode (Kibana Dev Tools syntax):*

```
GET /orders/_search
{"query": {"match": {"status": "open"}}}
```

```
GET /orders,products/_search
{"query": {"match_all": {}}, "sort": [{"total": "desc"}]}
```

*ES|QL mode:*

```
FROM orders | WHERE status == "open" AND total > 50 | LIMIT 100
```

Pick, rename and derive columns with `KEEP` / `DROP` / `RENAME` / `EVAL`:

```
FROM orders
| EVAL net = total - tax, day = DATE_TRUNC(1 day, @timestamp)
| KEEP day, customer, net
| SORT net DESC
| LIMIT 20
```

Aggregate with `STATS ... BY`, bucketing time with `BUCKET`:

```
FROM logs-*
| WHERE status >= 500
| STATS errors = COUNT(*) BY service, span = BUCKET(@timestamp, 1 hour)
| SORT span DESC, errors DESC
```

`STATS` output can be filtered again further down the pipe:

```
FROM traces
| STATS p95 = PERCENTILE(took_ms, 95), avg = AVG(took_ms), n = COUNT(*) BY service
| WHERE n > 100
| SORT p95 DESC
```

Branch with `CASE`, and fan a multi-valued field out one row per value:

```
FROM orders
| EVAL tier = CASE(total > 1000, "large", total > 100, "medium", "small")
| STATS n = COUNT(*) BY tier
```

```
FROM articles | MV_EXPAND tags | STATS n = COUNT(*) BY tags | SORT n DESC | LIMIT 10
```

Pull structure out of a text field with `GROK` (or `DISSECT`):

```
FROM logs
| GROK message "%{IP:client} %{WORD:method} %{URIPATHPARAM:path}"
| STATS hits = COUNT(*) BY path
| SORT hits DESC
| LIMIT 10
```

Match loosely, and ask for document metadata:

```
FROM logs-* METADATA _index, _id
| WHERE message LIKE "*timeout*" AND service IN ("api", "web")
| KEEP _index, @timestamp, service, message
| LIMIT 50
```

ES|QL requires Elasticsearch 8.11 or later.

Any Elasticsearch REST endpoint is accepted — the response is returned as a
flat table. Search responses unpack `hits.hits`; all other responses are
flattened as a single row.

System indices (names starting with `.`) are hidden in the resource tree.

**Comments:** Lucene has no comment syntax of its own; `--` to the end of
the line is stripped before the query is sent, wherever a term could start
(`a--b` stays one term). Dev Tools takes Kibana Console's `#` and `//` line
comments and `/* ... */` blocks; ES|QL has `//` and `/* ... */` natively.

**Time range and sorting:** five session settings — changeable any time via
`session.set`, no reconnect needed — apply to every Lucene and ES|QL query.
Dev Tools queries are sent exactly as written.

| Setting | Default | Meaning |
|---------|---------|---------|
| `time_field` | `@timestamp` | Date field the range filters on |
| `time_from` | — | Lower bound, inclusive |
| `time_to` | — | Upper bound, inclusive |
| `sort_field` | — | Field to order results by |
| `sort_order` | `desc` | `asc` or `desc` |

`time_from` / `time_to` accept `now`, an offset from now (`-1h`, `now-30m`,
`+15s`), an ISO-8601 timestamp (`2024-01-01T00:00:00Z`), or a Unix timestamp in
seconds. Either bound may stand alone; leave both empty for no time filter.
`time_to` must not be before `time_from`. Setting any of the five to an empty
string restores its default.

In Lucene mode the range becomes a `range` filter beside the query and the sort
a `sort` clause. In ES|QL mode the range is spliced in as a `WHERE` directly
after the source command — where the time field is still in scope, a `STATS`
further down the pipe having dropped it — and the sort appended as a `SORT`,
ahead of a trailing `LIMIT` so the limit takes the head of the sorted result. A
query carrying its own `SORT` keeps it, and one that reads no index (`ROW`,
`SHOW`) is left untouched.

**Documents over time:** `execute.histogram` counts the documents a Lucene
or ES|QL query matches per bucket of `time_field`, within the session time
range — an `auto_date_histogram` on the same search in Lucene mode, and
`STATS COUNT(*) BY BUCKET(...)` appended to the query (cut at its first
`STATS`) in ES|QL mode. Dev Tools requests are sent as written and cannot be
charted.

**Resources:**

```
(root)
└── <index>
    ├── mappings  → field name and type
    └── aliases   → alias names
```

Describing an index returns field metadata from its mapping (name, type).
"""
