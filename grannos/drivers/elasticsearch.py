"""Elasticsearch driver — requires: pip install elasticsearch aiohttp"""

import logging
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import elasticsearch

from ..log import log_query
from ..protocol import (
    DriverParam,
    DriverParamChoice,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    HistogramBucket,
    HistogramResult,
    Language,
    NodeType,
    ParamType,
    ReadResult,
    WriteResult,
)
from ..tabular import flatten_docs
from .base import BaseDriver, ConnectionLostError, DriverError, DriverSettings

_DEFAULT_SEARCH_SIZE = 1000
_DEFAULT_TIME_FIELD = "@timestamp"

_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
    "y": 365 * 86400,
}
_DURATION_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)$")
_EPOCH_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
# An ES|QL identifier that needs no backquoting.
_PLAIN_IDENTIFIER_RE = re.compile(r"^[A-Za-z_@][A-Za-z0-9_@.]*$")
# ES|QL source commands a time filter may be appended to.
_ESQL_SOURCE_RE = re.compile(r"^(from|ts|metrics)\b", re.IGNORECASE)
_ESQL_SORT_RE = re.compile(r"^sort\b", re.IGNORECASE)
_ESQL_LIMIT_RE = re.compile(r"^limit\b", re.IGNORECASE)
_ESQL_STATS_RE = re.compile(r"^stats\b", re.IGNORECASE)
# Explicit row cap on the generated histogram STATS — ES|QL's own default when
# none is given, spelled out so the server does not warn about it. Well above
# the bucket count the dispatcher allows.
_ESQL_HISTOGRAM_LIMIT = 1000


logger = logging.getLogger(__name__)


class ElasticsearchDriver(BaseDriver):
    """Elasticsearch driver backed by the official elasticsearch-py client.

    Args:
        params: Connect request fields (``host``, ``port``, ``username``, ``password``,
            ``query_mode``).
        client: Open Elasticsearch client. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "Elasticsearch"
    SUPPORTS_HISTOGRAM = True
    # The default query mode; Dev Tools and ES|QL buffers have no language
    # of their own here.
    LANGUAGES = [Language.LUCENE]

    FIND_PATHS = {
        NodeType.INDEX: [["*"]],
        NodeType.FIELD: [["*", "mappings", "*"]],
        NodeType.ALIAS: [["*", "aliases", "*"]],
    }

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
            default=_DEFAULT_TIME_FIELD,
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
Setting any of the five to an empty string restores its default.

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

    def __init__(
        self,
        params: dict[str, Any],
        client: elasticsearch.AsyncElasticsearch,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._client = client
        self._ever_connected = False
        self._session_values: dict[str, Any] = {
            p.key: p.default for p in self.SESSION_PARAMS
        }
        """Runtime SESSION_PARAMS values, seeded from their declared defaults."""

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "ElasticsearchDriver":
        return cls(params, cls._open(params), settings)

    @staticmethod
    def _open(params: dict[str, Any]) -> elasticsearch.AsyncElasticsearch:
        host = params.get("host", "localhost")
        port = int(params.get("port", 9200))
        protocol = params.get("protocol", "https")
        kwargs: dict[str, Any] = {"hosts": [f"{protocol}://{host}:{port}"]}
        username = params.get("username")
        password = params.get("password")
        if username and password:
            kwargs["basic_auth"] = (username, password)
        return elasticsearch.AsyncElasticsearch(**kwargs)

    async def reconnect(self) -> None:
        await self._client.close()
        self._client = self._open(self.params)
        self._ever_connected = False

    async def disconnect(self) -> None:
        await self._client.close()

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        try:
            result = await self._execute(query)
            self._ever_connected = True
            return result
        except Exception as exc:
            if isinstance(exc, elasticsearch.ConnectionError):
                if self._ever_connected:
                    raise ConnectionLostError(str(exc)) from exc
                raise DriverError(str(exc)) from exc
            raise DriverError(str(exc)) from exc

    async def _execute(self, query: str) -> ReadResult:
        mode = self.params.get("query_mode", "lucene")
        if mode == "lucene":
            return await self._execute_lucene(query)
        elif mode == "dev_tools":
            return await self._execute_dev_tools(query)
        elif mode == "esql":
            return await self._execute_esql(query)
        else:
            raise DriverError(f"Unknown query_mode: {mode!r}")

    async def _execute_lucene(self, query: str) -> ReadResult:
        kwargs, applied = self._lucene_search(query)
        kwargs["size"] = _DEFAULT_SEARCH_SIZE
        if sort := self._sort_clause():
            kwargs["sort"] = sort
            applied.append(f"sort={json.dumps(sort)}")
        log_query(logger, f"{query} [{' '.join(applied)}]" if applied else query)
        resp = await self._client.search(**kwargs)
        return self._hits_to_result(resp)

    def _lucene_search(self, query: str) -> tuple[dict[str, Any], list[str]]:
        """The `search` arguments selecting the documents a Lucene query names.

        Returns the index and query part of the call — no size or sort — and
        the list of session settings folded in, for the query log.
        """
        if " | " not in query:
            raise DriverError(
                "Query must be in the format: <index> | <query>\n"
                "Example: orders | status:open AND total:>50"
            )
        index, _, lucene = query.partition(" | ")
        kwargs: dict[str, Any] = {"index": index.strip()}
        applied = []
        # The session time range cannot ride along with `q` (a URI parameter),
        # so a bounded search restates the Lucene string as a `query_string`
        # inside a bool whose filter carries the range.
        time_filter = self._time_filter()
        if time_filter is None:
            kwargs["q"] = lucene.strip()
        else:
            kwargs["query"] = {
                "bool": {
                    "must": [{"query_string": {"query": lucene.strip()}}],
                    "filter": [time_filter],
                }
            }
            applied.append(f"range={json.dumps(time_filter['range'])}")
        return kwargs, applied

    async def _execute_dev_tools(self, query: str) -> ReadResult:
        _VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"}
        lines = query.strip().splitlines()
        tokens = lines[0].strip().split(None, 1)
        if len(tokens) != 2 or tokens[0].upper() not in _VALID_METHODS:
            raise DriverError(
                "Dev Tools query must be in Kibana Dev Tools format:\n"
                "  METHOD /path\n"
                "  {optional body}\n"
                "Example:\n"
                "  GET /orders/_search\n"
                '  {"query": {"match_all": {}}}'
            )
        method, path = tokens[0].upper(), tokens[1].strip()
        body_str = "\n".join(lines[1:]).strip()
        body, headers = self._parse_body(body_str)
        if isinstance(body, dict) and "_search" in path:
            body.setdefault("size", _DEFAULT_SEARCH_SIZE)
        log_query(logger, query)
        raw = await self._client.perform_request(
            method, path, body=body, headers=headers
        )
        resp = raw.body if hasattr(raw, "body") else raw
        if isinstance(resp, dict) and "error" in resp:
            error = resp["error"]
            status = resp.get("status", "error")
            if isinstance(error, dict):
                raise DriverError(
                    f"Elasticsearch error ({status}): [{error.get('type', 'error')}] {error.get('reason', error)}"
                )
            raise DriverError(f"Elasticsearch error ({status}): {error}")
        if isinstance(resp, dict) and "hits" in resp:
            return self._hits_to_result(resp)
        if isinstance(resp, dict):
            return flatten_docs(list(resp.keys()), [[resp[k] for k in resp]])  # ty: ignore[invalid-argument-type]
        return ReadResult(columns=["response"], rows=[[str(resp)]], rows_total=1)

    async def _execute_esql(self, query: str) -> ReadResult:
        effective = self._apply_esql_settings(query)
        log_query(logger, effective)
        try:
            resp = await self._client.esql.query(query=effective, format="json")
        except elasticsearch.ApiError as exc:
            raise _esql_error(exc) from exc
        columns = [col["name"] for col in resp["columns"]]
        rows = resp["values"]
        return ReadResult(columns=columns, rows=rows, rows_total=len(rows))

    async def histogram(self, query: str, buckets: int) -> HistogramResult:
        mode = self.params.get("query_mode", "lucene")
        if mode == "dev_tools":
            raise DriverError(
                "Histograms are not available in Dev Tools mode — a Dev Tools "
                "request is sent exactly as written. Use the Lucene or ES|QL "
                "query mode, or add a date_histogram aggregation to the request."
            )
        if mode not in ("lucene", "esql"):
            raise DriverError(f"Unknown query_mode: {mode!r}")
        try:
            if mode == "lucene":
                return await self._histogram_lucene(query, buckets)
            return await self._histogram_esql(query, buckets)
        except elasticsearch.ConnectionError as exc:
            if self._ever_connected:
                raise ConnectionLostError(str(exc)) from exc
            raise DriverError(str(exc)) from exc
        except elasticsearch.ApiError as exc:
            raise (
                _esql_error(exc) if mode == "esql" else DriverError(str(exc))
            ) from exc

    async def _histogram_lucene(self, query: str, buckets: int) -> HistogramResult:
        """Bucket a Lucene query's matches with an `auto_date_histogram`.

        Elasticsearch picks a round interval yielding at most `buckets`
        buckets over the span the matches cover, and fills the gaps with
        empty ones — so the session range needs no special handling: set, it
        is already in the query; unset, the span is the data's own.
        """
        field = self._time_field()
        kwargs, applied = self._lucene_search(query)
        kwargs["size"] = 0
        kwargs["track_total_hits"] = False
        kwargs["aggs"] = {
            "histogram": {"auto_date_histogram": {"field": field, "buckets": buckets}}
        }
        applied.append(f"histogram={field}/{buckets}")
        log_query(logger, f"{query} [{' '.join(applied)}]")
        resp = await self._client.search(**kwargs)
        self._ever_connected = True
        agg = resp.get("aggregations", {}).get("histogram", {})
        return HistogramResult(
            field=field,
            interval=str(agg.get("interval") or ""),
            buckets=[
                HistogramBucket(time=int(b["key"]), count=int(b["doc_count"]))
                for b in agg.get("buckets", [])
            ],
        )

    async def _histogram_esql(self, query: str, buckets: int) -> HistogramResult:
        """Bucket an ES|QL query's rows with `STATS COUNT(*) BY BUCKET(...)`.

        The query is kept up to its first `STATS` — the last point where the
        rows are still documents and the time field is still in scope — with
        the session time range spliced in as for `execute`. `BUCKET`'s
        target-count form needs the span it divides, so a bound the session
        leaves open is first read off the data with `MIN`/`MAX`. ES|QL emits
        no bucket for an empty interval; those are filled in afterwards.
        """
        esql = query.strip()
        commands = _split_commands(esql)
        if not _ESQL_SOURCE_RE.match(esql[slice(*commands[0])]):
            raise DriverError("A histogram needs a query that reads an index (FROM …)")
        if clause := self._esql_time_clause():
            at = commands[0][1]
            esql = f"{esql[:at]} | WHERE {clause}{esql[at:]}"
            commands = _split_commands(esql)
        for start, end in commands:
            if _ESQL_STATS_RE.match(esql[start:end]):
                esql = esql[:start].rstrip().rstrip("|").rstrip()
                break
        field = self._time_field()
        ident = _esql_identifier(field)
        bounds = self._time_bounds()
        lo, hi = bounds.get("gte"), bounds.get("lte")
        if lo is None or hi is None:
            extent = await self._esql(
                f"{esql} | STATS lo = MIN({ident}), hi = MAX({ident}) | LIMIT 1"
            )
            row = extent["values"][0] if extent["values"] else [None, None]
            lo = lo or row[0]
            hi = hi or row[1]
            if lo is None or hi is None:
                return HistogramResult(field=field, interval="", buckets=[])
        resp = await self._esql(
            f"{esql} | STATS count = COUNT(*) BY time = BUCKET({ident}, {buckets},"
            f' "{lo}", "{hi}") | SORT time | LIMIT {_ESQL_HISTOGRAM_LIMIT}'
        )
        raw = [
            HistogramBucket(time=_epoch_ms(time), count=int(count))
            for count, time in resp["values"]
            if time is not None
        ]
        return _fill_gaps(field, raw)

    async def _esql(self, esql: str) -> dict[str, Any]:
        """Run one ES|QL statement, logged, returning the raw JSON response."""
        log_query(logger, esql)
        resp = await self._client.esql.query(query=esql, format="json")
        self._ever_connected = True
        return resp  # ty: ignore[invalid-return-type]

    def _apply_esql_settings(self, query: str) -> str:
        """Fold the session time range and sort field into an ES|QL query.

        The time filter goes straight after the source command, where the
        timestamp field is still in scope (a `STATS` further down the pipe may
        drop it); the sort goes at the end, but ahead of a trailing `LIMIT` so
        the limit takes the first rows of the sorted result rather than sorting
        an arbitrary slice. Both are skipped when the query already says
        otherwise: an explicit `SORT` wins over the session setting, and a query
        drawing on no index at all (`ROW`, `SHOW`) is left alone entirely —
        neither the time field nor the sort field exists in its output.
        """
        esql = query.strip()
        commands = _split_commands(esql)
        if not _ESQL_SOURCE_RE.match(esql[slice(*commands[0])]):
            return esql
        if clause := self._esql_time_clause():
            at = commands[0][1]
            esql = f"{esql[:at]} | WHERE {clause}{esql[at:]}"
            commands = _split_commands(esql)
        sort_field = self._session_values.get("sort_field")
        if sort_field and not any(
            _ESQL_SORT_RE.match(esql[slice(*span)]) for span in commands
        ):
            order = str(self._session_values.get("sort_order") or "desc").upper()
            at = commands[-1][1]
            if len(commands) > 1 and _ESQL_LIMIT_RE.match(esql[slice(*commands[-1])]):
                at = commands[-2][1]
            sort = f" | SORT {_esql_identifier(str(sort_field))} {order}"
            esql = f"{esql[:at]}{sort}{esql[at:]}"
        return esql

    def _time_field(self) -> str:
        return str(self._session_values.get("time_field") or _DEFAULT_TIME_FIELD)

    def _time_bounds(self) -> dict[str, str]:
        """Resolved `gte`/`lte` bounds for the session time range, if any."""
        bounds = {}
        for key, op in (("time_from", "gte"), ("time_to", "lte")):
            if value := self._session_values.get(key):
                bounds[op] = _es_datetime(_resolve_time(str(value)))
        return bounds

    def _time_filter(self) -> dict[str, Any] | None:
        """The session time range as a `range` query clause, or None if unset."""
        bounds = self._time_bounds()
        return {"range": {self._time_field(): bounds}} if bounds else None

    def _esql_time_clause(self) -> str | None:
        """The session time range as an ES|QL boolean expression, or None."""
        field = _esql_identifier(self._time_field())
        bounds = self._time_bounds()
        parts = [
            f'{field} {op} TO_DATETIME("{value}")'
            for op, value in (
                (">=", bounds.get("gte")),
                ("<=", bounds.get("lte")),
            )
            if value
        ]
        return " AND ".join(parts) or None

    def _sort_clause(self) -> list[dict[str, Any]] | None:
        """The session sort field as a search `sort` clause, or None if unset."""
        field = self._session_values.get("sort_field")
        if not field:
            return None
        order = str(self._session_values.get("sort_order") or "desc")
        return [{str(field): {"order": order}}]

    async def set_session(self, values: dict[str, Any]) -> None:
        known = {p.key for p in self.SESSION_PARAMS}
        if unknown := sorted(set(values) - known):
            raise DriverError(f"Unknown session setting: {', '.join(unknown)}")
        defaults = {p.key: p.default for p in self.SESSION_PARAMS}
        updated = dict(self._session_values)
        for key, value in values.items():
            text = "" if value is None else str(value).strip()
            if not text:
                updated[key] = defaults[key]
                continue
            if key in ("time_from", "time_to"):
                _resolve_time(text)  # reject a bad bound here, not at query time
            elif key == "sort_order":
                text = text.lower()
                if text not in ("asc", "desc"):
                    raise DriverError(
                        f"Unknown sort_order: {value!r} (expected 'asc' or 'desc')"
                    )
            updated[key] = text
        self._session_values = updated

    def get_session(self) -> dict[str, Any]:
        return dict(self._session_values)

    @staticmethod
    def _parse_body(
        body_str: str,
    ) -> tuple[dict[str, Any] | bytes | None, dict[str, str] | None]:
        """Parse a Dev Tools request body into (body, headers).

        Uses raw_decode to consume one JSON object at a time so multi-line
        objects within a multi-doc payload (e.g. _msearch) are handled correctly.
        One document → JSON body; multiple documents → NDJSON bytes.
        """
        if not body_str:
            return None, None
        decoder = json.JSONDecoder()
        docs: list[Any] = []
        s = body_str.strip()
        while s:
            try:
                obj, idx = decoder.raw_decode(s)
            except json.JSONDecodeError as exc:
                raise DriverError(f"Invalid request body: {exc}") from exc
            docs.append(obj)
            s = s[idx:].strip()
        if len(docs) == 1:
            return docs[0], {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        ndjson = "\n".join(json.dumps(doc) for doc in docs) + "\n"
        return ndjson.encode(), {
            "Content-Type": "application/x-ndjson",
            "Accept": "application/json",
        }

    def _hits_to_result(self, resp: Any) -> ReadResult:
        hits = resp["hits"]["hits"]
        total = resp["hits"]["total"]
        rows_total = total["value"] if isinstance(total, dict) else int(total)
        if not hits:
            return ReadResult(columns=[], rows=[], rows_total=rows_total)
        docs = [{"_id": hit["_id"], **hit.get("_source", {})} for hit in hits]
        columns = list(dict.fromkeys(k for doc in docs for k in doc))
        rows = [[doc.get(col) for col in columns] for doc in docs]
        return flatten_docs(columns, rows, rows_total=rows_total)

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                log_query(logger, "cat.indices")
                resp = await self._client.cat.indices(
                    format="json", h="index", s="index"
                )
                return [
                    ExploreItem(name=entry["index"], type="index", expandable=True)  # ty: ignore[invalid-argument-type]
                    for entry in resp
                    if not entry["index"].startswith(".")  # ty: ignore[invalid-argument-type]
                ]
            case [_index]:
                return [
                    ExploreItem(name="mappings", type="group", expandable=True),
                    ExploreItem(name="aliases", type="group", expandable=True),
                ]
            case [index, "mappings"]:
                return [
                    ExploreItem(name=field, type=kind, expandable=False)
                    for field, kind in (await self._fields(index)).items()
                ]
            case [index, "aliases"]:
                log_query(logger, f"indices.get_alias {index}")
                resp = await self._client.indices.get_alias(index=index)
                aliases = resp.get(index, {}).get("aliases", {})
                return [
                    ExploreItem(name=alias, type="alias", expandable=False)
                    for alias in aliases
                ]
            case _:
                return []

    async def explore_describe(self, path: list[str]) -> EntityDescription | None:
        match path:
            case [index]:
                return EntityDescription(
                    name=index,
                    kind="document",
                    properties=[
                        FieldDescription(name=field, types=[kind])
                        for field, kind in (await self._fields(index)).items()
                    ],
                )
            case _:
                return None

    async def _fields(self, index: str) -> dict[str, str]:
        """The queryable fields of `index`, as dotted paths to their types.

        `index` may be a pattern or alias — a Lucene query names its indices
        that way — in which case the mappings of everything it matches are
        merged, in index order, the last word on a path's type winning.
        Every mapping is flattened as a query addresses it: `user.name` for a
        nested property, `message.keyword` for a multi-field.
        """
        log_query(logger, f"indices.get_mapping {index}")
        resp = await self._client.indices.get_mapping(index=index)
        fields: dict[str, str] = {}
        for name in sorted(resp):
            _flatten_fields(resp[name]["mappings"].get("properties", {}), "", fields)
        return fields


def _flatten_fields(props: dict[str, Any], prefix: str, into: dict[str, str]) -> None:
    """Flatten a mapping's `properties` into `into`, as dotted paths to types.

    A property with sub-`properties` and no type of its own is an `object`
    holder and listed as such; a multi-field's `fields` are listed under the
    parent's path like sub-properties are.
    """
    for name, info in props.items():
        path = f"{prefix}{name}"
        into[path] = info.get("type", "object")
        if children := info.get("properties"):
            _flatten_fields(children, f"{path}.", into)
        if multi := info.get("fields"):
            _flatten_fields(multi, f"{path}.", into)


def _resolve_time(value: str) -> datetime:
    """Resolve a session time bound to an absolute UTC datetime.

    Accepts `now`, an offset from now (`-1h`, `now-30m`, `+15s`), an ISO-8601
    timestamp, or a Unix timestamp in seconds. Resolving here rather than
    handing Elasticsearch its own date math keeps one syntax across the Lucene
    and ES|QL modes — ES|QL has no `now-1h` literal.

    Raises:
        DriverError: If the value is in none of those forms.
    """
    text = value.strip()
    now = datetime.now(UTC)
    if text.lower().startswith("now"):
        text = text[3:].strip()
        if not text:
            return now
    if match := _DURATION_RE.match(text):
        amount, unit = match.groups()
        return now + timedelta(seconds=float(amount) * _DURATION_UNITS[unit])
    if _EPOCH_RE.match(text):
        return datetime.fromtimestamp(float(text), tz=UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DriverError(
            f"Invalid time value {value!r} — expected 'now', an offset such as "
            "'-1h', an ISO-8601 timestamp, or a Unix timestamp"
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _es_datetime(when: datetime) -> str:
    """Format a datetime as the UTC millisecond form both ES and ES|QL read."""
    utc = when.astimezone(UTC)
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}.{utc.microsecond // 1000:03d}Z"


def _esql_identifier(name: str) -> str:
    """Quote a field name for ES|QL, leaving plain identifiers untouched."""
    if _PLAIN_IDENTIFIER_RE.match(name):
        return name
    return "`" + name.replace("`", "``") + "`"


def _split_commands(query: str) -> list[tuple[int, int]]:
    """Spans of the top-level `|`-separated commands of an ES|QL query.

    Each span excludes surrounding whitespace, so a command's end offset is
    exactly where a ` | NEW COMMAND` can be spliced in. Pipes inside string
    literals (including triple-quoted blocks) and backquoted identifiers do
    not split.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i < len(query):
        char = query[i]
        if char == '"':
            if query.startswith('"""', i):
                end = query.find('"""', i + 3)
                i = len(query) if end == -1 else end + 3
                continue
            i += 1
            while i < len(query):
                if query[i] == "\\":
                    i += 2
                    continue
                if query[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if char == "`":
            i += 1
            while i < len(query) and query[i] != "`":
                i += 1
            i += 1
            continue
        if char == "|":
            spans.append((start, i))
            start = i + 1
        i += 1
    spans.append((start, len(query)))
    return [
        (s + len(text) - len(text.lstrip()), e - (len(text) - len(text.rstrip())))
        for s, e in spans
        for text in [query[s:e]]
    ]


def _epoch_ms(value: Any) -> int:
    """Unix milliseconds for a datetime ES|QL returned (ISO string or number)."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def _fill_gaps(field: str, buckets: list[HistogramBucket]) -> HistogramResult:
    """Make `buckets` contiguous, inserting empty ones where ES|QL emitted none.

    The interval is taken as the smallest gap between neighbours, and gaps
    are filled only when every gap is a whole multiple of it — true of any
    fixed interval. A calendar interval (months vary in length) does not
    pass that test and is left as it came, with no interval reported.
    """
    if len(buckets) < 2:
        return HistogramResult(field=field, interval="", buckets=buckets)
    steps = [b.time - a.time for a, b in zip(buckets, buckets[1:])]
    step = min(steps)
    if step <= 0 or any(s % step for s in steps):
        return HistogramResult(field=field, interval="", buckets=buckets)
    filled: list[HistogramBucket] = [buckets[0]]
    for bucket in buckets[1:]:
        while filled[-1].time + step < bucket.time:
            filled.append(HistogramBucket(time=filled[-1].time + step, count=0))
        filled.append(bucket)
    return HistogramResult(field=field, interval=_duration(step), buckets=filled)


def _duration(ms: int) -> str:
    """Format a millisecond span the way ES names intervals: `5m`, `1h`, `1d`."""
    for unit, size in (("d", 86_400_000), ("h", 3_600_000), ("m", 60_000), ("s", 1000)):
        if ms % size == 0:
            return f"{ms // size}{unit}"
    return f"{ms}ms"


def _esql_error(exc: elasticsearch.ApiError) -> DriverError:
    """Turn an ES|QL API error into a DriverError, naming the likely cause.

    A server without the ES|QL endpoint matches `POST /_query` against its
    `/{index}` handlers instead, so the rejection is about the HTTP method
    rather than the query — unreadable unless it is spelled out.
    """
    message = str(exc)
    if "Incorrect HTTP method for uri" in message and "/_query" in message:
        return DriverError(
            "This server does not support ES|QL — it has no /_query endpoint. "
            "ES|QL requires Elasticsearch 8.11 or later; use the Lucene or Dev "
            "Tools query mode instead."
        )
    return DriverError(message)
