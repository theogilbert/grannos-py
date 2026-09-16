# Drivers

Each driver is loaded on demand. Only the packages required for the drivers you
actually use need to be installed.

---

## SQLite

**Install:** none (stdlib)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `database` | yes | — | File path or `:memory:` |

**Queries:** Standard SQL. Positional bind parameters use `?` placeholders.

```sql
SELECT * FROM users WHERE age > ?
```

**Explore tree:**

```
(root)
└── <table|view>
    ├── columns       → name, type
    ├── indices       → index name
    └── foreign_keys  → "col → ref_table.ref_col"
```

`explore.describe` is supported on:
- `[table|view]` — returns full column metadata (name, type, nullability, primary key flag)
- `[table, "indices", index_name]` — returns an `IndexDescription` with key fields (name + direction), `unique`, and `condition` (the SQL WHERE clause for partial indexes)

---

## DuckDB

**Install:** `pip install 'grannos-py[duckdb]'`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `database` | no | `:memory:` | File path or `:memory:` |

**Queries:** Standard SQL. Positional bind parameters use `?` placeholders.

```sql
SELECT * FROM read_parquet('/path/to/file.parquet')
SELECT * FROM read_csv('/path/to/file.csv', header = true)
SELECT * FROM 'glob/**/*.parquet'
```

**Explore tree:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns       → name, type
        ├── indices       → index name
        └── foreign_keys  → "col → ref_table.ref_col"
```

`explore.describe` is supported on:
- `[schema, table]` — returns full column metadata (name, type, nullability, primary key flag)
- `[schema, table, "indices", index_name]` — returns an `IndexDescription` with key fields (name + direction), `unique`, `entity` (table name), and `condition` (the SQL WHERE clause for partial indexes)

---

## SQL Server

**Install:** `pip install mssql-python`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `host` | yes | — | Server hostname or IP |
| `port` | yes | `1433` | TCP port |
| `database` | yes | — | Database name |
| `user` | yes | — | Login name |
| `password` | yes | — | Password (masked) |
| `applicationIntent` | yes | — | `READ_WRITE` or `READ_ONLY` |

**Queries:** Standard T-SQL. Positional bind parameters use `?` placeholders.

```sql
SELECT * FROM dbo.orders WHERE status = ?
```

**Explore tree:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indices      → name, type (e.g. CLUSTERED)
```

System schemas (`sys`, `INFORMATION_SCHEMA`, `guest`, `db_*`) are hidden.

`explore.describe` is supported on `[schema, table]` paths and returns full
column metadata (name, type, nullability, default).

---

## Oracle

**Install:** `pip install oracledb` — thin mode, no Oracle Instant Client required.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `host` | yes | — | Server hostname or IP |
| `port` | yes | `1521` | Listener port |
| `service_name` | yes | — | Database service name |
| `user` | yes | — | Username |
| `password` | yes | — | Password (masked) |

**Queries:** Standard SQL. Positional bind parameters use `:1`, `:2`, … placeholders.

```sql
SELECT * FROM employees WHERE department_id = :1 AND hire_date > :2
```

**Explore tree:**

```
(root)  ← non-system schemas (ALL_USERS where ORACLE_MAINTAINED = 'N')
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indexes      → name, index type
```

`explore.describe` is supported on `[schema, table]` paths and returns full
column metadata (name, type, nullability, primary key flag, default).

---

## Neo4j

**Install:** `pip install neo4j`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `uri` | yes | `bolt://localhost:7687` | Bolt URI |
| `user` | yes | `neo4j` | Username |
| `password` | yes | — | Password (masked) |
| `database` | no | `neo4j` | Database name |

**Queries:** Cypher. Positional bind parameters are referenced as `$0`, `$1`, …

```cypher
MATCH (u:User {name: $0})-[:BOUGHT]->(p:Product) RETURN u, p
```

Results are serialized and flattened: nodes expand to `col._labels`, `col.prop`,
…; relationships expand to `col._type`, `col.prop`, …

**Explore tree:**

```
(root)
├── entities       → <label>  → property names (sampled from existing nodes)
├── relationships  → <type>   → property names (sampled from existing relationships)
└── indexes        → index name
```

`explore.describe` is supported on `["indexes", index_name]` paths and returns an
`IndexDescription` with the indexed properties, `unique`, and `entity` (the node label or
relationship type the index operates on). The `direction` field on each `IndexKeyField`
holds the Neo4j index type (`RANGE`, `TEXT`, `POINT`, …).

---

## Elasticsearch

**Install:** `pip install elasticsearch aiohttp`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `host` | yes | — | Server hostname or IP |
| `port` | yes | `9200` | HTTP port |
| `username` | no | — | Username |
| `password` | no | — | Password (masked) |
| `query_mode` | yes | `lucene` | Query language: `lucene`, `dev_tools`, or `esql` |

**Session settings** (`session.set` / `session.get`, no reconnect needed):

| Setting | Default | Description |
|---------|---------|-------------|
| `time_field` | `@timestamp` | Date field the time range filters on |
| `time_from` | — | Lower bound of the time range, inclusive |
| `time_to` | — | Upper bound of the time range, inclusive |
| `sort_field` | — | Field to order results by |
| `sort_order` | `desc` | `asc` or `desc` |

**Queries:** Prefix with the target index name (pattern or alias) and ` | `.
This prefix is not needed in `esql` mode, where the target index is named
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

**Documents over time:** the driver answers `execute.histogram` for Lucene
and ES|QL queries, counting the matching documents per bucket of
`time_field` within the session time range. The bucket width is the range
divided into the requested count — not a round interval — so a chart fills
every column asked for (or one fewer, since buckets are aligned to the
epoch rather than to the range's start). A bound the session leaves open is
first read off the matches with `min`/`max`. In Lucene mode the chart is a
fixed-interval `date_histogram` aggregation on the same search (with
`size: 0`), `extended_bounds` pinning it to the range. In ES|QL mode the
query is kept up to its first `STATS` and
`| STATS COUNT(*) BY BUCKET(<time_field>, <width> milliseconds)` appended;
empty buckets, which ES|QL omits, are filled in out to the range's edges.
Dev Tools requests are sent as written and cannot be charted.

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

**Explore tree:**

```
(root)
└── <index>
    ├── mappings  → field name and type
    └── aliases   → alias names
```

System indices (names starting with `.`) are hidden. Mappings are listed
flat, as a Lucene query addresses them: `user.name` for a nested property,
`message.keyword` for a multi-field, the holder itself listed as `object`.
The `<index>` segment may be a pattern or alias as well as a name — a client
completing `logs-* | …` lists `["logs-*", "mappings"]` — in which case the
mappings of every index matched are merged.

`explore.describe` is supported on `[index]` paths and returns the same field
metadata (name, type).

The driver declares the `lucene` language: the `<index> | <query_string>`
form of its default query mode.

---

## Prometheus

**Install:** `pip install "grannos-py[prometheus]"` (or `pip install aiohttp`)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `url` | yes | `http://localhost:9090` | Base URL of the Prometheus server |
| `username` | no | — | Username (HTTP basic auth) |
| `password` | no | — | Password (masked) |
| `query_mode` | yes | `instant` | Query mode: `instant` or `range` |

**Queries:** PromQL, evaluated via the connection's configured query mode.

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
Scalar/string results return a single `timestamp`/`value` row.

**Explore tree:**

```
(root)
├── metrics
│   └── <metric>
│       └── <label>
├── jobs
│   └── <job>
├── configuration
└── runtime
```

Requires Prometheus >= 2.24 (`/api/v1/labels` with `match[]` support).

`explore.describe` is supported on `["metrics", metric]` paths and returns label
metadata (name, up to 3 sampled values). `kind` and `comment` are populated from
`/api/v1/metadata` (type and help text) when the server exposes it.
`["metrics", metric, label]` re-fetches a single label standalone (up to 3
sampled values).

`configuration`, `runtime`, and every node under `jobs` are leaf nodes — the
explore tree goes no deeper than a job; describing it is the detailed view.
- `["configuration"]` returns a `RawDocument` (`filetype: "yaml"`) holding the
  running configuration (`/api/v1/status/config`).
- `["runtime"]` returns a `GenericRecordDescription` (`kind: "prometheus.runtime"`)
  with CLI flags, runtime info (retention, start time, storage stats, …), and
  build info as label/value fields (labeled `Flag: *`, `Runtime: *`, `Build: *`),
  merged from `/api/v1/status/flags`, `/api/v1/status/runtimeinfo`, and
  `/api/v1/status/buildinfo`.
- `["jobs", job]` returns an array of `GenericRecordDescription`
  (`kind: "prometheus.target"`), one per target in the job (`/api/v1/targets`,
  grouped by the `job` label), named after its instance, with fields in this
  order: `URL` (the target's externally-reachable `globalUrl`), `Interval`,
  `Timeout`, `Last Scrape` (a relative age — `5s ago`, `3m ago`, `2h ago`),
  `Status` (`✓`/`✗`/`?` for up/down/unknown), and `Last Scrape Duration`
  (whole milliseconds, e.g. `12ms`). `Last Error` is appended (on every
  target's record) only when at least one target in the job has a non-empty
  error. Target labels and the scrape pool name are not exposed.

---

## MongoDB

**Install:** `pip install pymongo`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `uri` | yes | Connection URI (embed credentials and `authSource` here if needed) |
| `username` | no | Username (can also be embedded in the URI) |
| `password` | no | Password (masked; can also be embedded in the URI) |

**Queries:** MongoDB Extended JSON command objects. `"db"` is required and
names the target database. The top-level operation key names the collection.

**Read:**

```json
{"find": "orders", "db": "mydb", "filter": {"status": "open"}, "sort": {"createdAt": -1}, "limit": 100}
```

`filter`, `sort`, `projection`, and `limit` are all optional. `find` defaults
to a limit of 1000 rows when `"limit"` is omitted.

```json
{"aggregate": "orders", "db": "mydb", "pipeline": [
  {"$group": {"_id": "$status", "total": {"$sum": "$amount"}}},
  {"$sort": {"total": -1}}
]}
```

**Insert:**

```json
{"insertOne": "users", "db": "mydb", "document": {"name": "Alice", "age": 30}}
```

```json
{"insertMany": "users", "db": "mydb", "documents": [{"name": "Alice"}, {"name": "Bob"}]}
```

**Update:**

```json
{"updateOne": "users", "db": "mydb", "filter": {"name": "Alice"}, "update": {"$set": {"age": 31}}}
```

```json
{"updateMany": "users", "db": "mydb", "filter": {"role": "guest"}, "update": {"$set": {"active": false}}}
```

**Delete:**

```json
{"deleteOne": "orders", "db": "mydb", "filter": {"status": "cancelled"}}
```

```json
{"deleteMany": "orders", "db": "mydb", "filter": {"status": "cancelled"}}
```

Document values support Extended JSON, so BSON types that plain JSON can't
express — dates, ObjectIds, decimals — can be written directly:

```json
{"updateOne": "events", "db": "mydb",
 "filter": {"_id": {"$oid": "5f8d0d55b54764421b7156c0"}},
 "update": {"$set": {"occurredAt": {"$date": "2024-01-01T00:00:00Z"}}}}
```

**Collections and indexes:**

```json
{"createCollection": "events", "db": "mydb"}
{"dropCollection": "old_events", "db": "mydb"}
{"createIndex": "users", "db": "mydb", "keys": {"email": 1}, "options": {"unique": true}}
{"dropIndex": "users", "db": "mydb", "name": "email_1"}
```

`options` is optional for both `createCollection` and `createIndex` and is
passed through to the underlying pymongo call.

Results are flattened with dot-notation column names (`address.city`,
`address.zip`).

**Explore tree:**

```
(root)
└── <database>
    └── <collection>
        ├── fields   → top-level field names (sampled from up to 10 documents)
        └── indexes  → index names
```

`explore.describe` is supported on `[database, collection, "indexes", index_name]` paths
and returns an `IndexDescription` with key fields (name + direction), `unique`, and `condition`
(the `partialFilterExpression` serialized as JSON, if set).
