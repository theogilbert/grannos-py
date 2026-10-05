# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dev dependencies (run once)
uv sync --group dev --all-extras

# Run unit + integration tests (default pytest paths: tests/unit, tests/integration)
uv run pytest

# Run a single test file or test
uv run pytest tests/unit/test_dispatcher.py
uv run pytest tests/unit/test_dispatcher.py::test_name

# Lint, format, and type-check
uv run ruff check --fix
uv run ruff format
uv run ty check

# Run all CI checks in one go (includes external tests via Docker)
make ci

# Run only external tests (each script spins up a Docker container)
make external-tests
```

External tests (require a live service) live in `tests/external/`. The scripts in `scripts/` spin up the required service via Docker; `make ci` runs them all automatically.

## Architecture

Grannos is a **JSON-over-stdio server** that lets IDE clients (e.g. grannos.nvim) query databases. The process model is:

```
stdin → Server → Dispatcher → CachingDriver → BaseDriver implementation
                             ↓
                       ConnectionStore (per-connection state + idle timer)
```

**Wire protocol** (`protocol.py`): newline-delimited JSON. Each request `{id, method, params}` gets a response `{id, result, error}`. Progress notifications `{id, progress}` can be emitted before the final response. All types are dataclasses; `encode`/`decode` handle serialisation.

**Server** (`server.py`): reads stdin asynchronously, spawns one `asyncio.Task` per request. A lock serialises stdout writes. The `cancel` method cancels an in-flight task by request id.

**Dispatcher** (`dispatcher.py`): routes methods to handlers. `connect`/`disconnect`/`capabilities`/`driver.help` are connection-free; all others require a `connection_id`. Each connection is protected by a semaphore (`max_concurrency`) and an `IdleTimer` that auto-closes after inactivity. On `ConnectionLostError`, the dispatcher transparently reconnects and retries once.

**Drivers** (`drivers/`): each database backend subclasses `BaseDriver` and declares `LABEL`, `PARAMS`, `LANGUAGES`, `HELP`, and `DEFAULT_IDLE_TIMEOUT` as class attributes. The `drivers/__init__.py` registry lazy-imports drivers so missing optional packages don't break startup — only installed drivers appear in `capabilities`. To add a driver: implement `BaseDriver`, add it to `_REGISTRY`. Complex drivers may be packages (e.g. `drivers/oracle/`, `drivers/postgres/`, each with `driver.py` and `queries.py`) — the registry still imports only the outer package.

**Describe result types** (`protocol.py`): `explore_describe` returns a `DescribeResult` union — `TableDescription`, `IndexDescription`, `IndicesDescription`, `ColumnDescription`, or `ColumnsDescription` (discriminate on the `type` field). `ColumnDescription` uses `data_type` (not `type`) for the SQL data type to avoid collision with the discriminator. `ColumnDescription.exclusive_indices` / `composite_indices` carry full `IndexDescription` objects, not booleans like the lighter `ColumnInfo` inside `TableDescription`.

**Symbol lookup** (`explore_find.py`): the `explore.find` method turns a symbol name, a node type, and a list of `SearchScope` ancestors into `explore.describe` paths — letting a client resolve the symbol under the cursor without searching its own cache. Generic: each driver declares *where* each kind of node lives in its tree as `BaseDriver.FIND_PATHS` path templates (`"*"` = one level of children, any other segment = a literal group name), and `walk_find` expands them through the driver's `explore_list`, pruning each wildcard level by the scopes. Scopes sharing a `type` are alternatives, scopes of different types compound. `MAX_LIST_CALLS` caps the fan-out of an under-scoped search.

`CachingDriver.explore_find` resolves in two passes: (1) the **find cache** — `ConnectionCache.get_find`, keyed by `find_key(node_type, name, scopes)`; (2) the wrapped driver — its own `explore_find` where it has one, else `walk_find` over its `explore_list`. Both outcomes are written to the find cache. A find neither reads nor writes the list/describe cache: the two answer different questions (the list cache holds what a *path* contains, while a walk lists only the levels its templates pass through and stops at the one it was searching), so a browsed tree cannot settle a search and a search cannot fill in the tree. The find cache is persisted in the same JSON file as `list`/`describe`, so it survives restarts — the whole point, since a driver's catalog lookup populates nothing else. Because it is keyed by *search* rather than by path it cannot be evicted per-prefix: any `reset` clears it whole, in both directions (a find may now point into the reset subtree, or newly match something created there). A driver overriding `explore_find` raises `FindNotSupported` for the node types its query does not cover; Oracle's override (`drivers/oracle/queries.py`, the `fetch_find_*` functions) resolves table/view/column/index across every schema in one dictionary query, repeating `fetch_schemas`' non-system owner filter as a subquery so a find never returns a path the tree does not contain, and leaves `schema` to the walker. Its table/view branch also matches `ALL_SYNONYMS`, returning the *base* object's path — the tree holds no synonyms, and since the describe cache is keyed by path, every synonym of one table then shares that table's single describe. A synonym that is chained, dangling, or over a db link resolves to nothing rather than to a path the tree lacks.

Node types are the `NodeType` enum in `protocol.py` — one flat vocabulary across all drivers, shared by `explore.list`'s `type` field, `explore.find`'s `type` param, and `FIND_PATHS`. Note `ExploreItem.type` is still typed `str`: several drivers overload it on leaf field nodes to report the field's *data* type (`"int4"`, `"varchar2"`) rather than `NodeType.COLUMN`.

**Method results** (`protocol.py`): every dispatcher handler returns a dataclass (`ExploreListResult`, `ExecuteReadResult`, …) unioned as `MethodResult`, not an ad-hoc dict — these *are* the wire shape, since `encode` serialises them field-for-field. They are distinct from the driver-level types (`ReadResult`, `WriteResult`) because a method result may add fields the driver knows nothing about, such as the server-measured `duration_ms`. `session.get` is the sole exception, returning a bare dict whose shape its driver's `SESSION_PARAMS` define.

**Execute messages** (`protocol.py`): `ReadResult`/`WriteResult` — and the `Execute*Result` method results they feed — carry `messages: list[ExecuteMessage]`, out-of-band text a *successful* statement produced (Oracle `DBMS_OUTPUT` lines, PL/SQL compilation errors; Postgres `RAISE NOTICE` and SQL Server `PRINT` would fit the same field). `MessageLevel` is deliberately just `info`/`warning` — a failed request is reported through the response's `error`, so there is no `error` level. Position is structured as `line`/`col` (1-indexed, into the query string as submitted) rather than baked into `text`, so clients never parse it back out. `explore.preview` deliberately does not carry messages: its query is the server's, not the user's.

**Query logging** (`log.py`): `log_query(logger, statement, binds)` writes one DEBUG line per statement a driver sends, so `grannos --log -v` shows exactly what reached the database — user statements and the catalog queries that `explore.list`/`describe`/`find`/`diagram` generate alike. Whitespace is collapsed so a statement stays one grep-able line, and `truncate`/`LOG_CAP` (shared with `server.py`'s request/response lines) caps it. Each driver funnels its calls through one chokepoint rather than logging at every call site — `_exec(cur, sql, binds)` in the cursor-based drivers (`oracle/queries.py`, `postgres/queries.py`, `sqlserver.py`), `_sql` in `sqlite.py`/`duckdb.py`, `_cypher` in `neo4j.py`, `_get` in `prometheus.py`, `_run` in `s3.py`; `mongodb.py` and `elasticsearch.py` log per call site, their client calls being too heterogeneous to funnel. Every chokepoint takes `private=True`, set at the one site that runs the *user's* statement: a catalog query's binds are schema and object names, but the user's are their data and never reach the log.

**Request cancellation** (`drivers/base.py`): `cancel` cancels the request's task, which is enough for drivers on an async client — psycopg and python-oracledb send the server a cancel themselves (adding an explicit one on top hangs Oracle). The thread-based drivers (SQLite, DuckDB, SQL Server, S3) instead run every blocking call through `run_blocking(fn, interrupt)`. A cancel there does three things: it raises a per-call flag that `check_cancelled()` turns into `CallCancelled` (a `BaseException`, so a best-effort `except Exception` doesn't swallow it), it calls `interrupt` to abort the running statement, and it then *waits for `fn` to return* before re-raising, so the connection is idle again before the semaphore hands it to the next request. `check_cancelled()` is called from the same chokepoints as `log_query` (see above), and S3 calls it from a botocore `before-call` hook plus between files of a transfer. The interrupt is `conn.interrupt` for SQLite/DuckDB. mssql-python has no `SQLCancel`, so SQL Server's interrupt is `_sever_link`, which shuts down the connection's TCP socket (found by diffing `/dev/fd` across the connect, with pooling disabled to make that possible). **That kills the connection**: the next request reconnects through `ConnectionLostError`, and any open transaction rolls back. If the socket couldn't be identified, a warning is logged at connect time and a cancel then waits out the running statement. S3 has no interrupt at all, so a cancel finishes the request or file in flight. Connects can't be interrupted, so they use `open_blocking(fn, close)`: a cancel returns immediately, and the connection that later arrives is passed to `close`, so it doesn't leak.

**Explore cache** (`explore_cache.py`): `CachingDriver` wraps any `BaseDriver` and caches `explore_list` / `explore_describe` results in a per-connection JSON file under `~/.cache/grannos/`. The cache file is keyed by a SHA-256 of the non-sensitive connection params. Passwords and other `secret` params are never written to disk.

**Diagram rendering** (`diagram/`): the `explore.diagram` method (`dispatcher.py:_handle_explore_diagram`) renders a table and every table reachable from it via foreign keys as an ASCII box-and-connector diagram. The pipeline is a straight-line pass through the package: `graph.discover` walks `explore_describe` calls to find all connected tables/edges → `layout.compute_layout` assigns each table a hub-centered column/side via BFS → `place.place` turns that abstract layout into concrete box rectangles with overprovisioned routing channels → `route.route` A*-searches an orthogonal path per edge across a character grid, then `route.compact` strips unused channel space back out → `canvas.Canvas` blits the boxes and edges onto a character grid and renders it to text. Every table/column name drawn is tracked as a `DiagramRegion` (byte-offset span, see `protocol.py`) so a client can map a cursor position in the rendered diagram back to an `explore.describe` path. The place→route pair is not one-shot: `route.route` routes every edge or none (raising `NoRouteError` when one has no anchor or lane left), and `_place_and_route` re-places the same graph with a roomier `place.Spacing` until they all fit (`_spacing_ladder` — more rows first, since a `box_gap` of 1 costs every stacked box the anchors on the side facing its neighbour, then channels widened up to a lane per edge in the graph). A diagram is never drawn with a relationship missing: exhausting the ladder raises `DiagramError`. Column types are drawn without their length/precision modifier (`place._base_type`) — `explore.describe` keeps the full type.

## Test layout

| Path | Purpose |
|------|---------|
| `tests/unit/` | Fast, no I/O, mock all drivers |
| `tests/integration/` | Run against embedded/in-process databases (DuckDB, SQLite) |
| `tests/external/` | Require a live external service; not in default `pytest` paths |

The project uses `pytest-asyncio` in `auto` mode — all async test functions are picked up automatically.
