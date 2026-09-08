"""Oracle driver — requires: pip install oracledb"""

import asyncio
import csv
import logging
import re
from typing import Any

import oracledb
from oracledb import AsyncConnection

from ...protocol import (
    DescribeResult,
    DriverParam,
    EntityDescription,
    ExecuteMessage,
    ExploreItem,
    FieldDescription,
    IndexDescription,
    Language,
    MessageLevel,
    NodeType,
    ParamType,
    ReadResult,
    SearchScope,
    TableReference,
    WriteResult,
)
from ..base import (
    BaseDriver,
    ConnectionLostError,
    DriverError,
    DriverSettings,
    FindNotSupported,
    build_column_samples,
    find_reference,
    group_references_by_column,
    group_references_by_ref_column,
)
from .load import (
    LoadCommand,
    LoadOptions,
    build_insert_statement,
    parse_load,
    read_rows,
    validate_identifier,
)
from .queries import (
    _exec,
    _executemany,
    MAX_DBMS_OUTPUT_LINES,
    ColumnDetail,
    apply_metadata_transform,
    enable_dbms_output,
    fetch_compilation_errors,
    fetch_dbms_output,
    build_column_index_lists,
    build_preview_query,
    fetch_all_column_comments,
    fetch_column_details,
    fetch_column_names_and_types,
    fetch_explain_plan,
    fetch_find_columns,
    fetch_find_indexes,
    fetch_find_tables_and_views,
    fetch_index_ddl,
    fetch_index_fields_for_index,
    invalidate_cache,
    fetch_index_fields_for_table,
    fetch_index_meta,
    fetch_index_metas_for_table,
    fetch_incoming_references,
    fetch_index_names_and_types,
    fetch_join_tables_for_index,
    fetch_nls_formats,
    fetch_join_tables_for_table,
    fetch_column_sample,
    fetch_outgoing_references,
    fetch_pk_columns,
    fetch_schemas,
    fetch_table_comment,
    fetch_table_sample_rows,
    fetch_tables_and_views,
    render_lob,
    set_nls_format,
)

logger = logging.getLogger(__name__)


class OracleDriver(BaseDriver):
    """Oracle driver backed by python-oracledb (thin mode, no Instant Client required).

    Args:
        params: Connect request fields (``host``, ``port``, ``service_name``, ``user``, ``password``).
        conn: Open oracledb async connection. Use :meth:`create` instead of constructing directly.
        has_oracle_maintained: True when connected to Oracle 12c+.
    """

    LABEL = "Oracle"
    LANGUAGES = [Language.SQL]

    FIND_PATHS = {
        NodeType.SCHEMA: [["*"]],
        NodeType.TABLE: [["*", "*"]],
        NodeType.VIEW: [["*", "*"]],
        NodeType.COLUMN: [["*", "*", "columns", "*"]],
        NodeType.INDEX: [["*", "*", "indexes", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=1521),
        DriverParam(key="service_name", type=ParamType.STRING, label="Service Name"),
        DriverParam(key="user", type=ParamType.STRING, label="User"),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
    ]

    HELP: str = """\
## Oracle

Uses the `oracledb` driver in thin mode — no Oracle Instant Client required.

### Queries

Standard SQL.

```sql
SELECT * FROM employees WHERE department_id = 10 AND hire_date > DATE '2024-01-01'
SELECT e.id, d.name FROM employees e JOIN departments d ON d.id = e.department_id
INSERT INTO employees (department_id, name) VALUES (10, 'Alice')
ALTER SESSION SET NLS_LENGTH_SEMANTICS = CHAR
```

### Importing from a file

```
LOAD <table> [(column_list)] [FROM] '/local/path' [(options)]
```

Oracle has no `\\copy`: SQL*Loader is a separate binary and an external table
reads the *database server's* filesystem, so this does what SQLcl's `LOAD`
does — the file is read on the machine running grannos-py, parsed here, and
sent as batched INSERTs. `FROM` is optional, so SQLcl's
`LOAD <table> '/local/path'` works too.

Target columns come from the explicit `(column_list)` if given, else from the
file's header row if `HEADER` is set, else from the table's own column order.

**Options** — comma-separated inside the parentheses, in any order:

- `FORMAT csv` — the only format, and the default.
- `HEADER` — the first row names the columns instead of carrying data.
- `DELIMITER ','` — field separator, one character; `'\\t'` means tab.
- `QUOTE '"'` — quote character.
- `NULL 'text'` — field value to load as NULL. An empty field, quoted or not,
  is already NULL: Oracle stores a zero-length string that way.
- `SKIP 0` — leading lines to discard before the header.
- `BATCH 1000` — rows per round-trip.
- `ENCODING 'utf-8'` — how to decode the file.
- `DATEFORMAT 'YYYY-MM-DD'` — how the file writes its dates. Values are bound
  as text and converted by Oracle through the session's NLS format models,
  which default to `DD-MON-RR` and reject an ISO date with `ORA-01843`; this
  sets them for the load and puts the session's back afterwards. It covers
  DATE *and* TIMESTAMP columns, which convert through separate models.
- `TIMESTAMPFORMAT 'YYYY-MM-DD HH24:MI:SS'` — overrides `DATEFORMAT` for
  TIMESTAMP columns, for a file whose timestamps carry a time and whose dates
  don't.

**Examples:**

```sql
LOAD employees FROM '/tmp/employees.csv' (FORMAT csv, HEADER)
LOAD employees FROM '/tmp/employees.csv' (HEADER, DATEFORMAT 'YYYY-MM-DD')
LOAD hr.employees (name, department_id) FROM '/tmp/employees.csv' (HEADER, BATCH 5000)
LOAD employees FROM '/tmp/employees.tsv' (DELIMITER '\\t', SKIP 2, NULL 'NA')
```

A row Oracle rejects aborts the load, naming the line it occupies in the file
(counting from 1, header included); rows already sent stay uncommitted, so
`ROLLBACK` undoes a partial load.

### Resources

```
(root)  ← non-system schemas (ALL_USERS where ORACLE_MAINTAINED = 'N')
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indexes      → name, index type
```

Describing a table or view returns column metadata (name, type, nullability,
primary key flag, default). Describing an index returns its key fields,
direction, and uniqueness.

### Session properties

Run `ALTER SESSION SET <property> = <value>` like any other statement. It's remembered per-connection and automatically re-applied
if the connection is silently reconnected (e.g. after an idle timeout).

### PL/SQL

Anonymous blocks and `CREATE PROCEDURE`/`FUNCTION`/`PACKAGE` run as
ordinary statements — send the whole block, embedded `;` and all. Do *not*
include a trailing `/`: that's a SQL\\*Plus command, not SQL, and it's rejected
with an explicit error.

```sql
BEGIN
    INSERT INTO employees (department_id, name) VALUES (10, 'Alice');
    DBMS_OUTPUT.PUT_LINE('inserted');
END;
```

`DBMS_OUTPUT` is enabled on every connection; anything a statement prints comes
back as `info` messages alongside its result — no `SET SERVEROUTPUT ON` needed.
A PL/SQL object that compiles *with errors* is reported by Oracle as a
successful CREATE; its compilation errors come back as `warning` messages
carrying the line and column of the offending token.

### Transactions

Writes are not auto-committed. Run `COMMIT` (or `ROLLBACK`)
explicitly — anything uncommitted is lost when the connection closes, including
after an idle timeout.
"""

    def __init__(
        self,
        params: dict[str, Any],
        conn: AsyncConnection,
        has_oracle_maintained: bool,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._conn = conn
        self._has_oracle_maintained = has_oracle_maintained
        """True when connected to Oracle 12c+; enables ORACLE_MAINTAINED column filter."""
        self._metadata_transform_set = False
        """Set to True after DBMS_METADATA session transform params are applied.
        Cleared on reconnect so they are re-applied to the new session."""
        self._session_statements: dict[str, str] = {}
        """ALTER SESSION SET statements successfully run on this connection, keyed
        by upper-cased property name (a later SET for the same property replaces
        the earlier entry). Replayed in reconnect() since a fresh Oracle session
        starts without them."""

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "OracleDriver":
        conn, has_oracle_maintained = await cls._open(params)
        return cls(params, conn, has_oracle_maintained, settings)

    @staticmethod
    async def _open(params: dict[str, Any]) -> tuple[Any, bool]:
        try:
            conn = await oracledb.connect_async(
                user=params.get("user", ""),
                password=params.get("password", ""),
                dsn=(
                    f"{params.get('host', 'localhost')}"
                    f":{params.get('port', 1521)}"
                    f"/{params.get('service_name', 'FREEPDB1')}"
                ),
            )
            conn.outputtypehandler = _replace_undecodable_text
            major_version = int(conn.version.split(".")[0])
            # Session-scoped, so this has to be redone on every reconnect —
            # which is why it lives here rather than in create().
            await enable_dbms_output(conn)
            return conn, major_version >= 12
        except oracledb.DatabaseError as exc:
            raise DriverError(_exc_message(exc)) from exc

    async def reconnect(self) -> None:
        try:
            await self._conn.close()
        except oracledb.InterfaceError:
            pass
        self._conn, self._has_oracle_maintained = await self._open(self.params)
        self._metadata_transform_set = False
        await self._replay_session_statements()

    async def _replay_session_statements(self) -> None:
        for stmt in self._session_statements.values():
            try:
                cur = self._conn.cursor()
                await _exec(cur, stmt)
            except Exception as exc:
                _maybe_raise_connection_lost(exc)
                if isinstance(exc, oracledb.DatabaseError):
                    raise DriverError(_format_db_error(exc, stmt)) from exc
                raise

    async def disconnect(self) -> None:
        try:
            await self._conn.close()
        except oracledb.InterfaceError:
            pass

    async def execute(
        self, query: str, binds: list[Any] | None = None
    ) -> ReadResult | WriteResult:
        """Run a SQL statement. Positional bind values map to ``:1``, ``:2``, … in the query.

        Args:
            query: SQL statement to execute.
            binds: Optional positional bind parameters (referenced as ``:1``, ``:2``, … in the query).

        Returns:
            ReadResult for queries that return rows, WriteResult otherwise.

        Raises:
            ConnectionLostError: If the connection was lost during execution.
        """
        load_cmd = parse_load(query)
        if load_cmd is not None:
            return await self._execute_load(load_cmd)

        binds = binds or []
        _reject_sqlplus_terminator(query)

        try:
            cur = self._conn.cursor()
            await _exec(cur, query, binds, private=True)
            prop = _alter_session_property(query)
            if prop is not None:
                self._session_statements[prop] = query
            if _is_explain_plan(query):
                lines = await fetch_explain_plan(cur)
                rows = [[row] for row in lines]
                return ReadResult(
                    columns=["PLAN_TABLE_OUTPUT"],
                    rows=rows,
                    rows_total=len(rows),
                    messages=await self._collect_messages(cur, query),
                )
            if cur.description is not None:
                columns = [d[0] for d in cur.description]
                # Render each row's LOBs as it's fetched rather than after a
                # bulk fetchall(): a LOB locator is only valid until the
                # cursor's next internal fetch, and fetchall() on a result
                # bigger than one arraysize batch (default 100) advances past
                # earlier rows before we ever get to read their LOBs.
                rows = [
                    [await render_lob(v, self._register_lob) for v in row]
                    async for row in cur
                ]
                # Only now the rows are all in hand: collecting messages runs
                # statements of its own, which would strand the LOB locators.
                return ReadResult(
                    columns=columns,
                    rows=rows,
                    rows_total=len(rows),
                    messages=await self._collect_messages(cur, query),
                )
            invalidate_cache(self._conn)
            return WriteResult(
                rows_affected=cur.rowcount if cur.rowcount >= 0 else 0,
                messages=await self._collect_messages(cur, query),
            )
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            _maybe_raise_decode_error(exc)
            if isinstance(exc, oracledb.DatabaseError):
                raise DriverError(_format_db_error(exc, query)) from exc
            raise

    async def _execute_load(self, cmd: LoadCommand) -> WriteResult:
        """Load a local delimited file into a table, a batch of rows per round-trip.

        Oracle's own loaders are out of reach from here: SQL*Loader is a
        separate binary and an external table reads the *server's* filesystem,
        so — like SQLcl's LOAD — the file is parsed client-side and sent as
        ordinary array INSERTs. Values are bound as text and left to Oracle's
        implicit conversion, which for DATE and TIMESTAMP columns runs through
        the session's NLS formats — hence the DATEFORMAT option, applied to the
        session for the duration of the load and put back afterwards.
        """
        options = cmd.options
        try:
            handle = open(cmd.path, newline="", encoding=options.encoding)
        except OSError as exc:
            raise DriverError(f"could not read {cmd.path!r}: {exc}") from exc
        except LookupError as exc:
            raise DriverError(f"unknown encoding {options.encoding!r}") from exc

        loaded = 0
        cur = self._conn.cursor()
        restore = await self._apply_load_formats(cur, options)
        try:
            with handle:
                header, rows = read_rows(handle, options)
                columns = cmd.columns or header
                if columns is not None:
                    for name in columns:
                        validate_identifier(name, "column name")

                statement = ""
                width = 0
                batch: list[tuple[int, list[str | None]]] = []
                for line, row in rows:
                    if not statement:
                        width = len(columns) if columns else len(row)
                        statement = build_insert_statement(cmd.table, columns, width)
                    if len(row) != width:
                        raise DriverError(
                            f"line {line} of {cmd.path} has "
                            f"{len(row)} fields, expected {width}"
                        )
                    batch.append((line, row))
                    if len(batch) >= options.batch:
                        await self._load_batch(cur, statement, batch, cmd)
                        loaded += len(batch)
                        batch = []
                if batch:
                    await self._load_batch(cur, statement, batch, cmd)
                    loaded += len(batch)

            invalidate_cache(self._conn)
            return WriteResult(
                rows_affected=loaded,
                messages=await self._collect_messages(cur, ""),
            )
        except DriverError:
            raise
        except UnicodeDecodeError as exc:
            raise DriverError(
                f"{cmd.path!r} is not valid {exc.encoding} at byte {exc.start} "
                f"({exc.reason}) — set a different ENCODING"
            ) from exc
        except csv.Error as exc:
            raise DriverError(f"could not parse {cmd.path!r}: {exc}") from exc
        except OSError as exc:
            raise DriverError(f"could not read {cmd.path!r}: {exc}") from exc
        finally:
            await self._restore_load_formats(cur, restore)

    async def _apply_load_formats(
        self, cur: Any, options: LoadOptions
    ) -> dict[str, str]:
        """Apply the NLS formats a load asked for, returning what to put back.

        A text bind reaches a DATE or TIMESTAMP column through the session's
        format model, so this is set on the session rather than per value —
        and only for the load, since a query run afterwards would otherwise
        silently render its dates a different way.
        """
        # A DATE and a TIMESTAMP column are converted through different models,
        # and a file's dates are written one way whatever column they land in,
        # so DATEFORMAT alone covers both unless TIMESTAMPFORMAT overrides it.
        timestamp = options.timestamp_format or options.date_format
        wanted = {
            "NLS_DATE_FORMAT": options.date_format,
            "NLS_TIMESTAMP_FORMAT": timestamp,
            "NLS_TIMESTAMP_TZ_FORMAT": timestamp,
        }
        wanted = {k: v for k, v in wanted.items() if v is not None}
        if not wanted:
            return {}

        previous = await fetch_nls_formats(self._conn)
        for parameter, value in wanted.items():
            await set_nls_format(cur, parameter, value)
        return {k: v for k, v in previous.items() if k in wanted}

    async def _restore_load_formats(self, cur: Any, previous: dict[str, str]) -> None:
        """Put back what :meth:`_apply_load_formats` replaced.

        Never raises: this runs in a finally, where an exception of its own
        would replace the error the client actually needs to see.
        """
        for parameter, value in previous.items():
            try:
                await set_nls_format(cur, parameter, value)
            except Exception as exc:
                logger.debug("Failed to restore %s: %s", parameter, exc)

    async def _load_batch(
        self,
        cur: Any,
        statement: str,
        batch: list[tuple[int, list[str | None]]],
        cmd: LoadCommand,
    ) -> None:
        """Send one batch of rows, naming the offending line if Oracle rejects one.

        After a failed array DML ``rowcount`` holds the rows that did go in, so
        the row the client has to go and fix can be pointed at by the line it
        occupies in the file rather than left for them to find.
        """
        try:
            await _executemany(cur, statement, [values for _, values in batch])
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            if isinstance(exc, oracledb.DatabaseError):
                done = min(max(cur.rowcount or 0, 0), len(batch) - 1)
                message = _exc_message(exc)
                raise DriverError(
                    f"{message} (line {batch[done][0]} of {cmd.path})"
                    f"{await self._conversion_hint(message, cmd.options)}"
                ) from exc
            raise

    async def _conversion_hint(self, message: str, options: LoadOptions) -> str:
        """Name the format models behind a failed datetime conversion.

        ORA-01843 on a row that looks perfectly well-formed is a session
        setting, not a broken file — but the error says only that the month was
        invalid, leaving the reader to guess which layout Oracle expected. Both
        models are named because the error doesn't say which column it came
        from, and a DATE and a TIMESTAMP column don't share one.
        """
        if not any(code in message for code in _DATE_CONVERSION_ERRORS):
            return ""
        try:
            formats = await fetch_nls_formats(self._conn)
        except Exception as exc:
            logger.debug("Failed to read NLS formats: %s", exc)
            return ""

        shown = ", ".join(
            f"{parameter} {formats[parameter]!r}"
            for parameter in ("NLS_DATE_FORMAT", "NLS_TIMESTAMP_FORMAT")
            if parameter in formats
        )
        if options.date_format is not None or options.timestamp_format is not None:
            return f" — the value matches neither format the load set ({shown})"
        return (
            f" — datetime values in the file must match the session's format "
            f"models ({shown}); add e.g. (DATEFORMAT 'YYYY-MM-DD') to the LOAD, "
            "which sets both"
        )

    async def _collect_messages(self, cur: Any, query: str) -> list[ExecuteMessage]:
        """Gather the out-of-band messages a just-executed statement produced.

        Never raises: the statement already succeeded, so failing to read its
        DBMS_OUTPUT or compilation errors must not turn it into an error.
        """
        messages: list[ExecuteMessage] = []
        try:
            lines, truncated = await fetch_dbms_output(self._conn)
            messages.extend(
                ExecuteMessage(level=MessageLevel.INFO, text=line) for line in lines
            )
            if truncated:
                messages.append(
                    ExecuteMessage(
                        level=MessageLevel.WARNING,
                        text=(
                            f"DBMS_OUTPUT truncated at {MAX_DBMS_OUTPUT_LINES} lines; "
                            "the rest was discarded"
                        ),
                    )
                )
        except Exception as exc:
            logger.debug("Failed to drain DBMS_OUTPUT: %s", exc)

        try:
            messages.extend(await self._compilation_messages(cur, query))
        except Exception as exc:
            logger.debug("Failed to fetch compilation errors: %s", exc)

        return messages

    async def _compilation_messages(self, cur: Any, query: str) -> list[ExecuteMessage]:
        """Turn a "created with compilation errors" warning into per-error messages.

        Oracle reports a PL/SQL object that failed to compile as a *successful*
        CREATE, flagging it only through ``cursor.warning``. Without this the
        client is told the statement worked and the object is silently broken.
        """
        if getattr(cur, "warning", None) is None:
            return []
        target = _created_object(query)
        if target is None:
            return []

        name, object_type = target
        errors = await fetch_compilation_errors(self._conn, name, object_type)
        return [
            ExecuteMessage(
                level=MessageLevel.WARNING,
                text=text.strip(),
                # user_errors positions are relative to the CREATE keyword, which
                # is where the submitted query starts once its leading blank and
                # comment lines are accounted for.
                line=line + _statement_start_line(query) - 1,
                col=position or None,
            )
            for line, position, text in errors
        ]

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        try:
            return await self._explore_list(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            _maybe_raise_decode_error(exc)
            raise

    async def _explore_list(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                schemas = await fetch_schemas(self._conn, self._has_oracle_maintained)
                return [
                    ExploreItem(name=s, type="schema", expandable=True) for s in schemas
                ]

            case [schema]:
                pairs = await fetch_tables_and_views(self._conn, schema.upper())
                return [
                    ExploreItem(name=name, type=kind, expandable=True)
                    for name, kind in pairs
                ]

            case [_schema, _table]:
                return [
                    ExploreItem(name="columns", type="group", expandable=True),
                    ExploreItem(name="indexes", type="group", expandable=True),
                ]

            case [schema, table, "columns"]:
                pairs = await fetch_column_names_and_types(
                    self._conn, schema.upper(), table.upper()
                )
                return [
                    ExploreItem(name=name, type=kind, expandable=False)
                    for name, kind in pairs
                ]

            case [schema, table, "indexes"]:
                pairs = await fetch_index_names_and_types(
                    self._conn, schema.upper(), table.upper()
                )
                return [
                    ExploreItem(name=name, type=kind, expandable=False)
                    for name, kind in pairs
                ]

            case _:
                return []

    async def explore_find(
        self, node_type: str, name: str, scopes: list[SearchScope]
    ) -> list[list[str]]:
        """Resolve a symbol with one data-dictionary query instead of a descent.

        The generic walker would list every schema's tables just to learn which
        owner holds an unqualified name — N round trips, and the whole catalog
        over the wire, to answer with one row. Oracle's dictionary answers it
        directly, so a cold-cache hover costs a single query.

        ``schema`` is left to the walker: the root listing is one cheap call the
        explore cache already serves.
        """
        try:
            return await self._explore_find(node_type, name, scopes)
        except FindNotSupported:
            raise
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            _maybe_raise_decode_error(exc)
            raise

    async def _explore_find(
        self, node_type: str, name: str, scopes: list[SearchScope]
    ) -> list[list[str]]:
        schemas = _scope_names(scopes, NodeType.SCHEMA)
        tables = _scope_names(scopes, NodeType.TABLE, NodeType.VIEW)
        match node_type:
            case NodeType.TABLE | NodeType.VIEW:
                # Both kinds are returned for either search: the tree holds them
                # at one level, and a client naming a view "table" is guessing
                # from syntax that cannot tell them apart.
                rows = await fetch_find_tables_and_views(
                    self._conn, name, schemas, self._has_oracle_maintained
                )
                return [[owner, table] for owner, table in rows]

            case NodeType.COLUMN:
                cols = await fetch_find_columns(
                    self._conn, name, schemas, tables, self._has_oracle_maintained
                )
                return [
                    [owner, table, "columns", column] for owner, table, column in cols
                ]

            case NodeType.INDEX:
                idx = await fetch_find_indexes(
                    self._conn, name, schemas, tables, self._has_oracle_maintained
                )
                return [[owner, table, "indexes", index] for owner, table, index in idx]

            case _:
                raise FindNotSupported

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case [schema, table]:
                result = await self.execute(build_preview_query(schema, table))
                return result if isinstance(result, ReadResult) else None
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        try:
            return await self._explore_describe(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            _maybe_raise_decode_error(exc)
            raise

    async def _explore_describe(self, path: list[str]) -> DescribeResult:
        match path:
            case [schema, table]:
                return await self._describe_entity(schema.upper(), table.upper())

            case [schema, table, "indexes"]:
                return await self._describe_indices(schema.upper(), table.upper())

            case [schema, table, "indexes", index_name]:
                return await self._describe_index(
                    schema.upper(), table.upper(), index_name.upper()
                )

            case [schema, table, "columns", col_name]:
                return await self._describe_field(
                    schema.upper(), table.upper(), col_name.upper()
                )

            case [schema, table, "relationships", column]:
                return await self._describe_relationship(
                    schema.upper(), table.upper(), column.upper()
                )

            case _:
                return None

    async def _describe_entity(self, schema: str, table: str) -> EntityDescription:
        col_details = await fetch_column_details(self._conn, schema, table)
        pk_cols = await fetch_pk_columns(self._conn, schema, table)
        indices = await self._describe_indices(schema, table, fetch_ddl=False)
        fields_by_index = await fetch_index_fields_for_table(self._conn, schema, table)
        excl, comp = build_column_index_lists(fields_by_index, indices)
        comments = await fetch_all_column_comments(self._conn, schema, table)
        samples = await self._fetch_samples(schema, table)
        comment = await fetch_table_comment(self._conn, schema, table)
        outgoing_by_col = group_references_by_column(
            await fetch_outgoing_references(self._conn, schema, table)
        )
        incoming_by_col = group_references_by_ref_column(
            await fetch_incoming_references(self._conn, schema, table)
        )

        return EntityDescription(
            name=table,
            kind="table",
            schema=schema,
            comment=comment,
            properties=[
                FieldDescription(
                    name=col.name,
                    types=[_format_type(col)],
                    nullable=col.nullable,
                    pk=col.name in pk_cols,
                    default=col.default,
                    exclusive_indices=excl.get(col.name, []),
                    composite_indices=comp.get(col.name, []),
                    comment=comments.get(col.name),
                    sample=samples.get(col.name, []),
                    outgoing_references=outgoing_by_col.get(col.name, []),
                    incoming_references=incoming_by_col.get(col.name, []),
                )
                for col in col_details
            ],
        )

    async def _describe_indices(
        self, schema: str, table: str, *, fetch_ddl: bool = True
    ) -> list[IndexDescription]:
        metas = await fetch_index_metas_for_table(self._conn, schema, table)
        fields_by_index = await fetch_index_fields_for_table(self._conn, schema, table)
        join_tables = await fetch_join_tables_for_table(self._conn, schema, table)

        indices = []
        for meta in metas:
            ddl = (
                await self._get_index_ddl(meta.owner, meta.name, meta.index_type, table)
                if fetch_ddl and not meta.generated
                else None
            )
            indices.append(
                IndexDescription(
                    name=meta.name,
                    fields=fields_by_index.get(meta.name, []),
                    unique=meta.unique,
                    tables=join_tables.get(meta.name, [table]),
                    index_type=meta.index_type,
                    visible=meta.visible,
                    ddl=ddl,
                )
            )

        return indices

    async def _describe_index(
        self, schema: str, table: str, index_name: str
    ) -> IndexDescription | None:
        meta = await fetch_index_meta(self._conn, schema, index_name)
        if meta is None:
            return None

        fields = await fetch_index_fields_for_index(self._conn, schema, index_name)
        tables = await fetch_join_tables_for_index(
            self._conn, schema, index_name, table
        )
        ddl = (
            None
            if meta.generated
            else await self._get_index_ddl(schema, index_name, meta.index_type, table)
        )

        return IndexDescription(
            name=index_name,
            fields=fields,
            unique=meta.unique,
            tables=tables,
            index_type=meta.index_type,
            visible=meta.visible,
            ddl=ddl,
        )

    async def _describe_field(
        self, schema: str, table: str, col_name: str
    ) -> FieldDescription | None:
        col_details = await fetch_column_details(self._conn, schema, table)
        col = next((c for c in col_details if c.name == col_name), None)
        if col is None:
            return None

        pk_cols = await fetch_pk_columns(self._conn, schema, table)
        indices = await self._describe_indices(schema, table, fetch_ddl=False)
        fields_by_index = await fetch_index_fields_for_table(self._conn, schema, table)
        excl, comp = build_column_index_lists(fields_by_index, indices)
        comments = await fetch_all_column_comments(self._conn, schema, table)
        sample = await self._fetch_sample(schema, table, col_name)
        outgoing_by_col = group_references_by_column(
            await fetch_outgoing_references(self._conn, schema, table)
        )
        incoming_by_col = group_references_by_ref_column(
            await fetch_incoming_references(self._conn, schema, table)
        )

        return FieldDescription(
            name=col.name,
            types=[_format_type(col)],
            nullable=col.nullable,
            pk=col.name in pk_cols,
            default=col.default,
            exclusive_indices=excl.get(col_name, []),
            composite_indices=comp.get(col_name, []),
            comment=comments.get(col_name),
            sample=sample,
            outgoing_references=outgoing_by_col.get(col_name, []),
            incoming_references=incoming_by_col.get(col_name, []),
        )

    async def _describe_relationship(
        self, schema: str, table: str, column: str
    ) -> TableReference | None:
        refs = await fetch_outgoing_references(self._conn, schema, table)
        return find_reference(refs, column)

    async def _fetch_sample(self, schema: str, table: str, col_name: str) -> list[Any]:
        try:
            return await asyncio.wait_for(
                fetch_column_sample(
                    self._conn,
                    schema,
                    table,
                    col_name,
                    self._settings.column_sample_size,
                ),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return []

    async def _fetch_samples(self, schema: str, table: str) -> dict[str, list[Any]]:
        try:
            columns, rows = await asyncio.wait_for(
                fetch_table_sample_rows(self._conn, schema, table),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return {}
        return build_column_samples(columns, rows, self._settings.column_sample_size)

    async def _ensure_metadata_transform(self) -> None:
        if self._metadata_transform_set:
            return
        await apply_metadata_transform(self._conn)
        self._metadata_transform_set = True

    async def _get_index_ddl(
        self, schema: str, index_name: str, index_type: str | None, table: str
    ) -> str:
        try:
            await self._ensure_metadata_transform()
            return await fetch_index_ddl(self._conn, schema, index_name)
        except Exception as exc:
            logger.debug(
                "DBMS_METADATA.GET_DDL failed for index %s.%s on table %s.%s (index_type=%r): %s",
                schema,
                index_name,
                schema,
                table,
                index_type,
                exc,
            )
            return "-- DDL unavailable"


_TEXT_DB_TYPES = (
    oracledb.DB_TYPE_VARCHAR,
    oracledb.DB_TYPE_NVARCHAR,
    oracledb.DB_TYPE_CHAR,
    oracledb.DB_TYPE_NCHAR,
    oracledb.DB_TYPE_LONG,
)
"""Character types fetched as str, and so decoded with the database charset."""


_DATE_CONVERSION_ERRORS = frozenset(
    {
        "ORA-01830",  # date format picture ends before converting entire string
        "ORA-01839",  # date not valid for month specified
        "ORA-01840",  # input value not long enough for date format
        "ORA-01841",  # full year must be between -4713 and +9999
        "ORA-01843",  # not a valid month
        "ORA-01846",  # not a valid day of the week
        "ORA-01847",  # day of month must be between 1 and last day of month
        "ORA-01858",  # a non-numeric character was found where a numeric was expected
        "ORA-01861",  # literal does not match format string
        "ORA-01862",  # the numeric value does not match the length of the format item
        "ORA-01865",  # not a valid era
    }
)
"""Errors Oracle raises when a text value doesn't fit the format model it's
being converted through — in a load, almost always the session's own."""


def _replace_undecodable_text(cursor: Any, metadata: Any) -> Any:
    """Fetch character columns with undecodable bytes replaced by U+FFFD.

    Oracle hands back whatever bytes a column holds, even ones that aren't
    valid in the database's own character set — a Latin-1 string loaded into an
    AL32UTF8 database, say. python-oracledb then raises UnicodeDecodeError
    while fetching the row, which fails the entire request: one bad byte in one
    sampled value takes down a table description or a whole diagram. Decoding
    with ``errors="replace"`` keeps the rest of the row (and every other row)
    intact and shows the damage as U+FFFD where it actually is.

    Installed on the connection, so it covers every cursor: user queries,
    previews and the catalog queries behind explore.

    Returns:
        A var for character columns, None to leave any other type alone.
    """
    if metadata.type_code not in _TEXT_DB_TYPES:
        return None
    return cursor.var(
        metadata.type_code,
        size=metadata.display_size or 0,
        arraysize=cursor.arraysize,
        encoding_errors="replace",
    )


_VARCHAR_TYPES = {"VARCHAR2", "VARCHAR"}


def _format_type(col: ColumnDetail) -> str:
    """Appends the max length to a VARCHAR2/VARCHAR column's type name — just
    the char length (e.g. ``VARCHAR2(50)``), or both when byte length differs
    (e.g. ``VARCHAR2(50 CHAR, 200 BYTE)``, which happens under BYTE semantics
    or a multi-byte charset). Other types are returned unchanged."""
    if col.type not in _VARCHAR_TYPES or col.char_length is None:
        return col.type
    if col.byte_length is not None and col.byte_length != col.char_length:
        return f"{col.type}({col.char_length} CHAR, {col.byte_length} BYTE)"
    return f"{col.type}({col.char_length})"


def _is_explain_plan(query: str) -> bool:
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        return stripped.upper().startswith("EXPLAIN PLAN")
    return False


_CREATE_OBJECT_RE = re.compile(
    r"""(?isx)
    ^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:NON)?EDITIONABLE\s+)?
    (PACKAGE\s+BODY|TYPE\s+BODY|PACKAGE|PROCEDURE|FUNCTION|TRIGGER|TYPE|VIEW)\s+
    (?:(?:"(?P<sq>[^"]+)"|(?P<su>\w+))\s*\.\s*)?
    (?:"(?P<nq>[^"]+)"|(?P<nu>\w+))
    """
)


def _created_object(query: str) -> tuple[str, str] | None:
    """Return ``(name, user_errors type)`` for a CREATE of a compilable object.

    The name is upper-cased unless it was double-quoted, matching how Oracle
    stores it in ``user_errors``. Returns None for any other statement.
    """
    match = _CREATE_OBJECT_RE.match(_strip_leading_comments(query))
    if match is None:
        return None
    quoted, unquoted = match.group("nq"), match.group("nu")
    name = quoted if quoted is not None else (unquoted or "").upper()
    object_type = re.sub(r"\s+", " ", match.group(1).upper())
    return name, object_type


def _statement_start_line(query: str) -> int:
    """1-indexed line of the first line that isn't blank or a ``--`` comment.

    Oracle numbers compilation errors from the start of the object's source, so
    any preamble in the submitted query has to be added back to make the line
    numbers line up with what the user is looking at.
    """
    for offset, line in enumerate(query.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return offset + 1
    return 1


def _strip_leading_comments(query: str) -> str:
    lines = query.splitlines()
    start = _statement_start_line(query) - 1
    return "\n".join(lines[start:])


def _reject_sqlplus_terminator(query: str) -> None:
    """Raise if *query* ends with a lone ``/`` line.

    ``/`` is the SQL*Plus "run the buffer" command, not SQL — it commonly
    trails a PL/SQL block copied out of a script. Oracle would report it as
    ``PLS-00103: Encountered the symbol "/"``, which doesn't say what to do
    about it, so name it here instead. Only a *trailing* ``/`` is rejected: a
    lone ``/`` mid-statement can legitimately be a line-broken division.
    """
    lines = query.rstrip().splitlines()
    if lines and lines[-1].strip() == "/":
        raise DriverError(
            f"line {len(lines)}: '/' is a SQL*Plus terminator, not part of the "
            "statement — remove it (the statement is sent to Oracle as-is)"
        )


_ALTER_SESSION_RE = re.compile(
    r"(?is)^\s*ALTER\s+SESSION\s+SET\s+([A-Za-z_][A-Za-z0-9_]*)\s*="
)


def _alter_session_property(query: str) -> str | None:
    """Return the upper-cased property name if *query* is an ``ALTER SESSION
    SET <property> = ...`` statement, else None."""
    match = _ALTER_SESSION_RE.match(query)
    return match.group(1).upper() if match else None


def _scope_names(scopes: list[SearchScope], *types: NodeType) -> tuple[str, ...]:
    """Return the names of every scope of one of *types*, in both the form the
    client wrote and its upper-cased form — the two ways Oracle may have stored
    the identifier. Sorted so the query text is stable and cacheable.
    """
    wanted = {str(t) for t in types}
    names: set[str] = set()
    for scope in scopes:
        if scope.type in wanted:
            names.update({scope.name, scope.name.upper()})
    return tuple(sorted(names))


def _maybe_raise_decode_error(exc: Exception) -> None:
    """Report a decode failure the output type handler didn't cover.

    :func:`_replace_undecodable_text` disarms the common case (character
    columns), but a few paths decode outside it — a CLOB read, an array var —
    and a bare UnicodeDecodeError reaches the client as "internal error", which
    says nothing about what to do. Naming the charset and the offending byte at
    least points at the data.
    """
    if isinstance(exc, UnicodeDecodeError):
        raise DriverError(
            "the database returned bytes that are not valid "
            f"{exc.encoding} at byte {exc.start}: {exc.reason}"
        ) from exc


def _maybe_raise_connection_lost(exc: Exception) -> None:
    if isinstance(exc, (oracledb.OperationalError, oracledb.InterfaceError)):
        raise ConnectionLostError(_exc_message(exc)) from exc
    if isinstance(exc, oracledb.DatabaseError):
        error = exc.args[0] if exc.args else None
        if getattr(error, "is_session_dead", False):
            raise ConnectionLostError(_exc_message(exc)) from exc


def _format_db_error(exc: oracledb.DatabaseError, query: str) -> str:
    """Format a database error, pointing at the spot in the query it names.

    Oracle locates a syntax error two different ways: a character ``offset``
    into the statement (SQL), or a ``line``/``column`` written into the message
    text (PL/SQL, ORA-06550). Either way the client is left counting lines to
    find the place, so the offending line is quoted back with a caret under it.
    """
    msg = _exc_message(exc)
    error = exc.args[0] if exc.args else None
    offset = getattr(error, "offset", 0)
    if offset > 0:
        line, col = _offset_to_line_col(query, offset)
        msg = f"{msg} (line {line}, col {col})"
    elif position := _message_line_col(msg):
        # PL/SQL numbers from the start of the block, which is where the
        # submitted query starts once blank and comment lines are accounted for.
        line, col = position
        line += _statement_start_line(query) - 1
    else:
        return msg
    excerpt = _error_excerpt(query, line, col)
    return f"{msg}\n{excerpt}" if excerpt else msg


_MESSAGE_POSITION_RE = re.compile(r"line (\d+), column (\d+)", re.IGNORECASE)


def _message_line_col(message: str) -> tuple[int, int] | None:
    """The ``line N, column M`` position Oracle wrote into an error message."""
    match = _MESSAGE_POSITION_RE.search(message)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _error_excerpt(query: str, line: int, col: int) -> str | None:
    """Quote line *line* of *query* with a caret under column *col*.

    Returns None if the position falls outside the query — a line number
    relative to something other than what was submitted (a stored object's own
    source, say) would otherwise point at an innocent line.
    """
    lines = query.splitlines()
    if not 1 <= line <= len(lines):
        return None
    text = lines[line - 1]
    if not 1 <= col <= len(text) + 1:
        return None
    gutter = f"{line} | "
    # Tabs are one column to Oracle but many to a terminal: keeping them in the
    # quoted line while padding the caret with spaces would drift them apart.
    return f"{gutter}{text.expandtabs(1)}\n{' ' * (len(gutter) + col - 1)}^"


def _offset_to_line_col(query: str, offset: int) -> tuple[int, int]:
    segment = query[:offset]
    line = segment.count("\n") + 1
    col = offset - segment.rfind("\n")
    return line, col


def _exc_message(exc: BaseException) -> str:
    """Build an error message that includes the full exception cause chain."""
    msg = str(exc).rstrip()
    cause: BaseException | None = exc.__cause__ or exc.__context__
    seen: set[int] = {id(exc)}
    while cause is not None and id(cause) not in seen:
        detail = str(cause).strip()
        if detail and detail not in msg:
            msg = f"{msg} — {detail}"
        seen.add(id(cause))
        cause = cause.__cause__ or cause.__context__
    return msg
