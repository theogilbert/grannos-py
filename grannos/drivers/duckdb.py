import asyncio
import dataclasses
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar

import duckdb

from ..log import log_query
from ..protocol import (
    DescribeResult,
    DriverParam,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    IndexDescription,
    IndexKeyField,
    Language,
    LobPlaceholder,
    NodeType,
    ParamType,
    ReadResult,
    TableReference,
    WriteResult,
)
from .base import (
    BaseDriver,
    DriverError,
    DriverSettings,
    find_reference,
    group_references_by_column,
    group_references_by_ref_column,
    run_blocking,
)

T = TypeVar("T")


logger = logging.getLogger(__name__)


class DuckDBDriver(BaseDriver):
    """DuckDB driver backed by the duckdb package.

    Args:
        params: Connect request fields.
        conn: Open DuckDB connection. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "DuckDB"
    LANGUAGES = [Language.SQL]
    DEFAULT_IDLE_TIMEOUT: ClassVar[float] = 0
    """File-based driver; idle connections are never closed automatically."""

    FIND_PATHS = {
        NodeType.SCHEMA: [["*"]],
        NodeType.TABLE: [["*", "*"]],
        NodeType.VIEW: [["*", "*"]],
        NodeType.COLUMN: [["*", "*", "columns", "*"]],
        NodeType.INDEX: [["*", "*", "indices", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="database",
            type=ParamType.STRING,
            label="Database file path or :memory:",
            required=False,
            default=":memory:",
        ),
    ]

    HELP: str = """\
## DuckDB

Standard SQL. DuckDB's SQL dialect also reads files directly:

```sql
SELECT * FROM read_parquet('/path/to/file.parquet')
SELECT * FROM read_csv('/path/to/file.csv', header = true)
SELECT * FROM 'glob/**/*.parquet'
```

**Create a table from a CSV file:**

```sql
CREATE TABLE users AS SELECT * FROM read_csv('/path/to/users.csv', header = true)
CREATE TABLE users AS SELECT * FROM '/path/to/users.csv'   -- reader inferred from the extension
COPY users FROM '/path/to/users.csv' (FORMAT csv, HEADER)  -- into an existing table
```

Column names and types are auto-detected. Pass
`columns = {'id': 'INTEGER', 'name': 'VARCHAR'}` to `read_csv` to declare
them yourself, or `delim`/`quote`/`skip` to override the dialect it sniffs.

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns       → name, type
        ├── indices       → index name
        └── foreign_keys  → "col → ref_table.ref_col"
```
"""

    def __init__(
        self,
        params: dict[str, Any],
        conn: duckdb.DuckDBPyConnection,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._conn = conn

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "DuckDBDriver":
        """Open a DuckDB connection and return a ready-to-use driver.

        Args:
            params: May contain ``database`` (file path or ``:memory:``,
                defaults to ``:memory:``).
        """
        database = params.get("database") or ":memory:"
        try:
            conn = await asyncio.get_running_loop().run_in_executor(
                None, lambda: duckdb.connect(database)
            )
        except duckdb.Error as exc:
            raise DriverError(str(exc)) from exc
        return cls(params, conn, settings)

    async def reconnect(self) -> None:
        database = self.params.get("database") or ":memory:"
        self._conn = await asyncio.get_running_loop().run_in_executor(
            None, lambda: duckdb.connect(database)
        )

    async def disconnect(self) -> None:
        await self._run(self._conn.close)

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a SQL statement.

        Args:
            query: SQL statement to execute.
            binds: Positional bind parameters (``?`` placeholders).
        """
        return await self._run(self._execute_sync, query, binds)

    def _sql(
        self, sql: str, binds: Sequence[Any] = (), *, private: bool = False
    ) -> duckdb.DuckDBPyConnection:
        """Log a statement, then run it on the connection.

        Every query this driver sends goes through here, so debug logging of
        them needs no change at the call sites.

        Args:
            sql: Statement text.
            binds: Bind values, logged alongside the statement.
            private: Set for the *user's* own statement, whose binds are user
                data rather than the object names a catalog query binds.
        """
        log_query(logger, sql, None if private else binds)
        return self._conn.execute(sql, binds) if binds else self._conn.execute(sql)

    def _execute_sync(self, sql: str, binds: list[Any]) -> ReadResult | WriteResult:
        cur = self._sql(sql, binds, private=True)
        desc = cur.description
        # DuckDB returns a synthetic "Count" column for DML instead of description=None
        if desc is None or (len(desc) == 1 and desc[0][0] == "Count"):
            row = cur.fetchone() if desc is not None else None
            return WriteResult(rows_affected=int(row[0]) if row else 0)
        columns = [d[0] for d in desc]
        rows: list[list[Any]] = [
            [_render_lob(self._register_lob, v) for v in r] for r in cur.fetchall()
        ]
        return ReadResult(columns=columns, rows=rows, rows_total=len(rows))

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        """List child nodes at the given path in the DuckDB object tree.

        Args:
            path: Path segments (e.g. ``[]`` for schemas, ``["main"]`` for tables).
        """
        return await self._run(self._explore_list_sync, path)

    def _explore_list_sync(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                rows = self._sql(
                    "SELECT schema_name FROM information_schema.schemata"
                    " WHERE catalog_name = current_catalog()"
                    " AND schema_name NOT IN ('information_schema', 'pg_catalog')"
                    " ORDER BY schema_name"
                ).fetchall()
                return [
                    ExploreItem(name=r[0], type="schema", expandable=True) for r in rows
                ]

            case [schema]:
                rows = self._sql(
                    "SELECT table_name, table_type FROM information_schema.tables"
                    " WHERE table_schema = ? ORDER BY table_name",
                    [schema],
                ).fetchall()
                return [
                    ExploreItem(
                        name=r[0],
                        type="view" if r[1] == "VIEW" else "table",
                        expandable=True,
                    )
                    for r in rows
                ]

            case [_schema, _table]:
                return [
                    ExploreItem(name="columns", type="group", expandable=True),
                    ExploreItem(name="indices", type="group", expandable=True),
                    ExploreItem(name="foreign_keys", type="group", expandable=True),
                ]

            case [schema, table, "columns"]:
                rows = self._sql(
                    "SELECT column_name, data_type FROM information_schema.columns"
                    " WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                    [schema, table],
                ).fetchall()
                return [
                    ExploreItem(name=r[0], type=r[1], expandable=False) for r in rows
                ]

            case [schema, table, "indices"]:
                rows = self._sql(
                    "SELECT index_name FROM duckdb_indexes()"
                    " WHERE schema_name = ? AND table_name = ? ORDER BY index_name",
                    [schema, table],
                ).fetchall()
                return [
                    ExploreItem(name=r[0], type="index", expandable=False) for r in rows
                ]

            case [schema, table, "foreign_keys"]:
                items = []
                for src_cols, fk_table, fk_cols, _ in self._outgoing_fk_rows_sync(
                    schema, table
                ):
                    src = ", ".join(src_cols)
                    ref = ", ".join(fk_cols)
                    items.append(
                        ExploreItem(
                            name=f"{src} → {fk_table}.{ref}",
                            type="foreign_key",
                            expandable=False,
                        )
                    )
                return items

            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case [schema, table]:
                result = await self.execute(
                    f'SELECT * FROM "{schema}"."{table}" LIMIT 10', []
                )
                return result if isinstance(result, ReadResult) else None
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        """Return metadata for the node at the given path.

        Args:
            path: ``[schema, table]`` for a table description,
                ``[schema, table, "indices"]`` for all indexes, or
                ``[schema, table, "indices", index_name]`` for one index.
        """
        match path:
            case [schema, table]:
                base = await self._run(self._describe_entity_sync, schema, table)
                properties = []
                for f in base.properties:
                    sample = await self._fetch_sample(schema, table, f.name)
                    properties.append(dataclasses.replace(f, sample=sample))
                return dataclasses.replace(base, properties=properties)
            case [schema, table, "columns", col_name]:
                base = await self._run(
                    self._describe_field_sync, schema, table, col_name
                )
                if base is None:
                    return None
                sample = await self._fetch_sample(schema, table, col_name)
                return dataclasses.replace(base, sample=sample)
            case _:
                return await self._run(self._explore_describe_sync, path)

    def _explore_describe_sync(self, path: list[str]) -> DescribeResult:
        match path:
            case [schema, table, "indices"]:
                return self._describe_indices_sync(schema, table)

            case [schema, table, "indices", index_name]:
                rows = self._sql(
                    "SELECT is_unique, sql FROM duckdb_indexes()"
                    " WHERE schema_name = ? AND table_name = ? AND index_name = ?",
                    [schema, table, index_name],
                ).fetchall()
                if not rows:
                    return None
                is_unique, sql = rows[0]
                return IndexDescription(
                    name=index_name,
                    fields=_parse_index_columns(sql) if sql else [],
                    unique=bool(is_unique),
                    tables=[table],
                    ddl=sql,
                )

            case [schema, table, "relationships", column]:
                refs = self._outgoing_references_sync(schema, table)
                return find_reference(refs, column)

            case _:
                return None

    def _describe_entity_sync(self, schema: str, table: str) -> EntityDescription:
        col_rows = self._sql(
            "SELECT column_name, data_type, is_nullable"
            " FROM information_schema.columns"
            " WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [schema, table],
        ).fetchall()
        pk_rows = self._sql(
            "SELECT constraint_column_names FROM duckdb_constraints()"
            " WHERE schema_name = ? AND table_name = ? AND constraint_type = 'PRIMARY KEY'",
            [schema, table],
        ).fetchall()
        pk_cols: set[str] = set(pk_rows[0][0]) if pk_rows else set()

        idx_desc_list = self._describe_indices_sync(schema, table)
        col_excl: dict[str, list[IndexDescription]] = {}
        col_comp: dict[str, list[IndexDescription]] = {}
        for idx_desc in idx_desc_list:
            key_col_names = [f.name for f in idx_desc.fields]
            for cn in key_col_names:
                if len(key_col_names) == 1:
                    col_excl.setdefault(cn, []).append(idx_desc)
                else:
                    col_comp.setdefault(cn, []).append(idx_desc)

        col_comments: dict[str, str | None] = {}
        try:
            for r in self._sql(
                "SELECT column_name, comment FROM duckdb_columns()"
                " WHERE schema_name = ? AND table_name = ?",
                [schema, table],
            ).fetchall():
                col_comments[r[0]] = r[1] if r[1] else None
        except Exception:
            pass

        table_comment: str | None = None
        try:
            comment_rows = self._sql(
                "SELECT comment FROM duckdb_tables()"
                " WHERE schema_name = ? AND table_name = ?",
                [schema, table],
            ).fetchall()
            table_comment = (
                comment_rows[0][0] if comment_rows and comment_rows[0][0] else None
            )
        except Exception:
            pass

        outgoing_by_col = group_references_by_column(
            self._outgoing_references_sync(schema, table)
        )
        incoming_by_col = group_references_by_ref_column(
            self._incoming_references_sync(schema, table)
        )

        return EntityDescription(
            name=table,
            kind="table",
            schema=schema,
            comment=table_comment,
            properties=[
                FieldDescription(
                    name=r[0],
                    types=[r[1] or ""],
                    nullable=r[2] == "YES",
                    pk=r[0] in pk_cols,
                    exclusive_indices=col_excl.get(r[0], []),
                    composite_indices=col_comp.get(r[0], []),
                    comment=col_comments.get(r[0]),
                    outgoing_references=outgoing_by_col.get(r[0], []),
                    incoming_references=incoming_by_col.get(r[0], []),
                )
                for r in col_rows
            ],
        )

    def _describe_indices_sync(self, schema: str, table: str) -> list[IndexDescription]:
        rows = self._sql(
            "SELECT index_name, is_unique, sql FROM duckdb_indexes()"
            " WHERE schema_name = ? AND table_name = ? ORDER BY index_name",
            [schema, table],
        ).fetchall()
        return [
            IndexDescription(
                name=idx_name,
                fields=_parse_index_columns(sql) if sql else [],
                unique=bool(is_unique),
                tables=[table],
                ddl=sql,
            )
            for idx_name, is_unique, sql in rows
        ]

    def _describe_field_sync(
        self, schema: str, table: str, col_name: str
    ) -> FieldDescription | None:
        col_rows = self._sql(
            "SELECT column_name, data_type, is_nullable"
            " FROM information_schema.columns"
            " WHERE table_schema = ? AND table_name = ? AND column_name = ?",
            [schema, table, col_name],
        ).fetchall()
        if not col_rows:
            return None
        r = col_rows[0]

        pk_rows = self._sql(
            "SELECT constraint_column_names FROM duckdb_constraints()"
            " WHERE schema_name = ? AND table_name = ? AND constraint_type = 'PRIMARY KEY'",
            [schema, table],
        ).fetchall()
        pk_cols: set[str] = set(pk_rows[0][0]) if pk_rows else set()

        idx_desc_list = self._describe_indices_sync(schema, table)
        exclusive_indices = []
        composite_indices = []
        for idx_desc in idx_desc_list:
            key_col_names = [f.name for f in idx_desc.fields]
            if col_name not in key_col_names:
                continue
            if len(key_col_names) == 1:
                exclusive_indices.append(idx_desc)
            else:
                composite_indices.append(idx_desc)

        comment: str | None = None
        try:
            rows = self._sql(
                "SELECT comment FROM duckdb_columns()"
                " WHERE schema_name = ? AND table_name = ? AND column_name = ?",
                [schema, table, col_name],
            ).fetchall()
            comment = rows[0][0] if rows and rows[0][0] else None
        except Exception:
            pass
        outgoing_by_col = group_references_by_column(
            self._outgoing_references_sync(schema, table)
        )
        incoming_by_col = group_references_by_ref_column(
            self._incoming_references_sync(schema, table)
        )

        return FieldDescription(
            name=col_name,
            types=[r[1] or ""],
            nullable=r[2] == "YES",
            pk=col_name in pk_cols,
            exclusive_indices=exclusive_indices,
            composite_indices=composite_indices,
            comment=comment,
            outgoing_references=outgoing_by_col.get(col_name, []),
            incoming_references=incoming_by_col.get(col_name, []),
        )

    def _outgoing_fk_rows_sync(
        self, schema: str, table: str
    ) -> list[tuple[list[str], str, list[str], str]]:
        """Raw (local_columns, ref_table, ref_columns, constraint_name) rows for a
        table's foreign keys."""
        return self._sql(
            "SELECT constraint_column_names, referenced_table, referenced_column_names,"
            " constraint_name"
            " FROM duckdb_constraints()"
            " WHERE schema_name = ? AND table_name = ? AND constraint_type = 'FOREIGN KEY'",
            [schema, table],
        ).fetchall()

    def _outgoing_references_sync(
        self, schema: str, table: str
    ) -> list[TableReference]:
        rows = self._outgoing_fk_rows_sync(schema, table)
        unique_cols = self._unique_columns_sync(schema, table)
        return [
            TableReference(
                table=table,
                schema=schema,
                column=src_col,
                ref_table=fk_table,
                ref_schema=schema,
                ref_column=ref_col,
                unique=src_col in unique_cols,
                constraint_name=constraint_name,
            )
            for src_cols, fk_table, fk_cols, constraint_name in rows
            for src_col, ref_col in zip(src_cols, fk_cols)
        ]

    def _incoming_references_sync(
        self, schema: str, table: str
    ) -> list[TableReference]:
        rows = self._sql(
            "SELECT table_name, constraint_column_names, referenced_column_names,"
            " constraint_name"
            " FROM duckdb_constraints()"
            " WHERE schema_name = ? AND referenced_table = ? AND constraint_type = 'FOREIGN KEY'",
            [schema, table],
        ).fetchall()
        references = []
        for other_table, fk_cols, ref_cols, constraint_name in rows:
            unique_cols = self._unique_columns_sync(schema, other_table)
            references.extend(
                TableReference(
                    table=other_table,
                    schema=schema,
                    column=fk_col,
                    ref_table=table,
                    ref_schema=schema,
                    ref_column=ref_col,
                    unique=fk_col in unique_cols,
                    constraint_name=constraint_name,
                )
                for fk_col, ref_col in zip(fk_cols, ref_cols)
            )
        return references

    def _unique_columns_sync(self, schema: str, table: str) -> set[str]:
        """Columns constrained to unique values by a single-column PK or
        UNIQUE constraint. DuckDB doesn't back these with a listed entry in
        ``duckdb_indexes()``, so ``duckdb_constraints()`` is the source of truth."""
        rows = self._sql(
            "SELECT constraint_column_names FROM duckdb_constraints()"
            " WHERE schema_name = ? AND table_name = ?"
            " AND constraint_type IN ('PRIMARY KEY', 'UNIQUE')",
            [schema, table],
        ).fetchall()
        return {cols[0] for (cols,) in rows if len(cols) == 1}

    async def _fetch_sample(self, schema: str, table: str, col_name: str) -> list[Any]:
        try:
            return await asyncio.wait_for(
                self._run(self._fetch_sample_sync, schema, table, col_name),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return []

    def _fetch_sample_sync(self, schema: str, table: str, col_name: str) -> list[Any]:
        try:
            return [
                row[0]
                for row in self._sql(
                    f'SELECT DISTINCT "{col_name}" FROM "{schema}"."{table}"'
                    f' WHERE "{col_name}" IS NOT NULL LIMIT {self._settings.column_sample_size}'
                ).fetchall()
            ]
        except Exception:
            return []

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return await run_blocking(lambda: fn(*args, **kwargs), self._conn.interrupt)
        except duckdb.Error as exc:
            raise DriverError(str(exc)) from exc


def _render_lob(
    register_lob: Callable[[bytes | str, str], LobPlaceholder], value: Any
) -> Any:
    """Render a BLOB value as a :class:`LobPlaceholder` instead of inlining it in the row.

    DuckDB fully materializes BLOB columns as plain ``bytes``, but ``bytes``
    still isn't JSON-serialisable and can be arbitrarily large, so it's
    swapped for a placeholder like Oracle's CLOB/BLOB handling. Already fully
    in memory, so it's registered for later explore.download(ref=...) retrieval.
    """
    if not isinstance(value, (bytes, bytearray)):
        return value
    return register_lob(bytes(value), f"BLOB ({len(value)} bytes)")


def _parse_index_columns(sql: str) -> list[IndexKeyField]:
    """Extract column names and sort directions from a CREATE INDEX SQL string."""
    m = re.search(r"\(([^)]+)\)", sql)
    if not m:
        return []
    fields = []
    for part in m.group(1).split(","):
        tokens = part.strip().split()
        if not tokens:
            continue
        name = tokens[0].strip('"')
        direction = "desc" if len(tokens) > 1 and tokens[1].upper() == "DESC" else "asc"
        fields.append(IndexKeyField(name=name, direction=direction))
    return fields
