import asyncio
import dataclasses
import logging
import sqlite3
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar

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
    check_cancelled,
    group_references_by_ref_column,
    run_blocking,
)

T = TypeVar("T")


logger = logging.getLogger(__name__)


class SQLiteDriver(BaseDriver):
    """SQLite driver backed by the stdlib sqlite3 module.

    Args:
        params: Connect request fields; must include ``database`` (file path or ``":memory:"``).
        conn: Open sqlite3 connection. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "SQLite"
    LANGUAGES = [Language.SQL]
    DEFAULT_IDLE_TIMEOUT: ClassVar[float] = 0
    """File-based driver; idle connections are never closed automatically."""

    FIND_PATHS = {
        NodeType.TABLE: [["*"]],
        NodeType.VIEW: [["*"]],
        NodeType.COLUMN: [["*", "columns", "*"]],
        NodeType.INDEX: [["*", "indices", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(key="database", type=ParamType.STRING, label="Database file path"),
    ]

    HELP: str = """\
## SQLite

Standard SQL.

```sql
SELECT * FROM users WHERE age > 21
SELECT u.id, o.total FROM users u JOIN orders o ON o.user_id = u.id
INSERT INTO users (name, age) VALUES ('Alice', 30)
```

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <table|view>
    ├── columns       → name, type
    ├── indices       → index name
    └── foreign_keys  → "col → ref_table.ref_col"
```

Describing a table or view returns full column metadata (name, type,
nullability, primary key flag).
"""

    def __init__(
        self, params: dict[str, Any], conn: sqlite3.Connection, settings: DriverSettings
    ) -> None:
        super().__init__(params, settings)
        self._conn = conn

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "SQLiteDriver":
        """Open a SQLite connection and return a ready-to-use driver.

        Args:
            params: Must contain ``database`` — a file path or ``":memory:"``.

        Returns:
            A connected SQLiteDriver instance.
        """
        loop = asyncio.get_running_loop()
        try:
            conn = await loop.run_in_executor(
                None,
                lambda: sqlite3.connect(
                    params["database"], check_same_thread=False, isolation_level=None
                ),
            )
        except sqlite3.OperationalError as exc:
            raise DriverError(str(exc)) from exc
        return cls(params, conn, settings)

    async def reconnect(self) -> None:
        self._conn = await self._run(
            sqlite3.connect,
            self.params["database"],
            check_same_thread=False,
            isolation_level=None,
        )

    async def disconnect(self) -> None:
        await self._run(self._conn.close)

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a SQL statement.

        Args:
            query: SQL statement to execute.
            binds: Positional bind parameters (``?`` placeholders).

        Returns:
            ReadResult for queries that return rows, DMLResult otherwise.
        """
        return await self._run(self._execute_sync, query, binds)

    def _sql(
        self, sql: str, binds: Sequence[Any] = (), *, private: bool = False
    ) -> sqlite3.Cursor:
        """Log a statement, then run it on the connection.

        Every query this driver sends goes through here, so debug logging of
        them — and refusing them once the request is cancelled — needs no
        change at the call sites.

        Args:
            sql: Statement text.
            binds: Bind values, logged alongside the statement.
            private: Set for the *user's* own statement, whose binds are user
                data rather than the object names a catalog query binds.
        """
        check_cancelled()
        log_query(logger, sql, None if private else binds)
        return self._conn.execute(sql, binds) if binds else self._conn.execute(sql)

    def _execute_sync(self, sql: str, binds: list[Any]) -> ReadResult | WriteResult:

        cur = self._sql(sql, binds, private=True)
        if cur.description is not None:
            columns = [d[0] for d in cur.description]
            rows: list[list[Any]] = [
                [_render_lob(self._register_lob, v) for v in r] for r in cur.fetchall()
            ]
            return ReadResult(columns=columns, rows=rows, rows_total=len(rows))
        return WriteResult(rows_affected=cur.rowcount if cur.rowcount >= 0 else 0)

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        """List child nodes at the given path in the SQLite object tree.

        Args:
            path: Path segments (e.g. ``[]`` for root, ``["users"]`` for a table's groups).

        Returns:
            Child nodes, or an empty list if the path is unrecognised.
        """
        return await self._run(self._explore_list_sync, path)

    def _explore_list_sync(self, path: list[str]) -> list[ExploreItem]:

        match path:
            case []:
                rows = self._sql(
                    "SELECT name, type FROM sqlite_master"
                    " WHERE type IN ('table','view') ORDER BY name"
                ).fetchall()
                return [
                    ExploreItem(name=r[0], type=r[1], expandable=True) for r in rows
                ]

            case [_table]:
                return [
                    ExploreItem(name="columns", type="group", expandable=True),
                    ExploreItem(name="indices", type="group", expandable=True),
                    ExploreItem(name="foreign_keys", type="group", expandable=True),
                ]

            case [table, "columns"]:
                rows = self._sql(f"PRAGMA table_info({table})").fetchall()
                return [
                    ExploreItem(name=r[1], type=r[2], expandable=False) for r in rows
                ]

            case [table, "indices"]:
                rows = self._sql(f"PRAGMA index_list({table})").fetchall()
                return [
                    ExploreItem(name=r[1], type="index", expandable=False) for r in rows
                ]

            case [table, "foreign_keys"]:
                return [
                    ExploreItem(
                        name=f"{ref.column} → {ref.ref_table}.{ref.ref_column}",
                        type="foreign_key",
                        expandable=False,
                    )
                    for ref in self._outgoing_references_sync(table)
                ]

            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case [table]:
                result = await self.execute(f"SELECT * FROM {table} LIMIT 10", [])
                return result if isinstance(result, ReadResult) else None
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        """Return entity/field metadata for the node at the given path.

        Args:
            path: Single-element path with the table name (e.g. ``["users"]``),
                ``[table, "indices"]`` for all indexes, or
                ``[table, "indices", index_name]`` for a single index.

        Returns:
            EntityDescription, FieldDescription, IndexDescription, list[IndexDescription],
            or TableReference depending on the path.
        """
        match path:
            case [table]:
                base = await self._run(self._describe_entity_sync, table)
                properties = []
                for f in base.properties:
                    sample = await self._fetch_sample(table, f.name)
                    properties.append(dataclasses.replace(f, sample=sample))
                return dataclasses.replace(base, properties=properties)
            case [table, "columns", col_name]:
                base = await self._run(self._describe_field_sync, table, col_name)
                if base is None:
                    return None
                sample = await self._fetch_sample(table, col_name)
                return dataclasses.replace(base, sample=sample)
            case _:
                return await self._run(self._explore_describe_sync, path)

    def _explore_describe_sync(self, path: list[str]) -> DescribeResult:
        match path:
            case [table, "indices"]:
                return self._describe_indices_sync(table)

            case [table, "indices", index_name]:
                return self._describe_index_sync(table, index_name)

            case [table, "relationships", column]:
                refs = self._outgoing_references_sync(table)
                return find_reference(refs, column)

            case _:
                return None

    def _describe_entity_sync(self, table: str) -> EntityDescription:
        cols = self._sql(f"PRAGMA table_info({table})").fetchall()
        idx_desc_list = self._describe_indices_sync(table)
        col_excl: dict[str, list[IndexDescription]] = {}
        col_comp: dict[str, list[IndexDescription]] = {}
        for idx_desc in idx_desc_list:
            key_col_names = [f.name for f in idx_desc.fields]
            for cn in key_col_names:
                if len(key_col_names) == 1:
                    col_excl.setdefault(cn, []).append(idx_desc)
                else:
                    col_comp.setdefault(cn, []).append(idx_desc)
        outgoing_by_col = group_references_by_column(
            self._outgoing_references_sync(table)
        )
        incoming_by_col = group_references_by_ref_column(
            self._incoming_references_sync(table)
        )

        return EntityDescription(
            name=table,
            kind="table",
            properties=[
                FieldDescription(
                    name=r[1],
                    types=[r[2] or ""],
                    nullable=not bool(r[3]),
                    pk=bool(r[5]),
                    exclusive_indices=col_excl.get(r[1], []),
                    composite_indices=col_comp.get(r[1], []),
                    outgoing_references=outgoing_by_col.get(r[1], []),
                    incoming_references=incoming_by_col.get(r[1], []),
                )
                for r in cols
            ],
        )

    def _describe_indices_sync(self, table: str) -> list[IndexDescription]:
        index_list = self._sql(f"PRAGMA index_list({table})").fetchall()
        indices = []
        for idx_row in index_list:
            idx = self._describe_index_sync(table, idx_row[1])
            if idx is not None:
                indices.append(idx)
        return indices

    def _describe_index_sync(
        self, table: str, index_name: str
    ) -> IndexDescription | None:
        index_list = self._sql(f"PRAGMA index_list({table})").fetchall()
        index_row = next((r for r in index_list if r[1] == index_name), None)
        if index_row is None:
            return None
        unique = bool(index_row[2])
        xinfo = self._sql(f"PRAGMA index_xinfo({index_name})").fetchall()
        fields = [
            IndexKeyField(name=r[2], direction="desc" if r[3] else "asc")
            for r in xinfo
            if r[5]  # key=1: part of the index key; 0 = implicit rowid
        ]
        row = self._sql(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        ).fetchone()
        ddl: str | None = row[0] if row and row[0] else None
        return IndexDescription(
            name=index_name,
            fields=fields,
            unique=unique,
            tables=[table],
            index_type="btree",
            ddl=ddl,
        )

    def _describe_field_sync(
        self, table: str, col_name: str
    ) -> FieldDescription | None:
        cols = self._sql(f"PRAGMA table_info({table})").fetchall()
        row = next((r for r in cols if r[1] == col_name), None)
        if row is None:
            return None

        idx_desc_list = self._describe_indices_sync(table)
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
        outgoing_by_col = group_references_by_column(
            self._outgoing_references_sync(table)
        )
        incoming_by_col = group_references_by_ref_column(
            self._incoming_references_sync(table)
        )

        return FieldDescription(
            name=col_name,
            types=[row[2] or ""],
            nullable=not bool(row[3]),
            pk=bool(row[5]),
            exclusive_indices=exclusive_indices,
            composite_indices=composite_indices,
            outgoing_references=outgoing_by_col.get(col_name, []),
            incoming_references=incoming_by_col.get(col_name, []),
        )

    def _outgoing_references_sync(self, table: str) -> list[TableReference]:
        rows = self._sql(f"PRAGMA foreign_key_list({table})").fetchall()
        unique_cols = self._unique_columns_sync(table)
        return [
            TableReference(
                table=table,
                column=r[3],
                ref_table=r[2],
                ref_column=r[4],
                unique=r[3] in unique_cols,
            )
            for r in rows
        ]

    def _incoming_references_sync(self, table: str) -> list[TableReference]:
        other_tables = self._sql(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name != ?",
            (table,),
        ).fetchall()
        references = []
        for (other_table,) in other_tables:
            rows = self._sql(f"PRAGMA foreign_key_list({other_table})").fetchall()
            matching = [r for r in rows if r[2].lower() == table.lower()]
            if not matching:
                continue
            unique_cols = self._unique_columns_sync(other_table)
            references.extend(
                TableReference(
                    table=other_table,
                    column=r[3],
                    ref_table=table,
                    ref_column=r[4],
                    unique=r[3] in unique_cols,
                )
                for r in matching
            )
        return references

    def _unique_columns_sync(self, table: str) -> set[str]:
        """Columns constrained to unique values: the table's own PK (unless
        composite) or covered by a single-column UNIQUE index."""
        cols = self._sql(f"PRAGMA table_info({table})").fetchall()
        pk_cols = [r[1] for r in cols if r[5]]
        unique = set(pk_cols) if len(pk_cols) == 1 else set()
        index_list = self._sql(f"PRAGMA index_list({table})").fetchall()
        for idx_row in index_list:
            if not idx_row[2]:
                continue  # not a UNIQUE index
            xinfo = self._sql(f"PRAGMA index_xinfo({idx_row[1]})").fetchall()
            key_cols = [r[2] for r in xinfo if r[5]]
            if len(key_cols) == 1:
                unique.add(key_cols[0])
        return unique

    async def _fetch_sample(self, table: str, col_name: str) -> list[Any]:
        try:
            return await asyncio.wait_for(
                self._run(self._fetch_sample_sync, table, col_name),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return []

    def _fetch_sample_sync(self, table: str, col_name: str) -> list[Any]:
        try:
            return [
                r[0]
                for r in self._sql(
                    f'SELECT DISTINCT "{col_name}" FROM "{table}"'
                    f' WHERE "{col_name}" IS NOT NULL LIMIT {self._settings.column_sample_size}'
                ).fetchall()
            ]
        except Exception:
            return []

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return await run_blocking(lambda: fn(*args, **kwargs), self._conn.interrupt)
        except sqlite3.Error as exc:
            raise DriverError(str(exc)) from exc


def _render_lob(
    register_lob: Callable[[bytes | str, str], LobPlaceholder], value: Any
) -> Any:
    """Render a BLOB value as a :class:`LobPlaceholder` instead of inlining it in the row.

    sqlite3 fully materializes BLOB columns as plain ``bytes``, but ``bytes``
    still isn't JSON-serialisable and can be arbitrarily large, so it's
    swapped for a placeholder like Oracle's CLOB/BLOB handling. Unlike Oracle's
    lazy locator, the value is already fully in memory here, so it's registered
    for later explore.download(ref=...) retrieval via `register_lob`.
    """
    if not isinstance(value, (bytes, bytearray)):
        return value
    return register_lob(bytes(value), f"BLOB ({len(value)} bytes)")
