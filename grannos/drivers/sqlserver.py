"""SQL Server driver — requires: pip install mssql-python"""

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import mssql_python

from ..log import log_query
from ..protocol import (
    DescribeResult,
    DriverParam,
    DriverParamChoice,
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
    SAMPLE_SCAN_ROWS,
    BaseDriver,
    ConnectionLostError,
    DriverError,
    DriverSettings,
    build_column_samples,
    find_reference,
    group_references_by_column,
    group_references_by_ref_column,
)

T = TypeVar("T")

_READ_ONLY_INTENT = "READ_ONLY"


logger = logging.getLogger(__name__)


def _exec(
    cur: Any, sql: str, binds: Sequence[Any] | None = None, *, private: bool = False
) -> None:
    """Log *sql* and run it on *cur*.

    Every statement this module sends to the database goes through here, so
    debug logging of them needs no change at the call sites.

    Args:
        cur: Open cursor to execute on.
        sql: Statement text.
        binds: Bind values, logged alongside the statement.
        private: Set for the *user's* own statement, whose binds are user data
            rather than the schema and object names a catalog query binds.
    """
    log_query(logger, str(sql), None if private else binds)
    if binds:
        cur.execute(sql, list(binds))
    else:
        cur.execute(sql)


class SQLServerDriver(BaseDriver):
    """SQL Server driver backed by mssql-python.

    Args:
        params: Connect request fields (``host``, ``port``, ``user``, ``password``,
            ``database``, ``applicationIntent``).
        conn: Open mssql_python connection. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "SQL Server"
    LANGUAGES = [Language.SQL]

    FIND_PATHS = {
        NodeType.SCHEMA: [["*"]],
        NodeType.TABLE: [["*", "*"]],
        NodeType.VIEW: [["*", "*"]],
        NodeType.COLUMN: [["*", "*", "columns", "*"]],
        NodeType.INDEX: [["*", "*", "indices", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=1433),
        DriverParam(key="database", type=ParamType.STRING, label="Database"),
        DriverParam(key="user", type=ParamType.STRING, label="User"),
        DriverParam(
            key="applicationIntent",
            type=ParamType.ENUM,
            label="Application Intent",
            choices=[
                DriverParamChoice(value="READ_WRITE", label="READ_WRITE"),
                DriverParamChoice(value="READ_ONLY", label="READ_ONLY"),
            ],
        ),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
    ]

    HELP: str = """\
## SQL Server

**Queries:** Standard T-SQL.

```sql
SELECT * FROM dbo.orders WHERE status = 'open'
SELECT o.id, c.name FROM dbo.orders o JOIN dbo.customers c ON c.id = o.customer_id
INSERT INTO dbo.orders (customer_id, status) VALUES (1, 'open')
```

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indices      → name, type (e.g. CLUSTERED)
```

System schemas (`sys`, `INFORMATION_SCHEMA`, `guest`, `db_*`) are hidden.

Describing a table or view returns full column metadata (name, type,
nullability, default).
"""

    def __init__(
        self,
        params: dict[str, Any],
        conn: "mssql_python.Connection",
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._conn = conn

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "SQLServerDriver":
        """Open a SQL Server connection and return a ready-to-use driver.

        Args:
            params: Connect request fields; see class docstring for supported keys.

        Returns:
            A connected SQLServerDriver instance.
        """
        return cls(params, await cls._open(params), settings)

    async def reconnect(self) -> None:
        self._conn = await self._open(self.params)

    async def disconnect(self) -> None:
        await self._run(self._conn.close)

    @staticmethod
    async def _open(params: dict[str, Any]) -> mssql_python.Connection:
        intent = params.get("applicationIntent", "")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: mssql_python.connect(
                    server=f"{params.get('host', 'localhost')},{params.get('port', 1433)}",
                    uid=params.get("user", ""),
                    pwd=params.get("password", ""),
                    database=params.get("database", ""),
                    intent=intent,
                    autocommit=intent == _READ_ONLY_INTENT,
                    trustservercertificate="yes",
                ),
            )
        except Exception as exc:
            raise DriverError(str(exc)) from exc

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a SQL statement.

        Args:
            query: SQL statement to execute.
            binds: Positional bind parameters (``?`` placeholders).

        Returns:
            ReadResult for queries that return rows, DMLResult otherwise.

        Raises:
            ConnectionLostError: If the connection was lost during execution.
        """
        try:
            return await self._run(self._execute_sync, query, binds)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise DriverError(str(exc)) from exc

    def _execute_sync(self, sql: str, binds: list[Any]) -> ReadResult | WriteResult:

        log_query(logger, sql)  # the user's own statement: text only, no binds
        cur = self._conn.execute(sql, binds)
        if cur.description is not None:
            columns = [d[0] for d in cur.description]
            rows: list[list[Any]] = [
                [_render_lob(self._register_lob, v) for v in r]
                for r in cur.fetchall()  # ty: ignore[missing-argument]
            ]
            return ReadResult(columns=columns, rows=rows, rows_total=len(rows))
        return WriteResult(rows_affected=cur.rowcount if cur.rowcount >= 0 else 0)

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        """List child nodes at the given path in the SQL Server object tree.

        Args:
            path: Path segments (``[]`` for schemas, ``[schema]`` for tables,
                ``[schema, table]`` for groups, etc.).

        Returns:
            Child nodes, or an empty list if the path is unrecognised.
        """
        try:
            return await self._run(self._explore_list_sync, path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise DriverError(str(exc)) from exc

    def _explore_list_sync(self, path: list[str]) -> list[ExploreItem]:

        cur = self._conn.cursor()
        match path:
            case []:
                _exec(
                    cur,
                    "SELECT name FROM sys.schemas"
                    " WHERE name NOT IN ('sys','INFORMATION_SCHEMA','guest','db_owner',"
                    "'db_accessadmin','db_securityadmin','db_ddladmin','db_backupoperator',"
                    "'db_datareader','db_datawriter','db_denydatareader','db_denydatawriter')"
                    " ORDER BY name",
                )
                return [
                    ExploreItem(name=r[0], type="schema", expandable=True)
                    for r in cur.fetchall()  # ty: ignore[missing-argument]
                ]

            case [schema]:
                _exec(
                    cur,
                    "SELECT TABLE_NAME, TABLE_TYPE FROM INFORMATION_SCHEMA.TABLES"
                    " WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME",
                    (schema,),
                )
                return [
                    ExploreItem(name=r[0], type=r[1].lower(), expandable=True)
                    for r in cur.fetchall()  # ty: ignore[missing-argument]
                ]

            case [_schema, _table]:
                return [
                    ExploreItem(name="columns", type="group", expandable=True),
                    ExploreItem(name="indices", type="group", expandable=True),
                ]

            case [schema, table, "columns"]:
                _exec(
                    cur,
                    "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS"
                    " WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                    " ORDER BY ORDINAL_POSITION",
                    (schema, table),
                )
                return [
                    ExploreItem(name=r[0], type=r[1], expandable=False)
                    for r in cur.fetchall()  # ty: ignore[missing-argument]
                ]

            case [schema, table, "indices"]:
                _exec(
                    cur,
                    "SELECT i.name, i.type_desc"
                    " FROM sys.indexes i"
                    " JOIN sys.objects o ON i.object_id = o.object_id"
                    " JOIN sys.schemas s ON o.schema_id = s.schema_id"
                    " WHERE s.name = ? AND o.name = ? AND i.name IS NOT NULL"
                    " ORDER BY i.name",
                    (schema, table),
                )
                return [
                    ExploreItem(name=r[0], type=r[1].lower(), expandable=False)
                    for r in cur.fetchall()  # ty: ignore[missing-argument]
                ]

            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case [schema, table]:
                result = await self.execute(
                    f"SELECT TOP 10 * FROM [{schema}].[{table}]", []
                )
                return result if isinstance(result, ReadResult) else None
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        """Return entity/field metadata for the node at the given path.

        Args:
            path: Two-element path ``[schema, table]``, ``[schema, table, "indices"]``
                for all indexes, or ``[schema, table, "indices", index_name]`` for one.

        Returns:
            EntityDescription, FieldDescription, IndexDescription, list[IndexDescription],
            or TableReference depending on the path.
        """
        try:
            return await self._explore_describe(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise DriverError(str(exc)) from exc

    async def _explore_describe(self, path: list[str]) -> DescribeResult:
        match path:
            case [schema, table]:
                base = await self._run(self._describe_entity_sync, schema, table)
                samples = await self._fetch_samples(schema, table)
                return dataclasses.replace(
                    base,
                    properties=[
                        dataclasses.replace(f, sample=samples.get(f.name, []))
                        for f in base.properties
                    ],
                )
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
                return self._describe_index_sync(schema, table, index_name)

            case [schema, table, "relationships", column]:
                refs = self._outgoing_references_sync(schema, table)
                return find_reference(refs, column)

            case _:
                return None

    def _describe_entity_sync(self, schema: str, table: str) -> EntityDescription:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT"
            " FROM INFORMATION_SCHEMA.COLUMNS"
            " WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
            " ORDER BY ORDINAL_POSITION",
            (schema, table),
        )
        col_rows = cur.fetchall()  # ty: ignore[missing-argument]

        _exec(
            cur,
            "SELECT c.name FROM sys.key_constraints kc"
            " JOIN sys.index_columns ic"
            "  ON kc.unique_index_id = ic.index_id AND kc.parent_object_id = ic.object_id"
            " JOIN sys.columns c"
            "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id"
            " JOIN sys.objects o ON kc.parent_object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND kc.type = 'PK'",
            (schema, table),
        )
        pk_cols: set[str] = {r[0] for r in cur.fetchall()}  # ty: ignore[missing-argument]

        _exec(
            cur,
            "SELECT c.name, CAST(ep.value AS NVARCHAR(MAX))"
            " FROM sys.columns c"
            " JOIN sys.objects o ON c.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " LEFT JOIN sys.extended_properties ep"
            "  ON ep.major_id = c.object_id AND ep.minor_id = c.column_id"
            "  AND ep.name = 'MS_Description'"
            " WHERE s.name = ? AND o.name = ?",
            (schema, table),
        )
        col_comments: dict[str, str | None] = {
            r[0]: (r[1].strip() if r[1] else None)
            for r in cur.fetchall()  # ty: ignore[missing-argument]
        }

        _exec(
            cur,
            "SELECT CAST(ep.value AS NVARCHAR(MAX))"
            " FROM sys.objects o"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " LEFT JOIN sys.extended_properties ep"
            "  ON ep.major_id = o.object_id AND ep.minor_id = 0"
            "  AND ep.name = 'MS_Description'"
            " WHERE s.name = ? AND o.name = ?",
            (schema, table),
        )
        comment_row = cur.fetchone()  # ty: ignore[missing-argument]
        table_comment: str | None = (
            comment_row[0].strip() if comment_row and comment_row[0] else None
        )

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
                    types=[r[1]],
                    nullable=r[2] == "YES",
                    pk=r[0] in pk_cols,
                    default=r[3],
                    exclusive_indices=col_excl.get(r[0], []),
                    composite_indices=col_comp.get(r[0], []),
                    comment=col_comments.get(r[0]),
                    outgoing_references=outgoing_by_col.get(r[0], []),
                    incoming_references=incoming_by_col.get(r[0], []),
                )
                for r in col_rows
            ],
        )

    def _describe_field_sync(
        self, schema: str, table: str, col_name: str
    ) -> FieldDescription | None:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT"
            " FROM INFORMATION_SCHEMA.COLUMNS"
            " WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = ?",
            (schema, table, col_name),
        )
        rows = cur.fetchall()  # ty: ignore[missing-argument]
        if not rows:
            return None
        r = rows[0]

        _exec(
            cur,
            "SELECT c.name FROM sys.key_constraints kc"
            " JOIN sys.index_columns ic"
            "  ON kc.unique_index_id = ic.index_id AND kc.parent_object_id = ic.object_id"
            " JOIN sys.columns c"
            "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id"
            " JOIN sys.objects o ON kc.parent_object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND kc.type = 'PK'",
            (schema, table),
        )
        pk_cols: set[str] = {row[0] for row in cur.fetchall()}  # ty: ignore[missing-argument]

        _exec(
            cur,
            "SELECT CAST(ep.value AS NVARCHAR(MAX))"
            " FROM sys.columns c"
            " JOIN sys.objects o ON c.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " LEFT JOIN sys.extended_properties ep"
            "  ON ep.major_id = c.object_id AND ep.minor_id = c.column_id"
            "  AND ep.name = 'MS_Description'"
            " WHERE s.name = ? AND o.name = ? AND c.name = ?",
            (schema, table, col_name),
        )
        comment_row = cur.fetchone()  # ty: ignore[missing-argument]
        comment: str | None = (
            comment_row[0].strip() if comment_row and comment_row[0] else None
        )

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
            default=r[3],
            exclusive_indices=exclusive_indices,
            composite_indices=composite_indices,
            comment=comment,
            outgoing_references=outgoing_by_col.get(col_name, []),
            incoming_references=incoming_by_col.get(col_name, []),
        )

    def _describe_indices_sync(self, schema: str, table: str) -> list[IndexDescription]:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT i.name, i.type_desc, i.is_unique,"
            "       i.is_disabled, i.filter_definition"
            " FROM sys.indexes i"
            " JOIN sys.objects o ON i.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND i.name IS NOT NULL"
            " ORDER BY i.name",
            (schema, table),
        )
        index_rows = cur.fetchall()  # ty: ignore[missing-argument]

        _exec(
            cur,
            "SELECT i.name, c.name, ic.is_descending_key, ic.is_included_column"
            " FROM sys.indexes i"
            " JOIN sys.index_columns ic"
            "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id"
            " JOIN sys.columns c"
            "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id"
            " JOIN sys.objects o ON i.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND i.name IS NOT NULL"
            " ORDER BY i.name, ic.key_ordinal, ic.index_column_id",
            (schema, table),
        )
        index_fields: dict[str, list[IndexKeyField]] = {}
        index_included: dict[str, list[str]] = {}
        for idx_name, col_name, is_desc, is_included in cur.fetchall():  # ty: ignore[missing-argument]
            if is_included:
                index_included.setdefault(idx_name, []).append(col_name)
            else:
                index_fields.setdefault(idx_name, []).append(
                    IndexKeyField(name=col_name, direction="desc" if is_desc else "asc")
                )

        indices = []
        for (
            idx_name,
            type_desc,
            is_unique,
            is_disabled,
            filter_def,
        ) in index_rows:
            is_clustered = (type_desc or "").upper() == "CLUSTERED"
            fields = index_fields.get(idx_name, [])
            included = index_included.get(idx_name, [])
            condition = filter_def.strip() if filter_def else None
            ddl = _build_sqlserver_ddl(
                idx_name,
                schema,
                table,
                type_desc,
                bool(is_unique),
                fields,
                included,
                condition,
            )
            indices.append(
                IndexDescription(
                    name=idx_name,
                    fields=fields,
                    unique=bool(is_unique),
                    tables=[table],
                    index_type=type_desc.lower() if type_desc else None,
                    clustered=bool(is_clustered),
                    visible=not bool(is_disabled),
                    included_columns=included,
                    ddl=ddl,
                )
            )
        return indices

    def _describe_index_sync(
        self, schema: str, table: str, index_name: str
    ) -> IndexDescription | None:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT i.type_desc, i.is_unique,"
            "       i.is_disabled, i.filter_definition"
            " FROM sys.indexes i"
            " JOIN sys.objects o ON i.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND i.name = ?",
            (schema, table, index_name),
        )
        row = cur.fetchone()  # ty: ignore[missing-argument]
        if row is None:
            return None
        type_desc, is_unique, is_disabled, filter_def = row
        is_clustered = (type_desc or "").upper() == "CLUSTERED"

        _exec(
            cur,
            "SELECT c.name, ic.is_descending_key, ic.is_included_column"
            " FROM sys.indexes i"
            " JOIN sys.index_columns ic"
            "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id"
            " JOIN sys.columns c"
            "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id"
            " JOIN sys.objects o ON i.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND i.name = ?"
            " ORDER BY ic.key_ordinal, ic.index_column_id",
            (schema, table, index_name),
        )
        fields: list[IndexKeyField] = []
        included: list[str] = []
        for col_name, is_desc, is_included in cur.fetchall():  # ty: ignore[missing-argument]
            if is_included:
                included.append(col_name)
            else:
                fields.append(
                    IndexKeyField(name=col_name, direction="desc" if is_desc else "asc")
                )

        condition = filter_def.strip() if filter_def else None
        ddl = _build_sqlserver_ddl(
            index_name,
            schema,
            table,
            type_desc,
            bool(is_unique),
            fields,
            included,
            condition,
        )
        return IndexDescription(
            name=index_name,
            fields=fields,
            unique=bool(is_unique),
            tables=[table],
            index_type=type_desc.lower() if type_desc else None,
            clustered=bool(is_clustered),
            visible=not bool(is_disabled),
            included_columns=included,
            ddl=ddl,
        )

    def _outgoing_references_sync(
        self, schema: str, table: str
    ) -> list[TableReference]:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT pc.name, rs.name, ro.name, rc.name, fk.name"
            " FROM sys.foreign_keys fk"
            " JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id"
            " JOIN sys.objects po ON fk.parent_object_id = po.object_id"
            " JOIN sys.schemas ps ON po.schema_id = ps.schema_id"
            " JOIN sys.columns pc"
            "  ON fkc.parent_object_id = pc.object_id AND fkc.parent_column_id = pc.column_id"
            " JOIN sys.objects ro ON fk.referenced_object_id = ro.object_id"
            " JOIN sys.schemas rs ON ro.schema_id = rs.schema_id"
            " JOIN sys.columns rc"
            "  ON fkc.referenced_object_id = rc.object_id AND fkc.referenced_column_id = rc.column_id"
            " WHERE ps.name = ? AND po.name = ?",
            (schema, table),
        )
        rows = cur.fetchall()  # ty: ignore[missing-argument]
        unique_cols = self._unique_columns_sync(schema, table)
        return [
            TableReference(
                table=table,
                schema=schema,
                column=r[0],
                ref_table=r[2],
                ref_schema=r[1],
                ref_column=r[3],
                unique=r[0] in unique_cols,
                constraint_name=r[4],
            )
            for r in rows
        ]

    def _incoming_references_sync(
        self, schema: str, table: str
    ) -> list[TableReference]:
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT rc.name, ps.name, po.name, pc.name, fk.name"
            " FROM sys.foreign_keys fk"
            " JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id"
            " JOIN sys.objects po ON fk.parent_object_id = po.object_id"
            " JOIN sys.schemas ps ON po.schema_id = ps.schema_id"
            " JOIN sys.columns pc"
            "  ON fkc.parent_object_id = pc.object_id AND fkc.parent_column_id = pc.column_id"
            " JOIN sys.objects ro ON fk.referenced_object_id = ro.object_id"
            " JOIN sys.schemas rs ON ro.schema_id = rs.schema_id"
            " JOIN sys.columns rc"
            "  ON fkc.referenced_object_id = rc.object_id AND fkc.referenced_column_id = rc.column_id"
            " WHERE rs.name = ? AND ro.name = ?",
            (schema, table),
        )
        references = []
        for r in cur.fetchall():  # ty: ignore[missing-argument]
            unique_cols = self._unique_columns_sync(r[1], r[2])
            references.append(
                TableReference(
                    table=r[2],
                    schema=r[1],
                    column=r[3],
                    ref_table=table,
                    ref_schema=schema,
                    ref_column=r[0],
                    unique=r[3] in unique_cols,
                    constraint_name=r[4],
                )
            )
        return references

    def _unique_columns_sync(self, schema: str, table: str) -> set[str]:
        """Columns covered by a single-column UNIQUE index (PKs included,
        since SQL Server backs every PK with a unique index)."""
        cur = self._conn.cursor()
        _exec(
            cur,
            "SELECT c.name, i.name"
            " FROM sys.indexes i"
            " JOIN sys.index_columns ic"
            "  ON i.object_id = ic.object_id AND i.index_id = ic.index_id"
            " JOIN sys.columns c"
            "  ON ic.object_id = c.object_id AND ic.column_id = c.column_id"
            " JOIN sys.objects o ON i.object_id = o.object_id"
            " JOIN sys.schemas s ON o.schema_id = s.schema_id"
            " WHERE s.name = ? AND o.name = ? AND i.is_unique = 1",
            (schema, table),
        )
        index_cols: dict[str, set[str]] = {}
        for col_name, idx_name in cur.fetchall():  # ty: ignore[missing-argument]
            index_cols.setdefault(idx_name, set()).add(col_name)
        return {next(iter(cols)) for cols in index_cols.values() if len(cols) == 1}

    async def _fetch_sample(self, schema: str, table: str, col_name: str) -> list[Any]:
        try:
            return await asyncio.wait_for(
                self._run(self._fetch_sample_sync, schema, table, col_name),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return []

    def _fetch_sample_sync(self, schema: str, table: str, col_name: str) -> list[Any]:
        cur = self._conn.cursor()
        try:
            _exec(
                cur,
                f"SELECT TOP {self._settings.column_sample_size} [{col_name}] FROM"
                f" (SELECT DISTINCT [{col_name}] FROM [{schema}].[{table}]"
                f"  WHERE [{col_name}] IS NOT NULL) AS _s",
            )
            return [row[0] for row in cur.fetchall()]  # ty: ignore[missing-argument]
        except Exception:
            return []

    async def _fetch_samples(self, schema: str, table: str) -> dict[str, list[Any]]:
        try:
            columns, rows = await asyncio.wait_for(
                self._run(self._fetch_sample_rows_sync, schema, table),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return {}
        return build_column_samples(columns, rows, self._settings.column_sample_size)

    def _fetch_sample_rows_sync(
        self, schema: str, table: str
    ) -> tuple[list[str], list[tuple]]:
        cur = self._conn.cursor()
        try:
            _exec(cur, f"SELECT TOP {SAMPLE_SCAN_ROWS} * FROM [{schema}].[{table}]")
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchall()  # ty: ignore[missing-argument]
            return columns, [tuple(r) for r in rows]
        except Exception:
            return [], []

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: fn(*args, **kwargs)
        )


# mssql-python's ``driver_error`` for the SQLSTATEs that mean the link itself
# is gone (01002, 08003, 08007, 08S01). OperationalError also covers errors a
# reconnect cannot fix — timeouts, cancels, and 08004, which SQL Server reports
# for error 916 (principal cannot access a database) on a perfectly live link.
_LINK_FAILURES = frozenset(
    {
        "Disconnect error",
        "Connection not open",
        "Connection failure during transaction",
        "Communication link failure",
    }
)


def _maybe_raise_connection_lost(exc: Exception) -> None:
    if isinstance(exc, mssql_python.InterfaceError) or (
        isinstance(exc, mssql_python.OperationalError)
        and exc.driver_error in _LINK_FAILURES
    ):
        raise ConnectionLostError(str(exc)) from exc


def _render_lob(
    register_lob: Callable[[bytes | str, str], LobPlaceholder], value: Any
) -> Any:
    """Render a binary value as a :class:`LobPlaceholder` instead of inlining it in the row.

    mssql-python fully materializes BINARY/VARBINARY/IMAGE columns as plain
    ``bytes`` — unlike Oracle's streaming LOB locators — but ``bytes`` still
    isn't JSON-serialisable and can be arbitrarily large, so it's swapped for
    a placeholder like Oracle's CLOB/BLOB handling. Already fully in memory, so
    it's registered for later explore.download(ref=...) retrieval.
    """
    if not isinstance(value, (bytes, bytearray)):
        return value
    return register_lob(bytes(value), f"VARBINARY ({len(value)} bytes)")


def _build_sqlserver_ddl(
    index_name: str,
    schema: str,
    table: str,
    type_desc: str | None,
    is_unique: bool,
    fields: list[IndexKeyField],
    included: list[str],
    condition: str | None,
) -> str | None:
    """Construct a CREATE INDEX DDL string for standard CLUSTERED/NONCLUSTERED indexes."""
    td = (type_desc or "").upper()
    if td not in ("CLUSTERED", "NONCLUSTERED"):
        return None
    parts = ["CREATE"]
    if is_unique:
        parts.append("UNIQUE")
    parts.append(td)
    parts.append(f"INDEX [{index_name}]")
    parts.append(f"ON [{schema}].[{table}]")
    col_list = ", ".join(f"[{f.name}] {f.direction.upper()}" for f in fields)
    parts.append(f"({col_list})")
    if included:
        parts.append("INCLUDE (" + ", ".join(f"[{c}]" for c in included) + ")")
    if condition:
        parts.append(f"WHERE {condition}")
    return " ".join(parts)
