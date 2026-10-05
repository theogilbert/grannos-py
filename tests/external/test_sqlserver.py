"""
Integration tests for the SQL Server driver.

Requires a running SQL Server instance. Configure via environment variables:
  MSSQL_HOST      (default: localhost)
  MSSQL_PORT      (default: 1433)
  MSSQL_USER      (default: sa)
  MSSQL_PASSWORD  (required — no default)
  MSSQL_DATABASE  (default: tempdb)

Tests are skipped automatically when mssql_python is not installed or the
server is unreachable.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator

import pytest

from grannos.drivers.base import ConnectionLostError, DriverSettings
from grannos.drivers.sqlserver import SQLServerDriver
from grannos.protocol import (
    EntityDescription,
    FieldDescription,
    IndexDescription,
    ReadResult,
    TableReference,
    WriteResult,
)

pytestmark = pytest.mark.external


def _params() -> dict:
    return {
        "host": os.environ.get("MSSQL_HOST", "localhost"),
        "port": int(os.environ.get("MSSQL_PORT", "1433")),
        "user": os.environ.get("MSSQL_USER", "sa"),
        "password": os.environ.get("MSSQL_PASSWORD", ""),
        "database": os.environ.get("MSSQL_DATABASE", "tempdb"),
    }


@pytest.fixture
async def driver() -> AsyncGenerator[SQLServerDriver, None]:
    pytest.importorskip("mssql_python")
    try:
        d = await SQLServerDriver.create(_params(), DriverSettings())
    except Exception as exc:
        pytest.skip(f"SQL Server not available: {exc}")
    yield d
    await d.disconnect()


@pytest.fixture
async def table(driver: SQLServerDriver) -> AsyncGenerator[str, None]:
    """Yields a unique table name and drops it on teardown."""
    name = f"t_{uuid.uuid4().hex[:12]}"
    yield name
    await driver.execute(
        f"IF OBJECT_ID(N'dbo.{name}', N'U') IS NOT NULL DROP TABLE dbo.{name}", []
    )


@pytest.fixture
async def tables(driver: SQLServerDriver) -> AsyncGenerator[tuple[str, str], None]:
    """Yields two unique table names and drops both on teardown."""
    parent = f"t_{uuid.uuid4().hex[:12]}"
    child = f"t_{uuid.uuid4().hex[:12]}"
    yield parent, child
    for name in (child, parent):  # child first to satisfy FK
        await driver.execute(
            f"IF OBJECT_ID(N'dbo.{name}', N'U') IS NOT NULL DROP TABLE dbo.{name}", []
        )


class TestExecute:
    async def test_should_return_columns_and_rows(
        self, driver: SQLServerDriver
    ) -> None:
        result = await driver.execute("SELECT 1 AS n, 'hello' AS s", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["n", "s"]
        assert result.rows == [[1, "hello"]]

    async def test_should_return_rows_affected_for_insert(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        result = await driver.execute(
            f"INSERT INTO dbo.{table} VALUES (?, ?)", [1, "hello"]
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_should_return_rows_affected_for_delete(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        await driver.execute(f"INSERT INTO dbo.{table} VALUES (?)", [1])
        await driver.execute(f"INSERT INTO dbo.{table} VALUES (?)", [2])
        result = await driver.execute(f"DELETE FROM dbo.{table}", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

    async def test_should_persist_inserts_within_connection(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"INSERT INTO dbo.{table} VALUES (?, ?)", [1, "hello"])
        result = await driver.execute(f"SELECT * FROM dbo.{table}", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["id", "val"]
        assert result.rows == [[1, "hello"]]


class TestCancel:
    # Runs for minutes uncancelled: a CPU-bound cross join, single-threaded.
    SLOW = (
        "SELECT SUM(CAST(a.object_id AS bigint) * b.column_id % 7 + c.column_id)"
        " FROM sys.all_columns a CROSS JOIN sys.all_columns b"
        " CROSS JOIN (SELECT TOP 20 column_id FROM sys.all_columns) c"
        " OPTION (MAXDOP 1)"
    )

    async def test_should_abort_running_query_on_server(
        self, driver: SQLServerDriver
    ) -> None:
        result = await driver.execute("SELECT @@SPID", [])
        assert isinstance(result, ReadResult)
        spid = result.rows[0][0]
        task = asyncio.create_task(driver.execute(self.SLOW, []))
        await asyncio.sleep(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        monitor = await SQLServerDriver.create(_params(), DriverSettings())
        try:
            for _ in range(10):  # the server notices the dropped link shortly
                result = await monitor.execute(
                    # The freed session id may be the monitor's own by now.
                    "SELECT COUNT(*) FROM sys.dm_exec_requests"
                    " WHERE session_id = ? AND session_id <> @@SPID",
                    [spid],
                )
                assert isinstance(result, ReadResult)
                if result.rows == [[0]]:
                    break
                await asyncio.sleep(0.5)
            assert result.rows == [[0]]
        finally:
            await monitor.disconnect()

    async def test_should_report_lost_connection_then_reconnect(
        self, driver: SQLServerDriver
    ) -> None:
        task = asyncio.create_task(driver.execute(self.SLOW, []))
        await asyncio.sleep(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        # The dispatcher answers this by reconnecting and retrying.
        with pytest.raises(ConnectionLostError):
            await driver.execute("SELECT 1", [])
        await driver.reconnect()
        result = await driver.execute("SELECT 1", [])
        assert isinstance(result, ReadResult)
        assert result.rows == [[1]]


class TestExploreList:
    async def test_should_list_dbo_schema(self, driver: SQLServerDriver) -> None:
        items = await driver.explore_list([])
        assert any(i.name == "dbo" and i.type == "schema" for i in items)

    async def test_should_not_list_system_schemas(
        self, driver: SQLServerDriver
    ) -> None:
        items = await driver.explore_list([])
        names = {i.name for i in items}
        assert "sys" not in names
        assert "INFORMATION_SCHEMA" not in names

    async def test_should_list_created_table(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        items = await driver.explore_list(["dbo"])
        assert any(i.name == table for i in items)

    async def test_should_return_groups_for_table(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        items = await driver.explore_list(["dbo", table])
        assert {i.name for i in items} == {"columns", "indices"}
        assert all(i.type == "group" and i.expandable for i in items)

    async def test_should_list_columns_in_ordinal_order(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50), active BIT)", []
        )
        items = await driver.explore_list(["dbo", table, "columns"])
        assert [i.name for i in items] == ["id", "val", "active"]
        assert all(not i.expandable for i in items)

    async def test_should_list_explicit_index(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val)", [])
        items = await driver.explore_list(["dbo", table, "indices"])
        assert any(i.name == "ix_val" for i in items)
        assert all(not i.expandable for i in items)

    async def test_should_list_unique_index(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE UNIQUE INDEX uix_val ON dbo.{table}(val)", [])
        items = await driver.explore_list(["dbo", table, "indices"])
        assert any(i.name == "uix_val" for i in items)

    async def test_should_return_empty_for_unknown_path(
        self, driver: SQLServerDriver
    ) -> None:
        assert (
            await driver.explore_list(["dbo", "no_such_table", "columns", "extra"])
            == []
        )


class TestExploreDescribe:
    async def test_should_return_field_metadata(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT NOT NULL, val VARCHAR(50) NULL)", []
        )
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        assert desc.schema == "dbo"
        assert desc.name == table
        assert desc.kind == "table"
        fields = {f.name: f for f in desc.properties}
        assert list(fields) == ["id", "val"]
        assert fields["id"].types == ["int"]
        assert fields["id"].nullable is False
        assert fields["val"].types == ["varchar"]
        assert fields["val"].nullable is True

    async def test_should_return_none_for_unknown_path(
        self, driver: SQLServerDriver
    ) -> None:
        assert await driver.explore_describe([]) is None

    async def test_columns_group_path_no_longer_resolves(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        assert await driver.explore_describe(["dbo", table, "columns"]) is None

    async def test_comment(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        await driver.execute(
            "EXEC sp_addextendedproperty"
            " @name = N'MS_Description', @value = N'A test table comment',"
            " @level0type = N'Schema', @level0name = N'dbo',"
            f" @level1type = N'Table', @level1name = N'{table}'",
            [],
        )
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        assert desc.comment == "A test table comment"

    async def test_comment_defaults_to_none(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        assert desc.comment is None

    async def test_data_type(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].types == ["int"]
        assert fields["val"].types == ["varchar"]

    async def test_pk_and_nullable(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT PRIMARY KEY, val VARCHAR(50) NOT NULL)",
            [],
        )
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].pk is True
        assert fields["val"].pk is False
        assert fields["id"].nullable is False
        assert fields["val"].nullable is False

    async def test_exclusive_index(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val)", [])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert len(fields["val"].exclusive_indices) == 1
        assert fields["val"].exclusive_indices[0].name == "ix_val"
        assert fields["id"].exclusive_indices == []
        assert fields["val"].composite_indices == []

    async def test_composite_index(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50), other VARCHAR(50))", []
        )
        await driver.execute(f"CREATE INDEX ix_vc ON dbo.{table}(val, other)", [])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert len(fields["val"].composite_indices) == 1
        assert fields["val"].exclusive_indices == []

    async def test_field_comment(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(
            "EXEC sp_addextendedproperty"
            " @name = N'MS_Description', @value = N'A test comment',"
            " @level0type = N'Schema', @level0name = N'dbo',"
            f" @level1type = N'Table', @level1name = N'{table}',"
            " @level2type = N'Column', @level2name = N'val'",
            [],
        )
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["val"].comment == "A test comment"
        assert fields["id"].comment is None

    async def test_sample_values(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(10))", [])
        for i, v in enumerate(["x", "y", "z", "x"]):
            await driver.execute(f"INSERT INTO dbo.{table} VALUES (?, ?)", [i, v])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        sample = fields["val"].sample
        assert len(sample) <= 3
        assert set(sample).issubset({"x", "y", "z"})

    async def test_returns_outgoing_references(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", child])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["parent_id"].outgoing_references == [
            TableReference(
                table=child,
                schema="dbo",
                column="parent_id",
                ref_table=parent,
                ref_schema="dbo",
                ref_column="id",
                constraint_name=f"fk_{child}",
            )
        ]
        assert fields["id"].outgoing_references == []

    async def test_returns_incoming_references(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", parent])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].incoming_references == [
            TableReference(
                table=child,
                schema="dbo",
                column="parent_id",
                ref_table=parent,
                ref_schema="dbo",
                ref_column="id",
                constraint_name=f"fk_{child}",
            )
        ]

    async def test_references_default_to_empty(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        desc = await driver.explore_describe(["dbo", table])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].outgoing_references == []
        assert fields["id"].incoming_references == []

    async def test_outgoing_reference_is_unique_when_fk_column_has_unique_constraint(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  parent_id INT CONSTRAINT uq_{child} UNIQUE"
            f"    CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", child])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["parent_id"].outgoing_references[0].unique is True

    async def test_outgoing_reference_is_not_unique_by_default(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", child])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["parent_id"].outgoing_references[0].unique is False

    async def test_incoming_reference_is_unique_when_fk_column_has_unique_constraint(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  parent_id INT CONSTRAINT uq_{child} UNIQUE"
            f"    CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", parent])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].incoming_references[0].unique is True

    async def test_incoming_reference_is_not_unique_by_default(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", parent])
        assert isinstance(desc, EntityDescription)
        fields = {f.name: f for f in desc.properties}
        assert fields["id"].incoming_references[0].unique is False

    async def test_should_describe_a_relationship(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(
            ["dbo", child, "relationships", "parent_id"]
        )
        assert isinstance(desc, TableReference)
        assert desc.table == child
        assert desc.schema == "dbo"
        assert desc.column == "parent_id"
        assert desc.ref_table == parent
        assert desc.ref_schema == "dbo"
        assert desc.ref_column == "id"
        assert desc.constraint_name == f"fk_{child}"

    async def test_should_return_none_for_unknown_relationship_column(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        result = await driver.explore_describe(["dbo", table, "relationships", "id"])
        assert result is None


class TestExploreDescribeIndex:
    async def test_basic_fields_and_direction(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_val"])
        assert isinstance(desc, IndexDescription)
        assert desc.name == "ix_val"
        assert len(desc.fields) == 1
        assert desc.fields[0].name == "val"
        assert desc.fields[0].direction == "asc"

    async def test_descending_direction(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val INT)", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val DESC)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_val"])
        assert isinstance(desc, IndexDescription)
        assert desc.fields[0].direction == "desc"

    async def test_unique_index(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, email VARCHAR(100))", []
        )
        await driver.execute(f"CREATE UNIQUE INDEX uix_email ON dbo.{table}(email)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "uix_email"])
        assert isinstance(desc, IndexDescription)
        assert desc.unique is True

    async def test_clustered_index(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE CLUSTERED INDEX cix_id ON dbo.{table}(id)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "cix_id"])
        assert isinstance(desc, IndexDescription)
        assert desc.clustered is True
        assert desc.index_type == "clustered"

    async def test_included_columns(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50), extra VARCHAR(50))",
            [],
        )
        await driver.execute(
            f"CREATE INDEX ix_val ON dbo.{table}(val) INCLUDE (extra)", []
        )
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_val"])
        assert isinstance(desc, IndexDescription)
        assert desc.included_columns == ["extra"]

    async def test_filtered_index_ddl_contains_where(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50), active BIT)", []
        )
        await driver.execute(
            f"CREATE INDEX ix_active ON dbo.{table}(val) WHERE active = 1", []
        )
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_active"])
        assert isinstance(desc, IndexDescription)
        assert desc.ddl is not None
        assert "active" in desc.ddl

    async def test_multi_column_index_field_order(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, last VARCHAR(50), first VARCHAR(50))",
            [],
        )
        await driver.execute(f"CREATE INDEX ix_name ON dbo.{table}(last, first)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_name"])
        assert isinstance(desc, IndexDescription)
        assert [f.name for f in desc.fields] == ["last", "first"]

    async def test_ddl_populated(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val)", [])
        desc = await driver.explore_describe(["dbo", table, "indices", "ix_val"])
        assert isinstance(desc, IndexDescription)
        assert desc.ddl is not None
        assert "ix_val" in desc.ddl

    async def test_unknown_index_returns_none(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        assert (
            await driver.explore_describe(["dbo", table, "indices", "no_such_idx"])
            is None
        )


class TestExploreDescribeField:
    async def test_basic_fields(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50) NOT NULL)", []
        )
        desc = await driver.explore_describe(["dbo", table, "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert desc.name == "val"
        assert desc.types == ["varchar"]
        assert desc.nullable is False
        assert desc.pk is False

    async def test_pk_column(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE dbo.{table} (id INT PRIMARY KEY, val VARCHAR(50))", []
        )
        desc = await driver.explore_describe(["dbo", table, "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.pk is True

    async def test_exclusive_index(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(f"CREATE INDEX ix_val ON dbo.{table}(val)", [])
        desc = await driver.explore_describe(["dbo", table, "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.exclusive_indices) == 1
        assert desc.exclusive_indices[0].name == "ix_val"
        assert desc.composite_indices == []

    async def test_comment(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(50))", [])
        await driver.execute(
            "EXEC sp_addextendedproperty"
            " @name = N'MS_Description', @value = N'Column comment',"
            " @level0type = N'Schema', @level0name = N'dbo',"
            f" @level1type = N'Table', @level1name = N'{table}',"
            " @level2type = N'Column', @level2name = N'val'",
            [],
        )
        desc = await driver.explore_describe(["dbo", table, "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert desc.comment == "Column comment"

    async def test_sample_values(self, driver: SQLServerDriver, table: str) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT, val VARCHAR(10))", [])
        for i, v in enumerate(["a", "b", "c", "a"]):
            await driver.execute(f"INSERT INTO dbo.{table} VALUES (?, ?)", [i, v])
        desc = await driver.explore_describe(["dbo", table, "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.sample) <= 3
        assert set(desc.sample).issubset({"a", "b", "c"})

    async def test_outgoing_references(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", child, "columns", "parent_id"])
        assert isinstance(desc, FieldDescription)
        assert desc.outgoing_references == [
            TableReference(
                table=child,
                schema="dbo",
                column="parent_id",
                ref_table=parent,
                ref_schema="dbo",
                ref_column="id",
                constraint_name=f"fk_{child}",
            )
        ]

    async def test_incoming_references(
        self, driver: SQLServerDriver, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE dbo.{parent} (id INT CONSTRAINT pk_{parent} PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE dbo.{child} ("
            f"  id INT,"
            f"  parent_id INT CONSTRAINT fk_{child} REFERENCES dbo.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(["dbo", parent, "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.incoming_references == [
            TableReference(
                table=child,
                schema="dbo",
                column="parent_id",
                ref_table=parent,
                ref_schema="dbo",
                ref_column="id",
                constraint_name=f"fk_{child}",
            )
        ]

    async def test_empty_outgoing_references_when_not_fk(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        desc = await driver.explore_describe(["dbo", table, "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.outgoing_references == []

    async def test_unknown_column_returns_none(
        self, driver: SQLServerDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE dbo.{table} (id INT)", [])
        assert (
            await driver.explore_describe(["dbo", table, "columns", "no_such_col"])
            is None
        )
