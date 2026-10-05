"""
Integration tests for the Oracle driver.

Requires a running Oracle instance. Configure via environment variables:
  ORACLE_HOST      (default: localhost)
  ORACLE_PORT      (default: 1521)
  ORACLE_USER      (default: testuser)
  ORACLE_PASSWORD  (required — no default)
  ORACLE_SERVICE   (default: FREEPDB1)

Tests are skipped automatically when oracledb is not installed or the
server is unreachable.
"""

import asyncio
import dataclasses
import os
import uuid
from collections.abc import AsyncGenerator

import pytest

from grannos.drivers.base import DriverSettings
from grannos.drivers.oracle import OracleDriver
from grannos.protocol import (
    EntityDescription,
    FieldDescription,
    IndexDescription,
    MessageLevel,
    NodeType,
    ReadResult,
    SearchScope,
    TableReference,
    WriteResult,
)

pytestmark = pytest.mark.external


def _params() -> dict:
    return {
        "host": os.environ.get("ORACLE_HOST", "localhost"),
        "port": int(os.environ.get("ORACLE_PORT", "1521")),
        "user": os.environ.get("ORACLE_USER", "testuser"),
        "password": os.environ.get("ORACLE_PASSWORD", ""),
        "service_name": os.environ.get("ORACLE_SERVICE", "FREEPDB1"),
    }


@pytest.fixture
async def driver() -> AsyncGenerator[OracleDriver, None]:
    pytest.importorskip("oracledb")
    try:
        d = await OracleDriver.create(_params(), DriverSettings())
    except Exception as exc:
        pytest.skip(f"Oracle not available: {exc}")
    yield d
    await d.disconnect()


@pytest.fixture
async def schema(driver: OracleDriver) -> str:
    result = await driver.execute("SELECT USER FROM DUAL", [])
    assert isinstance(result, ReadResult)
    return result.rows[0][0]


@pytest.fixture
async def driver2() -> AsyncGenerator[OracleDriver, None]:
    pytest.importorskip("oracledb")
    password2 = os.environ.get("ORACLE_PASSWORD2", "")
    if not password2:
        pytest.skip("ORACLE_PASSWORD2 not set; skipping cross-schema index tests")
    params = {
        **_params(),
        "user": os.environ.get("ORACLE_USER2", "testuser2"),
        "password": password2,
    }
    try:
        d = await OracleDriver.create(params, DriverSettings())
    except Exception as exc:
        pytest.skip(f"Second Oracle user not available: {exc}")
    yield d
    await d.disconnect()


@pytest.fixture
async def schema2(driver2: OracleDriver) -> str:
    result = await driver2.execute("SELECT USER FROM DUAL", [])
    assert isinstance(result, ReadResult)
    return result.rows[0][0]


@pytest.fixture
async def table(driver: OracleDriver, schema: str) -> AsyncGenerator[str, None]:
    name = "T_" + uuid.uuid4().hex[:12].upper()
    yield name
    try:
        await driver.execute(
            f"DROP TABLE {schema}.{name} CASCADE CONSTRAINTS PURGE", []
        )
    except Exception:
        pass


@pytest.fixture
async def synonym(driver: OracleDriver, schema: str) -> AsyncGenerator[str, None]:
    name = "S_" + uuid.uuid4().hex[:12].upper()
    yield name
    try:
        await driver.execute(f"DROP SYNONYM {schema}.{name}", [])
    except Exception:
        pass


@pytest.fixture
async def synonyms(
    driver: OracleDriver, schema: str
) -> AsyncGenerator[tuple[str, str], None]:
    first = "S_" + uuid.uuid4().hex[:12].upper()
    second = "S_" + uuid.uuid4().hex[:12].upper()
    yield first, second
    for name in (second, first):
        try:
            await driver.execute(f"DROP SYNONYM {schema}.{name}", [])
        except Exception:
            pass


@pytest.fixture
async def public_synonym(driver: OracleDriver) -> AsyncGenerator[str, None]:
    name = "S_" + uuid.uuid4().hex[:12].upper()
    yield name
    try:
        await driver.execute(f"DROP PUBLIC SYNONYM {name}", [])
    except Exception:
        pass


@pytest.fixture
async def tables(
    driver: OracleDriver, schema: str
) -> AsyncGenerator[tuple[str, str], None]:
    parent = "T_" + uuid.uuid4().hex[:12].upper()
    child = "T_" + uuid.uuid4().hex[:12].upper()
    yield parent, child
    for name in (child, parent):
        try:
            await driver.execute(
                f"DROP TABLE {schema}.{name} CASCADE CONSTRAINTS PURGE", []
            )
        except Exception:
            pass


class TestReconnect:
    async def test_reconnect_succeeds_when_connection_is_dead(
        self, driver: OracleDriver
    ) -> None:
        await driver._conn.close()
        await driver.reconnect()
        result = await driver.execute("SELECT 1 FROM DUAL", [])
        assert isinstance(result, ReadResult)


class TestExecute:
    async def test_should_return_columns_and_rows(self, driver: OracleDriver) -> None:
        result = await driver.execute("SELECT 1 AS n, 'hello' AS s FROM DUAL", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["N", "S"]
        assert result.rows == [[1, "hello"]]

    async def test_should_support_positional_params(self, driver: OracleDriver) -> None:
        result = await driver.execute("SELECT :1 AS val FROM DUAL", [42])
        assert isinstance(result, ReadResult)
        assert result.rows == [[42]]

    async def test_should_return_dml_result_for_insert(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        result = await driver.execute(
            f"INSERT INTO {schema}.{table} VALUES (:1, :2)", [1, "hello"]
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_should_return_dml_result_for_delete(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        await driver.execute(f"INSERT INTO {schema}.{table} VALUES (:1)", [1])
        await driver.execute(f"INSERT INTO {schema}.{table} VALUES (:1)", [2])
        result = await driver.execute(f"DELETE FROM {schema}.{table}", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

    async def test_should_persist_inserts_within_connection(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        await driver.execute(
            f"INSERT INTO {schema}.{table} VALUES (:1, :2)", [1, "hello"]
        )
        result = await driver.execute(f"SELECT id, val FROM {schema}.{table}", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["ID", "VAL"]
        assert result.rows == [[1, "hello"]]


class TestCancel:
    async def test_should_abort_running_query(self, driver: OracleDriver) -> None:
        # No explicit conn.cancel(): oracledb's async thin mode already sends
        # a break when the awaiting task is cancelled, and calling it on top
        # of that leaves the connection hanging.
        task = asyncio.create_task(
            driver.execute(
                "SELECT COUNT(*) FROM all_objects a, all_objects b, all_objects c"
            )
        )
        await asyncio.sleep(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        result = await asyncio.wait_for(
            driver.execute("SELECT 1 AS n FROM dual"), timeout=5
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [[1]]


class TestUndecodableBytes:
    """Bytes that aren't valid in the database character set — a Latin-1 string
    loaded into an AL32UTF8 database, say — reach the client as U+FFFD instead
    of failing the whole request."""

    BAD_BYTES = "UTL_RAW.CAST_TO_VARCHAR2(HEXTORAW('41FF42'))"
    """A VARCHAR2 holding 'A', 0xFF, 'B' — 0xFF is not valid UTF-8."""

    async def test_execute_replaces_undecodable_bytes(
        self, driver: OracleDriver
    ) -> None:
        result = await driver.execute(f"SELECT {self.BAD_BYTES} AS bad FROM DUAL", [])
        assert isinstance(result, ReadResult)
        assert result.rows == [["A\ufffdB"]]

    async def test_describe_samples_replace_undecodable_bytes(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        await driver.execute(
            f"INSERT INTO {schema}.{table} VALUES (1, {self.BAD_BYTES})", []
        )
        description = await driver.explore_describe([schema, table])
        assert isinstance(description, EntityDescription)
        val = next(p for p in description.properties if p.name == "VAL")
        assert val.sample == ["A\ufffdB"]


class TestExploreList:
    async def test_root_lists_non_system_schemas(
        self, driver: OracleDriver, schema: str
    ) -> None:
        items = await driver.explore_list([])
        names = {i.name for i in items}
        assert schema in names
        assert "SYS" not in names
        assert "SYSTEM" not in names
        assert all(i.type == "schema" for i in items)
        assert all(i.expandable for i in items)

    async def test_schema_lists_created_table(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        items = await driver.explore_list([schema])
        assert any(i.name == table and i.type == "table" for i in items)

    async def test_schema_lists_view(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        view = table + "_V"
        try:
            await driver.execute(
                f"CREATE VIEW {schema}.{view} AS SELECT * FROM {schema}.{table}", []
            )
            items = await driver.explore_list([schema])
            assert any(i.name == view and i.type == "view" for i in items)
        finally:
            try:
                await driver.execute(f"DROP VIEW {schema}.{view}", [])
            except Exception:
                pass

    async def test_table_returns_groups(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        items = await driver.explore_list([schema, table])
        assert {i.name for i in items} == {"columns", "indexes"}
        assert all(i.type == "group" and i.expandable for i in items)

    async def test_columns_lists_in_ordinal_order(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50), active NUMBER(1))",
            [],
        )
        items = await driver.explore_list([schema, table, "columns"])
        assert [i.name for i in items] == ["ID", "VAL", "ACTIVE"]
        assert all(not i.expandable for i in items)

    async def test_columns_reflect_data_type(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, name VARCHAR2(100))", []
        )
        items = await driver.explore_list([schema, table, "columns"])
        by_name = {i.name: i for i in items}
        assert by_name["ID"].type == "NUMBER"
        assert by_name["NAME"].type == "VARCHAR2"

    async def test_indexes_lists_created_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val)", []
        )
        items = await driver.explore_list([schema, table, "indexes"])
        assert any(i.name == idx for i in items)
        assert all(not i.expandable for i in items)

    async def test_indexes_empty_when_none_exist(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        items = await driver.explore_list([schema, table, "indexes"])
        assert items == []

    async def test_unknown_path_returns_empty(self, driver: OracleDriver) -> None:
        assert (
            await driver.explore_list(["NO_SUCH_SCHEMA", "NO_SUCH_TABLE", "extra"])
            == []
        )


class TestExploreDescribe:
    async def test_returns_field_metadata(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER NOT NULL, val VARCHAR2(50))", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        assert desc.schema == schema
        assert desc.name == table
        assert desc.kind == "table"
        by_name = {f.name: f for f in desc.properties}
        assert list(by_name) == ["ID", "VAL"]
        assert by_name["ID"].types == ["NUMBER"]
        assert by_name["ID"].nullable is False
        assert by_name["VAL"].types == ["VARCHAR2(50)"]
        assert by_name["VAL"].nullable is True

    async def test_returns_pk_flag(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER PRIMARY KEY, val VARCHAR2(50))",
            [],
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].pk is True
        assert by_name["VAL"].pk is False

    async def test_returns_default_value(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER DEFAULT 0, val VARCHAR2(50))", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].default == "0"
        assert by_name["VAL"].default is None

    async def test_returns_none_for_unknown_path(self, driver: OracleDriver) -> None:
        assert await driver.explore_describe([]) is None

    async def test_columns_group_path_no_longer_resolves(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        assert await driver.explore_describe([schema, table, "columns"]) is None

    async def test_comment(self, driver: OracleDriver, schema: str, table: str) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        await driver.execute(
            f"COMMENT ON TABLE {schema}.{table} IS 'A test table comment'", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        assert desc.comment == "A test table comment"

    async def test_comment_defaults_to_none(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        assert desc.comment is None
        assert await driver.explore_describe(["SYS"]) is None

    async def test_data_type(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].types == ["NUMBER"]
        assert by_name["VAL"].types == ["VARCHAR2(50)"]

    async def test_exclusive_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val)", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert len(by_name["VAL"].exclusive_indices) == 1
        assert by_name["VAL"].exclusive_indices[0].name == idx
        assert by_name["ID"].exclusive_indices == []
        assert by_name["VAL"].composite_indices == []

    async def test_composite_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table}"
            f" (id NUMBER, val VARCHAR2(50), other VARCHAR2(50))",
            [],
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val, other)", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert len(by_name["VAL"].composite_indices) == 1
        assert by_name["VAL"].exclusive_indices == []

    async def test_field_comment(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        await driver.execute(
            f"COMMENT ON COLUMN {schema}.{table}.val IS 'A test comment'", []
        )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["VAL"].comment == "A test comment"
        assert by_name["ID"].comment is None

    async def test_sample_values(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(10))", []
        )
        for i, v in enumerate(["x", "y", "z", "x"]):
            await driver.execute(
                f"INSERT INTO {schema}.{table} VALUES (:1, :2)", [i, v]
            )
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        sample = by_name["VAL"].sample
        assert len(sample) <= 3
        assert set(sample).issubset({"x", "y", "z"})

    async def test_returns_outgoing_references(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, child])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        ref = by_name["PARENT_ID"].outgoing_references[0]
        # The constraint is unnamed, so Oracle auto-generates an unpredictable
        # SYS_C* name — assert it's present without pinning its exact value.
        assert ref.constraint_name is not None
        assert dataclasses.replace(ref, constraint_name=None) == TableReference(
            table=child,
            schema=schema,
            column="PARENT_ID",
            ref_table=parent,
            ref_schema=schema,
            ref_column="ID",
        )
        assert by_name["ID"].outgoing_references == []

    async def test_returns_incoming_references(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, parent])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        ref = by_name["ID"].incoming_references[0]
        assert ref.constraint_name is not None
        assert dataclasses.replace(ref, constraint_name=None) == TableReference(
            table=child,
            schema=schema,
            column="PARENT_ID",
            ref_table=parent,
            ref_schema=schema,
            ref_column="ID",
        )

    async def test_references_default_to_empty(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        desc = await driver.explore_describe([schema, table])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].outgoing_references == []
        assert by_name["ID"].incoming_references == []

    async def test_outgoing_reference_is_unique_when_fk_column_has_unique_constraint(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  parent_id NUMBER UNIQUE,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, child])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["PARENT_ID"].outgoing_references[0].unique is True

    async def test_outgoing_reference_is_not_unique_by_default(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, child])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["PARENT_ID"].outgoing_references[0].unique is False

    async def test_incoming_reference_is_unique_when_fk_column_has_unique_constraint(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  parent_id NUMBER UNIQUE,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, parent])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].incoming_references[0].unique is True

    async def test_incoming_reference_is_not_unique_by_default(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, parent])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["ID"].incoming_references[0].unique is False

    async def test_should_describe_a_relationship(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe(
            [schema, child, "relationships", "parent_id"]
        )
        assert isinstance(desc, TableReference)
        assert desc.table == child
        assert desc.schema == schema
        assert desc.column == "PARENT_ID"
        assert desc.ref_table == parent
        assert desc.ref_schema == schema
        assert desc.ref_column == "ID"
        assert desc.constraint_name is not None

    async def test_should_return_none_for_unknown_relationship_column(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        result = await driver.explore_describe([schema, table, "relationships", "id"])
        assert result is None


class TestExploreDescribeIndex:
    async def test_basic_fields_and_type(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes", idx])
        assert isinstance(desc, IndexDescription)
        assert desc.name == idx
        assert len(desc.fields) == 1
        assert desc.fields[0].name == "VAL"
        assert desc.fields[0].direction == "asc"

    async def test_descending_direction(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val NUMBER)", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val DESC)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes", idx])
        assert isinstance(desc, IndexDescription)
        assert desc.fields[0].direction == "desc"

    async def test_unique_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, email VARCHAR2(100))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE UNIQUE INDEX {schema}.{idx} ON {schema}.{table}(email)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes", idx])
        assert isinstance(desc, IndexDescription)
        assert desc.unique is True

    async def test_multi_column_index_field_order(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table}"
            f" (id NUMBER, last VARCHAR2(50), first VARCHAR2(50))",
            [],
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(last, first)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes", idx])
        assert isinstance(desc, IndexDescription)
        assert [f.name for f in desc.fields] == ["LAST", "FIRST"]

    async def test_ddl_populated_for_user_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes", idx])
        assert isinstance(desc, IndexDescription)
        assert desc.ddl is not None
        assert idx in desc.ddl

    async def test_unknown_index_returns_none(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        assert (
            await driver.explore_describe([schema, table, "indexes", "NO_SUCH_IDX"])
            is None
        )


class TestExploreDescribeField:
    async def test_basic_fields(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50) NOT NULL)", []
        )
        desc = await driver.explore_describe([schema, table, "columns", "VAL"])
        assert isinstance(desc, FieldDescription)
        assert desc.name == "VAL"
        assert desc.types == ["VARCHAR2(50)"]
        assert desc.nullable is False
        assert desc.pk is False

    async def test_pk_column(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER PRIMARY KEY, val VARCHAR2(50))",
            [],
        )
        desc = await driver.explore_describe([schema, table, "columns", "ID"])
        assert isinstance(desc, FieldDescription)
        assert desc.pk is True

    async def test_exclusive_index(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver.execute(
            f"CREATE INDEX {schema}.{idx} ON {schema}.{table}(val)", []
        )
        desc = await driver.explore_describe([schema, table, "columns", "VAL"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.exclusive_indices) == 1
        assert desc.exclusive_indices[0].name == idx
        assert desc.composite_indices == []

    async def test_comment(self, driver: OracleDriver, schema: str, table: str) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        await driver.execute(
            f"COMMENT ON COLUMN {schema}.{table}.val IS 'Column comment'", []
        )
        desc = await driver.explore_describe([schema, table, "columns", "VAL"])
        assert isinstance(desc, FieldDescription)
        assert desc.comment == "Column comment"

    async def test_sample_values(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(10))", []
        )
        for i, v in enumerate(["a", "b", "c", "a"]):
            await driver.execute(
                f"INSERT INTO {schema}.{table} VALUES (:1, :2)", [i, v]
            )
        desc = await driver.explore_describe([schema, table, "columns", "VAL"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.sample) <= 3
        assert set(desc.sample).issubset({"a", "b", "c"})

    async def test_outgoing_references(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, child, "columns", "PARENT_ID"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.outgoing_references) == 1
        ref = desc.outgoing_references[0]
        assert dataclasses.replace(ref, constraint_name=None) == TableReference(
            table=child,
            schema=schema,
            column="PARENT_ID",
            ref_table=parent,
            ref_schema=schema,
            ref_column="ID",
        )

    async def test_incoming_references(
        self, driver: OracleDriver, schema: str, tables: tuple[str, str]
    ) -> None:
        parent, child = tables
        await driver.execute(
            f"CREATE TABLE {schema}.{parent} (id NUMBER PRIMARY KEY)", []
        )
        await driver.execute(
            f"CREATE TABLE {schema}.{child} ("
            f"  id NUMBER, parent_id NUMBER,"
            f"  FOREIGN KEY (parent_id) REFERENCES {schema}.{parent}(id)"
            ")",
            [],
        )
        desc = await driver.explore_describe([schema, parent, "columns", "ID"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.incoming_references) == 1
        ref = desc.incoming_references[0]
        assert dataclasses.replace(ref, constraint_name=None) == TableReference(
            table=child,
            schema=schema,
            column="PARENT_ID",
            ref_table=parent,
            ref_schema=schema,
            ref_column="ID",
        )

    async def test_empty_outgoing_references_when_not_fk(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        desc = await driver.explore_describe([schema, table, "columns", "ID"])
        assert isinstance(desc, FieldDescription)
        assert desc.outgoing_references == []

    async def test_unknown_column_returns_none(
        self, driver: OracleDriver, schema: str, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (id NUMBER)", [])
        assert (
            await driver.explore_describe([schema, table, "columns", "NO_SUCH_COL"])
            is None
        )


class TestCrossSchemaIndex:
    async def test_indices_ddl_for_index_owned_by_different_schema(
        self,
        driver: OracleDriver,
        driver2: OracleDriver,
        schema: str,
        schema2: str,
        table: str,
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (id NUMBER, val VARCHAR2(50))", []
        )
        idx = "IDX_" + table
        await driver2.execute(
            f"CREATE INDEX {schema2}.{idx} ON {schema}.{table}(val)", []
        )
        desc = await driver.explore_describe([schema, table, "indexes"])
        assert isinstance(desc, list)
        indices = [i for i in desc if isinstance(i, IndexDescription)]
        by_name = {i.name: i for i in indices}
        assert idx in by_name
        ddl = by_name[idx].ddl
        assert ddl is not None
        assert idx in ddl


@pytest.fixture
async def proc(driver: OracleDriver) -> AsyncGenerator[str, None]:
    name = "P_" + uuid.uuid4().hex[:12].upper()
    yield name
    try:
        await driver.execute(f"DROP PROCEDURE {name}", [])
    except Exception:
        pass


class TestExecuteMessages:
    async def test_dbms_output_is_returned_as_info_messages(
        self, driver: OracleDriver
    ) -> None:
        result = await driver.execute(
            "BEGIN DBMS_OUTPUT.PUT_LINE('one'); DBMS_OUTPUT.PUT_LINE('two'); END;", []
        )
        assert [(m.level, m.text) for m in result.messages] == [
            (MessageLevel.INFO, "one"),
            (MessageLevel.INFO, "two"),
        ]

    async def test_statement_without_output_has_no_messages(
        self, driver: OracleDriver
    ) -> None:
        result = await driver.execute("SELECT 1 FROM DUAL", [])
        assert result.messages == []

    async def test_output_does_not_leak_into_the_next_statement(
        self, driver: OracleDriver
    ) -> None:
        await driver.execute("BEGIN DBMS_OUTPUT.PUT_LINE('first'); END;", [])
        result = await driver.execute("SELECT 1 FROM DUAL", [])
        assert result.messages == []

    async def test_output_from_a_function_called_by_a_select(
        self, driver: OracleDriver
    ) -> None:
        name = "F_" + uuid.uuid4().hex[:12].upper()
        await driver.execute(
            f"CREATE OR REPLACE FUNCTION {name} RETURN NUMBER AS "
            "BEGIN DBMS_OUTPUT.PUT_LINE('from function'); RETURN 1; END;",
            [],
        )
        try:
            result = await driver.execute(f"SELECT {name} FROM DUAL", [])
            assert isinstance(result, ReadResult)
            assert result.rows == [[1]]
            assert [m.text for m in result.messages] == ["from function"]
        finally:
            await driver.execute(f"DROP FUNCTION {name}", [])

    async def test_output_survives_reconnect(self, driver: OracleDriver) -> None:
        await driver.reconnect()
        result = await driver.execute("BEGIN DBMS_OUTPUT.PUT_LINE('alive'); END;", [])
        assert [m.text for m in result.messages] == ["alive"]

    async def test_compilation_errors_are_returned_as_warnings(
        self, driver: OracleDriver, proc: str
    ) -> None:
        result = await driver.execute(
            f"CREATE OR REPLACE PROCEDURE {proc} AS\nBEGIN\n    no_such_thing();\nEND;",
            [],
        )
        assert isinstance(result, WriteResult)
        warnings = [m for m in result.messages if m.level == MessageLevel.WARNING]
        assert any("PLS-00201" in m.text for m in warnings)

    async def test_compilation_error_position_points_at_the_query(
        self, driver: OracleDriver, proc: str
    ) -> None:
        query = (
            "-- leading comment\n"
            f"CREATE OR REPLACE PROCEDURE {proc} AS\n"
            "BEGIN\n"
            "    no_such_thing();\n"
            "END;"
        )
        result = await driver.execute(query, [])
        bad = next(m for m in result.messages if "PLS-00201" in m.text)
        # Line 4 col 5 is the "no_such_thing()" call in the submitted text.
        assert bad.line is not None
        assert (bad.line, bad.col) == (4, 5)
        assert query.splitlines()[bad.line - 1].strip().startswith("no_such_thing")

    async def test_procedure_that_compiles_has_no_warnings(
        self, driver: OracleDriver, proc: str
    ) -> None:
        result = await driver.execute(
            f"CREATE OR REPLACE PROCEDURE {proc} AS BEGIN NULL; END;", []
        )
        assert result.messages == []


class TestExploreFind:
    """The find queries run against the real data dictionary — the unit tests
    mock the cursor and so never prove the SQL parses."""

    async def test_finds_a_table_without_a_schema_scope(
        self, driver: OracleDriver, table: str, schema: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        assert await driver.explore_find(NodeType.TABLE, table, []) == [[schema, table]]

    async def test_found_path_is_describable(
        self, driver: OracleDriver, table: str
    ) -> None:
        """The whole point of matching the tree's own schema filter: a path a
        find returns has to be one explore.describe accepts."""
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        (path,) = await driver.explore_find(NodeType.TABLE, table, [])
        described = await driver.explore_describe(path)
        assert isinstance(described, EntityDescription)
        assert described.name == table

    async def test_lower_case_symbol_matches_the_folded_identifier(
        self, driver: OracleDriver, table: str, schema: str
    ) -> None:
        """A symbol read out of a query buffer is rarely upper-cased."""
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        assert await driver.explore_find(NodeType.TABLE, table.lower(), []) == [
            [schema, table]
        ]

    async def test_finds_a_view(self, driver: OracleDriver, table: str) -> None:
        view = f"V_{table}"
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE VIEW {view} AS SELECT * FROM {table}", [])
        try:
            paths = await driver.explore_find(NodeType.VIEW, view, [])
            assert [p[1] for p in paths] == [view]
        finally:
            await driver.execute(f"DROP VIEW {view}", [])

    async def test_finds_a_column_scoped_by_its_table(
        self, driver: OracleDriver, tables: tuple[str, str], schema: str
    ) -> None:
        """A column name is never hovered without a table in the query, and the
        same name in two tables must resolve to only the scoped one."""
        first, second = tables
        await driver.execute(f"CREATE TABLE {first} (shared_col NUMBER)", [])
        await driver.execute(f"CREATE TABLE {second} (shared_col NUMBER)", [])
        paths = await driver.explore_find(
            NodeType.COLUMN,
            "shared_col",
            [SearchScope(name=first, type=NodeType.TABLE)],
        )
        assert paths == [[schema, first, "columns", "SHARED_COL"]]

    async def test_unscoped_column_returns_every_candidate(
        self, driver: OracleDriver, tables: tuple[str, str]
    ) -> None:
        """Ambiguity is the client's picker to resolve, not an error."""
        first, second = tables
        await driver.execute(f"CREATE TABLE {first} (shared_col NUMBER)", [])
        await driver.execute(f"CREATE TABLE {second} (shared_col NUMBER)", [])
        paths = await driver.explore_find(NodeType.COLUMN, "shared_col", [])
        assert sorted(p[1] for p in paths) == sorted([first, second])

    async def test_found_column_path_is_describable(
        self, driver: OracleDriver, table: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        (path,) = await driver.explore_find(
            NodeType.COLUMN, "id", [SearchScope(name=table, type=NodeType.TABLE)]
        )
        described = await driver.explore_describe(path)
        assert isinstance(described, FieldDescription)
        assert described.name == "ID"

    async def test_finds_an_index_under_its_table(
        self, driver: OracleDriver, table: str, schema: str
    ) -> None:
        index = f"I_{table}"
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE INDEX {index} ON {table} (id)", [])
        assert await driver.explore_find(NodeType.INDEX, index, []) == [
            [schema, table, "indexes", index]
        ]

    async def test_found_index_path_is_describable(
        self, driver: OracleDriver, table: str
    ) -> None:
        index = f"I_{table}"
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE INDEX {index} ON {table} (id)", [])
        (path,) = await driver.explore_find(NodeType.INDEX, index, [])
        described = await driver.explore_describe(path)
        assert isinstance(described, IndexDescription)
        assert described.name == index

    async def test_schema_scope_excludes_another_schema(
        self, driver: OracleDriver, table: str, schema: str, schema2: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        assert (
            await driver.explore_find(
                NodeType.TABLE, table, [SearchScope(name=schema2, type=NodeType.SCHEMA)]
            )
            == []
        )
        assert await driver.explore_find(
            NodeType.TABLE, table, [SearchScope(name=schema, type=NodeType.SCHEMA)]
        ) == [[schema, table]]

    async def test_finds_a_table_through_a_synonym(
        self, driver: OracleDriver, table: str, synonym: str, schema: str
    ) -> None:
        """The tree holds no synonyms, so the synonym resolves to the path of
        the table it points at."""
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE SYNONYM {synonym} FOR {table}", [])
        assert await driver.explore_find(NodeType.TABLE, synonym, []) == [
            [schema, table]
        ]

    async def test_found_synonym_path_is_describable(
        self, driver: OracleDriver, table: str, synonym: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE SYNONYM {synonym} FOR {table}", [])
        (path,) = await driver.explore_find(NodeType.TABLE, synonym, [])
        described = await driver.explore_describe(path)
        assert isinstance(described, EntityDescription)
        assert described.name == table

    async def test_every_synonym_of_a_table_resolves_to_the_same_path(
        self, driver: OracleDriver, table: str, synonyms: tuple[str, str], schema: str
    ) -> None:
        """What makes one describe serve them all: the explore cache is keyed
        by path, so the second name costs no describe."""
        first, second = synonyms
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE SYNONYM {first} FOR {table}", [])
        await driver.execute(f"CREATE SYNONYM {second} FOR {table}", [])
        assert (
            await driver.explore_find(NodeType.TABLE, first, [])
            == await driver.explore_find(NodeType.TABLE, second, [])
            == [[schema, table]]
        )

    async def test_finds_a_table_through_a_public_synonym(
        self, driver: OracleDriver, table: str, public_synonym: str, schema: str
    ) -> None:
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE PUBLIC SYNONYM {public_synonym} FOR {table}", [])
        assert await driver.explore_find(NodeType.TABLE, public_synonym, []) == [
            [schema, table]
        ]

    async def test_public_synonym_is_excluded_by_a_schema_scope(
        self, driver: OracleDriver, table: str, public_synonym: str, schema: str
    ) -> None:
        """A public synonym is owned by PUBLIC, which is no schema in the tree —
        and a client scoping the search wrote a qualified name, which by
        definition does not go through one."""
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE PUBLIC SYNONYM {public_synonym} FOR {table}", [])
        assert (
            await driver.explore_find(
                NodeType.TABLE,
                public_synonym,
                [SearchScope(name=schema, type=NodeType.SCHEMA)],
            )
            == []
        )

    async def test_dangling_synonym_resolves_to_nothing(
        self, driver: OracleDriver, synonym: str
    ) -> None:
        """Oracle lets a synonym name an object that does not exist; returning
        it would hand the client a path describe cannot follow."""
        await driver.execute(f"CREATE SYNONYM {synonym} FOR T_NO_SUCH_TABLE", [])
        assert await driver.explore_find(NodeType.TABLE, synonym, []) == []

    async def test_chained_synonym_resolves_to_nothing(
        self, driver: OracleDriver, table: str, synonyms: tuple[str, str], schema: str
    ) -> None:
        """Chains are deliberately not followed: the first link resolves, the
        one pointing at it does not, and neither yields a bad path."""
        first, second = synonyms
        await driver.execute(f"CREATE TABLE {table} (id NUMBER)", [])
        await driver.execute(f"CREATE SYNONYM {first} FOR {table}", [])
        await driver.execute(f"CREATE SYNONYM {second} FOR {first}", [])
        assert await driver.explore_find(NodeType.TABLE, first, []) == [[schema, table]]
        assert await driver.explore_find(NodeType.TABLE, second, []) == []

    async def test_does_not_resolve_to_a_system_table(
        self, driver: OracleDriver
    ) -> None:
        """DUAL lives in SYS, which the object tree does not list — returning it
        would hand the client a path describe cannot follow."""
        assert await driver.explore_find(NodeType.TABLE, "DUAL", []) == []

    async def test_absent_symbol_resolves_to_nothing(
        self, driver: OracleDriver
    ) -> None:
        assert await driver.explore_find(NodeType.TABLE, "T_NO_SUCH_TABLE", []) == []


_NLS_DATE_FORMAT_QUERY = (
    "SELECT value FROM nls_session_parameters WHERE parameter = 'NLS_DATE_FORMAT'"
)


class TestLoad:
    async def test_loads_a_csv_with_a_header(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (ID NUMBER, NAME VARCHAR2(50))", []
        )
        path = tmp_path / "rows.csv"
        path.write_text("ID,NAME\n1,alice\n2,bob\n")

        result = await driver.execute(f"LOAD {schema}.{table} FROM '{path}' (HEADER)")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

        rows = await driver.execute(
            f"SELECT ID, NAME FROM {schema}.{table} ORDER BY ID"
        )
        assert isinstance(rows, ReadResult)
        assert rows.rows == [[1, "alice"], [2, "bob"]]

    async def test_loads_a_headerless_file_in_table_column_order(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (ID NUMBER, NAME VARCHAR2(50))", []
        )
        path = tmp_path / "rows.csv"
        path.write_text("1,alice\n")

        result = await driver.execute(f"LOAD {schema}.{table} '{path}'")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

        rows = await driver.execute(f"SELECT NAME FROM {schema}.{table}")
        assert isinstance(rows, ReadResult)
        assert rows.rows == [["alice"]]

    async def test_loads_a_subset_of_columns_and_nulls(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(
            f"CREATE TABLE {schema}.{table} "
            "(ID NUMBER, NAME VARCHAR2(50), NOTE VARCHAR2(50))",
            [],
        )
        path = tmp_path / "rows.tsv"
        path.write_text("1\tNA\n2\tzed\n")

        result = await driver.execute(
            f"LOAD {schema}.{table} (ID, NAME) FROM '{path}' "
            "(DELIMITER '\\t', NULL 'NA')"
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

        rows = await driver.execute(f"SELECT NAME FROM {schema}.{table} ORDER BY ID")
        assert isinstance(rows, ReadResult)
        assert rows.rows == [[None], ["zed"]]

    async def test_empty_fields_load_as_null_whatever_the_column_type(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        """What lets the reader stop telling a quoted \"\" from an unquoted empty
        field: Oracle stores a zero-length string as NULL in every column type,
        so neither form can reach a table as anything else."""
        await driver.execute(
            f"CREATE TABLE {schema}.{table} (ID NUMBER, NAME VARCHAR2(50), HIRED DATE)",
            [],
        )
        path = tmp_path / "rows.csv"
        path.write_text('1,"",\n')

        result = await driver.execute(f"LOAD {schema}.{table} FROM '{path}'")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

        rows = await driver.execute(f"SELECT ID, NAME, HIRED FROM {schema}.{table}")
        assert isinstance(rows, ReadResult)
        assert rows.rows == [[1, None, None]]

    async def test_dates_convert_through_the_session_format(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (HIRED DATE)", [])
        await driver.execute("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'", [])
        path = tmp_path / "rows.csv"
        path.write_text("2024-03-01\n")

        result = await driver.execute(f"LOAD {schema}.{table} FROM '{path}'")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

        rows = await driver.execute(
            f"SELECT TO_CHAR(HIRED, 'YYYY-MM-DD') FROM {schema}.{table}"
        )
        assert isinstance(rows, ReadResult)
        assert rows.rows == [["2024-03-01"]]

    async def test_date_format_option_converts_iso_dates(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (HIRED DATE)", [])
        path = tmp_path / "rows.csv"
        path.write_text("HIRED\n2026-09-04\n")

        result = await driver.execute(
            f"LOAD {schema}.{table} FROM '{path}' (HEADER, DATEFORMAT 'YYYY-MM-DD')"
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

        rows = await driver.execute(
            f"SELECT TO_CHAR(HIRED, 'YYYY-MM-DD') FROM {schema}.{table}"
        )
        assert isinstance(rows, ReadResult)
        assert rows.rows == [["2026-09-04"]]

    async def test_date_format_option_is_restored_after_the_load(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (HIRED DATE)", [])
        before = await driver.execute(_NLS_DATE_FORMAT_QUERY, [])
        assert isinstance(before, ReadResult)
        path = tmp_path / "rows.csv"
        path.write_text("2026-09-04\n")

        await driver.execute(
            f"LOAD {schema}.{table} FROM '{path}' (DATEFORMAT 'YYYY-MM-DD')"
        )

        after = await driver.execute(_NLS_DATE_FORMAT_QUERY, [])
        assert isinstance(after, ReadResult)
        assert after.rows == before.rows

    async def test_date_conversion_failure_names_the_session_format(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (HIRED DATE)", [])
        path = tmp_path / "rows.csv"
        path.write_text("2026-09-04\n")

        with pytest.raises(Exception, match="NLS_DATE_FORMAT"):
            await driver.execute(f"LOAD {schema}.{table} FROM '{path}'")

    async def test_date_format_covers_timestamp_columns(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        """A TIMESTAMP column converts through NLS_TIMESTAMP_FORMAT, which
        DATEFORMAT has to reach as well or an ISO file still fails."""
        await driver.execute(f"CREATE TABLE {schema}.{table} (SEEN TIMESTAMP(6))", [])
        path = tmp_path / "rows.csv"
        path.write_text("SEEN\n2026-09-04\n")

        result = await driver.execute(
            f"LOAD {schema}.{table} FROM '{path}' (HEADER, DATEFORMAT 'YYYY-MM-DD')"
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

        rows = await driver.execute(
            f"SELECT TO_CHAR(SEEN, 'YYYY-MM-DD') FROM {schema}.{table}"
        )
        assert isinstance(rows, ReadResult)
        assert rows.rows == [["2026-09-04"]]

    async def test_timestamp_conversion_failure_names_both_models(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (SEEN TIMESTAMP(6))", [])
        path = tmp_path / "rows.csv"
        path.write_text("2026-09-04\n")

        with pytest.raises(Exception, match="NLS_TIMESTAMP_FORMAT"):
            await driver.execute(f"LOAD {schema}.{table} FROM '{path}'")

    async def test_rejected_row_is_named_by_its_file_line(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (ID NUMBER)", [])
        path = tmp_path / "rows.csv"
        path.write_text("ID\n1\n2\nnot-a-number\n")

        with pytest.raises(Exception, match="line 4 of"):
            await driver.execute(f"LOAD {schema}.{table} FROM '{path}' (HEADER)")

    async def test_a_partial_load_can_be_rolled_back(
        self, driver: OracleDriver, schema: str, table: str, tmp_path
    ) -> None:
        await driver.execute(f"CREATE TABLE {schema}.{table} (ID NUMBER)", [])
        path = tmp_path / "rows.csv"
        path.write_text("1\n2\nnot-a-number\n")

        with pytest.raises(Exception):
            await driver.execute(f"LOAD {schema}.{table} FROM '{path}' (BATCH 1)")
        await driver.execute("ROLLBACK", [])

        rows = await driver.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        assert isinstance(rows, ReadResult)
        assert rows.rows == [[0]]
