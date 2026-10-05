import asyncio
import logging
import pathlib
from collections.abc import AsyncGenerator

import pytest

from grannos.drivers.base import DriverSettings
from grannos.drivers.sqlite import SQLiteDriver
from grannos.explore_cache import CachingDriver, ConnectionCache
from grannos.protocol import (
    EntityDescription,
    ExploreItem,
    FieldDescription,
    IndexDescription,
    NodeType,
    ReadResult,
    SearchScope,
    TableReference,
    WriteResult,
)


@pytest.fixture
async def driver() -> AsyncGenerator[SQLiteDriver, None]:
    d = await SQLiteDriver.create({"database": ":memory:"}, DriverSettings())
    yield d
    await d.disconnect()


class TestExecute:
    async def test_should_return_columns_and_rows(self, driver: SQLiteDriver) -> None:
        result = await driver.execute("SELECT 1 AS n, 'a' AS s", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["n", "s"]
        assert result.rows == [[1, "a"]]

    async def test_should_return_rows_affected_for_insert(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        result = await driver.execute("INSERT INTO t VALUES (?, ?)", [1, "hello"])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_should_return_rows_affected_for_update(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("INSERT INTO t VALUES (1, 'a')", [])
        await driver.execute("INSERT INTO t VALUES (2, 'b')", [])
        result = await driver.execute("UPDATE t SET val = 'x'", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

    async def test_should_return_rows_affected_for_delete(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        await driver.execute("INSERT INTO t VALUES (1)", [])
        await driver.execute("INSERT INTO t VALUES (2)", [])
        result = await driver.execute("DELETE FROM t WHERE id = 1", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_should_persist_inserts_within_connection(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("INSERT INTO t VALUES (?, ?)", [1, "hello"])
        result = await driver.execute("SELECT * FROM t", [])
        assert isinstance(result, ReadResult)
        assert result.columns == ["id", "val"]
        assert result.rows == [[1, "hello"]]


class TestExploreList:
    async def test_should_list_tables_at_top_level(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE users (id INTEGER)", [])
        items = await driver.explore_list([])
        assert ExploreItem(name="users", type="table", expandable=True) in items

    async def test_should_list_views_at_top_level(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        await driver.execute("CREATE VIEW v AS SELECT * FROM t", [])
        items = await driver.explore_list([])
        names = {i.name for i in items}
        assert "t" in names
        assert "v" in names
        view = next(i for i in items if i.name == "v")
        assert view.type == "view"

    async def test_should_return_groups_for_table(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        items = await driver.explore_list(["t"])
        assert {i.name for i in items} == {"columns", "indices", "foreign_keys"}
        assert all(i.type == "group" and i.expandable for i in items)

    async def test_should_list_columns_in_definition_order(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        items = await driver.explore_list(["t", "columns"])
        assert [i.name for i in items] == ["id", "val"]
        assert all(not i.expandable for i in items)

    async def test_should_list_index_by_name(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx_val ON t(val)", [])
        items = await driver.explore_list(["t", "indices"])
        assert any(i.name == "idx_val" for i in items)
        assert all(not i.expandable for i in items)

    async def test_should_return_empty_indices_when_none_exist(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_list(["t", "indices"]) == []

    async def test_should_list_foreign_key_reference(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        items = await driver.explore_list(["child", "foreign_keys"])
        assert len(items) == 1
        assert items[0].name == "parent_id → parent.id"
        assert items[0].type == "foreign_key"
        assert not items[0].expandable

    async def test_should_return_empty_foreign_keys_when_none_exist(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_list(["t", "foreign_keys"]) == []

    async def test_should_return_empty_when_path_is_too_deep(
        self, driver: SQLiteDriver
    ) -> None:
        assert await driver.explore_list(["a", "b", "c", "d"]) == []


class TestExploreDescribe:
    async def test_should_return_field_names_and_types(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        desc = await driver.explore_describe(["t"])
        assert desc is not None
        assert isinstance(desc, EntityDescription)
        assert desc.name == "t"
        assert desc.kind == "table"
        assert [f.name for f in desc.properties] == ["id", "val"]
        assert [f.types for f in desc.properties] == [["INTEGER"], ["TEXT"]]

    async def test_should_return_nullable_flag(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (a INTEGER NOT NULL, b INTEGER)", [])
        desc = await driver.explore_describe(["t"])
        assert desc is not None
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["a"].nullable is False
        assert by_name["b"].nullable is True

    async def test_should_return_pk_flag(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", [])
        desc = await driver.explore_describe(["t"])
        assert desc is not None
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].pk is True
        assert by_name["val"].pk is False

    async def test_should_return_exclusive_index_for_single_column_index(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT, other TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val)", [])
        desc = await driver.explore_describe(["t"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert len(by_name["val"].exclusive_indices) == 1
        assert by_name["val"].composite_indices == []
        assert by_name["id"].exclusive_indices == []
        assert by_name["id"].composite_indices == []
        assert by_name["other"].exclusive_indices == []
        assert by_name["other"].composite_indices == []

    async def test_should_return_composite_index_for_multi_column_index(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT, other TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val, other)", [])
        desc = await driver.explore_describe(["t"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["val"].exclusive_indices == []
        assert len(by_name["val"].composite_indices) == 1
        assert by_name["other"].exclusive_indices == []
        assert len(by_name["other"].composite_indices) == 1
        assert by_name["id"].exclusive_indices == []
        assert by_name["id"].composite_indices == []

    async def test_should_return_both_when_column_has_exclusive_and_composite_index(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT, other TEXT)", [])
        await driver.execute("CREATE INDEX idx1 ON t(val)", [])
        await driver.execute("CREATE INDEX idx2 ON t(val, other)", [])
        desc = await driver.explore_describe(["t"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert len(by_name["val"].exclusive_indices) == 1
        assert len(by_name["val"].composite_indices) == 1

    async def test_should_return_outgoing_references(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["child"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["parent_id"].outgoing_references == [
            TableReference(
                table="child", column="parent_id", ref_table="parent", ref_column="id"
            )
        ]
        assert by_name["id"].outgoing_references == []

    async def test_should_return_empty_outgoing_references_when_none_exist(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        desc = await driver.explore_describe(["t"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].outgoing_references == []

    async def test_should_return_incoming_references(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["parent"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].incoming_references == [
            TableReference(
                table="child", column="parent_id", ref_table="parent", ref_column="id"
            )
        ]

    async def test_should_return_empty_incoming_references_when_none_exist(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        desc = await driver.explore_describe(["t"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].incoming_references == []

    async def test_should_mark_outgoing_reference_unique_when_fk_column_is_pk(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY REFERENCES parent(id))", []
        )
        desc = await driver.explore_describe(["child"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].outgoing_references[0].unique is True

    async def test_should_mark_outgoing_reference_not_unique_by_default(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["child"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["parent_id"].outgoing_references[0].unique is False

    async def test_should_mark_incoming_reference_unique_when_fk_column_is_unique(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (parent_id INTEGER UNIQUE REFERENCES parent(id))", []
        )
        desc = await driver.explore_describe(["parent"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].incoming_references[0].unique is True

    async def test_should_mark_incoming_reference_not_unique_by_default(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["parent"])
        assert isinstance(desc, EntityDescription)
        by_name = {f.name: f for f in desc.properties}
        assert by_name["id"].incoming_references[0].unique is False

    async def test_should_return_none_when_path_is_invalid(
        self, driver: SQLiteDriver
    ) -> None:
        assert await driver.explore_describe([]) is None

    async def test_columns_group_path_no_longer_resolves(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_describe(["t", "columns"]) is None

    async def test_should_describe_a_relationship(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["child", "relationships", "parent_id"])
        assert isinstance(desc, TableReference)
        assert desc.table == "child"
        assert desc.column == "parent_id"
        assert desc.ref_table == "parent"
        assert desc.ref_column == "id"
        assert desc.constraint_name is None

    async def test_should_return_none_for_unknown_relationship_column(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_describe(["t", "relationships", "id"]) is None


class TestExploreDescribeIndex:
    async def test_basic_index_fields_and_direction(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.name == "idx"
        assert len(desc.fields) == 1
        assert desc.fields[0].name == "val"
        assert desc.fields[0].direction == "asc"

    async def test_descending_direction(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val DESC)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.fields[0].direction == "desc"

    async def test_unique_index(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, email TEXT)", [])
        await driver.execute("CREATE UNIQUE INDEX idx ON t(email)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.unique is True

    async def test_non_unique_index(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.unique is False

    async def test_multi_column_index(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, first TEXT, last TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(last, first)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert [f.name for f in desc.fields] == ["last", "first"]

    async def test_partial_index_ddl_contains_where(self, driver: SQLiteDriver) -> None:
        await driver.execute(
            "CREATE TABLE t (id INTEGER, email TEXT, active INTEGER)", []
        )
        await driver.execute("CREATE INDEX idx ON t(email) WHERE active = 1", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.ddl is not None
        assert "active = 1" in desc.ddl

    async def test_non_partial_index_ddl_has_no_where(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val)", [])
        desc = await driver.explore_describe(["t", "indices", "idx"])
        assert isinstance(desc, IndexDescription)
        assert desc.ddl is None or "WHERE" not in desc.ddl.upper()

    async def test_unknown_index_returns_none(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_describe(["t", "indices", "no_such_idx"]) is None


class TestExploreDescribeField:
    async def test_single_field_basic_fields(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT NOT NULL)", [])
        desc = await driver.explore_describe(["t", "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert desc.name == "val"
        assert desc.types == ["TEXT"]
        assert desc.nullable is False
        assert desc.pk is False

    async def test_single_field_pk(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", [])
        desc = await driver.explore_describe(["t", "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.pk is True

    async def test_single_field_exclusive_index(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val)", [])
        desc = await driver.explore_describe(["t", "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.exclusive_indices) == 1
        assert desc.exclusive_indices[0].name == "idx"
        assert desc.composite_indices == []

    async def test_single_field_composite_index(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT, other TEXT)", [])
        await driver.execute("CREATE INDEX idx ON t(val, other)", [])
        desc = await driver.explore_describe(["t", "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.composite_indices) == 1
        assert desc.composite_indices[0].name == "idx"
        assert desc.exclusive_indices == []

    async def test_single_field_sample_values(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        for i, v in enumerate(["x", "y", "z", "x"]):
            await driver.execute("INSERT INTO t VALUES (?, ?)", [i, v])
        desc = await driver.explore_describe(["t", "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert len(desc.sample) <= 3
        assert set(desc.sample).issubset({"x", "y", "z"})

    async def test_single_field_outgoing_references(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["child", "columns", "parent_id"])
        assert isinstance(desc, FieldDescription)
        assert desc.outgoing_references == [
            TableReference(
                table="child", column="parent_id", ref_table="parent", ref_column="id"
            )
        ]

    async def test_single_field_incoming_references(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)", [])
        await driver.execute(
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id))",
            [],
        )
        desc = await driver.explore_describe(["parent", "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.incoming_references == [
            TableReference(
                table="child", column="parent_id", ref_table="parent", ref_column="id"
            )
        ]

    async def test_single_field_empty_outgoing_references_when_not_fk(
        self, driver: SQLiteDriver
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        desc = await driver.explore_describe(["t", "columns", "id"])
        assert isinstance(desc, FieldDescription)
        assert desc.outgoing_references == []

    async def test_unknown_column_returns_none(self, driver: SQLiteDriver) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert await driver.explore_describe(["t", "columns", "no_such_col"]) is None

    async def test_sample_timeout_returns_empty(self) -> None:
        driver = await SQLiteDriver.create(
            {"database": ":memory:"}, DriverSettings(column_sample_timeout=0.0)
        )
        await driver.execute("CREATE TABLE t (id INTEGER, val TEXT)", [])
        await driver.execute("INSERT INTO t VALUES (1, 'x')", [])
        desc = await driver.explore_describe(["t", "columns", "val"])
        assert isinstance(desc, FieldDescription)
        assert desc.sample == []


class TestExploreFind:
    """End-to-end symbol resolution: the generic walker over a real database,
    through the caching driver a live connection actually uses."""

    @pytest.fixture
    async def searchable(
        self, driver: SQLiteDriver, tmp_path: pathlib.Path
    ) -> CachingDriver:
        await driver.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", []
        )
        await driver.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER,"
            " REFERENCES users(id))".replace(
                ", REFERENCES", ", FOREIGN KEY (user_id) REFERENCES"
            ),
            [],
        )
        await driver.execute("CREATE INDEX orders_user_id ON orders (user_id)", [])
        await driver.execute("CREATE VIEW active_users AS SELECT * FROM users", [])
        return CachingDriver(driver, ConnectionCache({}, tmp_path / "cache.json"))

    async def test_finds_a_column_within_its_table(
        self, searchable: CachingDriver
    ) -> None:
        result = await searchable.explore_find(
            NodeType.COLUMN, "name", [SearchScope(name="users", type=NodeType.TABLE)]
        )
        assert result == [["users", "columns", "name"]]

    async def test_reports_every_candidate_for_an_unqualified_column(
        self, searchable: CachingDriver
    ) -> None:
        result = await searchable.explore_find(NodeType.COLUMN, "id", [])
        assert result == [
            ["active_users", "columns", "id"],  # the view selects users.*
            ["orders", "columns", "id"],
            ["users", "columns", "id"],
        ]

    async def test_scope_disambiguates_an_unqualified_column(
        self, searchable: CachingDriver
    ) -> None:
        result = await searchable.explore_find(
            NodeType.COLUMN, "id", [SearchScope(name="orders", type=NodeType.TABLE)]
        )
        assert result == [["orders", "columns", "id"]]

    async def test_finds_a_table(self, searchable: CachingDriver) -> None:
        assert await searchable.explore_find(NodeType.TABLE, "orders", []) == [
            ["orders"]
        ]

    async def test_finds_a_view_asked_for_as_a_table(
        self, searchable: CachingDriver
    ) -> None:
        """A client reading a FROM clause cannot tell a view from a table."""
        assert await searchable.explore_find(NodeType.TABLE, "active_users", []) == [
            ["active_users"]
        ]

    async def test_finds_an_index(self, searchable: CachingDriver) -> None:
        assert await searchable.explore_find(NodeType.INDEX, "orders_user_id", []) == [
            ["orders", "indices", "orders_user_id"]
        ]

    async def test_returns_catalog_casing_for_a_differently_cased_symbol(
        self, searchable: CachingDriver
    ) -> None:
        """SQL identifiers are case-insensitive, so the query text's casing need
        not match the catalog's — the path returned has to be the catalog's, or
        explore.describe could not consume it."""
        result = await searchable.explore_find(
            NodeType.COLUMN, "NAME", [SearchScope(name="USERS", type=NodeType.TABLE)]
        )
        assert result == [["users", "columns", "name"]]

    async def test_returned_path_describes(self, searchable: CachingDriver) -> None:
        """The whole point of a find: hand the path straight back to describe."""
        (path,) = await searchable.explore_find(
            NodeType.COLUMN, "name", [SearchScope(name="users", type=NodeType.TABLE)]
        )
        desc = await searchable.explore_describe(path)
        assert isinstance(desc, FieldDescription)
        assert desc.name == "name"

    async def test_unknown_symbol_finds_nothing(
        self, searchable: CachingDriver
    ) -> None:
        assert await searchable.explore_find(NodeType.COLUMN, "nope", []) == []


class TestQueryLogging:
    """Debug logging has to show every statement that reached the database —
    without writing the user's own data into the log file."""

    async def test_catalog_queries_are_logged_with_their_binds(
        self, driver: SQLiteDriver, caplog: pytest.LogCaptureFixture
    ) -> None:
        await driver.execute("CREATE TABLE t (id INTEGER, name TEXT)", [])
        with caplog.at_level(logging.DEBUG):
            await driver.explore_list([])
        assert "query SELECT name, type FROM sqlite_master" in caplog.text

    async def test_user_statement_is_logged(
        self, driver: SQLiteDriver, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            await driver.execute("CREATE TABLE t (id INTEGER, name TEXT)", [])
        assert "query CREATE TABLE t (id INTEGER, name TEXT)" in caplog.text

    async def test_user_bind_values_are_never_logged(
        self, driver: SQLiteDriver, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one guarantee worth a test of its own: a driver's own catalog
        binds are object names, but the user's are their data."""
        await driver.execute("CREATE TABLE t (id INTEGER, name TEXT)", [])
        with caplog.at_level(logging.DEBUG):
            await driver.execute("INSERT INTO t VALUES (?, ?)", [1, "s3cret-value"])
        assert "INSERT INTO t VALUES (?, ?)" in caplog.text
        assert "s3cret-value" not in caplog.text

    async def test_nothing_is_logged_above_debug(
        self, driver: SQLiteDriver, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            await driver.execute("CREATE TABLE t (id INTEGER)", [])
        assert "query" not in caplog.text


class TestCancel:
    async def test_should_abort_running_query(self, driver: SQLiteDriver) -> None:
        task = asyncio.create_task(
            driver.execute(
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
                [],
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        result = await driver.execute("SELECT 1", [])
        assert isinstance(result, ReadResult)
        assert result.rows == [[1]]
