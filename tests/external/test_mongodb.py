"""
Integration tests for the MongoDB driver.

Requires a running MongoDB instance. Configure via environment variables:
  MONGODB_URI       (default: mongodb://localhost:27017)
  MONGODB_DATABASE  (default: grannos_test)

Tests are skipped automatically when pymongo is not installed or the
server is unreachable.
"""

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from grannos.drivers.base import DriverSettings
from grannos.drivers.mongodb import MongoDriver
from grannos.protocol import ExploreItem, ReadResult, WriteResult

pytestmark = pytest.mark.external

_TEST_DB = os.environ.get("MONGODB_DATABASE", "grannos_test")


def _params() -> dict:
    return {"uri": os.environ.get("MONGODB_URI", "mongodb://localhost:27017")}


@pytest.fixture
async def driver() -> AsyncGenerator[MongoDriver, None]:
    pytest.importorskip("pymongo")
    try:
        d = await MongoDriver.create(_params(), DriverSettings())
    except Exception as exc:
        pytest.skip(f"MongoDB not available: {exc}")
    yield d
    await d.disconnect()


@pytest.fixture(autouse=True)
async def clean_db(driver: MongoDriver) -> AsyncGenerator[None, None]:
    await driver._client[_TEST_DB].drop_collection("users")
    await driver._client[_TEST_DB].drop_collection("orders")
    yield
    await driver._client[_TEST_DB].drop_collection("users")
    await driver._client[_TEST_DB].drop_collection("orders")


def _cmd(**kwargs: Any) -> str:
    return json.dumps({"db": _TEST_DB, **kwargs})


class TestExecuteFind:
    async def test_returns_columns_and_rows(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"name": "Alice", "age": 30})
        result = await driver.execute(_cmd(find="users", filter={"name": "Alice"}), [])
        assert isinstance(result, ReadResult)
        row = dict(zip(result.columns, result.rows[0]))
        assert row["name"] == "Alice"
        assert row["age"] == "30"

    async def test_flattens_nested_documents(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one(
            {"name": "Bob", "address": {"city": "NYC", "zip": "10001"}}
        )
        result = await driver.execute(_cmd(find="users", filter={"name": "Bob"}), [])
        assert isinstance(result, ReadResult)
        row = dict(zip(result.columns, result.rows[0]))
        assert row["address.city"] == "NYC"
        assert row["address.zip"] == "10001"

    async def test_serializes_object_id(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"name": "Carol"})
        result = await driver.execute(_cmd(find="users"), [])
        assert isinstance(result, ReadResult)
        row = dict(zip(result.columns, result.rows[0]))
        assert "_id" in row
        assert row["_id"] is not None

    async def test_returns_empty_result_for_no_matches(
        self, driver: MongoDriver
    ) -> None:
        result = await driver.execute(_cmd(find="users", filter={"name": "Ghost"}), [])
        assert isinstance(result, ReadResult)
        assert result.rows == []

    async def test_respects_limit(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many([{"n": i} for i in range(20)])
        result = await driver.execute(_cmd(find="users", limit=5), [])
        assert isinstance(result, ReadResult)
        assert len(result.rows) == 5


class TestExecuteAggregate:
    async def test_aggregate_groups_correctly(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["orders"].insert_many(
            [
                {"status": "open", "amount": 10},
                {"status": "open", "amount": 20},
                {"status": "closed", "amount": 5},
            ]
        )
        result = await driver.execute(
            _cmd(
                aggregate="orders",
                pipeline=[
                    {"$group": {"_id": "$status", "total": {"$sum": "$amount"}}},
                    {"$sort": {"_id": 1}},
                ],
            ),
            [],
        )
        assert isinstance(result, ReadResult)
        rows = [dict(zip(result.columns, r)) for r in result.rows]
        by_status = {r["_id"]: r["total"] for r in rows}
        assert by_status["closed"] == "5"
        assert by_status["open"] == "30"


class TestExecuteDML:
    async def test_insert_one_returns_rows_affected(self, driver: MongoDriver) -> None:
        result = await driver.execute(
            _cmd(insertOne="users", document={"name": "Alice"}), []
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_insert_many_returns_rows_affected(self, driver: MongoDriver) -> None:
        result = await driver.execute(
            _cmd(insertMany="users", documents=[{"name": "Alice"}, {"name": "Bob"}]), []
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

    async def test_update_one_returns_rows_affected(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many(
            [{"name": "Alice", "active": False}]
        )
        result = await driver.execute(
            _cmd(
                updateOne="users",
                filter={"name": "Alice"},
                update={"$set": {"active": True}},
            ),
            [],
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_update_many_returns_rows_affected(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many(
            [{"role": "admin"}, {"role": "admin"}]
        )
        result = await driver.execute(
            _cmd(
                updateMany="users",
                filter={"role": "admin"},
                update={"$set": {"active": True}},
            ),
            [],
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2

    async def test_delete_one_returns_rows_affected(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many(
            [{"name": "Alice"}, {"name": "Bob"}]
        )
        result = await driver.execute(
            _cmd(deleteOne="users", filter={"name": "Alice"}), []
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_delete_many_returns_rows_affected(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many(
            [{"role": "admin"}, {"role": "admin"}, {"role": "user"}]
        )
        result = await driver.execute(
            _cmd(deleteMany="users", filter={"role": "admin"}), []
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2


class TestExecuteCountAndDistinct:
    async def test_count_documents_applies_filter(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["orders"].insert_many(
            [{"status": "open"}, {"status": "open"}, {"status": "closed"}]
        )
        result = await driver.execute(
            _cmd(countDocuments="orders", filter={"status": "open"}), []
        )
        assert isinstance(result, ReadResult)
        assert result.columns == ["count"]
        assert result.rows == [["2"]]

    async def test_count_is_count_documents(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["orders"].insert_many(
            [{"n": i} for i in range(5)]
        )
        result = await driver.execute(_cmd(count="orders", skip=1, limit=3), [])
        assert isinstance(result, ReadResult)
        assert result.rows == [["3"]]

    async def test_count_on_missing_collection_is_zero(
        self, driver: MongoDriver
    ) -> None:
        result = await driver.execute(_cmd(countDocuments="orders"), [])
        assert isinstance(result, ReadResult)
        assert result.rows == [["0"]]

    async def test_estimated_document_count(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["orders"].insert_many(
            [{"n": i} for i in range(4)]
        )
        result = await driver.execute(_cmd(estimatedDocumentCount="orders"), [])
        assert isinstance(result, ReadResult)
        assert result.rows == [["4"]]

    async def test_distinct_returns_one_row_per_value(
        self, driver: MongoDriver
    ) -> None:
        await driver._client[_TEST_DB]["orders"].insert_many(
            [
                {"status": "open", "amount": 5},
                {"status": "open", "amount": 50},
                {"status": "closed", "amount": 50},
                {"status": "void", "amount": 1},
            ]
        )
        result = await driver.execute(
            _cmd(distinct="orders", key="status", filter={"amount": {"$gt": 2}}), []
        )
        assert isinstance(result, ReadResult)
        assert result.columns == ["status"]
        assert sorted(r[0] for r in result.rows) == ["closed", "open"]

    async def test_distinct_on_nested_key(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_many(
            [{"address": {"city": "NYC"}}, {"address": {"city": "LA"}}]
        )
        result = await driver.execute(_cmd(distinct="users", key="address.city"), [])
        assert isinstance(result, ReadResult)
        assert sorted(r[0] for r in result.rows) == ["LA", "NYC"]


class TestExecuteFindOne:
    async def test_returns_first_by_sort(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_many(
            [{"name": "Bob", "age": 40}, {"name": "Alice", "age": 30}]
        )
        result = await driver.execute(
            _cmd(findOne="users", sort={"age": 1}, projection={"_id": 0}), []
        )
        assert isinstance(result, ReadResult)
        assert result.columns == ["name", "age"]
        assert result.rows == [["Alice", "30"]]

    async def test_no_match_is_empty(self, driver: MongoDriver) -> None:
        result = await driver.execute(_cmd(findOne="users"), [])
        assert isinstance(result, ReadResult)
        assert result.rows == []

    async def test_find_skips(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_many(
            [{"n": i} for i in range(5)]
        )
        result = await driver.execute(
            _cmd(find="users", sort={"n": 1}, skip=3, projection={"_id": 0}), []
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [["3"], ["4"]]


class TestExecuteReplaceAndUpsert:
    async def test_replace_one(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_one({"name": "Alice", "age": 30})
        result = await driver.execute(
            _cmd(
                replaceOne="users", filter={"name": "Alice"}, replacement={"name": "Al"}
            ),
            [],
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        doc = await driver._client[_TEST_DB]["users"].find_one({}, {"_id": 0})
        assert doc == {"name": "Al"}

    async def test_upsert_counts_inserted_document(self, driver: MongoDriver) -> None:
        result = await driver.execute(
            _cmd(
                updateOne="users",
                filter={"name": "Dave"},
                update={"$set": {"age": 40}},
                options={"upsert": True},
            ),
            [],
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        assert await driver._client[_TEST_DB]["users"].count_documents({}) == 1


class TestExecuteFindOneAnd:
    async def test_update_returns_document_after(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_one({"_id": "c", "seq": 1})
        result = await driver.execute(
            _cmd(
                findOneAndUpdate="users",
                filter={"_id": "c"},
                update={"$inc": {"seq": 1}},
                returnDocument="after",
            ),
            [],
        )
        assert isinstance(result, ReadResult)
        assert dict(zip(result.columns, result.rows[0]))["seq"] == "2"

    async def test_update_returns_document_before_by_default(
        self, driver: MongoDriver
    ) -> None:
        await driver._client[_TEST_DB]["users"].insert_one({"_id": "c", "seq": 1})
        result = await driver.execute(
            _cmd(
                findOneAndUpdate="users",
                filter={"_id": "c"},
                update={"$inc": {"seq": 1}},
            ),
            [],
        )
        assert isinstance(result, ReadResult)
        assert dict(zip(result.columns, result.rows[0]))["seq"] == "1"

    async def test_update_upserts(self, driver: MongoDriver) -> None:
        result = await driver.execute(
            _cmd(
                findOneAndUpdate="users",
                filter={"_id": "c"},
                update={"$set": {"seq": 1}},
                upsert=True,
                returnDocument="after",
            ),
            [],
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [["c", "1"]]

    async def test_replace(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_one({"_id": 1, "name": "Alice"})
        await driver.execute(
            _cmd(
                findOneAndReplace="users", filter={"_id": 1}, replacement={"name": "Al"}
            ),
            [],
        )
        assert await driver._client[_TEST_DB]["users"].find_one({"_id": 1}) == {
            "_id": 1,
            "name": "Al",
        }

    async def test_delete_returns_deleted_document(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_many(
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 40}]
        )
        result = await driver.execute(
            _cmd(findOneAndDelete="users", sort={"age": -1}, projection={"_id": 0}), []
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [["Bob", "40"]]
        assert await driver._client[_TEST_DB]["users"].count_documents({}) == 1


class TestExecuteRenameCollection:
    async def test_renames(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["users"].insert_one({"name": "Alice"})
        result = await driver.execute(_cmd(renameCollection="users", to="orders"), [])
        assert isinstance(result, WriteResult)
        names = await driver._client[_TEST_DB].list_collection_names()
        assert "orders" in names
        assert "users" not in names


class TestExecuteRunCommand:
    async def test_returns_reply_as_one_row(self, driver: MongoDriver) -> None:
        await driver._client[_TEST_DB]["orders"].insert_one({"n": 1})
        result = await driver.execute(
            _cmd(runCommand={"count": "orders", "query": {"n": 1}}), []
        )
        assert isinstance(result, ReadResult)
        row = dict(zip(result.columns, result.rows[0]))
        assert row["n"] == "1"
        assert row["ok"] == "1.0"


class TestExecuteCreateCollection:
    async def test_creates_collection(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db].drop_collection("new_events")
        result = await driver.execute(_cmd(createCollection="new_events"), [])
        assert isinstance(result, WriteResult)
        names = await driver._client[db].list_collection_names()
        assert "new_events" in names
        await driver._client[db].drop_collection("new_events")


class TestExecuteDropCollection:
    async def test_drops_collection(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db].create_collection("temp_events")
        result = await driver.execute(_cmd(dropCollection="temp_events"), [])
        assert isinstance(result, WriteResult)
        names = await driver._client[db].list_collection_names()
        assert "temp_events" not in names


class TestExecuteCreateIndex:
    async def test_creates_index(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"email": "a@example.com"})
        result = await driver.execute(
            _cmd(createIndex="users", keys={"email": 1}, options={"name": "email_idx"}),
            [],
        )
        assert isinstance(result, WriteResult)
        info = await driver._client[db]["users"].index_information()
        assert "email_idx" in info


class TestExecuteDropIndex:
    async def test_drops_index(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].create_index("email", name="email_idx")
        result = await driver.execute(_cmd(dropIndex="users", name="email_idx"), [])
        assert isinstance(result, WriteResult)
        info = await driver._client[db]["users"].index_information()
        assert "email_idx" not in info


class TestExecuteExtendedJsonRoundTrip:
    async def test_date_round_trips_through_insert_and_find(
        self, driver: MongoDriver
    ) -> None:
        await driver.execute(
            _cmd(
                insertOne="users",
                document={
                    "name": "Dave",
                    "joinedAt": {"$date": "2024-01-01T00:00:00Z"},
                },
            ),
            [],
        )
        result = await driver.execute(_cmd(find="users", filter={"name": "Dave"}), [])
        assert isinstance(result, ReadResult)
        row = dict(zip(result.columns, result.rows[0]))
        assert row["joinedAt"].startswith("2024-01-01")


class TestExploreList:
    async def test_root_lists_databases(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"seed": True})
        items = await driver.explore_list([])
        names = [i.name for i in items]
        assert db in names
        assert all(i.type == "database" for i in items)
        assert all(i.expandable for i in items)

    async def test_database_lists_collections(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"x": 1})
        await driver._client[db]["orders"].insert_one({"x": 1})
        items = await driver.explore_list([db])
        names = [i.name for i in items]
        assert "users" in names
        assert "orders" in names
        assert all(i.type == "collection" for i in items)
        assert all(i.expandable for i in items)

    async def test_collection_lists_fields_and_indexes_groups(
        self, driver: MongoDriver
    ) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"x": 1})
        items = await driver.explore_list([db, "users"])
        assert items == [
            ExploreItem(name="fields", type="group", expandable=True),
            ExploreItem(name="indexes", type="group", expandable=True),
        ]

    async def test_fields_samples_top_level_keys(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_many(
            [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "email": "b@b.com"},
            ]
        )
        items = await driver.explore_list([db, "users", "fields"])
        names = [i.name for i in items]
        assert "name" in names
        assert "age" in names
        assert "email" in names
        assert all(i.type == "field" for i in items)
        assert all(not i.expandable for i in items)

    async def test_indexes_lists_index_names(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({"name": "Alice"})
        await driver._client[db]["users"].create_index("name", name="name_idx")
        items = await driver.explore_list([db, "users", "indexes"])
        names = [i.name for i in items]
        assert "_id_" in names
        assert "name_idx" in names
        assert all(i.type == "index" for i in items)
        assert all(not i.expandable for i in items)

    async def test_fields_empty_for_empty_collection(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        await driver._client[db]["users"].insert_one({})
        await driver._client[db]["users"].delete_many({})
        items = await driver.explore_list([db, "users", "fields"])
        assert items == []

    async def test_unknown_path_returns_empty(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        assert await driver.explore_list([db, "users", "fields", "extra"]) == []


class TestExploreDescribe:
    async def test_returns_none_for_all_paths(self, driver: MongoDriver) -> None:
        db = _TEST_DB
        assert await driver.explore_describe([]) is None
        assert await driver.explore_describe([db]) is None
        assert await driver.explore_describe([db, "users"]) is None
