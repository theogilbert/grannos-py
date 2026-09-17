"""Unit tests for MongoDriver — no live database required."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pymongo.errors
import pytest
from bson import ObjectId

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.mongodb import (
    MongoDriver,
    _gridfs_buckets,
    _is_gridfs_internal,
    _make_mongo_client,
    _serialize,
    _strip_comments,
)
from grannos.protocol import (
    DownloadResult,
    ExploreItem,
    GenericRecordDescription,
    LobPlaceholder,
    ReadResult,
    WriteResult,
)

_CLOSED_EXC = pymongo.errors.InvalidOperation("Cannot use AsyncMongoClient after close")


def _closed_client() -> MagicMock:
    """A client mock raising InvalidOperation, as pymongo does once closed by the idle timer."""
    cursor = MagicMock()
    cursor.to_list = AsyncMock(side_effect=_CLOSED_EXC)
    col = MagicMock()
    col.find.return_value.limit.return_value = cursor
    col.insert_one = AsyncMock(side_effect=_CLOSED_EXC)
    col.index_information = AsyncMock(side_effect=_CLOSED_EXC)
    db = MagicMock()
    db.__getitem__.return_value = col
    client = MagicMock()
    client.__getitem__.return_value = db
    client.list_database_names = AsyncMock(side_effect=_CLOSED_EXC)
    return client


def _open_client() -> tuple[MagicMock, MagicMock, MagicMock]:
    """A client mock with a stubbed db/collection chain for happy-path execute() calls."""
    col = MagicMock()
    col.create_index = AsyncMock()
    col.drop_index = AsyncMock()
    col.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    db = MagicMock()
    db.__getitem__.return_value = col
    db.create_collection = AsyncMock()
    db.drop_collection = AsyncMock()
    client = MagicMock()
    client.__getitem__.return_value = db
    return client, db, col


def _make_driver(client: MagicMock) -> MongoDriver:
    return MongoDriver({}, client, DriverSettings())


class TestConnectionLostAfterIdleClose:
    async def test_execute_raises_connection_lost(self) -> None:
        driver = _make_driver(_closed_client())
        with pytest.raises(ConnectionLostError):
            await driver.execute(
                '{"insertOne": "users", "db": "test", "document": {}}', []
            )

    async def test_explore_list_raises_connection_lost(self) -> None:
        driver = _make_driver(_closed_client())
        with pytest.raises(ConnectionLostError):
            await driver.explore_list([])

    async def test_explore_describe_raises_connection_lost(self) -> None:
        driver = _make_driver(_closed_client())
        with pytest.raises(ConnectionLostError):
            await driver.explore_describe(["db", "col", "indexes"])

    async def test_explore_preview_raises_connection_lost(self) -> None:
        driver = _make_driver(_closed_client())
        with pytest.raises(ConnectionLostError):
            await driver.explore_preview(["db", "col"])


class TestMakeMongoClientInvalidUri:
    async def test_invalid_uri_raises_driver_error(self) -> None:
        with pytest.raises(DriverError):
            await _make_mongo_client({"uri": "mongodb://"})

    async def test_invalid_uri_message_is_preserved(self) -> None:
        with pytest.raises(DriverError, match="at least one hostname"):
            await _make_mongo_client({"uri": "mongodb://"})

    async def test_malformed_host_raises_driver_error(self) -> None:
        # pymongo raises a plain ValueError (not InvalidURI) for unescaped
        # reserved characters in the host portion of the URI.
        with pytest.raises(DriverError, match="Reserved characters"):
            await _make_mongo_client({"uri": "mongodb://localhost:270:1213:"})


class TestExecuteCollectionAndIndexOps:
    async def test_create_collection(self) -> None:
        client, db, _ = _open_client()
        driver = _make_driver(client)
        result = await driver.execute(
            '{"createCollection": "events", "db": "mydb", "options": {"capped": true}}',
            [],
        )
        db.create_collection.assert_awaited_once_with("events", capped=True)
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_drop_collection(self) -> None:
        client, db, _ = _open_client()
        driver = _make_driver(client)
        result = await driver.execute(
            '{"dropCollection": "old_events", "db": "mydb"}', []
        )
        db.drop_collection.assert_awaited_once_with("old_events")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_create_index(self) -> None:
        client, _, col = _open_client()
        driver = _make_driver(client)
        result = await driver.execute(
            '{"createIndex": "users", "db": "mydb", "keys": {"email": 1}, '
            '"options": {"unique": true}}',
            [],
        )
        col.create_index.assert_awaited_once_with([("email", 1)], unique=True)
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    async def test_drop_index(self) -> None:
        client, _, col = _open_client()
        driver = _make_driver(client)
        result = await driver.execute(
            '{"dropIndex": "users", "db": "mydb", "name": "email_1"}', []
        )
        col.drop_index.assert_awaited_once_with("email_1")
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1


class TestExecuteExtendedJson:
    async def test_update_one_parses_date_literal(self) -> None:
        client, _, col = _open_client()
        driver = _make_driver(client)
        await driver.execute(
            '{"updateOne": "events", "db": "mydb", "filter": {}, '
            '"update": {"$set": {"occurredAt": {"$date": "2024-01-01T00:00:00Z"}}}}',
            [],
        )
        _, update = col.update_one.call_args.args
        occurred_at = update["$set"]["occurredAt"]
        assert isinstance(occurred_at, datetime)
        assert occurred_at.year == 2024


def _null_register_lob(value: object, text: str) -> LobPlaceholder:
    return LobPlaceholder(text=text)


class TestSerialize:
    def test_passes_through_plain_values(self) -> None:
        assert _serialize(_null_register_lob, "hello") == "hello"
        assert _serialize(_null_register_lob, 42) == 42
        assert _serialize(_null_register_lob, None) is None

    def test_renders_binary_as_byte_count(self) -> None:
        assert _serialize(_null_register_lob, b"\x01\x02\x03") == LobPlaceholder(
            text="BSON Binary (3 bytes)"
        )

    def test_renders_binary_nested_in_dict(self) -> None:
        result = _serialize(_null_register_lob, {"blob": b"\x00\x01"})
        assert result == {"blob": LobPlaceholder(text="BSON Binary (2 bytes)")}


class TestStripComments:
    def test_removes_line_and_block_comments(self) -> None:
        query = (
            "// open orders\n"
            '{"find": "orders", // the collection\n'
            ' "db": "mydb", /* the database */ "filter": {}}\n'
            "/* done */"
        )
        assert json.loads(_strip_comments(query)) == {
            "find": "orders",
            "db": "mydb",
            "filter": {},
        }

    def test_leaves_markers_inside_strings_alone(self) -> None:
        query = '{"find": "a//b", "db": "x", "filter": {"url": "http://h/*p*/"}}'
        assert _strip_comments(query) == query

    def test_leaves_an_escaped_quote_inside_a_string(self) -> None:
        query = '{"find": "a\\"//b", "db": "x"} // c'
        assert _strip_comments(query) == '{"find": "a\\"//b", "db": "x"} '

    async def test_execute_accepts_a_commented_command(self) -> None:
        client, _, col = _open_client()
        driver = _make_driver(client)
        await driver.execute(
            '// drop it\n{"dropCollection": "old", /* gone */ "db": "mydb"}', []
        )
        client["mydb"].drop_collection.assert_awaited_once_with("old")


class TestExecuteMalformedCommand:
    async def test_invalid_json_raises_driver_error(self) -> None:
        driver = _make_driver(_open_client()[0])
        with pytest.raises(DriverError, match="must be valid JSON"):
            await driver.execute("{not json", [])

    async def test_invalid_oid_raises_driver_error(self) -> None:
        driver = _make_driver(_open_client()[0])
        with pytest.raises(DriverError):
            await driver.execute(
                '{"find": "users", "db": "mydb", "filter": {"_id": {"$oid": "bad"}}}',
                [],
            )


class TestGridfsBucketDetection:
    def test_buckets_inferred_from_files_collections(self) -> None:
        names = ["fs.files", "fs.chunks", "users", "images.files", "images.chunks"]
        assert _gridfs_buckets(names) == {"fs", "images"}

    def test_no_buckets_when_no_files_collection(self) -> None:
        assert _gridfs_buckets(["users", "orders"]) == set()

    def test_internal_collections_hidden_for_known_buckets(self) -> None:
        buckets = {"fs"}
        assert _is_gridfs_internal("fs.files", buckets)
        assert _is_gridfs_internal("fs.chunks", buckets)
        assert not _is_gridfs_internal("users", buckets)

    def test_files_like_name_not_hidden_unless_bucket_known(self) -> None:
        # "other.files" only hidden if "other" was actually detected as a bucket
        assert not _is_gridfs_internal("other.files", {"fs"})


class TestExploreListGridfs:
    async def test_gridfs_group_shown_only_when_bucket_exists(self) -> None:
        client = MagicMock()
        client["mydb"].list_collection_names = AsyncMock(
            return_value=["users", "fs.files", "fs.chunks"]
        )
        driver = _make_driver(client)
        items = await driver.explore_list(["mydb"])
        assert items == [
            ExploreItem(name="users", type="collection", expandable=True),
            ExploreItem(name="gridfs", type="group", expandable=True),
        ]

    async def test_no_gridfs_group_when_no_bucket(self) -> None:
        client = MagicMock()
        client["mydb"].list_collection_names = AsyncMock(
            return_value=["users", "orders"]
        )
        driver = _make_driver(client)
        items = await driver.explore_list(["mydb"])
        assert items == [
            ExploreItem(name="orders", type="collection", expandable=True),
            ExploreItem(name="users", type="collection", expandable=True),
        ]

    async def test_gridfs_lists_bucket_names_as_leaves(self) -> None:
        client = MagicMock()
        client["mydb"].list_collection_names = AsyncMock(
            return_value=["fs.files", "fs.chunks", "images.files", "images.chunks"]
        )
        driver = _make_driver(client)
        items = await driver.explore_list(["mydb", "gridfs"])
        assert items == [
            ExploreItem(name="fs", type="gridfs_bucket", expandable=False),
            ExploreItem(name="images", type="gridfs_bucket", expandable=False),
        ]


class TestDescribeGridfsBucket:
    async def test_returns_stats_not_a_file_listing(self) -> None:
        client = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[{"count": 12345, "total_size": 999}])
        client["mydb"]["fs.files"].aggregate = AsyncMock(return_value=cursor)
        driver = _make_driver(client)
        result = await driver.explore_describe(["mydb", "gridfs", "fs"])
        assert isinstance(result, GenericRecordDescription)
        assert result.kind == "mongodb.gridfs_bucket"
        labels = {f.label: f.value for f in result.fields}
        assert labels["Files"] == "12,345"
        assert labels["Total Size"] == "999 bytes"
        assert "gridfs.fs" in labels["Query"]

    async def test_empty_bucket_reports_zero(self) -> None:
        client = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=[])
        client["mydb"]["fs.files"].aggregate = AsyncMock(return_value=cursor)
        driver = _make_driver(client)
        result = await driver.explore_describe(["mydb", "gridfs", "fs"])
        assert isinstance(result, GenericRecordDescription)
        labels = {f.label: f.value for f in result.fields}
        assert labels["Files"] == "0"


class TestFindGridfs:
    async def test_find_routes_to_gridfs_files_collection(self) -> None:
        client = MagicMock()
        cursor = MagicMock()
        file_id = ObjectId()
        cursor.to_list = AsyncMock(
            return_value=[
                {
                    "_id": file_id,
                    "filename": "report.pdf",
                    "length": 4096,
                    "uploadDate": datetime(2026, 1, 1),
                    "contentType": "application/pdf",
                    "metadata": {"owner": "alice"},
                }
            ]
        )
        db = client["mydb"]
        db.name = "mydb"
        db["fs.files"].find.return_value.limit.return_value.sort.return_value = cursor
        driver = _make_driver(client)
        result = await driver.execute(
            '{"find": "gridfs.fs", "db": "mydb", "filter": {}, "limit": 50}', []
        )
        assert isinstance(result, ReadResult)
        assert "_id" in result.columns
        assert "filename" in result.columns
        assert "content" in result.columns
        row = dict(zip(result.columns, result.rows[0]))
        assert row["_id"] == str(file_id)
        assert row["filename"] == "report.pdf"
        assert row["length"] == "4096"  # flatten_docs stringifies scalar values
        lob = row["content"]
        assert isinstance(lob, LobPlaceholder)
        assert lob.ref is not None
        assert lob.ref.startswith("gridfs:")
        db_name, bucket, ref_id = json.loads(lob.ref[len("gridfs:") :])
        assert (db_name, bucket, ref_id) == ("mydb", "fs", str(file_id))

    async def test_aggregate_on_gridfs_collection_raises_driver_error(self) -> None:
        driver = _make_driver(_open_client()[0])
        with pytest.raises(DriverError, match="only support"):
            await driver.execute(
                '{"aggregate": "gridfs.fs", "db": "mydb", "pipeline": []}', []
            )


class TestExplorePreviewGridfs:
    async def test_preview_routes_to_gridfs_files_collection(self) -> None:
        client = MagicMock()
        cursor = MagicMock()
        file_id = ObjectId()
        cursor.to_list = AsyncMock(
            return_value=[
                {
                    "_id": file_id,
                    "filename": "report.pdf",
                    "length": 4096,
                    "uploadDate": datetime(2026, 1, 1),
                    "contentType": "application/pdf",
                    "metadata": {"owner": "alice"},
                }
            ]
        )
        db = client["mydb"]
        db.name = "mydb"
        db["fs.files"].find.return_value.limit.return_value.sort.return_value = cursor
        driver = _make_driver(client)
        result = await driver.explore_preview(["mydb", "gridfs", "fs"])
        assert isinstance(result, ReadResult)
        assert "filename" in result.columns
        row = dict(zip(result.columns, result.rows[0]))
        assert row["filename"] == "report.pdf"

    async def test_preview_unknown_path_returns_none(self) -> None:
        driver = _make_driver(_open_client()[0])
        result = await driver.explore_preview(["mydb"])
        assert result is None


class TestExploreDownloadRefGridfs:
    async def test_gridfs_ref_delegates_to_gridfs_download(self) -> None:
        client = MagicMock()
        driver = _make_driver(client)
        file_id = ObjectId()
        ref = "gridfs:" + json.dumps(["mydb", "fs", str(file_id)])
        with patch("grannos.drivers.mongodb.AsyncGridFSBucket") as bucket_cls:
            grid_out = MagicMock()
            grid_out.filename = "report.pdf"
            grid_out.content_type = "application/pdf"
            grid_out.read = AsyncMock(return_value=b"%PDF-1.4")
            grid_out.close = AsyncMock()
            bucket_cls.return_value.open_download_stream = AsyncMock(
                return_value=grid_out
            )
            result = await driver.explore_download_ref(ref, None)
        assert isinstance(result, DownloadResult)
        assert result.filename == "report.pdf"
        assert result.content_type == "application/pdf"
        bucket_cls.assert_called_once()
        _, kwargs = bucket_cls.call_args
        assert kwargs["bucket_name"] == "fs"
        bucket_cls.return_value.open_download_stream.assert_called_once_with(file_id)

    async def test_non_gridfs_ref_falls_back_to_cache_lookup(self) -> None:
        driver = _make_driver(_open_client()[0])
        with pytest.raises(DriverError):
            # Not a "gridfs:" ref and not in the base cache -> base class's
            # "no longer available" error, proving the fallback ran.
            await driver.explore_download_ref("some-uuid-ref", None)
