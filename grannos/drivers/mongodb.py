"""MongoDB driver — requires: pip install pymongo"""

import logging
import base64
import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import gridfs
import pymongo
import pymongo.errors
from bson import ObjectId, json_util
from bson.errors import InvalidId
from gridfs import AsyncGridFSBucket

from ..log import log_query
from ..protocol import (
    DescribeResult,
    DownloadResult,
    DriverParam,
    ExploreItem,
    GenericRecordDescription,
    IndexDescription,
    IndexKeyField,
    Language,
    LobPlaceholder,
    NodeType,
    ParamType,
    ReadResult,
    RecordField,
    WriteResult,
)
from ..tabular import flatten_docs
from .base import BaseDriver, ConnectionLostError, DriverError, DriverSettings

_DEFAULT_FIND_LIMIT = 1000
_GRIDFS_PREFIX = "gridfs."
"""Collection-name prefix that routes a find to a GridFS bucket's file
metadata instead of a real collection — e.g. `"gridfs.fs"` for the default
bucket. See MongoDriver._find_gridfs."""

_GRIDFS_REF_PREFIX = "gridfs:"
"""LobPlaceholder.ref prefix identifying a GridFS file cell (as opposed to an
ordinary in-memory-cached LOB ref) — see MongoDriver.explore_download_ref."""


logger = logging.getLogger(__name__)


class _Op(StrEnum):
    FIND = "find"
    AGGREGATE = "aggregate"
    INSERT_ONE = "insertOne"
    INSERT_MANY = "insertMany"
    UPDATE_ONE = "updateOne"
    UPDATE_MANY = "updateMany"
    DELETE_ONE = "deleteOne"
    DELETE_MANY = "deleteMany"
    CREATE_COLLECTION = "createCollection"
    DROP_COLLECTION = "dropCollection"
    CREATE_INDEX = "createIndex"
    DROP_INDEX = "dropIndex"


class MongoDriver(BaseDriver):
    """MongoDB driver backed by the pymongo async API.

    Args:
        params: Connect request fields (``uri``).
        client: Open AsyncMongoClient. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "MongoDB"
    LANGUAGES = [Language.MONGO]

    FIND_PATHS = {
        NodeType.DATABASE: [["*"]],
        NodeType.COLLECTION: [["*", "*"]],
        NodeType.FIELD: [["*", "*", "fields", "*"]],
        NodeType.INDEX: [["*", "*", "indexes", "*"]],
        NodeType.GRIDFS_BUCKET: [["*", "gridfs", "*"]],
    }

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="uri",
            type=ParamType.STRING,
            label="Connection URI",
            default="mongodb://localhost:27017",
        ),
        DriverParam(
            key="username", type=ParamType.STRING, label="Username", required=False
        ),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            required=False,
            secret=True,
        ),
    ]

    HELP: str = """\
## MongoDB

**Queries:** MongoDB Extended JSON command objects, one per statement — a
file may hold any number, and `//` and `/* */` comments between or inside
them are ignored. `"db"` is required and names the target database. The
top-level operation key names the collection.

```json
{"find": "users", "db": "auth"}
```

**Read:**

```json
{"find": "orders", "db": "mydb", "filter": {"status": "open"}, "sort": {"createdAt": -1}, "limit": 100}
```

`filter`, `sort`, `projection`, and `limit` are all optional. `find` defaults
to a limit of 1000 rows when `"limit"` is omitted.

```json
{"aggregate": "orders", "db": "mydb", "pipeline": [
  {"$group": {"_id": "$status", "total": {"$sum": "$amount"}}},
  {"$sort": {"total": -1}}
]}
```

**Write:**

```json
{"insertOne": "users", "db": "mydb", "document": {"name": "Alice", "age": 30}}
{"updateOne": "users", "db": "mydb", "filter": {"name": "Alice"}, "update": {"$set": {"age": 31}}}
{"deleteOne": "orders", "db": "mydb", "filter": {"status": "cancelled"}}
```

Document values support Extended JSON, so BSON types that plain JSON can't
express — dates, ObjectIds, decimals — can be written directly:

```json
{"updateOne": "events", "db": "mydb",
 "filter": {"_id": {"$oid": "5f8d0d55b54764421b7156c0"}},
 "update": {"$set": {"occurredAt": {"$date": "2024-01-01T00:00:00Z"}}}}
```

**Collections and indexes:**

```json
{"createCollection": "events", "db": "mydb"}
{"dropCollection": "old_events", "db": "mydb"}
{"createIndex": "users", "db": "mydb", "keys": {"email": 1}, "options": {"unique": true}}
{"dropIndex": "users", "db": "mydb", "name": "email_1"}
```

`options` is optional for both `createCollection` and `createIndex` and is
passed through to the underlying pymongo call.

Results are flattened with dot-notation column names (`address.city`, `address.zip`).

**GridFS:**

A bucket named `<bucket>` (backed by `<bucket>.files`/`<bucket>.chunks`) is
queried with `find` on a synthetic collection name `"gridfs.<bucket>"` —
`filter`/`sort`/`limit` apply to the bucket's file metadata, not raw chunks
(`aggregate` isn't supported for it, only `find`):

```json
{"find": "gridfs.fs", "db": "mydb", "filter": {"filename": {"$regex": "^report-2026"}}, "limit": 50}
```

Each row is one file: `_id`, `filename`, `length`, `uploadDate`, `contentType`,
`md5`, `metadata.*`, and a `content` cell — a LOB placeholder, not the actual bytes.
A bucket can hold tens of thousands of files, so nothing here ever reads file
content up front; `content` is fetched lazily via `explore.download`'s `ref`
param once you actually want it.

**Resources:**

```
(root)
└── <database>
    ├── <collection>
    │   ├── fields   → top-level field names (sampled from up to 10 documents)
    │   └── indexes  → index names
    └── gridfs                    (only shown when the database has any)
        └── <bucket>              (leaf — query it, see GridFS above)
```

Describing an index returns its key fields with their sort direction (`asc` / `desc`).
A GridFS bucket is inferred from a `<bucket>.files` collection; its backing
`.files`/`.chunks` collections are hidden from the plain collection list once
represented under `gridfs`. Describing a bucket returns file count, total
size, and example query syntax — not a file listing. `explore.preview` on a
bucket runs `{"find": "gridfs.<bucket>", "limit": 10}`, same as typing it in
the query bar.

Any LOB cell in a result row (a GridFS `content` cell, or an ordinary BSON
Binary value) carries a `ref` — pass that to `explore.download`'s `ref` param
to fetch its full content later without re-running the query.
"""

    def __init__(
        self,
        params: dict[str, Any],
        client: pymongo.AsyncMongoClient,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._client = client

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "MongoDriver":
        return cls(params, await _make_mongo_client(params), settings)

    async def reconnect(self) -> None:
        await self._client.close()
        self._client = await _make_mongo_client(self.params)

    async def disconnect(self) -> None:
        await self._client.close()

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a MongoDB command expressed as a JSON string.

        Args:
            query: MongoDB Extended JSON object following MongoDB command syntax
                (``$date``, ``$oid``, ``$numberDecimal``, etc. are accepted in
                addition to plain JSON). The top-level key selects the operation;
                its value is the collection name. Supported operations:

                - ``find``: ``{"find": "col", "filter": {}, "projection": {},
                  "sort": {}, "limit": N}``
                - ``aggregate``: ``{"aggregate": "col", "pipeline": [...]}``
                - ``insertOne``: ``{"insertOne": "col", "document": {...}}``
                - ``insertMany``: ``{"insertMany": "col", "documents": [...]}``
                - ``updateOne``: ``{"updateOne": "col", "filter": {}, "update": {}}``
                - ``updateMany``: ``{"updateMany": "col", "filter": {}, "update": {}}``
                - ``deleteOne``: ``{"deleteOne": "col", "filter": {}}``
                - ``deleteMany``: ``{"deleteMany": "col", "filter": {}}``
                - ``createCollection``: ``{"createCollection": "col", "options": {}}``
                - ``dropCollection``: ``{"dropCollection": "col"}``
                - ``createIndex``: ``{"createIndex": "col", "keys": {}, "options": {}}``
                - ``dropIndex``: ``{"dropIndex": "col", "name": "..."}``

                ``"db"`` is required and names the target database.
            binds: Unused for MongoDB.
        """
        # The command as submitted is the statement here; the driver calls it
        # fans out to are logged separately at their own call sites.
        log_query(logger, query)
        try:
            cmd: dict[str, Any] = json_util.loads(_strip_comments(query))
            if "db" not in cmd:
                raise DriverError(
                    'MongoDB command must include a "db" key specifying the target database'
                )
            db = self._client[cmd.pop("db")]
            if _Op.FIND in cmd:
                return await self._find(db, cmd)
            if _Op.AGGREGATE in cmd:
                return await self._aggregate(db, cmd)
            if _Op.INSERT_ONE in cmd:
                return await self._insert_one(db, cmd)
            if _Op.INSERT_MANY in cmd:
                return await self._insert_many(db, cmd)
            if _Op.UPDATE_ONE in cmd:
                return await self._update_one(db, cmd)
            if _Op.UPDATE_MANY in cmd:
                return await self._update_many(db, cmd)
            if _Op.DELETE_ONE in cmd:
                return await self._delete_one(db, cmd)
            if _Op.DELETE_MANY in cmd:
                return await self._delete_many(db, cmd)
            if _Op.CREATE_COLLECTION in cmd:
                return await self._create_collection(db, cmd)
            if _Op.DROP_COLLECTION in cmd:
                return await self._drop_collection(db, cmd)
            if _Op.CREATE_INDEX in cmd:
                return await self._create_index(db, cmd)
            if _Op.DROP_INDEX in cmd:
                return await self._drop_index(db, cmd)
            raise DriverError(f"Unsupported command keys: {list(cmd.keys())}")
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            if isinstance(exc, DriverError):
                raise
            if isinstance(exc, json.JSONDecodeError):
                raise DriverError(f"MongoDB command must be valid JSON: {exc}") from exc
            raise DriverError(str(exc)) from exc

    async def _find(self, db: Any, cmd: dict[str, Any]) -> ReadResult:
        collection_name = cmd.pop(_Op.FIND)
        filter_ = cmd.pop("filter", {})
        projection = cmd.pop("projection", None)
        sort = cmd.pop("sort", None)
        limit = cmd.pop("limit", _DEFAULT_FIND_LIMIT)
        if collection_name.startswith(_GRIDFS_PREFIX):
            bucket = collection_name[len(_GRIDFS_PREFIX) :]
            return await self._find_gridfs(db, bucket, filter_, sort, limit)
        cursor = db[collection_name].find(filter_, projection).limit(limit)
        if sort:
            cursor = cursor.sort(list(sort.items()))
        return _docs_to_result(self._register_lob, await cursor.to_list())

    async def _aggregate(self, db: Any, cmd: dict[str, Any]) -> ReadResult:
        collection_name = cmd.pop(_Op.AGGREGATE)
        if collection_name.startswith(_GRIDFS_PREFIX):
            raise DriverError(
                f'GridFS collections only support "find", not "aggregate" — query '
                f'{collection_name!r} with {{"find": {collection_name!r}, "filter": {{...}}}}'
            )
        cursor = await db[collection_name].aggregate(cmd.pop("pipeline", []))
        return _docs_to_result(self._register_lob, await cursor.to_list())

    async def _find_gridfs(
        self,
        db: Any,
        bucket: str,
        filter_: dict[str, Any],
        sort: dict[str, Any] | None,
        limit: int,
    ) -> ReadResult:
        """Query a GridFS bucket's `.files` metadata collection, one row per
        matching file: filename, size, upload date, content-type, MD5, custom
        metadata, and a `content` LOB cell (fetched lazily via its `ref`, never
        eagerly read here — a bucket can hold tens of thousands of files, so
        this leans on the same filter/sort/limit machinery as a normal `find`
        instead of an unbounded tree listing).

        `projection` isn't supported here since the row shape is synthesized,
        not passed through — a projection would apply to the wrong doc shape.
        """
        cursor = db[f"{bucket}.files"].find(filter_).limit(limit)
        cursor = (
            cursor.sort(list(sort.items())) if sort else cursor.sort([("filename", 1)])
        )
        docs = await cursor.to_list()
        rows = [_gridfs_file_row(db.name, bucket, doc) for doc in docs]
        return _docs_to_result(self._register_lob, rows)

    async def _insert_one(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.INSERT_ONE)]
        await col.insert_one(cmd.pop("document", {}))
        return WriteResult(rows_affected=1)

    async def _insert_many(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.INSERT_MANY)]
        docs = cmd.pop("documents", [])
        if not docs:
            return WriteResult(rows_affected=0)
        result = await col.insert_many(docs)
        return WriteResult(rows_affected=len(result.inserted_ids))

    async def _update_one(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.UPDATE_ONE)]
        result = await col.update_one(cmd.pop("filter", {}), cmd.pop("update", {}))
        return WriteResult(rows_affected=result.modified_count)

    async def _update_many(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.UPDATE_MANY)]
        result = await col.update_many(cmd.pop("filter", {}), cmd.pop("update", {}))
        return WriteResult(rows_affected=result.modified_count)

    async def _delete_one(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.DELETE_ONE)]
        result = await col.delete_one(cmd.pop("filter", {}))
        return WriteResult(rows_affected=result.deleted_count)

    async def _delete_many(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.DELETE_MANY)]
        result = await col.delete_many(cmd.pop("filter", {}))
        return WriteResult(rows_affected=result.deleted_count)

    async def _create_collection(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        name = cmd.pop(_Op.CREATE_COLLECTION)
        await db.create_collection(name, **cmd.pop("options", {}))
        return WriteResult(rows_affected=1)

    async def _drop_collection(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        await db.drop_collection(cmd.pop(_Op.DROP_COLLECTION))
        return WriteResult(rows_affected=1)

    async def _create_index(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.CREATE_INDEX)]
        await col.create_index(
            list(cmd.pop("keys", {}).items()), **cmd.pop("options", {})
        )
        return WriteResult(rows_affected=1)

    async def _drop_index(self, db: Any, cmd: dict[str, Any]) -> WriteResult:
        col = db[cmd.pop(_Op.DROP_INDEX)]
        await col.drop_index(cmd.pop("name"))
        return WriteResult(rows_affected=1)

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        try:
            return await self._explore_list(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise

    async def _explore_list(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                log_query(logger, "list_database_names")
                return [
                    ExploreItem(name=n, type="database", expandable=True)
                    for n in sorted(await self._client.list_database_names())
                ]
            case [db_name]:
                log_query(logger, f"list_collection_names {db_name}")
                names = sorted(await self._client[db_name].list_collection_names())
                buckets = _gridfs_buckets(names)
                items = [
                    ExploreItem(name=n, type="collection", expandable=True)
                    for n in names
                    if not _is_gridfs_internal(n, buckets)
                ]
                if buckets:
                    items.append(
                        ExploreItem(name="gridfs", type="group", expandable=True)
                    )
                return items
            case [db_name, "gridfs"]:
                # Buckets are a leaf, not expandable — a bucket can hold tens
                # of thousands of files, so individual files aren't tree-listed
                # at all; describing the bucket gives stats + example query
                # syntax, and files are reached via a "gridfs.<bucket>" find.
                log_query(logger, f"list_collection_names {db_name}")
                names = await self._client[db_name].list_collection_names()
                return [
                    ExploreItem(name=b, type="gridfs_bucket", expandable=False)
                    for b in sorted(_gridfs_buckets(names))
                ]
            case [_, _]:
                return [
                    ExploreItem(name="fields", type="group", expandable=True),
                    ExploreItem(name="indexes", type="group", expandable=True),
                ]
            case [db_name, collection_name, "fields"]:
                return [
                    ExploreItem(name=f, type="field", expandable=False)
                    for f in await self._sample_fields(db_name, collection_name)
                ]
            case [db_name, collection_name, "indexes"]:
                return [
                    ExploreItem(name=i, type="index", expandable=False)
                    for i in await self._list_indexes(db_name, collection_name)
                ]
            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        try:
            return await self._explore_preview(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise

    async def _explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case [db_name, collection_name]:
                db = self._client[db_name]
                return await self._find(db, {_Op.FIND: collection_name, "limit": 10})
            case [db_name, "gridfs", bucket]:
                db = self._client[db_name]
                return await self._find(
                    db, {_Op.FIND: f"{_GRIDFS_PREFIX}{bucket}", "limit": 10}
                )
            case _:
                return None

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        try:
            return await self._explore_describe(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise

    async def _explore_describe(self, path: list[str]) -> DescribeResult:
        match path:
            case [db_name, collection_name, "indexes"]:
                log_query(logger, f"index_information {db_name}.{collection_name}")
                info = await self._client[db_name][collection_name].index_information()
                return [
                    _spec_to_index_description(name, spec, collection_name)
                    for name, spec in sorted(info.items())
                ]
            case [db_name, collection_name, "indexes", index_name]:
                log_query(logger, f"index_information {db_name}.{collection_name}")
                info = await self._client[db_name][collection_name].index_information()
                spec = info.get(index_name)
                if spec is None:
                    return None
                return _spec_to_index_description(index_name, spec, collection_name)
            case [db_name, "gridfs", bucket]:
                return await self._describe_gridfs_bucket(db_name, bucket)
            case _:
                return None

    async def explore_download_ref(
        self, ref: str, dest_path: str | None
    ) -> DownloadResult:
        """GridFS file cells (from a "gridfs.<bucket>" find) carry a ref
        encoding (db, bucket, file _id) rather than a cached in-memory value —
        unlike an ordinary BSON Binary cell, the content was deliberately never
        read up front (that's the whole point of GridFS: values too large for
        a normal document). The _id (not filename) identifies the file since
        GridFS allows multiple files to share the same filename. Falls back
        to BaseDriver's cache-based lookup for ordinary LOB refs.
        """
        if ref.startswith(_GRIDFS_REF_PREFIX):
            try:
                db_name, bucket, file_id = json.loads(ref[len(_GRIDFS_REF_PREFIX) :])
                file_id = ObjectId(file_id)
            except (json.JSONDecodeError, ValueError, InvalidId) as exc:
                raise DriverError("Malformed GridFS ref") from exc
            try:
                return await self._download_gridfs_file(
                    db_name, bucket, file_id, dest_path
                )
            except gridfs.NoFile as exc:
                raise DriverError(f"No such GridFS file: {file_id!r}") from exc
        return await super().explore_download_ref(ref, dest_path)

    async def _describe_gridfs_bucket(
        self, db_name: str, bucket_name: str
    ) -> GenericRecordDescription:
        """Cheap aggregate stats (count + total size), not a file listing —
        a bucket can hold tens of thousands of files. Includes example query
        syntax, since that's now the only way to reach individual files."""
        log_query(logger, f"aggregate {db_name}.{bucket_name}.files")
        cursor = await self._client[db_name][f"{bucket_name}.files"].aggregate(
            [
                {
                    "$group": {
                        "_id": None,
                        "count": {"$sum": 1},
                        "total_size": {"$sum": "$length"},
                    }
                }
            ]
        )
        docs = await cursor.to_list()
        count = docs[0]["count"] if docs else 0
        total_size = docs[0]["total_size"] if docs else 0
        example = json.dumps(
            {
                "find": f"{_GRIDFS_PREFIX}{bucket_name}",
                "db": db_name,
                "filter": {},
                "limit": 50,
            }
        )
        return GenericRecordDescription(
            kind="mongodb.gridfs_bucket",
            name=bucket_name,
            fields=[
                RecordField(label="Files", value=f"{count:,}"),
                RecordField(label="Total Size", value=f"{total_size:,} bytes"),
                RecordField(label="Query", value=example),
            ],
        )

    async def _download_gridfs_file(
        self, db_name: str, bucket_name: str, file_id: ObjectId, dest_path: str | None
    ) -> DownloadResult:
        grid_bucket = AsyncGridFSBucket(self._client[db_name], bucket_name=bucket_name)
        grid_out = await grid_bucket.open_download_stream(file_id)
        try:
            filename = grid_out.filename
            content_type = grid_out.content_type or "application/octet-stream"
            if dest_path is not None:
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = await grid_out.readchunk()
                        if not chunk:
                            break
                        f.write(chunk)
                return DownloadResult(
                    filename=filename,
                    content_type=content_type,
                    size=grid_out.length,
                    written_to=dest_path,
                )
            content = await grid_out.read()
            return DownloadResult(
                filename=filename,
                content_type=content_type,
                size=len(content),
                content_base64=base64.b64encode(content).decode(),
            )
        finally:
            await grid_out.close()

    async def _sample_fields(self, db_name: str, collection_name: str) -> list[str]:
        log_query(logger, f"aggregate {db_name}.{collection_name}")
        cursor = await self._client[db_name][collection_name].aggregate(
            [{"$sample": {"size": 10}}]
        )
        docs = await cursor.to_list()
        seen: dict[str, None] = {}
        for doc in docs:
            for key in doc:
                seen[key] = None
        return list(seen)

    async def _list_indexes(self, db_name: str, collection_name: str) -> list[str]:
        log_query(logger, f"index_information {db_name}.{collection_name}")
        return sorted(await self._client[db_name][collection_name].index_information())


def _strip_comments(query: str) -> str:
    """Return ``query`` with ``//`` line and ``/* */`` block comments removed.

    Extended JSON has no comment syntax, but a query file does (the ``mongo``
    editor grammar treats both as whitespace), and a comment sent along with
    the command must not make it invalid JSON. Comment markers inside a
    string are left alone; a ``//`` in a URL value is not a comment.
    """
    out: list[str] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch == '"':
            end = i + 1
            while end < n and query[end] != '"':
                end += 2 if query[end] == "\\" else 1
            out.append(query[i : end + 1])
            i = end + 1
        elif query.startswith("//", i):
            eol = query.find("\n", i)
            i = n if eol < 0 else eol
        elif query.startswith("/*", i):
            close = query.find("*/", i + 2)
            i = n if close < 0 else close + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _gridfs_file_row(db_name: str, bucket: str, doc: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic row for one `<bucket>.files` document: _id, filename,
    size, metadata, and a `content` LOB cell carrying a ref the client can
    pass to explore.download later — never reads the actual file content.
    The ref encodes the file's _id rather than its filename, since GridFS
    allows multiple files in a bucket to share the same filename."""
    file_id = doc["_id"]
    filename = doc.get("filename", "")
    length = doc.get("length", 0)
    ref = _GRIDFS_REF_PREFIX + json.dumps([db_name, bucket, str(file_id)])
    return {
        "_id": file_id,
        "filename": filename,
        "length": length,
        "uploadDate": doc.get("uploadDate"),
        "contentType": doc.get("contentType"),
        "md5": doc.get("md5"),
        "metadata": doc.get("metadata"),
        "content": LobPlaceholder(text=f"GridFS file ({length:,} bytes)", ref=ref),
    }


def _gridfs_buckets(collection_names: list[str]) -> set[str]:
    """Bucket names inferred from `<bucket>.files` collections."""
    return {n[: -len(".files")] for n in collection_names if n.endswith(".files")}


def _is_gridfs_internal(name: str, buckets: set[str]) -> bool:
    """Whether `name` is a `.files`/`.chunks` collection backing a GridFS
    bucket already represented under the "gridfs" tree branch — hidden from
    the plain collection list to avoid showing it twice."""
    for suffix in (".files", ".chunks"):
        if name.endswith(suffix) and name[: -len(suffix)] in buckets:
            return True
    return False


def _index_direction(direction: Any) -> str:
    if direction == 1:
        return "asc"
    if direction == -1:
        return "desc"
    return str(direction)


def _spec_to_index_description(
    index_name: str, spec: dict, collection_name: str
) -> IndexDescription:
    fields = [
        IndexKeyField(name=field, direction=_index_direction(direction))
        for field, direction in spec.get("key", [])
    ]
    # Determine index_type from non-numeric key direction values (e.g. "text", "hashed", "2dsphere").
    non_numeric = {
        str(d) for _, d in spec.get("key", []) if not isinstance(d, (int, float))
    }
    index_type = next(iter(non_numeric)).lower() if non_numeric else "regular"
    partial = spec.get("partialFilterExpression")
    return IndexDescription(
        name=index_name,
        fields=fields,
        unique=bool(spec.get("unique", False)),
        tables=[collection_name],
        index_type=index_type,
        visible=not bool(spec.get("hidden", False)),
        ddl=json.dumps(partial, separators=(",", ":")) if partial is not None else None,
    )


def _docs_to_result(
    register_lob: Callable[[bytes | str, str], LobPlaceholder],
    docs: list[dict[str, Any]],
) -> ReadResult:
    if not docs:
        return ReadResult(columns=[], rows=[], rows_total=0)
    serialized = [
        {k: _serialize(register_lob, v) for k, v in doc.items()} for doc in docs
    ]
    # dict.fromkeys deduplicates while preserving first-seen order (set would not)
    columns: list[str] = list(dict.fromkeys(k for doc in serialized for k in doc))
    rows = [[doc.get(col) for col in columns] for doc in serialized]
    return flatten_docs(columns, rows, rows_total=len(docs))


def _serialize(
    register_lob: Callable[[bytes | str, str], LobPlaceholder], value: Any
) -> Any:
    """Recursively convert BSON types to plain Python values."""
    try:
        from bson import Decimal128, ObjectId

        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, Decimal128):
            return str(value)
    except ImportError:
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return register_lob(bytes(value), f"BSON Binary ({len(value)} bytes)")
    if isinstance(value, dict):
        return {k: _serialize(register_lob, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(register_lob, v) for v in value]
    return value


async def _make_mongo_client(params: dict[str, Any]) -> pymongo.AsyncMongoClient:
    kwargs: dict[str, Any] = {}
    if params.get("username"):
        kwargs["username"] = params["username"]
    if params.get("password"):
        kwargs["password"] = params["password"]
    client: pymongo.AsyncMongoClient | None = None
    try:
        # the constructor parses the URI eagerly and can raise (InvalidURI, ValueError,
        # ConfigurationError, ...) before any connection is made
        client = pymongo.AsyncMongoClient(params["uri"], **kwargs)
        # pymongo is lazy — force a connection to verify credentials
        log_query(logger, "admin.command ping")
        await client.admin.command("ping")
    except Exception as exc:
        if client is not None:
            await client.close()
        raise DriverError(str(exc)) from exc
    return client


def _maybe_raise_connection_lost(exc: Exception) -> None:
    if isinstance(
        exc,
        (
            pymongo.errors.AutoReconnect,
            pymongo.errors.ConnectionFailure,
            pymongo.errors.NetworkTimeout,
        ),
    ):
        raise ConnectionLostError(str(exc)) from exc
    # The idle timer closes the client out-of-band; pymongo surfaces the next
    # use as InvalidOperation rather than a network error.
    if isinstance(exc, pymongo.errors.InvalidOperation) and "after close" in str(exc):
        raise ConnectionLostError(str(exc)) from exc
