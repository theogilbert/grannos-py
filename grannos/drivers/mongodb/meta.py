"""Capabilities of the MongoDB driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class MongoMeta(DriverMeta):
    """What the MongoDB driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("pymongo",)

    LABEL = "MongoDB"

    LANGUAGES = [Language.MONGO]

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

`filter`, `sort`, `projection`, `skip`, and `limit` are all optional. `find`
defaults to a limit of 1000 rows when `"limit"` is omitted. `findOne` takes the
same keys bar `skip`/`limit`, and returns the first match.

```json
{"countDocuments": "orders", "db": "mydb", "filter": {"status": "open"}}
{"estimatedDocumentCount": "orders", "db": "mydb"}
{"distinct": "orders", "db": "mydb", "key": "status", "filter": {"amount": {"$gt": 10}}}
```

`countDocuments` (or its shorthand `count`) counts exactly, and also takes
`skip`/`limit`; `estimatedDocumentCount` reads the collection's metadata, so it
is instant but ignores any filter. Both return one `count` row. `distinct`
returns one row per distinct value of `key`, a dotted path for a nested field.

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
{"replaceOne": "users", "db": "mydb", "filter": {"name": "Alice"}, "replacement": {"name": "Alice", "age": 32}}
```

`updateOne`, `updateMany`, and `replaceOne` take an optional `options` object:
`upsert`, `arrayFilters`, `hint`, `collation`.

```json
{"updateOne": "users", "db": "mydb", "filter": {"name": "Dave"}, "update": {"$set": {"age": 40}}, "options": {"upsert": true}}
```

`findOneAndUpdate`, `findOneAndReplace`, and `findOneAndDelete` write like their
`updateOne`/`replaceOne`/`deleteOne` counterparts, but return the document
itself — as it was before the write, or after it with `"returnDocument":
"after"`. `sort` picks which one when several match, and `projection` and
`upsert` are accepted as well:

```json
{"findOneAndUpdate": "counters", "db": "mydb", "filter": {"_id": "orders"},
 "update": {"$inc": {"seq": 1}}, "upsert": true, "returnDocument": "after"}
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
{"renameCollection": "events", "db": "mydb", "to": "events_2026", "dropTarget": false}
{"createIndex": "users", "db": "mydb", "keys": {"email": 1}, "options": {"unique": true}}
{"dropIndex": "users", "db": "mydb", "name": "email_1"}
```

`options` is optional for both `createCollection` and `createIndex` and is
passed through to the underlying pymongo call.

**Any other command:** `runCommand` sends a raw [database
command](https://www.mongodb.com/docs/manual/reference/command/) and returns its
reply as one row:

```json
{"runCommand": {"collStats": "orders"}, "db": "mydb"}
{"runCommand": {"explain": {"find": "orders", "filter": {"status": "open"}}}, "db": "mydb"}
```

Results are flattened with dot-notation column names (`address.city`, `address.zip`).

**GridFS:**

A bucket named `<bucket>` (backed by `<bucket>.files`/`<bucket>.chunks`) is
queried with `find` on a synthetic collection name `"gridfs.<bucket>"` —
`filter`/`sort`/`limit` apply to the bucket's file metadata, not raw chunks
(`countDocuments`, `estimatedDocumentCount`, `distinct`, and `findOne` work the
same way; `aggregate` and the write commands don't):

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
