"""Capabilities of the PostgreSQL driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class PostgresMeta(DriverMeta):
    """What the PostgreSQL driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("psycopg",)

    LABEL = "PostgreSQL"

    LANGUAGES = [Language.SQL]

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=5432),
        DriverParam(key="database", type=ParamType.STRING, label="Database"),
        DriverParam(key="user", type=ParamType.STRING, label="User"),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
    ]

    HELP: str = """\
## PostgreSQL

**Queries:** Standard SQL.

```sql
SELECT * FROM orders WHERE status = 'open'
SELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id
INSERT INTO orders (customer_id, status) VALUES (1, 'open') RETURNING id
```

**Exporting to a file:** `\\copy { table | (query) } TO '/local/path' [(options)]`,
same syntax as psql's `\\copy`. The result is streamed straight to a file on
the machine running grannos-py rather than returned as rows. `(options)` is
passed straight through to Postgres's `COPY` command, so any option it
accepts works here too — common ones are `FORMAT csv|text|binary`, `HEADER`,
`DELIMITER '|'`, and `NULL 'value'`. Full list:
https://www.postgresql.org/docs/current/sql-copy.html

```sql
\\copy orders TO '/tmp/orders.csv' (FORMAT csv, HEADER)
\\copy (SELECT * FROM orders WHERE status = 'open') TO '/tmp/open_orders.csv' (FORMAT csv, HEADER)
```

**Importing from a file:** `\\copy table [(column_list)] FROM '/local/path'
[(options)]`, same syntax as psql's `\\copy`. The file is streamed straight
from the machine running grannos-py into the table. `(options)` is passed
straight through to Postgres's `COPY` command, same options as above.

```sql
\\copy orders FROM '/tmp/orders.csv' (FORMAT csv, HEADER)
\\copy orders (id, status) FROM '/tmp/orders_partial.csv' (FORMAT csv, HEADER)
```

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)  ← non-system schemas (pg_namespace, excluding pg_catalog/information_schema/pg_*)
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indexes      → name, access method (e.g. btree, gin)
```

Describing a table or view returns column metadata (name, type, nullability,
primary key flag, default). Describing an index returns its key fields,
direction, uniqueness, INCLUDE columns, and DDL (as reported by `pg_indexes`).

**Session properties:** run `SET <parameter> = <value>` (or `SET <parameter>
TO <value>`) like any other statement. It's remembered per-connection and
automatically re-applied if the connection is silently reconnected (e.g.
after an idle timeout).
"""
