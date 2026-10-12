"""Capabilities of the Oracle driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class OracleMeta(DriverMeta):
    """What the Oracle driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("oracledb",)

    LABEL = "Oracle"

    LANGUAGES = [Language.SQL]

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=1521),
        DriverParam(key="service_name", type=ParamType.STRING, label="Service Name"),
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
## Oracle

Uses the `oracledb` driver in thin mode — no Oracle Instant Client required.

### Queries

Standard SQL.

```sql
SELECT * FROM employees WHERE department_id = 10 AND hire_date > DATE '2024-01-01'
SELECT e.id, d.name FROM employees e JOIN departments d ON d.id = e.department_id
INSERT INTO employees (department_id, name) VALUES (10, 'Alice')
ALTER SESSION SET NLS_LENGTH_SEMANTICS = CHAR
```

**Comments:** `--` to the end of the line, and `/* ... */`. They reach
Oracle with the statement, so an optimizer hint (`/*+ ... */`) takes effect.

### Importing from a file

```
LOAD <table> [(column_list)] [FROM] '/local/path' [(options)]
```

Oracle has no `\\copy`: SQL*Loader is a separate binary and an external table
reads the *database server's* filesystem, so this does what SQLcl's `LOAD`
does — the file is read on the machine running grannos-py, parsed here, and
sent as batched INSERTs. `FROM` is optional, so SQLcl's
`LOAD <table> '/local/path'` works too.

Target columns come from the explicit `(column_list)` if given, else from the
file's header row if `HEADER` is set, else from the table's own column order.

**Options** — comma-separated inside the parentheses, in any order:

- `FORMAT csv` — the only format, and the default.
- `HEADER` — the first row names the columns instead of carrying data.
- `DELIMITER ','` — field separator, one character; `'\\t'` means tab.
- `QUOTE '"'` — quote character.
- `NULL 'text'` — field value to load as NULL. An empty field, quoted or not,
  is already NULL: Oracle stores a zero-length string that way.
- `SKIP 0` — leading lines to discard before the header.
- `BATCH 1000` — rows per round-trip.
- `ENCODING 'utf-8'` — how to decode the file.
- `DATEFORMAT 'YYYY-MM-DD'` — how the file writes its dates. Values are bound
  as text and converted by Oracle through the session's NLS format models,
  which default to `DD-MON-RR` and reject an ISO date with `ORA-01843`; this
  sets them for the load and puts the session's back afterwards. It covers
  DATE *and* TIMESTAMP columns, which convert through separate models.
- `TIMESTAMPFORMAT 'YYYY-MM-DD HH24:MI:SS'` — overrides `DATEFORMAT` for
  TIMESTAMP columns, for a file whose timestamps carry a time and whose dates
  don't.

**Examples:**

```sql
LOAD employees FROM '/tmp/employees.csv' (FORMAT csv, HEADER)
LOAD employees FROM '/tmp/employees.csv' (HEADER, DATEFORMAT 'YYYY-MM-DD')
LOAD hr.employees (name, department_id) FROM '/tmp/employees.csv' (HEADER, BATCH 5000)
LOAD employees FROM '/tmp/employees.tsv' (DELIMITER '\\t', SKIP 2, NULL 'NA')
```

A row Oracle rejects aborts the load, naming the line it occupies in the file
(counting from 1, header included); rows already sent stay uncommitted, so
`ROLLBACK` undoes a partial load.

### Resources

```
(root)  ← non-system schemas (ALL_USERS where ORACLE_MAINTAINED = 'N')
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indexes      → name, index type
```

Describing a table or view returns column metadata (name, type, nullability,
primary key flag, default). Describing an index returns its key fields,
direction, and uniqueness.

### Session properties

Run `ALTER SESSION SET <property> = <value>` like any other statement. It's remembered per-connection and automatically re-applied
if the connection is silently reconnected (e.g. after an idle timeout).

### PL/SQL

Anonymous blocks and `CREATE PROCEDURE`/`FUNCTION`/`PACKAGE` run as
ordinary statements — send the whole block, embedded `;` and all. Do *not*
include a trailing `/`: that's a SQL\\*Plus command, not SQL, and it's rejected
with an explicit error.

```sql
BEGIN
    INSERT INTO employees (department_id, name) VALUES (10, 'Alice');
    DBMS_OUTPUT.PUT_LINE('inserted');
END;
```

`DBMS_OUTPUT` is enabled on every connection; anything a statement prints comes
back as `info` messages alongside its result — no `SET SERVEROUTPUT ON` needed.
A PL/SQL object that compiles *with errors* is reported by Oracle as a
successful CREATE; its compilation errors come back as `warning` messages
carrying the line and column of the offending token.

### Transactions

Writes are not auto-committed. Run `COMMIT` (or `ROLLBACK`)
explicitly — anything uncommitted is lost when the connection closes, including
after an idle timeout.
"""
