"""Capabilities of the DuckDB driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class DuckDBMeta(DriverMeta):
    """What the DuckDB driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("duckdb",)

    LABEL = "DuckDB"

    LANGUAGES = [Language.SQL]

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="database",
            type=ParamType.STRING,
            label="Database file path or :memory:",
            required=False,
            default=":memory:",
        ),
    ]

    HELP: str = """\
## DuckDB

Standard SQL. DuckDB's SQL dialect also reads files directly:

```sql
SELECT * FROM read_parquet('/path/to/file.parquet')
SELECT * FROM read_csv('/path/to/file.csv', header = true)
SELECT * FROM 'glob/**/*.parquet'
```

**Create a table from a CSV file:**

```sql
CREATE TABLE users AS SELECT * FROM read_csv('/path/to/users.csv', header = true)
CREATE TABLE users AS SELECT * FROM '/path/to/users.csv'   -- reader inferred from the extension
COPY users FROM '/path/to/users.csv' (FORMAT csv, HEADER)  -- into an existing table
```

Column names and types are auto-detected. Pass
`columns = {'id': 'INTEGER', 'name': 'VARCHAR'}` to `read_csv` to declare
them yourself, or `delim`/`quote`/`skip` to override the dialect it sniffs.

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns       → name, type
        ├── indices       → index name
        └── foreign_keys  → "col → ref_table.ref_col"
```
"""
