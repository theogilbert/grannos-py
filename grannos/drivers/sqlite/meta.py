"""Capabilities of the SQLite driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class SQLiteMeta(DriverMeta):
    """What the SQLite driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ()

    LABEL = "SQLite"

    LANGUAGES = [Language.SQL]

    PARAMS: list[DriverParam] = [
        DriverParam(key="database", type=ParamType.STRING, label="Database file path"),
    ]

    HELP: str = """\
## SQLite

Standard SQL.

```sql
SELECT * FROM users WHERE age > 21
SELECT u.id, o.total FROM users u JOIN orders o ON o.user_id = u.id
INSERT INTO users (name, age) VALUES ('Alice', 30)
```

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <table|view>
    ├── columns       → name, type
    ├── indices       → index name
    └── foreign_keys  → "col → ref_table.ref_col"
```

Describing a table or view returns full column metadata (name, type,
nullability, primary key flag).
"""
