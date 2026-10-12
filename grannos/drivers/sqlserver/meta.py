"""Capabilities of the SQL Server driver, importable without its client package."""

from ...protocol import DriverParam, DriverParamChoice, Language, ParamType
from ..base import DriverMeta


class SQLServerMeta(DriverMeta):
    """What the SQL Server driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("mssql_python",)

    LABEL = "SQL Server"

    LANGUAGES = [Language.SQL]

    PARAMS: list[DriverParam] = [
        DriverParam(key="host", type=ParamType.STRING, label="Host"),
        DriverParam(key="port", type=ParamType.INTEGER, label="Port", default=1433),
        DriverParam(key="database", type=ParamType.STRING, label="Database"),
        DriverParam(key="user", type=ParamType.STRING, label="User"),
        DriverParam(
            key="applicationIntent",
            type=ParamType.ENUM,
            label="Application Intent",
            choices=[
                DriverParamChoice(value="READ_WRITE", label="READ_WRITE"),
                DriverParamChoice(value="READ_ONLY", label="READ_ONLY"),
            ],
        ),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
    ]

    HELP: str = """\
## SQL Server

**Queries:** Standard T-SQL.

```sql
SELECT * FROM dbo.orders WHERE status = 'open'
SELECT o.id, c.name FROM dbo.orders o JOIN dbo.customers c ON c.id = o.customer_id
INSERT INTO dbo.orders (customer_id, status) VALUES (1, 'open')
```

**Comments:** `--` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
└── <schema>
    └── <table|view>
        ├── columns      → name, data type
        └── indices      → name, type (e.g. CLUSTERED)
```

System schemas (`sys`, `INFORMATION_SCHEMA`, `guest`, `db_*`) are hidden.

Describing a table or view returns full column metadata (name, type,
nullability, default).
"""
