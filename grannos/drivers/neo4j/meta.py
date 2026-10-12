"""Capabilities of the Neo4j driver, importable without its client package."""

from ...protocol import DriverParam, Language, ParamType
from ..base import DriverMeta


class Neo4jMeta(DriverMeta):
    """What the Neo4j driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("neo4j",)

    LABEL = "Neo4j"

    LANGUAGES = [Language.CYPHER]

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="uri",
            type=ParamType.STRING,
            label="Bolt URI",
            default="bolt://localhost:7687",
        ),
        DriverParam(key="user", type=ParamType.STRING, label="User", default="neo4j"),
        DriverParam(
            key="password",
            type=ParamType.STRING,
            label="Password",
            secret=True,
            required=False,
        ),
        DriverParam(
            key="database", type=ParamType.STRING, label="Database", default="neo4j"
        ),
    ]

    HELP: str = """\
## Neo4j

**Queries:** Cypher.

```cypher
MATCH (u:User {name: "Alice"})-[:BOUGHT]->(p:Product) RETURN u, p
```

Results are serialized and flattened: nodes expand to `col._labels`, `col.prop`,
…; relationships expand to `col._type`, `col.prop`, …

**Comments:** `//` to the end of the line, and `/* ... */`.

**Resources:**

```
(root)
├── entities       → <label>  → properties → property name (sampled from existing nodes)
├── relationships  → <type>   → properties → property name (sampled from existing relationships)
└── indexes        → index name
```

Describing an index returns the indexed properties (direction = index type,
e.g. `RANGE`, `TEXT`, `POINT`) and whether it's unique.

Describing a label or relationship type returns its properties (name,
observed types, whether mandatory) and the relationship types connecting it
to other labels (or, for a relationship type, the label pairs it connects).
Describing the properties group returns the same property list on its own;
describing a single property adds a value sample.
"""
