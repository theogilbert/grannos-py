"""Neo4j driver — requires: pip install neo4j"""

import logging
import asyncio
from collections.abc import Callable
from typing import Any, LiteralString

import neo4j
import neo4j.exceptions

from ..log import log_query
from ..protocol import (
    Connection,
    DescribeResult,
    DriverParam,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    IndexDescription,
    IndexKeyField,
    Language,
    LobPlaceholder,
    NodeType,
    ParamType,
    ReadResult,
    WriteResult,
)
from ..tabular import flatten_docs
from .comments import blank_comments
from .base import (
    SAMPLE_SCAN_ROWS,
    BaseDriver,
    ConnectionLostError,
    DriverError,
    DriverSettings,
    build_column_samples,
)


logger = logging.getLogger(__name__)


async def _cypher(
    session: Any, query: str, params: Any = None, *, private: bool = False
) -> Any:
    """Log *query* and run it on *session*.

    Every Cypher statement this driver sends goes through here, so debug
    logging of them needs no change at the call sites.

    Args:
        session: Open Neo4j async session.
        query: Cypher text.
        params: Query parameters, logged alongside the statement.
        private: Set for the *user's* own statement, whose parameters are user
            data rather than the labels and property names a catalog query binds.
    """
    log_query(logger, query, None if private else params)
    return await session.run(query, params) if params else await session.run(query)


class Neo4jDriver(BaseDriver):
    """Neo4j driver backed by the official neo4j async Python client.

    Args:
        params: Connect request fields (``uri``, ``user``, ``password``, ``database``).
        driver: Open AsyncDriver instance. Use :meth:`create` instead of constructing directly.
    """

    LABEL = "Neo4j"
    LANGUAGES = [Language.CYPHER]

    FIND_PATHS = {
        NodeType.LABEL: [["entities", "*"]],
        NodeType.RELATIONSHIP_TYPE: [["relationships", "*"]],
        NodeType.PROPERTY: [
            ["entities", "*", "properties", "*"],
            ["relationships", "*", "properties", "*"],
        ],
        NodeType.INDEX: [["indexes", "*"]],
    }

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

    def __init__(
        self,
        params: dict[str, Any],
        driver: neo4j.AsyncDriver,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._driver = driver

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "Neo4jDriver":
        return cls(params, await _make_neo4j_driver(params), settings)

    async def reconnect(self) -> None:
        await self._driver.close()
        self._driver = await _make_neo4j_driver(self.params)

    async def disconnect(self) -> None:
        await self._driver.close()

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a Cypher statement. Positional bind values map to ``$0``, ``$1``, …

        Args:
            query: Cypher statement to execute.
            binds: Positional bind parameters (referenced as ``$0``, ``$1``, … in the query).

        Returns:
            ReadResult for queries that RETURN rows, DMLResult otherwise.

        Raises:
            ConnectionLostError: If the connection was lost during execution.
        """
        params = {str(i): v for i, v in enumerate(binds)}
        db = self.params.get("database", "neo4j")
        try:
            async with self._driver.session(database=db) as session:
                result = await _cypher(session, query, params, private=True)
                keyword = _plan_keyword(query)
                if keyword:
                    summary = await result.consume()
                    plan = summary.profile if keyword == "profile" else summary.plan
                    if plan is not None:
                        return _plan_to_result(plan, keyword == "profile")
                    return WriteResult(rows_affected=0)
                keys = result.keys()
                if keys:
                    rows = []
                    async for record in result:
                        rows.append(
                            [_serialize(self._register_lob, record[k]) for k in keys]
                        )
                    return flatten_docs(list(keys), rows)
                summary = await result.consume()
                c = summary.counters
                affected = (
                    c.nodes_created
                    + c.nodes_deleted
                    + c.relationships_created
                    + c.relationships_deleted
                    + c.properties_set
                )
                return WriteResult(rows_affected=affected)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise DriverError(str(exc)) from exc

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        try:
            return await self._explore_list(path)
        except Exception as exc:
            _maybe_raise_connection_lost(exc)
            raise

    async def _explore_list(self, path: list[str]) -> list[ExploreItem]:
        match path:
            case []:
                return [
                    ExploreItem(name="entities", type="group", expandable=True),
                    ExploreItem(name="relationships", type="group", expandable=True),
                    ExploreItem(name="indexes", type="group", expandable=True),
                ]
            case ["indexes"]:
                names = await self._query_column(
                    "SHOW INDEXES YIELD name RETURN name ORDER BY name", "name"
                )
                return [
                    ExploreItem(name=n, type="index", expandable=False) for n in names
                ]
            case ["entities"]:
                labels = await self._query_column(
                    "CALL db.labels() YIELD label RETURN label ORDER BY label", "label"
                )
                return [
                    ExploreItem(name=label, type="label", expandable=True)
                    for label in labels
                ]
            case ["relationships"]:
                types = await self._query_column(
                    "CALL db.relationshipTypes() YIELD relationshipType"
                    " RETURN relationshipType ORDER BY relationshipType",
                    "relationshipType",
                )
                return [
                    ExploreItem(name=t, type="relationship_type", expandable=True)
                    for t in types
                ]
            case ["entities", _label]:
                return [ExploreItem(name="properties", type="group", expandable=True)]
            case ["relationships", _rel_type]:
                return [ExploreItem(name="properties", type="group", expandable=True)]
            case ["entities", label, "properties"]:
                props = await self._node_properties(label)
                return [
                    ExploreItem(name=p, type="property", expandable=False)
                    for p in props
                ]
            case ["relationships", rel_type, "properties"]:
                props = await self._relationship_properties(rel_type)
                return [
                    ExploreItem(name=p, type="property", expandable=False)
                    for p in props
                ]
            case _:
                return []

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        match path:
            case ["entities", label]:
                result = await self.execute(
                    f"MATCH (n:`{label}`) RETURN n LIMIT 10", []
                )
                return result if isinstance(result, ReadResult) else None
            case ["relationships", rel_type]:
                result = await self.execute(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN r LIMIT 10", []
                )
                return result if isinstance(result, ReadResult) else None
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
            case ["indexes"]:
                return await self._describe_all_indices()
            case ["indexes", index_name]:
                return await self._describe_index(index_name)
            case ["entities", label]:
                return await self._describe_node_entity(label)
            case ["entities", label, "properties"]:
                return await self._node_field_descriptions(label)
            case ["entities", label, "properties", prop]:
                return await self._describe_node_field(label, prop)
            case ["relationships", rel_type]:
                return await self._describe_relationship_entity(rel_type)
            case ["relationships", rel_type, "properties"]:
                return await self._relationship_field_descriptions(rel_type)
            case ["relationships", rel_type, "properties", prop]:
                return await self._describe_relationship_field(rel_type, prop)
            case _:
                return None

    async def _describe_all_indices(self) -> list[IndexDescription]:
        specs = await self._all_index_specs()
        return [self._spec_to_description(s) for s in specs]

    async def _describe_index(self, index_name: str) -> IndexDescription | None:
        specs = await self._all_index_specs()
        spec = next((s for s in specs if s["name"] == index_name), None)
        if spec is None:
            return None
        return self._spec_to_description(spec)

    def _spec_to_description(self, spec: dict) -> IndexDescription:
        labels_or_types: list[str] = spec.get("labelsOrTypes") or []
        idx_type: str = spec.get("type") or ""
        return IndexDescription(
            name=spec["name"],
            fields=[
                IndexKeyField(name=prop, direction=idx_type)
                for prop in (spec.get("properties") or [])
            ],
            unique=spec.get("owningConstraint") is not None,
            tables=labels_or_types,
            index_type=idx_type.lower() if idx_type else None,
            ddl=spec.get("createStatement"),
        )

    async def _describe_node_entity(self, label: str) -> EntityDescription:
        return EntityDescription(
            name=label,
            kind="node",
            properties=await self._node_field_descriptions(label),
            connections=await self._node_connections(label),
        )

    async def _node_field_descriptions(self, label: str) -> list[FieldDescription]:
        properties = await self._node_type_properties(label)
        samples = await self._node_samples(label)
        return [
            FieldDescription(
                name=name,
                types=types,
                nullable=not mandatory,
                sample=samples.get(name, []),
            )
            for name, types, mandatory in properties
        ]

    async def _describe_node_field(
        self, label: str, prop: str
    ) -> FieldDescription | None:
        match = next(
            (p for p in await self._node_type_properties(label) if p[0] == prop),
            None,
        )
        if match is None:
            return None
        name, types, mandatory = match
        samples = await self._node_samples(label)
        return FieldDescription(
            name=name, types=types, nullable=not mandatory, sample=samples.get(name, [])
        )

    async def _describe_relationship_entity(self, rel_type: str) -> EntityDescription:
        return EntityDescription(
            name=rel_type,
            kind="relationship",
            properties=await self._relationship_field_descriptions(rel_type),
            connections=await self._relationship_type_connections(rel_type),
        )

    async def _relationship_field_descriptions(
        self, rel_type: str
    ) -> list[FieldDescription]:
        properties = await self._relationship_type_properties(rel_type)
        samples = await self._relationship_samples(rel_type)
        return [
            FieldDescription(
                name=name,
                types=types,
                nullable=not mandatory,
                sample=samples.get(name, []),
            )
            for name, types, mandatory in properties
        ]

    async def _describe_relationship_field(
        self, rel_type: str, prop: str
    ) -> FieldDescription | None:
        match = next(
            (
                p
                for p in await self._relationship_type_properties(rel_type)
                if p[0] == prop
            ),
            None,
        )
        if match is None:
            return None
        name, types, mandatory = match
        samples = await self._relationship_samples(rel_type)
        return FieldDescription(
            name=name, types=types, nullable=not mandatory, sample=samples.get(name, [])
        )

    async def _all_index_specs(self) -> list[dict]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                "SHOW INDEXES YIELD name, type, properties, labelsOrTypes,"
                " owningConstraint, createStatement"
                " RETURN name, type, properties, labelsOrTypes,"
                " owningConstraint, createStatement",
            )
            return await result.data()

    async def _query_column(self, query: LiteralString, key: str) -> list[str]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(session, query)
            return [r[key] for r in await result.data()]

    async def _node_properties(self, label: str) -> list[str]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH (n:`{label}`) UNWIND keys(n) AS prop"
                " RETURN DISTINCT prop ORDER BY prop",
            )
            return [r["prop"] for r in await result.data()]

    async def _relationship_properties(self, rel_type: str) -> list[str]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH ()-[r:`{rel_type}`]->() UNWIND keys(r) AS prop"
                " RETURN DISTINCT prop ORDER BY prop",
            )
            return [r["prop"] for r in await result.data()]

    async def _node_type_properties(
        self, label: str
    ) -> list[tuple[str, list[str], bool]]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                "CALL db.schema.nodeTypeProperties()"
                " YIELD nodeLabels, propertyName, propertyTypes, mandatory"
                " WHERE $label IN nodeLabels AND propertyName IS NOT NULL"
                " RETURN propertyName, propertyTypes, mandatory",
                {"label": label},
            )
            return _aggregate_properties(await result.data())

    async def _relationship_type_properties(
        self, rel_type: str
    ) -> list[tuple[str, list[str], bool]]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                "CALL db.schema.relTypeProperties()"
                " YIELD relType, propertyName, propertyTypes, mandatory"
                " WHERE propertyName IS NOT NULL"
                " RETURN relType, propertyName, propertyTypes, mandatory",
            )
            rows = [
                r
                for r in await result.data()
                if _strip_rel_type(r["relType"]) == rel_type
            ]
            return _aggregate_properties(rows)

    async def _node_connections(self, label: str) -> list[Connection]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH (n:`{label}`)-[r]-(m)"
                " RETURN DISTINCT type(r) AS relType,"
                " labels(startNode(r)) AS fromLabels, labels(endNode(r)) AS toLabels",
            )
            rows = await result.data()
        seen: set[tuple[str, str, str]] = set()
        connections: list[Connection] = []
        for row in rows:
            for from_label in row["fromLabels"]:
                for to_label in row["toLabels"]:
                    key = (row["relType"], from_label, to_label)
                    if key not in seen:
                        seen.add(key)
                        connections.append(
                            Connection(
                                rel_type=row["relType"],
                                from_label=from_label,
                                to_label=to_label,
                            )
                        )
        connections.sort(key=lambda c: (c.rel_type, c.from_label, c.to_label))
        return connections

    async def _relationship_type_connections(self, rel_type: str) -> list[Connection]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH (a)-[r:`{rel_type}`]->(b)"
                " RETURN DISTINCT labels(a) AS fromLabels, labels(b) AS toLabels",
            )
            rows = await result.data()
        seen: set[tuple[str, str]] = set()
        connections: list[Connection] = []
        for row in rows:
            for from_label in row["fromLabels"]:
                for to_label in row["toLabels"]:
                    key = (from_label, to_label)
                    if key not in seen:
                        seen.add(key)
                        connections.append(
                            Connection(
                                rel_type=rel_type,
                                from_label=from_label,
                                to_label=to_label,
                            )
                        )
        connections.sort(key=lambda c: (c.from_label, c.to_label))
        return connections

    async def _node_samples(self, label: str) -> dict[str, list[Any]]:
        try:
            return await asyncio.wait_for(
                self._fetch_node_samples(label),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return {}

    async def _fetch_node_samples(self, label: str) -> dict[str, list[Any]]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH (n:`{label}`) RETURN n LIMIT $limit",
                {"limit": SAMPLE_SCAN_ROWS},
            )
            nodes = [dict(record["n"]) async for record in result]
        columns = sorted({k for n in nodes for k in n})
        rows = [tuple(n.get(c) for c in columns) for n in nodes]
        return build_column_samples(columns, rows, self._settings.column_sample_size)

    async def _relationship_samples(self, rel_type: str) -> dict[str, list[Any]]:
        try:
            return await asyncio.wait_for(
                self._fetch_relationship_samples(rel_type),
                timeout=self._settings.column_sample_timeout,
            )
        except asyncio.TimeoutError:
            return {}

    async def _fetch_relationship_samples(self, rel_type: str) -> dict[str, list[Any]]:
        db = self.params.get("database", "neo4j")
        async with self._driver.session(database=db) as session:
            result = await _cypher(
                session,
                f"MATCH ()-[r:`{rel_type}`]->() RETURN r LIMIT $limit",
                {"limit": SAMPLE_SCAN_ROWS},
            )
            rels = [dict(record["r"]) async for record in result]
        columns = sorted({k for r in rels for k in r})
        rows = [tuple(r.get(c) for c in columns) for r in rels]
        return build_column_samples(columns, rows, self._settings.column_sample_size)


def _aggregate_properties(rows: list[dict]) -> list[tuple[str, list[str], bool]]:
    """Group db.schema.*Properties() rows by propertyName, merging types across
    the label/rel-type combinations that carry it and requiring *mandatory*
    in every combination for the aggregated property to count as mandatory."""
    by_name: dict[str, list[tuple[list[str], bool]]] = {}
    for row in rows:
        name = row.get("propertyName")
        if name is None:
            continue
        by_name.setdefault(name, []).append(
            (row.get("propertyTypes") or [], bool(row.get("mandatory")))
        )
    return [
        (
            name,
            sorted({t for types, _ in entries for t in types}),
            all(m for _, m in entries),
        )
        for name, entries in sorted(by_name.items())
    ]


def _strip_rel_type(rel_type: str) -> str:
    """Convert db.schema.relTypeProperties()'s ``":`TYPE`"`` format to ``"TYPE"``."""
    return rel_type.removeprefix(":").strip("`")


def _serialize(
    register_lob: Callable[[bytes | str, str], LobPlaceholder], value: Any
) -> Any:
    """Recursively convert neo4j graph objects to plain Python values."""
    if isinstance(value, neo4j.graph.Node):
        return {"_labels": sorted(value.labels), **dict(value)}
    if isinstance(value, neo4j.graph.Relationship):
        return {"_type": value.type, **dict(value)}
    if isinstance(value, neo4j.graph.Path):
        return [_serialize(register_lob, n) for n in value.nodes]
    if isinstance(value, list):
        return [_serialize(register_lob, v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(register_lob, v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return register_lob(bytes(value), f"ByteArray ({len(value)} bytes)")
    return value


def _plan_keyword(query: str) -> str | None:
    """Returns 'explain' or 'profile' if the query's first real keyword is EXPLAIN/PROFILE."""
    words = blank_comments(query, line=("//",), block=True, quotes="'\"`").split(
        None, 1
    )
    keyword = words[0].lower() if words else ""
    return keyword if keyword in ("explain", "profile") else None


def _plan_to_result(root: dict, is_profile: bool) -> ReadResult:
    rows: list[list[Any]] = []
    _collect_plan_rows(root, rows, depth=0, is_profile=is_profile)
    columns = (
        ["operator", "rows", "db_hits", "identifiers"]
        if is_profile
        else ["operator", "estimated_rows", "identifiers"]
    )
    return ReadResult(columns=columns, rows=rows, rows_total=len(rows))


def _collect_plan_rows(
    plan: dict, out: list[list[Any]], depth: int, is_profile: bool
) -> None:
    op = "  " * depth + plan.get("operatorType", "?")
    identifiers = ", ".join(plan.get("identifiers", []))
    if is_profile:
        out.append([op, plan.get("rows", 0), plan.get("dbHits", 0), identifiers])
    else:
        estimated = plan.get("args", {}).get("EstimatedRows", "")
        if isinstance(estimated, float) and estimated == int(estimated):
            estimated = int(estimated)
        out.append([op, estimated, identifiers])
    for child in plan.get("children", []):
        _collect_plan_rows(child, out, depth + 1, is_profile)


def _maybe_raise_connection_lost(exc: Exception) -> None:
    if isinstance(
        exc, (neo4j.exceptions.ServiceUnavailable, neo4j.exceptions.SessionExpired)
    ):
        raise ConnectionLostError(str(exc)) from exc
    # The idle timer closes the driver out-of-band; the client surfaces the
    # next use as a plain DriverError rather than a network error.
    if isinstance(exc, neo4j.exceptions.DriverError) and "closed" in str(exc).lower():
        raise ConnectionLostError(str(exc)) from exc


async def _make_neo4j_driver(params: dict[str, Any]) -> neo4j.AsyncDriver:
    auth = (params.get("user", "neo4j"), params.get("password", ""))
    driver = neo4j.AsyncGraphDatabase.driver(params["uri"], auth=auth)
    try:
        await driver.verify_connectivity()
    except Exception as exc:
        await driver.close()
        raise DriverError(str(exc)) from exc
    return driver
