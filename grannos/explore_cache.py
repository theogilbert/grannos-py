import hashlib
import json
import logging
import pathlib
from dataclasses import asdict
from typing import Any, Self

from grannos.drivers import SENSITIVE_PARAM_KEYS

from .drivers.base import (
    BaseDriver,
    DriverSettings,
    FindNotSupported,
    HistogramResult,
    ReadResult,
    WriteResult,
)
from .explore_find import walk_find
from .protocol import (
    Connection,
    DescribeResult,
    DownloadResult,
    EntityDescription,
    ExploreItem,
    FieldDescription,
    GenericRecordDescription,
    IndexDescription,
    IndexKeyField,
    RawDocument,
    RecordField,
    SearchScope,
    TableReference,
    json_default,
)

logger = logging.getLogger(__name__)

CachedDescribe = (
    EntityDescription
    | FieldDescription
    | IndexDescription
    | TableReference
    | RawDocument
    | GenericRecordDescription
)
"""Non-None explore.describe results storable in the cache, at a single path.
A group path (e.g. an indices group node, or Neo4j's per-entity properties
group node) stores ``GroupDescribe`` instead — see ``_describe``'s value type."""

GroupDescribe = (
    list[IndexDescription] | list[FieldDescription] | list[GenericRecordDescription]
)
"""Describe result for a group path — a bare array of a single describable type."""


class CachingDriver(BaseDriver):
    """BaseDriver decorator that intercepts explore calls for caching."""

    def __init__(self, inner: BaseDriver, cache: "ConnectionCache") -> None:
        super().__init__({}, DriverSettings())
        self._inner = inner
        self._cache = cache

    @classmethod
    async def create(cls, params: dict[str, Any], settings: DriverSettings) -> Self:
        raise NotImplementedError

    async def reconnect(self) -> None:
        await self._inner.reconnect()

    async def disconnect(self) -> None:
        await self._inner.disconnect()

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        return await self._inner.execute(query, binds)

    async def histogram(self, query: str, buckets: int) -> HistogramResult:
        return await self._inner.histogram(query, buckets)

    async def set_session(self, values: dict[str, Any]) -> None:
        await self._inner.set_session(values)

    def get_session(self) -> dict[str, Any]:
        return self._inner.get_session()

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        items = self._cache.get_list(path)
        if items is None:
            items = await self._inner.explore_list(path)
            self._cache.set_list(path, items)
        else:
            logger.debug(f"explore.list cache hit for path {path}")
        return items

    async def explore_find(
        self, node_type: str, name: str, scopes: list[SearchScope]
    ) -> list[list[str]]:
        """Resolve a symbol, preferring the find cache over any round trip.

        Two passes:

        1. The **find cache** — this exact search, answered before and kept on
           disk. Keyed by the search itself, it serves a repeat whichever pass
           answered it the first time, and survives a restart.
        2. The wrapped driver: its own catalog lookup where it has one, else the
           generic walk over its ``explore_list``.

        The walk runs on the wrapped driver rather than on this class, so a find
        neither reads nor writes the list cache. The two caches answer different
        questions — the list cache holds what a *path* contains, and a walk stops
        at the level it was searching, leaving everything below it unlisted — so
        a tree the user has browsed can neither settle a search nor be completed
        by one. Every answer, from either pass, is cached under the search that
        asked for it.
        """
        cached = self._cache.get_find(node_type, name, scopes)
        if cached is not None:
            logger.debug(f"explore.find cache hit for {name!r}")
            return cached
        try:
            paths = await self._inner.explore_find(node_type, name, scopes)
        except FindNotSupported:
            paths = await walk_find(
                self._inner.explore_list,
                self._inner.FIND_PATHS,
                node_type,
                name,
                scopes,
            )
        self._cache.set_find(node_type, name, scopes, paths)
        return paths

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        return await self._inner.explore_preview(path)

    async def explore_download(
        self, path: list[str], dest_path: str | None
    ) -> DownloadResult:
        return await self._inner.explore_download(path, dest_path)

    async def explore_download_ref(
        self, ref: str, dest_path: str | None
    ) -> DownloadResult:
        return await self._inner.explore_download_ref(ref, dest_path)

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        if self._cache.has_describe(path):
            logger.debug(f"explore.describe cache hit for path {path}")
            return self._cache.get_describe(path)
        desc = await self._inner.explore_describe(path)
        if desc is None:
            return None
        entries: list[tuple[list[str], CachedDescribe | GroupDescribe]] = [(path, desc)]
        if isinstance(desc, EntityDescription):
            entries += [([*path, "columns", f.name], f) for f in desc.properties]
            for f in desc.properties:
                entries += [
                    ([*path, "relationships", ref.column], ref)
                    for ref in f.outgoing_references
                ]
        elif isinstance(desc, list):
            indices = [idx for idx in desc if isinstance(idx, IndexDescription)]
            entries += [([*path, idx.name], idx) for idx in indices]
            fields = [f for f in desc if isinstance(f, FieldDescription)]
            entries += [([*path, f.name], f) for f in fields]
            records = [r for r in desc if isinstance(r, GenericRecordDescription)]
            entries += [([*path, r.name], r) for r in records]
        self._cache.set_describes(entries)
        return desc

    def reset_cache(self, path: list[str]) -> None:
        self._cache.reset(path)


def cache_file(params: dict[str, Any], cache_dir: pathlib.Path) -> pathlib.Path:
    """Return the cache file path derived from the connection params.

    Args:
        params: Raw connection params; sensitive fields are excluded from the hash.
        cache_dir: Directory where cache files are stored.

    Returns:
        Path of the form ``<cache_dir>/<driver>_<sha256[:12]>.json``.
    """
    safe = {k: v for k, v in sorted(params.items()) if k not in SENSITIVE_PARAM_KEYS}
    digest = hashlib.sha256(json.dumps(safe).encode()).hexdigest()[:12]
    driver = params.get("driver", "unknown")
    return cache_dir / f"{driver}_{digest}.json"


def find_key(node_type: str, name: str, scopes: list[SearchScope]) -> str:
    """Return the cache key identifying one explore.find search.

    Scopes are sorted and case-folded: they are a *set* of constraints, and both
    resolvers match a scope name case-insensitively, so neither their order nor
    their casing can change the answer. *name* is kept verbatim — it cannot be
    folded, since an exact-case match outranks an inexact one (``_prefer_exact``).
    """
    folded = sorted([s.type, s.name.casefold()] for s in scopes)
    return json.dumps([node_type, name, folded])


class ConnectionCache:
    """Explore result cache for a single database connection, backed by a JSON file.

    Loads from disk on construction and persists atomically after every write.
    Callers interact only through ``get_*`` / ``set_*`` / ``reset``.

    Args:
        params: Raw connection params used to populate the ``_connection``
            metadata block written to disk.
        path: Path to the JSON cache file.
    """

    def __init__(self, params: dict[str, Any], path: pathlib.Path) -> None:
        self._params = params
        """Raw connection params; sensitive fields are stripped before writing to disk."""
        self._path = path
        """Path to the backing JSON cache file."""
        self._list: dict[tuple[str, ...], list[ExploreItem]] = {}
        """In-memory cache mapping path tuples to their explore.list results."""
        self._describe: dict[tuple[str, ...], CachedDescribe | GroupDescribe] = {}
        """In-memory cache mapping path tuples to their explore.describe results."""
        self._find: dict[str, list[list[str]]] = {}
        """In-memory cache mapping a :func:`find_key` to the paths it resolved to.
        Keyed by search rather than by path, so unlike the other two it cannot be
        evicted per-prefix — :meth:`reset` clears it whole."""
        self._load()

    def reset(self, path: list[str]) -> None:
        """Clear cached results at or below path, then persist or delete the disk file.

        Args:
            path: Path prefix to reset. Entries whose keys start with this prefix
                (including the prefix itself) are removed, along with any ancestor
                list/describe entries (since their children may have changed — an
                ancestor's own describe result, e.g. an EntityDescription, may embed
                a now-stale copy of what we just reset). An empty list resets the
                entire cache. Cached finds are dropped in full whatever the path:
                they are keyed by search, not by path, and a reset invalidates
                them in both directions — a find that resolved *into* the reset
                subtree may now point at nothing, and one that resolved to
                nothing may now match something newly created there.
        """
        prefix = tuple(path)
        n = len(prefix)
        for d in (self._list, self._describe):
            for k in [k for k in d if k[:n] == prefix]:
                del d[k]
        # Evict ancestor entries for all ancestor paths — their children may have
        # changed, or (for _describe) may now embed stale data of their own.
        for i in range(n):
            self._list.pop(prefix[:i], None)
            self._describe.pop(prefix[:i], None)
        self._find.clear()
        if self._list or self._describe:
            self._persist()
        else:
            self._path.unlink(missing_ok=True)

    def get_list(self, path: list[str]) -> list[ExploreItem] | None:
        """Return cached explore.list results for path, or None on a miss.

        Args:
            path: Path segments identifying the node.

        Returns:
            Cached items, or None if the path has not been fetched yet.
        """
        return self._list.get(tuple(path))

    def set_list(self, path: list[str], items: list[ExploreItem]) -> None:
        """Store explore.list results for path and persist to disk.

        Args:
            path: Path segments identifying the node.
            items: Items to cache.
        """
        self._list[tuple(path)] = items
        self._persist()

    def get_find(
        self, node_type: str, name: str, scopes: list[SearchScope]
    ) -> list[list[str]] | None:
        """Return the cached paths for an explore.find search, or None on a miss.

        An empty list is a cached *answer* — the symbol resolved to nothing —
        and is deliberately distinct from None.
        """
        return self._find.get(find_key(node_type, name, scopes))

    def set_find(
        self,
        node_type: str,
        name: str,
        scopes: list[SearchScope],
        paths: list[list[str]],
    ) -> None:
        """Store the paths an explore.find search resolved to and persist to disk."""
        self._find[find_key(node_type, name, scopes)] = paths
        self._persist()

    def has_describe(self, path: list[str]) -> bool:
        """Return True if a non-None explore.describe result for path is cached.

        Args:
            path: Path segments identifying the node.
        """
        return tuple(path) in self._describe

    def get_describe(self, path: list[str]) -> DescribeResult:
        """Return cached explore.describe results for path, or None on a miss."""
        return self._describe.get(tuple(path))

    def set_describe(
        self, path: list[str], desc: CachedDescribe | GroupDescribe
    ) -> None:
        """Store explore.describe results for path and persist to disk.

        Args:
            path: Path segments identifying the node.
            desc: Description to cache.
        """
        self.set_describes([(path, desc)])

    def set_describes(
        self, entries: list[tuple[list[str], CachedDescribe | GroupDescribe]]
    ) -> None:
        """Store several explore.describe results, persisting to disk once.

        Args:
            entries: ``(path, description)`` pairs to cache.
        """
        for path, desc in entries:
            self._describe[tuple(path)] = desc
        self._persist()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for str_path, items in data.get("list", {}).items():
                key = tuple(json.loads(str_path))
                self._list[key] = [ExploreItem(**item) for item in items]
            for str_path, desc in data.get("describe", {}).items():
                key = tuple(json.loads(str_path))
                if desc is None:
                    pass  # legacy: None was cached before; skip it
                elif isinstance(desc, list):
                    first_type = desc[0].get("type") if desc else None
                    if first_type == "field":
                        self._describe[key] = [_deserialize_field(f) for f in desc]
                    elif first_type == "generic_record":
                        self._describe[key] = [_deserialize_record(r) for r in desc]
                    else:
                        self._describe[key] = [_deserialize_index(idx) for idx in desc]
                elif desc.get("type") == "index":
                    self._describe[key] = _deserialize_index(desc)
                elif desc.get("type") == "field":
                    self._describe[key] = _deserialize_field(desc)
                elif desc.get("type") == "relationship":
                    self._describe[key] = _deserialize_reference(desc)
                elif desc.get("type") == "document":
                    self._describe[key] = _deserialize_document(desc)
                elif desc.get("type") == "generic_record":
                    self._describe[key] = _deserialize_record(desc)
                else:
                    self._describe[key] = _deserialize_entity(desc)
            for key_str, paths in data.get("find", {}).items():
                self._find[key_str] = [list(p) for p in paths]
        except Exception:
            logger.warning(f"Discarding unreadable explore cache at {self._path}")
            self._list.clear()
            self._describe.clear()
            self._find.clear()

    def _persist(self) -> None:
        try:
            data: dict[str, Any] = {
                "_connection": {
                    k: v
                    for k, v in self._params.items()
                    if k not in SENSITIVE_PARAM_KEYS
                },
                "list": {
                    json.dumps(list(key)): [asdict(item) for item in items]
                    for key, items in self._list.items()
                },
                "describe": {
                    json.dumps(list(key)): (
                        [asdict(item) for item in desc]  # ty: ignore[invalid-argument-type]
                        if isinstance(desc, list)
                        else asdict(desc)
                    )
                    for key, desc in self._describe.items()
                },
                "find": self._find,
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=json_default))
            tmp.replace(self._path)
        except Exception:
            logger.warning(f"Failed to persist explore cache to {self._path}")


def _deserialize_index(d: dict[str, Any]) -> IndexDescription:
    return IndexDescription(
        name=d["name"],
        fields=[IndexKeyField(**f) for f in d.get("fields", [])],
        unique=d.get("unique", False),
        tables=d.get("tables", []),
        index_type=d.get("index_type"),
        clustered=d.get("clustered", False),
        visible=d.get("visible", True),
        included_columns=d.get("included_columns", []),
        ddl=d.get("ddl"),
    )


def _deserialize_document(d: dict[str, Any]) -> RawDocument:
    return RawDocument(filetype=d["filetype"], content=d["content"])


def _deserialize_record(d: dict[str, Any]) -> GenericRecordDescription:
    return GenericRecordDescription(
        kind=d["kind"],
        name=d["name"],
        fields=[RecordField(**f) for f in d.get("fields", [])],
    )


def _deserialize_reference(d: dict[str, Any]) -> TableReference:
    return TableReference(
        table=d["table"],
        column=d["column"],
        ref_table=d["ref_table"],
        ref_column=d["ref_column"],
        schema=d.get("schema"),
        ref_schema=d.get("ref_schema"),
        unique=d.get("unique", False),
        constraint_name=d.get("constraint_name"),
    )


def _deserialize_field(d: dict[str, Any]) -> FieldDescription:
    return FieldDescription(
        name=d["name"],
        types=d.get("types", []),
        nullable=d.get("nullable"),
        pk=d.get("pk", False),
        default=d.get("default"),
        exclusive_indices=[
            _deserialize_index(i) for i in d.get("exclusive_indices", [])
        ],
        composite_indices=[
            _deserialize_index(i) for i in d.get("composite_indices", [])
        ],
        comment=d.get("comment"),
        sample=d.get("sample", []),
        outgoing_references=[
            _deserialize_reference(r) for r in d.get("outgoing_references", [])
        ],
        incoming_references=[
            _deserialize_reference(r) for r in d.get("incoming_references", [])
        ],
    )


def _deserialize_entity(d: dict[str, Any]) -> EntityDescription:
    return EntityDescription(
        name=d["name"],
        kind=d.get("kind", ""),
        properties=[_deserialize_field(f) for f in d.get("properties", [])],
        schema=d.get("schema"),
        comment=d.get("comment"),
        connections=[Connection(**c) for c in d.get("connections", [])],
    )
