import asyncio
import base64
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from typing import Any, ClassVar, Self

from ..protocol import (
    DescribeResult,
    DownloadResult,
    DriverParam,
    ExploreItem,
    Language,
    LobPlaceholder,
    NodeType,
    HistogramResult,
    ReadResult,
    SearchScope,
    TableReference,
    WriteResult,
)


def find_reference(refs: list[TableReference], column: str) -> TableReference | None:
    """Find one of a table's own outgoing FK references by local column, for a
    describe path ending in ``["relationships", column]``. Since
    :class:`TableReference` is self-contained (carries both the local and
    referenced side), the match can be returned directly — no reconstruction
    needed.

    Returns:
        None if *column* does not name one of the table's own foreign keys.
    """
    return next((r for r in refs if r.column == column), None)


def group_references_by_column(
    refs: list[TableReference],
) -> dict[str, list[TableReference]]:
    """Group a table's own outgoing FK references by local column, for populating
    ``FieldDescription.outgoing_references``."""
    by_column: dict[str, list[TableReference]] = {}
    for ref in refs:
        by_column.setdefault(ref.column, []).append(ref)
    return by_column


def group_references_by_ref_column(
    refs: list[TableReference],
) -> dict[str, list[TableReference]]:
    """Group FK references that point *at* a table by the local column they
    target (``ref_column``, not ``column`` — the owning side lives on another
    table), for populating ``FieldDescription.incoming_references``."""
    by_ref_column: dict[str, list[TableReference]] = {}
    for ref in refs:
        by_ref_column.setdefault(ref.ref_column, []).append(ref)
    return by_ref_column


@dataclass(frozen=True)
class DriverSettings:
    """Server-level configuration injected into every driver on connect."""

    column_sample_size: int = 3
    """Number of distinct non-null values sampled per column in describe results."""

    column_sample_timeout: float = 5.0
    """Seconds to wait for each column sample query before returning no sample."""


class DriverError(Exception):
    """Raised by drivers for errors that should be surfaced verbatim to the client."""


class ConnectionLostError(Exception):
    """Raised when the database connection is lost and a reconnect should be attempted."""


class FindNotSupported(Exception):
    """Raised by :meth:`BaseDriver.explore_find` to hand a search back to the
    generic tree walker.

    Raised by the base implementation for every search, and by an overriding
    driver for the node types its own catalog lookup does not cover — so an
    override can special-case one kind of node without reimplementing the rest.
    Deliberately not ``NotImplementedError``, which a driver's underlying
    library may raise for unrelated reasons; swallowing that would silently
    convert a bug into a slow tree walk."""


class BaseDriver(ABC):
    """Base class for all database drivers.

    Args:
        params: Raw connect request fields (e.g. ``{"driver": "sqlite", "database": "..."}``).
    """

    LABEL: str = ""
    """Human-readable display name declared by each driver subclass."""

    PARAMS: list[DriverParam] = []
    """Connection parameters declared by each driver subclass."""

    SESSION_PARAMS: list[DriverParam] = []
    """Runtime-only settings declared by a driver subclass, changeable on a live
    connection via :meth:`set_session`/:meth:`get_session` — never sent as part
    of ``connect`` params and never persisted alongside a saved connection.
    Empty for drivers with no such settings."""

    LANGUAGES: ClassVar[list[Language]] = []
    """Query languages this driver supports (see :class:`~grannos.protocol.Language`).

    Drivers with no language affinity leave this empty.
    """

    HELP: str = ""
    """Markdown help text declared by each driver subclass."""

    FIND_PATHS: ClassVar[dict[NodeType, list[list[str]]]] = {}
    """Where each kind of node lives in this driver's object tree, as path
    templates keyed by node type — the declaration backing ``explore.find``.

    A ``"*"`` segment stands for one level of children, any other segment for a
    literal group name; a template's last segment is the node itself. A type may
    list several templates (a Neo4j property lives under both entities and
    relationships), and several types may share one (a SQL table and view sit at
    the same level — nothing at that level says which is which, and a client
    reading a ``FROM`` clause cannot tell either).

    Empty for a driver whose tree has no fixed shape (e.g. S3's arbitrarily
    nested prefixes), which makes it unsearchable. See
    :mod:`grannos.explore_find`.
    """

    DEFAULT_IDLE_TIMEOUT: ClassVar[float] = 600
    """The default idle timeout for this driver.

    Connections idle for longer than the specified time will be automatically closed.
    The value can be set to 0 to disable closing the connection when idle too long.
    """

    SUPPORTS_WRITES: ClassVar[bool] = True
    """Whether this driver's query language can express write operations.

    True for nearly every driver; a genuinely read-only driver (e.g. Prometheus,
    whose PromQL has no write syntax at all) overrides this to False so clients
    can skip write-related connection settings (e.g. "always allow writes")
    that would otherwise never apply.
    """

    SUPPORTS_HISTOGRAM: ClassVar[bool] = False
    """Whether this driver implements :meth:`histogram`. Advertised to clients
    as ``supports_histogram`` so they only offer the view where it works."""

    _LOB_CACHE_MAX: ClassVar[int] = 200
    """Bound on the number of LobPlaceholder values kept in memory for later
    explore.download(ref=...) retrieval; oldest entries are evicted first."""

    def __init__(self, params: dict[str, Any], settings: DriverSettings) -> None:
        self.params = params
        self._settings = settings
        self._lob_cache: dict[str, bytes | str] = {}
        """ref -> raw value, populated by _register_lob."""
        self._lob_cache_order: list[str] = []
        """Insertion order of _lob_cache keys, for FIFO eviction."""

    @classmethod
    @abstractmethod
    async def create(cls, params: dict[str, Any], settings: DriverSettings) -> Self:
        """Open a new connection and return a ready-to-use driver instance.

        Args:
            params: Raw connect request fields.

        Returns:
            A fully connected driver instance.
        """
        ...

    @abstractmethod
    async def reconnect(self) -> None:
        """Re-establish the database connection on the current instance."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run a database query and return the result.

        Args:
            query: Query to execute.
            binds: Positional bind parameters.

        Returns:
            ReadResult for queries that return rows, DMLResult for write operations.

        Raises:
            ConnectionLostError: If the connection was lost during execution.
        """
        ...

    @abstractmethod
    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        """List child nodes at the given path in the object tree.

        Args:
            path: Ordered path segments from the root (e.g. ``["dbo", "users"]``).

        Returns:
            Child nodes, or an empty list if the path is unrecognised.
        """
        ...

    async def explore_find(
        self, node_type: str, name: str, scopes: list[SearchScope]
    ) -> list[list[str]]:
        """Resolve a symbol to the paths of the nodes matching it.

        Not implemented here: the base behaviour is to defer to the generic
        walker over :data:`FIND_PATHS`, which runs one level up (in
        ``CachingDriver``) so its ``explore_list`` calls are served from the
        connection's explore cache. A driver whose catalog can answer a search
        in a single query overrides this and raises :class:`FindNotSupported`
        for whatever node types that query does not cover.

        Args:
            node_type: Kind of node to find (e.g. ``"column"``).
            name: Symbol name as written by the client; match it
                case-insensitively.
            scopes: Ancestor restrictions — see
                :class:`~grannos.protocol.SearchScope`.

        Returns:
            Describe-paths of the matching nodes: empty if the symbol resolves
            to nothing, more than one entry if it is ambiguous.

        Raises:
            FindNotSupported: To hand the search to the generic walker.
        """
        raise FindNotSupported

    async def explore_preview(self, path: list[str]) -> ReadResult | None:
        """Return a sample of rows for the node at the given path.

        Args:
            path: Ordered path segments identifying a table or collection node.

        Returns:
            Up to 10 rows as a ReadResult, or None if the node type does not
            support row preview.
        """
        return None

    @abstractmethod
    async def explore_describe(self, path: list[str]) -> DescribeResult:
        """Return metadata for the node at the given path.

        Contract: when a path yields an ``EntityDescription``, describing the
        child path ``[*path, "columns", field_name]`` must yield the exact
        same ``FieldDescription`` object found in ``EntityDescription.properties``
        — the explore cache pre-populates per-field (and per-relationship)
        entries from the entity result relying on this being exact, not just
        equivalent.

        Args:
            path: Ordered path segments identifying a node.

        Returns:
            Entity/field/index/relationship metadata, or None if the path does
            not resolve to a describable node.
        """
        ...

    def _register_lob(self, value: bytes | str, text: str) -> LobPlaceholder:
        """Build a LobPlaceholder for `value`, caching it under a fresh ref so
        explore_download_ref can retrieve the full value later.

        Only safe for values already fully materialized in memory (e.g. a
        `bytes` column value already fetched from the driver) — never for a
        lazy handle whose later use could fail or crash (e.g. Oracle's LOB
        locator, which crashes the process if read after its cursor closes).
        Drivers with such a handle should keep constructing a plain
        `LobPlaceholder(text=...)` with no ref instead of calling this.

        Args:
            value: The raw cell value (already fetched, not a lazy handle).
            text: Server-formatted placeholder text to display.

        Returns:
            A LobPlaceholder carrying a ref to `value`.
        """
        ref = uuid.uuid4().hex
        self._lob_cache[ref] = value
        self._lob_cache_order.append(ref)
        if len(self._lob_cache_order) > self._LOB_CACHE_MAX:
            oldest = self._lob_cache_order.pop(0)
            self._lob_cache.pop(oldest, None)
        return LobPlaceholder(text=text, ref=ref)

    async def explore_download(
        self, path: list[str], dest_path: str | None
    ) -> DownloadResult:
        """Fetch the full content of the node at the given path (e.g. an S3 object).

        Args:
            path: Ordered path segments identifying a downloadable node.
            dest_path: When given, write content directly to this local path
                instead of returning it inline (the backend is always a local
                subprocess, so this needs no network hop) — the result then
                carries `written_to` instead of `content_base64`.

        Returns:
            The node's full content.

        Raises:
            DriverError: If this driver has no downloadable content, or path
                does not resolve to a downloadable node.
        """
        raise DriverError(f"{type(self).__name__} does not support explore.download")

    async def explore_download_ref(
        self, ref: str, dest_path: str | None
    ) -> DownloadResult:
        """Fetch the full content of a previously-registered LobPlaceholder value.

        Concrete (not per-driver): every driver that registers LOB values via
        `_register_lob` gets this for free.

        Args:
            ref: A `LobPlaceholder.ref` value from an earlier result row.
            dest_path: When given, write content directly to this local path
                instead of returning it inline.

        Returns:
            The cell's full content.

        Raises:
            DriverError: If `ref` is unknown or has been evicted from the
                cache (e.g. the query was re-run, or the cache filled up) —
                the client should re-run the query to get a fresh ref.
        """
        value = self._lob_cache.get(ref)
        if value is None:
            raise DriverError(
                "This value is no longer available for download — re-run the query to fetch it again"
            )
        if isinstance(value, str):
            data = value.encode("utf-8")
            content_type = "text/plain"
            filename = "lob.txt"
        else:
            data = bytes(value)
            content_type = "application/octet-stream"
            filename = "lob.bin"
        return await asyncio.get_running_loop().run_in_executor(
            None, _write_or_encode, data, filename, content_type, dest_path
        )

    async def histogram(self, query: str, buckets: int) -> HistogramResult:
        """Count the documents `query` matches per time bucket.

        The same query string ``execute`` takes, answered with a count per
        bucket of its time field instead of its rows — the shape behind a
        "documents over time" chart. Only drivers with ``SUPPORTS_HISTOGRAM``
        implement it.

        Args:
            query: The query, exactly as it would be passed to :meth:`execute`.
            buckets: Target number of buckets. A driver picks a round interval
                that yields about this many; the actual count may differ.

        Raises:
            DriverError: If this driver has no notion of a time histogram.
        """
        raise DriverError(f"{type(self).__name__} does not support histograms")

    async def set_session(self, values: dict[str, Any]) -> None:
        """Update one or more runtime session settings declared in SESSION_PARAMS.

        Args:
            values: New values for a subset of this driver's SESSION_PARAMS keys.

        Raises:
            DriverError: If this driver declares no SESSION_PARAMS.
        """
        raise DriverError(f"{type(self).__name__} has no session-level settings")

    def get_session(self) -> dict[str, Any]:
        """Return the current values of this driver's SESSION_PARAMS settings.

        Returns:
            A dict keyed by SESSION_PARAMS keys; empty for drivers with none.
        """
        return {}


def _write_or_encode(
    data: bytes, filename: str, content_type: str, dest_path: str | None
) -> DownloadResult:
    """Blocking helper for explore_download/explore_download_ref: write `data`
    to `dest_path` if given, else base64-encode it inline. Run via
    run_in_executor since both branches do blocking work.
    """
    if dest_path is not None:
        with open(dest_path, "wb") as f:
            f.write(data)
        return DownloadResult(
            filename=filename,
            content_type=content_type,
            size=len(data),
            written_to=dest_path,
        )
    return DownloadResult(
        filename=filename,
        content_type=content_type,
        size=len(data),
        content_base64=base64.b64encode(data).decode(),
    )


SAMPLE_SCAN_ROWS = 50
"""Table rows a driver scans in one query to derive per-column samples."""

_SAMPLEABLE_TYPES = (str, int, float, date, time, Decimal)
"""Sample value types the wire protocol can serialise (``date`` covers ``datetime``).
Others (LOB handles, bytes, intervals) are skipped."""


def build_column_samples(
    columns: list[str], rows: list[tuple], n: int
) -> dict[str, list[Any]]:
    """Map each column name to up to *n* distinct non-null values from *rows*.

    Values whose type is not in :data:`_SAMPLEABLE_TYPES` are skipped. Sparse
    or low-cardinality columns may yield fewer than *n* values.
    """
    samples: dict[str, list[Any]] = {name: [] for name in columns}
    for i, name in enumerate(columns):
        seen = samples[name]
        for row in rows:
            value = row[i]
            if value is None or not isinstance(value, _SAMPLEABLE_TYPES):
                continue
            if value not in seen:
                seen.append(value)
                if len(seen) == n:
                    break
    return samples
