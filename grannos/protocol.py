"""
Wire format: newline-delimited JSON (one message per line).

Request  (client → server): {id: int, method: str, params: dict}
Response (server → client): {id: int, result: any, error: str|None}
Progress (server → client): {id: int, progress: {status: str, message: str}}
"""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = "1.1"
"""Wire-protocol version this server implements, as ``"<major>.<minor>"``.

Bump ``major`` for changes that break existing clients (removed/renamed
fields, changed method semantics); bump ``minor`` for additive,
backward-compatible changes (new optional fields, new methods). Reported to
clients via ``capabilities`` — clients only need to check ``major`` for
compatibility, since a minor bump is guaranteed not to break them.
"""


class Method(StrEnum):
    """Supported request methods."""

    CAPABILITIES = "capabilities"
    DRIVER_HELP = "driver.help"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    EXECUTE = "execute"
    EXECUTE_HISTOGRAM = "execute.histogram"
    CANCEL = "cancel"
    EXPLORE_LIST = "explore.list"
    EXPLORE_FIND = "explore.find"
    EXPLORE_DESCRIBE = "explore.describe"
    EXPLORE_PREVIEW = "explore.preview"
    EXPLORE_DIAGRAM = "explore.diagram"
    EXPLORE_DOWNLOAD = "explore.download"
    SESSION_SET = "session.set"
    SESSION_GET = "session.get"


@dataclass
class Request:
    """Incoming request from the client."""

    id: int
    """Caller-chosen identifier echoed in the response."""
    method: Method
    """Method name (e.g. ``"execute"``, ``"connect"``)."""
    params: dict[str, Any]
    """Method-specific parameters."""


@dataclass
class Result:
    """Final response sent to the client."""

    id: int | None
    """Matches the originating request id; None for parse errors."""
    result: Any
    """Return value on success; None on error."""
    error: str | None
    """Error message on failure; None on success."""


@dataclass
class ProgressDetail:
    """Status update payload within a progress notification."""

    status: str
    """Machine-readable status key (e.g. ``"reconnecting"``)."""
    message: str
    """Human-readable description of the current step."""


@dataclass
class Progress:
    """Mid-request progress notification sent before the final result."""

    id: int
    """Matches the originating request id."""
    progress: ProgressDetail
    """Status update payload."""


class NodeType(StrEnum):
    """Canonical kinds of node in the object tree, shared by explore.list's
    ``type`` field, explore.find's ``type`` param, and drivers' ``FIND_PATHS``.

    Deliberately a flat vocabulary across all drivers rather than one enum per
    driver: a client picks an icon, or names a symbol's kind, without knowing
    which backend it is talking to. A given driver uses only the subset its tree
    actually contains.

    A ``StrEnum`` so a member is usable anywhere the raw wire string is, and
    compares equal to it — drivers and clients that pass plain strings keep
    working.
    """

    GROUP = "group"
    """Organisational node bundling sub-categories (``columns``, ``indexes``, …).
    Not a database object, and never an explore.find result."""

    SCHEMA = "schema"
    DATABASE = "database"
    TABLE = "table"
    VIEW = "view"
    COLLECTION = "collection"
    LABEL = "label"
    """Graph node label (Neo4j), or a metric label (Prometheus)."""
    LABEL_VALUE = "label_value"
    """One value of a metric label (Prometheus)."""
    RELATIONSHIP_TYPE = "relationship_type"
    """Graph relationship type (Neo4j)."""

    COLUMN = "column"
    FIELD = "field"
    """Field of a schemaless document or mapping (MongoDB, Elasticsearch)."""
    PROPERTY = "property"
    """Property of a graph node or relationship (Neo4j)."""

    INDEX = "index"
    """A SQL/Mongo/Neo4j index — or, for Elasticsearch, an index in its own
    entity-shaped sense."""
    FOREIGN_KEY = "foreign_key"
    ALIAS = "alias"

    GRIDFS_BUCKET = "gridfs_bucket"
    BUCKET = "bucket"
    """S3 bucket."""
    PREFIX = "prefix"
    """S3 key prefix, i.e. a directory-like level."""
    OBJECT = "object"
    """S3 object."""

    METRIC = "metric"
    JOB = "job"
    CONFIGURATION = "configuration"
    SETTINGS = "settings"


@dataclass
class ExploreItem:
    """A single node in the database object tree returned by explore.list."""

    name: str
    """Display name of the node."""
    type: str
    """Node kind — a :class:`NodeType` for structural nodes.

    Not typed as ``NodeType`` because drivers currently overload this field on
    leaf field nodes, reporting the field's *data* type there instead (e.g.
    ``"int4"``, ``"varchar2"``) so a tree client can show it next to the name.
    """
    expandable: bool
    """Whether the node has children that can be listed."""


@dataclass
class SearchScope:
    """One restriction narrowing an explore.find search to a region of the tree.

    A scope names an *ancestor* the matched node must sit under. Scopes sharing
    a ``type`` are alternatives (the node may sit under any one of them); scopes
    of different types all have to hold. So searching a column with scopes
    ``[(users, table), (orders, table), (public, schema)]`` looks for the column
    under ``public.users`` or ``public.orders``.

    A scope whose ``type`` names no level of the driver's tree is ignored rather
    than failing the search — clients infer scopes from source text and cannot
    know each driver's vocabulary.
    """

    name: str
    """Display name of the ancestor node (matched case-insensitively)."""
    type: str
    """Ancestor's node kind, normally a :class:`NodeType`. Plain ``str`` because
    it arrives from a client, which may name a kind this driver does not have."""


@dataclass
class TableReference:
    """One foreign key, read as ``table.column -> ref_table.ref_column``: ``table``/
    ``column`` always name the side that owns the FK constraint, ``ref_table``/
    ``ref_column`` always name the side it points at — regardless of which
    direction this instance was reached from.

    Self-contained: identifies both sides explicitly, so it can be returned
    either standalone as an explore.describe result for a path ending in
    ``["relationships", column]`` (e.g. as emitted by explore.diagram's
    ``regions``), or embedded in a :class:`FieldDescription`'s
    ``outgoing_references`` (where ``table``/``schema`` restate the embedding
    field's own entity, since it owns the FK) or ``incoming_references``
    (where ``ref_table``/``ref_schema`` restate the embedding field's own
    entity instead, since some other table owns the FK there)."""

    table: str
    """Name of the table that owns the FK constraint."""
    column: str
    """The owning table's own FK column."""
    ref_table: str
    """Name of the referenced table."""
    ref_column: str
    """Column on the referenced table."""
    schema: str | None = None
    """Schema of the owning table, or None for databases without schema support."""
    ref_schema: str | None = None
    """Schema of the referenced table, or None for databases without schema support."""
    unique: bool = False
    """Whether ``column`` is itself constrained to unique values on ``table`` (by a
    PK or a single-column UNIQUE index), making the relationship one-to-one
    rather than many-to-one."""
    constraint_name: str | None = None
    """Name of the FK constraint, or None if unnamed/unsupported by the database."""
    type: str = "relationship"
    """Discriminator — always ``"relationship"``."""


@dataclass
class Connection:
    """One observed (relationship type, start label, end label) triple for a graph
    database, embedded in :class:`EntityDescription`'s ``connections``. Unlike
    :class:`TableReference`, a graph relationship isn't anchored to any field —
    it's a free-floating typed edge between node instances — so this has no
    per-field home and is never independently describable on its own path."""

    rel_type: str
    """Relationship type name."""
    from_label: str
    """Label of the relationship's start node."""
    to_label: str
    """Label of the relationship's end node."""


@dataclass
class IndexKeyField:
    """One field in an index key."""

    name: str
    """Field name."""
    direction: str
    """Sort direction or index kind (``"asc"``, ``"desc"``, ``"text"``, ``"hashed"``, …)."""


@dataclass
class IndexDescription:
    """Key field metadata for an index returned by explore.describe."""

    name: str
    """Index name."""
    fields: list[IndexKeyField]
    """Ordered list of key fields."""
    unique: bool = False
    """Whether the index enforces uniqueness."""
    tables: list[str] = field(default_factory=list)
    """Tables (or labels/collections) the index operates on.
    Typically one entry; multiple for Oracle cluster indexes and SQL Server indexed views."""
    index_type: str | None = None
    """Storage type as reported by the driver (e.g. ``"btree"``, ``"hash"``, ``"bitmap"``); None if unknown."""
    clustered: bool = False
    """Whether the index defines the physical row order of the table."""
    visible: bool = True
    """Whether the query optimiser considers this index; False for Oracle INVISIBLE or SQL Server DISABLED."""
    included_columns: list[str] = field(default_factory=list)
    """Non-key columns stored in index leaf pages for covering queries (PostgreSQL / SQL Server INCLUDE)."""
    ddl: str | None = None
    """CREATE INDEX statement as stored by the database; None when the driver cannot produce it."""
    type: str = "index"
    """Discriminator — always ``"index"``."""


@dataclass
class FieldDescription:
    """Full metadata for a single field (column, property, …) returned by
    explore.describe — either embedded in an :class:`EntityDescription`'s
    ``properties``, or fetched standalone for a path ending in
    ``["<fields-group>", name]``. One shape for both; no lighter variant."""

    name: str
    """Field name."""
    types: list[str]
    """Data type(s) as reported by the database. Single-element for SQL columns;
    schemaless stores (e.g. Neo4j properties) may report more than one when
    the same key holds different types across instances."""
    nullable: bool | None = None
    """Whether the field allows a missing/NULL value; None if unknown."""
    pk: bool = False
    """Whether the field is part of the primary key. Always False where not applicable."""
    default: str | None = None
    """Default expression, or None if not set/not applicable."""
    exclusive_indices: list[IndexDescription] = field(default_factory=list)
    """Indices that cover only this field."""
    composite_indices: list[IndexDescription] = field(default_factory=list)
    """Indices that cover this field and at least one other field."""
    comment: str | None = None
    """Field comment as stored in the database; None if unsupported or not set."""
    sample: list[Any] = field(default_factory=list)
    """Up to 3 distinct non-null representative values sampled from the field."""
    outgoing_references: list[TableReference] = field(default_factory=list)
    """Foreign keys defined on this field that reference another entity. Empty if this
    field is not a foreign key. A field can carry more than one entry — either because
    it participates in more than one single-column FK constraint (each naming a different
    target), or because it is one leg of multiple composite FK constraints."""
    incoming_references: list[TableReference] = field(default_factory=list)
    """Foreign keys on other entities that reference this field. Empty if nothing
    references this field."""
    type: str = "field"
    """Discriminator — always ``"field"``."""


@dataclass
class EntityDescription:
    """Full metadata for a table, node label, relationship type, or document
    collection returned by explore.describe."""

    name: str
    """Entity name."""
    kind: str
    """Domain-specific classification (e.g. ``"table"``, ``"view"``, ``"node"``,
    ``"relationship"``, ``"document"``), for clients that want a domain-appropriate
    icon/label. Not a wire discriminator — use ``type`` for that."""
    properties: list[FieldDescription]
    """Full metadata for every field on this entity."""
    schema: str | None = None
    """Schema name, or None for databases without schema support."""
    comment: str | None = None
    """Entity comment as stored in the database; None if unsupported or not set."""
    connections: list[Connection] = field(default_factory=list)
    """Graph databases only: relationship types touching this entity and the
    label(s) they connect to/from. Empty for non-graph entities."""
    type: str = "entity"
    """Discriminator — always ``"entity"``."""


@dataclass
class RawDocument:
    """A node whose natural representation is a single opaque text document
    rather than tabular metadata — e.g. a driver's running configuration file.
    ``filetype`` lets the client pick a syntax highlighter; it is a free-form
    hint (``"yaml"``, ``"json"``, ``"ini"``, …), not a fixed enum, since drivers
    surface whatever format their backend natively produces."""

    filetype: str
    """Content's format, as a lowercase language/filetype hint (e.g. ``"yaml"``)."""
    content: str
    """The document's full text content."""
    type: str = "document"
    """Discriminator — always ``"document"``."""


@dataclass
class RecordField:
    """One label/value pair in a :class:`GenericRecordDescription`'s ``fields``."""

    label: str
    """Display label (e.g. ``"Scrape URL"``)."""
    value: str
    """Display value, pre-formatted by the driver."""


@dataclass
class GenericRecordDescription:
    """Escape hatch for driver-specific detail views whose natural shape is a
    flat label/value list that doesn't fit entity/field/index/relationship/
    document — e.g. a Prometheus scrape target. Returned as ``details`` either
    standalone, or as an element of the bare array for a group node. Unlike the
    other describe shapes, this makes no wire-shape guarantee beyond label/value
    pairs, so introducing a new kind of record never requires a protocol version
    bump."""

    kind: str
    """Namespaced, driver-owned label identifying what this record represents
    (e.g. ``"prometheus.target"``). Clients may key an optional dedicated
    renderer off this; unrecognized kinds should fall back to a generic
    label/value rendering of ``fields``."""
    name: str
    """Display name for this record (e.g. the target's instance label)."""
    fields: list[RecordField]
    """Ordered list of label/value pairs."""
    type: str = "generic_record"
    """Discriminator — always ``"generic_record"``."""


DescribeResult = (
    EntityDescription
    | FieldDescription
    | IndexDescription
    | TableReference
    | RawDocument
    | GenericRecordDescription
    | list[IndexDescription]
    | list[FieldDescription]
    | list[GenericRecordDescription]
    | None
)
"""Return type of ``explore_describe`` across all drivers. A path resolving to a
group of items (e.g. an indices group node, or Neo4j's per-entity properties
group node) returns a bare array of the singular type rather than a wrapper
object."""


@dataclass
class DownloadResult:
    """Content of a node fetched via explore.download, for a client to either load
    into a buffer or have written straight to a local file. Not cached — unlike
    explore.list/describe, a download always re-fetches from the driver.

    Exactly one of ``content_base64``/``written_to`` is set, depending on
    whether the request carried a ``dest_path``: with one, the driver writes
    bytes directly to that local path server-side (the backend is always a
    local subprocess, so this needs no network hop) and reports back
    ``written_to`` instead of inlining content into the response; without one,
    ``content_base64`` carries the full content for the client to decode."""

    filename: str
    """Suggested filename, e.g. the S3 object's key basename."""
    content_type: str
    """MIME type as reported by the driver, e.g. ``"text/plain"``, ``"application/octet-stream"``."""
    size: int
    """Size of the content in bytes."""
    content_base64: str | None = None
    """Full content, base64-encoded (content may be binary). Set when the request had no ``dest_path``."""
    written_to: str | None = None
    """Local path the driver wrote the content to. Set when the request had a ``dest_path``."""


@dataclass
class LobPlaceholder:
    """Stands in for a large object cell value a driver did not inline into a row.

    Tagging the value with an object — rather than a formatted string — lets
    clients distinguish it from a real string cell without pattern-matching
    cell contents.
    """

    text: str
    """Server-formatted placeholder text to display, e.g. ``"CLOB (3423 chars)"``."""
    ref: str | None = None
    """Opaque token a client can pass as explore.download's ``ref`` param to fetch
    this specific cell's full content later. None when the driver didn't cache the
    value for re-fetching (e.g. Oracle's schema-browsing sample previews, which skip
    an eager read of every sampled LOB to keep those cheap) — the cell just stays
    inert in that case."""
    type: str = "lob"
    """Discriminator — always ``"lob"``."""


@dataclass
class SpecialFloat:
    """Stands in for a non-finite float value (NaN, +Inf, -Inf) inside a ``rows`` cell.

    Plain JSON has no way to represent these (``json.dumps`` would emit the
    non-standard ``NaN``/``Infinity``/``-Infinity`` tokens, which strict decoders
    reject). Tagging the value with an object — rather than a bare string like
    ``"NaN"`` — also lets clients distinguish it from a real string cell without
    pattern-matching cell contents.
    """

    text: str
    """Display text, e.g. ``"NaN"``, ``"+Inf"``, ``"-Inf"``."""
    type: str = "special_float"
    """Discriminator — always ``"special_float"``."""


@dataclass
class DiagramRegion:
    """One span in the ``diagram`` string returned by explore.diagram that names a
    table, column, or relationship, letting a client resolve a cursor position to
    an explore.describe path without parsing the diagram text itself.

    A relationship (``kind="edge"``) is typically covered by several regions —
    one per row its connector line touches — all sharing the same ``path``.

    A table (``kind="table"``) is likewise typically covered by several regions:
    its box's top and bottom border rows in full, and — on each interior row —
    just the left and right border characters (never the whole row, so these
    never overlap that row's ``kind="column"`` region). All share the table's
    ``path``.

    A box whose columns don't all fit ends its body with a ``...`` row, emitted
    as a single ``kind="columns"`` region pointing at the columns group path.
    """

    row: int
    """0-indexed line number within ``diagram`` (lines split on ``\\n``)."""
    col_start: int
    """0-indexed byte offset (not codepoints) where the span starts."""
    col_end: int
    """0-indexed byte offset where the span ends (exclusive)."""
    kind: str
    """``"table"``, ``"column"``, ``"columns"``, or ``"edge"`` — discriminates what
    ``path`` names. ``"columns"`` marks the ``...`` row standing in for the columns
    a box has no room to list: its ``path`` is the columns *group* path, not a leaf
    column's. Since a table may itself have a column named ``columns``, the path
    alone cannot tell the two apart — only this field can."""
    path: list[str]
    """Path to pass as explore.describe's ``path`` param to describe this table,
    column, column list, or relationship."""


class MessageLevel(StrEnum):
    """Severity of an :class:`ExecuteMessage`.

    Deliberately has no ``error`` member: a failed request is reported through
    the response's ``error`` field, so a level that sometimes means "the request
    failed" would be ambiguous. A statement that *succeeded* while producing
    something wrong — PL/SQL that compiled with errors, say — is a ``warning``.
    """

    INFO = "info"
    """Output the statement chose to emit (Oracle ``DBMS_OUTPUT``, Postgres
    ``RAISE NOTICE``, SQL Server ``PRINT``)."""
    WARNING = "warning"
    """The statement succeeded, but the server flagged a problem with it."""


@dataclass
class ExecuteMessage:
    """Out-of-band text a statement produced alongside (or instead of) its result.

    Distinct from an error: the statement succeeded. Clients are expected to
    render these near the result, highlighted by ``level``.
    """

    level: MessageLevel
    """``"info"`` or ``"warning"`` — see :class:`MessageLevel`."""
    text: str
    """The message itself, without any position prefix — the position is carried
    in ``line``/``col``, so a client never has to parse it back out."""
    line: int | None = None
    """1-indexed line in the submitted ``query`` this message refers to, or None
    if it refers to no particular position."""
    col: int | None = None
    """1-indexed column within ``line``. Always None when ``line`` is None; may
    be None when the server knows the line but not the column."""


@dataclass
class ReadResult:
    """Result of a read-only query."""

    columns: list[str]
    """Column names in order."""
    rows: list[list[Any]]
    """Each row as a list of values; a value may be a :class:`LobPlaceholder` or a
    :class:`SpecialFloat`."""
    rows_total: int
    """Total number of rows matching the query (may exceed len(rows) when the driver applies a default limit)."""
    messages: list[ExecuteMessage] = field(default_factory=list)
    """Out-of-band messages the statement produced, in emission order."""


@dataclass
class WriteResult:
    """Result of a write query."""

    rows_affected: int
    """Number of rows inserted, updated, or deleted."""
    messages: list[ExecuteMessage] = field(default_factory=list)
    """Out-of-band messages the statement produced, in emission order."""


@dataclass
class HistogramBucket:
    """One bar of a :class:`HistogramResult`."""

    time: int
    """Start of the bucket, as Unix milliseconds (UTC)."""
    count: int
    """Documents whose time field falls in ``[time, time + interval)``."""


@dataclass
class HistogramResult:
    """Result of a driver's :meth:`~grannos.drivers.base.BaseDriver.histogram`.

    Buckets are contiguous and ascending: every interval between the first and
    the last is present, an empty one carrying a count of zero, so a client
    can draw them side by side without knowing the interval.
    """

    field: str
    """The time field the documents were bucketed on."""
    interval: str
    """Width of every bucket, as a short duration such as ``"5m"`` or ``"1d"``;
    empty when the driver could not tell (a single bucket, say)."""
    buckets: list[HistogramBucket]
    """Buckets in ascending time order. Empty when the query matched nothing."""


@dataclass
class DriverParamChoice:
    """A single option within an ``"enum"`` driver parameter."""

    value: str
    """Machine-readable value sent in ``connect.params``."""
    label: str
    """Human-readable display name shown in the UI."""


class Language(StrEnum):
    """Standard query-language identifiers reported by drivers in ``capabilities``.

    These are language-neutral identifiers — mapping them to editor-specific
    concepts (e.g. Vim filetypes) is the client's responsibility.
    """

    SQL = "sql"
    CYPHER = "cypher"
    PROMQL = "promql"
    LUCENE = "lucene"
    """Grannos' Elasticsearch Lucene mode: ``<index> | <query_string>``."""


class ParamType(StrEnum):
    """Represent the possible types a driver parameter can have."""

    STRING = "string"
    INTEGER = "integer"
    ENUM = "enum"


@dataclass
class DriverParam:
    """A single connection parameter announced by a driver."""

    key: str
    """Parameter key sent in ``connect.params``."""
    type: ParamType
    """The type of values accepted for this parameter."""
    label: str
    """Human-readable label for UI display."""
    required: bool = True
    """Whether a non-empty value is required."""
    default: str | int | None = None
    """Default value pre-filled in the UI."""
    choices: list[DriverParamChoice] | None = None
    """Allowed options for ``"enum"`` params."""
    secret: bool = False
    """Mask input in the UI; value is never persisted to disk by this server."""


@dataclass
class Driver:
    """A driver and its connection parameters, as announced by ``capabilities``."""

    driver: str
    """Driver identifier passed as ``driver`` in ``connect.params``."""
    label: str
    """Human-readable display name (e.g. ``"SQLite"``)."""
    params: list[DriverParam]
    """Connection parameters in display order."""
    session_params: list[DriverParam] = field(default_factory=list)
    """Runtime-only settings that can be changed on a live connection via
    ``session.set``/``session.get`` — never persisted to a saved connection and
    never sent as part of ``connect``. Empty for drivers with no such settings."""
    supports_writes: bool = True
    """Whether this driver's query language can express write operations.
    False for a genuinely read-only driver (e.g. Prometheus/PromQL) — clients
    should hide write-related connection settings (e.g. "always allow writes")
    for such drivers, since they'd never apply."""
    languages: list[Language] = field(default_factory=list)
    """Query languages this driver supports, using the standard :class:`Language` enum.

    Clients map these to editor-specific concepts (e.g. Vim filetypes) and may
    use them to prioritise matching drivers in a connection picker.  An empty
    list means the driver has no language affinity and is treated as generic.
    """
    supports_histogram: bool = False
    """Whether this driver answers ``execute.histogram`` — a count of matching
    documents per time bucket for a query. Clients should offer a histogram
    view only for drivers that do."""


# --- Method results -------------------------------------------------------
#
# One dataclass per method, carrying exactly the fields that method puts in a
# response's ``result``. These *are* the wire shape: ``encode`` serialises them
# field-for-field, so renaming a field here changes the protocol.
#
# Distinct from the driver-level types above (:class:`ReadResult`,
# :class:`WriteResult`, …), which are what a driver hands back internally — a
# method result may add fields the driver has no knowledge of, such as the
# server-measured ``duration_ms``.


@dataclass
class CapabilitiesResult:
    """Result of ``capabilities``."""

    server: str
    """Server implementation name — always ``"grannos"``."""
    protocol_version: str
    """Wire-protocol version, as ``"<major>.<minor>"``; see :data:`PROTOCOL_VERSION`."""
    drivers: list[Driver]
    """Every driver whose optional dependencies are installed."""


@dataclass
class DriverHelpResult:
    """Result of ``driver.help``."""

    content: str
    """Driver documentation, in Markdown."""


@dataclass
class ConnectResult:
    """Result of ``connect``."""

    connection_id: str
    """Identifier to pass as ``connection_id`` in subsequent requests."""


@dataclass
class OkResult:
    """Result of a method whose only outcome is success — ``disconnect``,
    ``session.set``. Failure arrives as an ``error`` on the response envelope
    instead, never as ``ok: false``."""

    ok: bool = True


@dataclass
class ExecuteReadResult:
    """Result of ``execute`` for a query that returned rows."""

    columns: list[str]
    """Column names in order."""
    rows: list[list[Any]]
    """Each row as a list of values; a value may be a :class:`LobPlaceholder` or a
    :class:`SpecialFloat`."""
    rows_total: int
    """Total rows matching the query, which may exceed ``len(rows)`` when the
    driver applied a default limit."""
    duration_ms: float
    """Wall-clock execution time in milliseconds, measured server-side."""
    messages: list[ExecuteMessage] = field(default_factory=list)
    """Out-of-band messages the statement produced, in emission order. Empty for
    drivers and statements that produce none."""


@dataclass
class ExecuteWriteResult:
    """Result of ``execute`` for a query that wrote rows."""

    rows_affected: int
    """Number of rows inserted, updated, or deleted."""
    duration_ms: float
    """Wall-clock execution time in milliseconds, measured server-side."""
    messages: list[ExecuteMessage] = field(default_factory=list)
    """Out-of-band messages the statement produced, in emission order. Empty for
    drivers and statements that produce none."""


@dataclass
class ExecuteHistogramResult:
    """Result of ``execute.histogram``."""

    field: str
    """The time field the documents were bucketed on."""
    interval: str
    """Width of every bucket, as a short duration such as ``"5m"``; empty when
    unknown."""
    buckets: list[HistogramBucket]
    """Contiguous buckets in ascending time order — see :class:`HistogramResult`."""
    duration_ms: float
    """Wall-clock execution time in milliseconds, measured server-side."""


@dataclass
class ExploreListResult:
    """Result of ``explore.list``."""

    items: list[ExploreItem]
    """Child nodes of the requested path; empty if it has none."""


@dataclass
class ExploreFindResult:
    """Result of ``explore.find``."""

    paths: list[list[str]]
    """Describe-paths of the nodes matching the search. Empty when the symbol
    resolves to nothing; more than one entry when it is ambiguous — the client
    decides how to report either case."""


@dataclass
class ExploreDescribeResult:
    """Result of ``explore.describe``."""

    details: "DescribeResult"
    """Metadata for the node, or null if the path names nothing describable.
    Discriminate on each object's own ``type`` field."""


@dataclass
class ExplorePreviewResult:
    """Result of ``explore.preview``. Every row field is null for a node type
    that cannot be previewed (i.e. anything but a table, view, collection, or
    GridFS bucket) — which is not an error."""

    columns: list[str] | None
    """Column names in order; null if the node does not support preview."""
    rows: list[list[Any]] | None
    """Up to 10 sample rows; null if the node does not support preview."""
    rows_total: int | None
    """Number of rows returned; null if the node does not support preview."""
    duration_ms: float
    """Wall-clock execution time in milliseconds, measured server-side."""


@dataclass
class ExploreDiagramResult:
    """Result of ``explore.diagram``."""

    diagram: str
    """The rendered diagram, as a multi-line string."""
    regions: list[DiagramRegion]
    """Spans mapping points in ``diagram`` back to describe-paths."""


MethodResult = (
    CapabilitiesResult
    | DriverHelpResult
    | ConnectResult
    | OkResult
    | ExecuteReadResult
    | ExecuteWriteResult
    | ExecuteHistogramResult
    | ExploreListResult
    | ExploreFindResult
    | ExploreDescribeResult
    | ExplorePreviewResult
    | ExploreDiagramResult
    | DownloadResult
    | dict[str, Any]
)
"""What a dispatched method returns, to be carried as a :class:`Result`'s
``result``.

The bare ``dict`` is for ``session.get`` alone: its shape is whatever the
driver's ``SESSION_PARAMS`` declare, so there is no fixed dataclass to give it.
"""


Response = Result | Progress | ExploreItem


ProgressCallback = Callable[[str, str], Awaitable[None]]
"""Async callback invoked to report a status update during a long-running operation.

Args:
    status: Machine-readable status key.
    message: Human-readable description of the current step.
"""


def json_default(obj: Any) -> Any:
    """Convert database values ``json.dumps`` cannot handle natively.

    Pass as the ``default`` argument to ``json.dumps`` wherever protocol
    types (and the sample values they carry) are serialized.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def encode(msg: Response) -> bytes:
    """Serialize a response message to a newline-terminated JSON byte string.

    Args:
        msg: The response to encode.

    Returns:
        UTF-8 encoded JSON line ending with ``\\n``.
    """
    return (
        json.dumps(asdict(msg), separators=(",", ":"), default=json_default) + "\n"
    ).encode()


class DecodeError(Exception):
    """Raised when a raw line cannot be parsed into a valid Request."""

    def __init__(self, message: str, id: int | None = None) -> None:
        super().__init__(message)
        self.id = id


def decode(line: bytes) -> Request:
    """Parse a raw JSON line into a Request.

    Args:
        line: A single newline-terminated JSON byte string.

    Returns:
        The decoded Request.

    Raises:
        DecodeError: If the line is not valid JSON, is not a JSON object,
            has missing or unexpected fields, or has a non-object params value.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise DecodeError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise DecodeError(f"Request must be a JSON object, got {type(data).__name__}")

    req_id = data.get("id")
    if not isinstance(req_id, int):
        raise DecodeError(f"Invalid request ID: {req_id!r}. Must be an integer.")

    try:
        data["method"] = Method(data.pop("method"))
    except (ValueError, KeyError) as e:
        raise DecodeError("Invalid or missing method in request", id=req_id) from e

    try:
        req = Request(**data)
    except TypeError as e:
        raise DecodeError(str(e), id=req_id) from e

    if not isinstance(req.params, dict):
        raise DecodeError(
            f"params must be a JSON object, got {type(req.params).__name__}",
            id=req.id,
        )

    return req
