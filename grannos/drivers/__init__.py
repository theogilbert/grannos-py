import importlib
import importlib.util
import logging
from dataclasses import dataclass

from ..protocol import Driver
from .base import BaseDriver, DriverMeta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredDriver:
    module: str
    class_name: str
    meta_class: str


_REGISTRY: list[RegisteredDriver] = [
    # SQL drivers — ordered by pick priority for the "sql" filetype.
    RegisteredDriver(
        module="oracle", class_name="OracleDriver", meta_class="OracleMeta"
    ),
    RegisteredDriver(
        module="postgres", class_name="PostgresDriver", meta_class="PostgresMeta"
    ),
    RegisteredDriver(
        module="sqlserver", class_name="SQLServerDriver", meta_class="SQLServerMeta"
    ),
    RegisteredDriver(
        module="sqlite", class_name="SQLiteDriver", meta_class="SQLiteMeta"
    ),
    RegisteredDriver(
        module="duckdb", class_name="DuckDBDriver", meta_class="DuckDBMeta"
    ),
    # Cypher driver.
    RegisteredDriver(module="neo4j", class_name="Neo4jDriver", meta_class="Neo4jMeta"),
    # PromQL driver.
    RegisteredDriver(
        module="prometheus", class_name="PrometheusDriver", meta_class="PrometheusMeta"
    ),
    # Generic drivers (no filetype affinity) — appear first for unknown filetypes.
    RegisteredDriver(
        module="mongodb", class_name="MongoDriver", meta_class="MongoMeta"
    ),
    RegisteredDriver(
        module="elasticsearch",
        class_name="ElasticsearchDriver",
        meta_class="ElasticsearchMeta",
    ),
    RegisteredDriver(module="s3", class_name="S3Driver", meta_class="S3Meta"),
]


def get_driver_help(name: str) -> str:
    """Return the HELP string for a named driver without importing its optional package."""
    return _load_meta(_find(name)).HELP


def get_driver(name: str) -> type[BaseDriver]:
    try:
        return _load_class(_find(name))
    except ImportError as exc:
        raise ValueError(str(exc)) from exc


def list_drivers() -> list[Driver]:
    """Return capabilities for every driver available in the current environment.

    Reads each driver's :class:`DriverMeta` and checks its packages are installed
    without importing them, so this answers in milliseconds.
    """
    result = []
    for entry in _REGISTRY:
        meta = _load_meta(entry)
        missing = [p for p in meta.REQUIRES if importlib.util.find_spec(p) is None]
        if missing:
            logger.info(
                "Driver %r unavailable: %s not installed",
                entry.module,
                ", ".join(missing),
            )
            continue
        result.append(
            Driver(
                driver=entry.module,
                label=meta.LABEL,
                params=meta.PARAMS,
                session_params=meta.SESSION_PARAMS,
                supports_writes=meta.SUPPORTS_WRITES,
                languages=meta.LANGUAGES,
                supports_histogram=meta.SUPPORTS_HISTOGRAM,
            )
        )
    return result


def _find(name: str) -> RegisteredDriver:
    for entry in _REGISTRY:
        if entry.module == name.lower():
            return entry
    raise ValueError(f"Unknown driver: {name!r}")


def _load_class(entry: RegisteredDriver) -> type[BaseDriver]:
    mod = importlib.import_module(f".{entry.module}.driver", package=__package__)
    return getattr(mod, entry.class_name)  # type: ignore[return-value]


def _load_meta(entry: RegisteredDriver) -> type[DriverMeta]:
    mod = importlib.import_module(f".{entry.module}.meta", package=__package__)
    return getattr(mod, entry.meta_class)  # type: ignore[return-value]


SENSITIVE_PARAM_KEYS: frozenset[str] = frozenset(
    p.key for e in _REGISTRY for p in _load_meta(e).PARAMS if p.secret
)
"""Every driver's secret param keys, installed or not — masked in logs and cache keys."""
