import importlib
import logging
from dataclasses import dataclass

from .base import BaseDriver
from ..protocol import Driver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredDriver:
    module: str
    class_name: str


_REGISTRY: list[RegisteredDriver] = [
    # SQL drivers — ordered by pick priority for the "sql" filetype.
    RegisteredDriver(module="oracle", class_name="OracleDriver"),
    RegisteredDriver(module="postgres", class_name="PostgresDriver"),
    RegisteredDriver(module="sqlserver", class_name="SQLServerDriver"),
    RegisteredDriver(module="sqlite", class_name="SQLiteDriver"),
    RegisteredDriver(module="duckdb", class_name="DuckDBDriver"),
    # Cypher driver.
    RegisteredDriver(module="neo4j", class_name="Neo4jDriver"),
    # PromQL driver.
    RegisteredDriver(module="prometheus", class_name="PrometheusDriver"),
    # Generic drivers (no filetype affinity) — appear first for unknown filetypes.
    RegisteredDriver(module="mongodb", class_name="MongoDriver"),
    RegisteredDriver(module="elasticsearch", class_name="ElasticsearchDriver"),
    RegisteredDriver(module="s3", class_name="S3Driver"),
]


def get_driver_help(name: str) -> str:
    """Return the HELP string for a named driver without importing its optional package."""
    return _load_class(_find(name)).HELP


def get_driver(name: str) -> type[BaseDriver]:
    try:
        return _load_class(_find(name))
    except ImportError as exc:
        raise ValueError(str(exc)) from exc


def list_drivers() -> list[Driver]:
    """Return capabilities for every driver available in the current environment."""
    result = []
    for entry in _REGISTRY:
        try:
            cls = _load_class(entry)
        except ImportError:
            logger.info("Driver %r unavailable: package not installed", entry.module)
            continue
        result.append(
            Driver(
                driver=entry.module,
                label=cls.LABEL,
                params=cls.PARAMS,
                session_params=cls.SESSION_PARAMS,
                supports_writes=cls.SUPPORTS_WRITES,
                languages=cls.LANGUAGES,
                supports_histogram=cls.SUPPORTS_HISTOGRAM,
            )
        )
    return result


def _find(name: str) -> RegisteredDriver:
    for entry in _REGISTRY:
        if entry.module == name.lower():
            return entry
    raise ValueError(f"Unknown driver: {name!r}")


def _load_class(entry: RegisteredDriver) -> type[BaseDriver]:
    mod = importlib.import_module(f".{entry.module}", package=__package__)
    return getattr(mod, entry.class_name)  # type: ignore[return-value]


SENSITIVE_PARAM_KEYS: frozenset[str] = frozenset(
    p.key for d in list_drivers() for p in d.params if p.secret
)
