"""S3 driver — requires: pip install boto3 pyyaml"""

import logging
import asyncio
import base64
import fnmatch
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import boto3
import yaml
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from ..log import log_query
from ..protocol import (
    DescribeResult,
    DownloadResult,
    DriverParam,
    ExploreItem,
    GenericRecordDescription,
    ParamType,
    RawDocument,
    ReadResult,
    RecordField,
    WriteResult,
)
from .base import BaseDriver, ConnectionLostError, DriverError, DriverSettings
from .comments import blank_comments

T = TypeVar("T")

_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/?(?P<key>.*)$")

_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
"""Objects larger than this are refused by explore.download — too large to
usefully load into a Neovim buffer."""

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}


logger = logging.getLogger(__name__)


class S3Driver(BaseDriver):
    """S3 driver backed by boto3 (sync client run in a thread pool).

    Args:
        params: Connect request fields (``access_key_id``, ``secret_access_key``,
            ``region``, ``endpoint``).
        client: Open boto3 S3 client. Use :meth:`create` instead of constructing
            directly.
    """

    LABEL = "S3"

    SUPPORTS_WRITES = True

    PARAMS: list[DriverParam] = [
        DriverParam(
            key="access_key_id",
            type=ParamType.STRING,
            label="Access Key ID",
            required=False,
        ),
        DriverParam(
            key="secret_access_key",
            type=ParamType.STRING,
            label="Secret Access Key",
            required=False,
            secret=True,
        ),
        DriverParam(
            key="region",
            type=ParamType.STRING,
            label="Region",
            default="us-east-1",
        ),
        DriverParam(
            key="endpoint",
            type=ParamType.STRING,
            label="Endpoint URL (for S3-compatible stores, e.g. MinIO/R2)",
            required=False,
        ),
    ]

    HELP: str = """\
## S3

**Queries:** a small `aws s3`-style command language. One command per query.

```
ls s3://bucket/prefix [--recursive] [--pattern GLOB]
cp <src> <dst> [--recursive] [--pattern GLOB]
mv <src> <dst> [--recursive] [--pattern GLOB]
rm s3://bucket/key [--recursive] [--pattern GLOB]
presign s3://bucket/key [--expires-in SECONDS]
```

`ls` with no URI lists buckets. `--pattern` is a glob (`fnmatch`) filter applied
to full object keys — a deliberate deviation from the real `aws` CLI, which has
no filtering on `ls` at all and splits filtering into `--exclude`/`--include`
on the other verbs.

`cp`/`mv` infer direction from which side is `s3://`:

```
cp ./file.txt s3://bucket/key          # upload
cp s3://bucket/key ./file.txt          # download to disk
cp s3://bucket-a/key s3://bucket-b/key # server-side copy, no local round-trip
```

Both sides local, or neither side `s3://`, is an error. `mv` is `cp` followed
by deleting the source (local file or S3 key, whichever side was the source).
`--recursive` copies/moves/removes everything under a prefix or local
directory, preserving relative paths.

**Comments:** `#` to the end of the line, starting a word as in a shell —
`s3://bucket/a#b` names a key, it does not start a comment.

**Resources:**

```
(root)
└── <bucket>
    └── <prefix>/...
        └── <object>
```

Describing a bucket returns a `RawDocument` (`filetype: "yaml"`) with region,
creation date, versioning status, and lifecycle rules. Describing an object
returns a `GenericRecordDescription` (`kind: "s3.object"`) with size,
content-type, last-modified, ETag, storage class, and (when present)
server-side encryption, version ID, user-defined metadata, and tags. A prefix
("folder") node is not describable.

`explore.download` fetches an object's full content for loading into a buffer;
objects larger than 25 MB are refused.
"""

    def __init__(
        self,
        params: dict[str, Any],
        client: Any,
        settings: DriverSettings,
    ) -> None:
        super().__init__(params, settings)
        self._client = client

    @classmethod
    async def create(
        cls, params: dict[str, Any], settings: DriverSettings
    ) -> "S3Driver":
        client = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _open_client(params)
        )
        driver = cls(params, client, settings)
        await driver._run(driver._client.list_buckets)
        return driver

    async def reconnect(self) -> None:
        self._client = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _open_client(self.params)
        )

    async def disconnect(self) -> None:
        await self._run(self._client.close)

    # -- execute (ls/cp/mv/rm/presign) --------------------------------------

    async def execute(self, query: str, binds: list[Any]) -> ReadResult | WriteResult:
        """Run an ``aws s3``-style command.

        Args:
            query: One command — ``ls``, ``cp``, ``mv``, ``rm``, or ``presign``
                (see :attr:`HELP` for full syntax).
            binds: Unused for S3.
        """
        try:
            # `#` comments as in a shell: only at the start of a word, so a key
            # holding one (`s3://b/a#b`) is left whole.
            tokens = shlex.split(
                blank_comments(query, line=("#",), quotes="'\"", after="")
            )
        except ValueError as exc:
            raise DriverError(f"Could not parse command: {exc}") from exc
        if not tokens:
            raise DriverError("Empty command")
        verb, args = tokens[0], tokens[1:]
        match verb:
            case "ls":
                return await self._cmd_ls(args)
            case "cp":
                return await self._cmd_transfer(args, delete_source=False)
            case "mv":
                return await self._cmd_transfer(args, delete_source=True)
            case "rm":
                return await self._cmd_rm(args)
            case "presign":
                return await self._cmd_presign(args)
            case _:
                raise DriverError(
                    f"Unknown command: {verb!r}. Supported: ls, cp, mv, rm, presign"
                )

    async def _cmd_ls(self, args: list[str]) -> ReadResult:
        recursive, pattern, positional = _parse_flags(args, "ls", expect_positional=1)
        if not positional:
            buckets = await self._run(self._list_buckets_sync)
            rows = [[name, None, iso, None, None] for name, iso in buckets]
            return ReadResult(
                columns=["key", "size", "last_modified", "storage_class", "etag"],
                rows=rows,
                rows_total=len(rows),
            )
        bucket, prefix = _parse_s3_uri(positional[0])
        rows = await self._run(self._ls_sync, bucket, prefix, recursive, pattern)
        return ReadResult(
            columns=["key", "size", "last_modified", "storage_class", "etag"],
            rows=rows,
            rows_total=len(rows),
        )

    async def _cmd_transfer(self, args: list[str], delete_source: bool) -> WriteResult:
        recursive, pattern, positional = _parse_flags(
            args, "mv" if delete_source else "cp", expect_positional=2
        )
        src, dst = positional
        opts = _TransferOpts(recursive=recursive, pattern=pattern)
        src_s3 = _try_parse_s3_uri(src)
        dst_s3 = _try_parse_s3_uri(dst)
        if src_s3 and dst_s3:
            keys = await self._run(self._copy_s3_to_s3_sync, src_s3, dst_s3, opts)
            if delete_source:
                await self._run(self._delete_keys_sync, src_s3[0], keys)
            return WriteResult(rows_affected=len(keys))
        if src_s3 and not dst_s3:
            keys = await self._run(self._download_sync, src_s3, dst, opts)
            if delete_source:
                await self._run(self._delete_keys_sync, src_s3[0], keys)
            return WriteResult(rows_affected=len(keys))
        if not src_s3 and dst_s3:
            paths = await self._run(self._upload_sync, src, dst_s3, opts)
            if delete_source:
                for p in paths:
                    p.unlink()
            return WriteResult(rows_affected=len(paths))
        raise DriverError("cp/mv requires at least one s3:// path")

    async def _cmd_rm(self, args: list[str]) -> WriteResult:
        recursive, pattern, positional = _parse_flags(args, "rm", expect_positional=1)
        bucket, prefix = _parse_s3_uri(positional[0])
        opts = _TransferOpts(recursive=recursive, pattern=pattern)
        keys = await self._run(self._match_keys_sync, bucket, prefix, opts)
        if not keys:
            return WriteResult(rows_affected=0)
        await self._run(self._delete_keys_sync, bucket, keys)
        return WriteResult(rows_affected=len(keys))

    async def _cmd_presign(self, args: list[str]) -> ReadResult:
        expires_in = 3600
        positional = []
        i = 0
        while i < len(args):
            if args[i] == "--expires-in":
                if i + 1 >= len(args):
                    raise DriverError("--expires-in requires a value")
                expires_in = int(args[i + 1])
                i += 2
            else:
                positional.append(args[i])
                i += 1
        if len(positional) != 1:
            raise DriverError("presign requires exactly one s3:// object path")
        bucket, key = _parse_s3_uri(positional[0])
        if not key:
            raise DriverError("presign requires an object key, not just a bucket")
        url = await self._run(self._presign_sync, bucket, key, expires_in)
        return ReadResult(columns=["url"], rows=[[url]], rows_total=1)

    # -- explore --------------------------------------------------------------

    async def explore_list(self, path: list[str]) -> list[ExploreItem]:
        return await self._run(self._explore_list_sync, path)

    def _explore_list_sync(self, path: list[str]) -> list[ExploreItem]:
        if not path:
            buckets = self._list_buckets_sync()
            return [
                ExploreItem(name=name, type="bucket", expandable=True)
                for name, _ in buckets
            ]
        bucket, *segments = path
        prefix = "/".join(segments) + "/" if segments else ""
        items: list[ExploreItem] = []
        for page in self._client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                items.append(ExploreItem(name=name, type="prefix", expandable=True))
            for obj in page.get("Contents", []):
                if obj["Key"] == prefix:
                    continue
                name = obj["Key"][len(prefix) :]
                items.append(ExploreItem(name=name, type="object", expandable=False))
        return items

    async def explore_describe(self, path: list[str]) -> DescribeResult:
        return await self._run(self._explore_describe_sync, path)

    def _explore_describe_sync(self, path: list[str]) -> DescribeResult:
        match path:
            case []:
                return None
            case [bucket]:
                return self._describe_bucket_sync(bucket)
            case [bucket, *segments] if segments:
                key = "/".join(segments)
                try:
                    head = self._client.head_object(Bucket=bucket, Key=key)
                except ClientError as exc:
                    if _error_code(exc) in _NOT_FOUND_CODES:
                        return None
                    raise
                return self._describe_object_sync(bucket, key, head)
            case _:
                return None

    def _describe_bucket_sync(self, bucket: str) -> RawDocument:
        region = (
            self._client.get_bucket_location(Bucket=bucket).get("LocationConstraint")
            or "us-east-1"
        )
        try:
            versioning_status = self._client.get_bucket_versioning(Bucket=bucket).get(
                "Status"
            )
        except ClientError:
            versioning_status = None
        versioning = "enabled" if versioning_status == "Enabled" else "disabled"
        created = self._bucket_creation_date_sync(bucket)
        doc: dict[str, Any] = {
            "region": region,
            "versioning": versioning,
        }
        if created:
            doc["created"] = created
        rules = self._bucket_lifecycle_rules_sync(bucket)
        if rules:
            doc["lifecycle_rules"] = rules
        content = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
        return RawDocument(filetype="yaml", content=content)

    def _bucket_creation_date_sync(self, bucket: str) -> str | None:
        for name, iso in self._list_buckets_sync():
            if name == bucket:
                return iso
        return None

    def _bucket_lifecycle_rules_sync(self, bucket: str) -> list[dict[str, Any]]:
        try:
            resp = self._client.get_bucket_lifecycle_configuration(Bucket=bucket)
        except ClientError as exc:
            if _error_code(exc) in ("NoSuchLifecycleConfiguration",):
                return []
            raise
        rules = []
        for rule in resp.get("Rules", []):
            entry: dict[str, Any] = {
                "id": rule.get("ID", ""),
                "status": (rule.get("Status") or "").lower(),
            }
            filt = rule.get("Filter") or {}
            prefix = filt.get("Prefix") if "Prefix" in filt else rule.get("Prefix")
            if prefix:
                entry["filter"] = {"prefix": prefix}
            expiration = rule.get("Expiration") or {}
            if "Days" in expiration:
                entry["expiration"] = {"days": expiration["Days"]}
            transitions = rule.get("Transitions") or []
            if transitions:
                entry["transitions"] = [
                    {"days": t.get("Days"), "storage_class": t.get("StorageClass")}
                    for t in transitions
                ]
            rules.append(entry)
        return rules

    def _describe_object_sync(
        self, bucket: str, key: str, head: dict[str, Any]
    ) -> GenericRecordDescription:
        size = head.get("ContentLength", 0)
        fields = [
            RecordField(label="Size", value=_format_size(size)),
            RecordField(label="Content-Type", value=head.get("ContentType", "") or ""),
            RecordField(
                label="Last Modified",
                value=_iso(head.get("LastModified")) or "",
            ),
            RecordField(label="ETag", value=head.get("ETag", "") or ""),
            RecordField(
                label="Storage Class", value=head.get("StorageClass") or "STANDARD"
            ),
        ]
        sse = head.get("ServerSideEncryption")
        if sse:
            fields.append(RecordField(label="Server-Side Encryption", value=sse))
        version_id = head.get("VersionId")
        if version_id and version_id != "null":
            fields.append(RecordField(label="Version ID", value=version_id))
        for meta_key, meta_value in (head.get("Metadata") or {}).items():
            fields.append(RecordField(label=meta_key, value=str(meta_value)))
        for tag_key, tag_value in self._object_tags_sync(bucket, key).items():
            fields.append(RecordField(label=f"Tag: {tag_key}", value=tag_value))
        return GenericRecordDescription(kind="s3.object", name=key, fields=fields)

    def _object_tags_sync(self, bucket: str, key: str) -> dict[str, str]:
        try:
            resp = self._client.get_object_tagging(Bucket=bucket, Key=key)
        except ClientError:
            return {}
        return {t["Key"]: t["Value"] for t in resp.get("TagSet", [])}

    # -- download ---------------------------------------------------------

    async def explore_download(
        self, path: list[str], dest_path: str | None
    ) -> DownloadResult:
        if len(path) < 2:
            raise DriverError("explore.download requires a path to an object")
        bucket, *segments = path
        key = "/".join(segments)
        if dest_path is not None:
            return await self._run(self._download_to_file_sync, bucket, key, dest_path)
        return await self._run(self._download_inline_sync, bucket, key)

    def _download_to_file_sync(
        self, bucket: str, key: str, dest_path: str
    ) -> DownloadResult:
        # Streams straight to disk via boto3's managed transfer (same call the
        # `cp` DSL command uses) — no size cap, since content never touches
        # Python-process memory as a whole blob the way the inline path does.
        head = self._client.head_object(Bucket=bucket, Key=key)
        self._client.download_file(bucket, key, dest_path)
        return DownloadResult(
            filename=key.rsplit("/", 1)[-1],
            content_type=head.get("ContentType", "application/octet-stream"),
            size=head.get("ContentLength", 0),
            written_to=dest_path,
        )

    def _download_inline_sync(self, bucket: str, key: str) -> DownloadResult:
        head = self._client.head_object(Bucket=bucket, Key=key)
        size = head.get("ContentLength", 0)
        if size > _MAX_DOWNLOAD_BYTES:
            raise DriverError(
                f"Object is {_format_size(size)}, larger than the "
                f"{_format_size(_MAX_DOWNLOAD_BYTES)} explore.download limit "
                "— use the save-to-disk download instead"
            )
        obj = self._client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return DownloadResult(
            content_base64=base64.b64encode(body).decode(),
            filename=key.rsplit("/", 1)[-1],
            content_type=obj.get("ContentType", "application/octet-stream"),
            size=len(body),
        )

    # -- transfer helpers (sync, run in executor) --------------------------

    def _list_buckets_sync(self) -> list[tuple[str, str | None]]:
        resp = self._client.list_buckets()
        return [
            (b["Name"], _iso(b.get("CreationDate"))) for b in resp.get("Buckets", [])
        ]

    def _ls_sync(
        self, bucket: str, prefix: str, recursive: bool, pattern: str | None
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"
        for page in self._client.get_paginator("list_objects_v2").paginate(**kwargs):
            for cp in page.get("CommonPrefixes", []):
                key = cp["Prefix"]
                if pattern and not fnmatch.fnmatch(key, pattern):
                    continue
                rows.append([key, None, None, None, None])
            for obj in page.get("Contents", []):
                if pattern and not fnmatch.fnmatch(obj["Key"], pattern):
                    continue
                rows.append(
                    [
                        obj["Key"],
                        obj.get("Size"),
                        _iso(obj.get("LastModified")),
                        obj.get("StorageClass") or "STANDARD",
                        obj.get("ETag"),
                    ]
                )
        return rows

    def _match_keys_sync(
        self, bucket: str, prefix: str, opts: "_TransferOpts"
    ) -> list[str]:
        if not opts.recursive:
            keys = [prefix] if prefix else []
        else:
            keys = []
            for page in self._client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix
            ):
                keys += [obj["Key"] for obj in page.get("Contents", [])]
        if opts.pattern:
            keys = [k for k in keys if fnmatch.fnmatch(k, opts.pattern)]
        return sorted(keys)

    def _delete_keys_sync(self, bucket: str, keys: list[str]) -> None:
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self._client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in batch]},
            )

    def _copy_s3_to_s3_sync(
        self,
        src: tuple[str, str],
        dst: tuple[str, str],
        opts: "_TransferOpts",
    ) -> list[str]:
        src_bucket, src_key = src
        dst_bucket, dst_key = dst
        keys = self._match_keys_sync(src_bucket, src_key, opts)
        for key in keys:
            target = _join_relative(src_key, dst_key, key, opts.recursive)
            self._client.copy_object(
                Bucket=dst_bucket,
                Key=target,
                CopySource={"Bucket": src_bucket, "Key": key},
            )
        return keys

    def _download_sync(
        self, src: tuple[str, str], dst: str, opts: "_TransferOpts"
    ) -> list[str]:
        bucket, prefix = src
        keys = self._match_keys_sync(bucket, prefix, opts)
        dst_path = Path(dst)
        for key in keys:
            target = _join_relative_local(prefix, dst_path, key, opts.recursive)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket, key, str(target))
        return keys

    def _upload_sync(
        self, src: str, dst: tuple[str, str], opts: "_TransferOpts"
    ) -> list[Path]:
        dst_bucket, dst_key = dst
        src_path = Path(src)
        if opts.recursive:
            if not src_path.is_dir():
                raise DriverError(f"--recursive requires a local directory: {src}")
            paths = sorted(p for p in src_path.rglob("*") if p.is_file())
            if opts.pattern:
                paths = [p for p in paths if fnmatch.fnmatch(str(p), opts.pattern)]
            for p in paths:
                rel = p.relative_to(src_path).as_posix()
                target_key = f"{dst_key.rstrip('/')}/{rel}" if dst_key else rel
                self._client.upload_file(str(p), dst_bucket, target_key)
            return paths
        if not src_path.is_file():
            raise DriverError(f"Local file not found: {src}")
        target_key = dst_key
        if not target_key or target_key.endswith("/"):
            target_key = f"{target_key}{src_path.name}"
        self._client.upload_file(str(src_path), dst_bucket, target_key)
        return [src_path]

    def _presign_sync(self, bucket: str, key: str, expires_in: int) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    # -- plumbing -----------------------------------------------------------

    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        # Every S3 API call goes through here; the operation name plus its
        # keyword arguments (Bucket, Key, Prefix) is the useful record.
        op = getattr(fn, "__name__", repr(fn))
        log_query(logger, f"s3 {op} {kwargs}" if kwargs else f"s3 {op}")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, lambda: fn(*args, **kwargs)
            )
        except (EndpointConnectionError, NoCredentialsError) as exc:
            raise ConnectionLostError(str(exc)) from exc
        except ClientError as exc:
            raise DriverError(_client_error_message(exc)) from exc
        except DriverError:
            raise
        except OSError as exc:
            raise DriverError(str(exc)) from exc


@dataclass
class _TransferOpts:
    recursive: bool
    pattern: str | None


def _open_client(params: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {"region_name": params.get("region") or "us-east-1"}
    if params.get("access_key_id"):
        kwargs["aws_access_key_id"] = params["access_key_id"]
    if params.get("secret_access_key"):
        kwargs["aws_secret_access_key"] = params["secret_access_key"]
    if params.get("endpoint"):
        kwargs["endpoint_url"] = params["endpoint"]
    try:
        return boto3.client("s3", **kwargs)
    except Exception as exc:
        raise DriverError(str(exc)) from exc


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = _try_parse_s3_uri(uri)
    if parsed is None:
        raise DriverError(f"Not an s3:// URI: {uri!r}")
    return parsed


def _try_parse_s3_uri(uri: str) -> tuple[str, str] | None:
    m = _S3_URI_RE.match(uri)
    if not m:
        return None
    return m.group("bucket"), m.group("key")


def _parse_flags(
    args: list[str], verb: str, expect_positional: int
) -> tuple[bool, str | None, list[str]]:
    """Parse ``--recursive``/``--pattern`` and return (recursive, pattern, positional_args)."""
    recursive = False
    pattern: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--recursive":
            recursive = True
            i += 1
        elif arg == "--pattern":
            if i + 1 >= len(args):
                raise DriverError("--pattern requires a value")
            pattern = args[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1
    if len(positional) > expect_positional:
        raise DriverError(f"{verb}: too many arguments")
    if verb != "ls" and len(positional) != expect_positional:
        raise DriverError(f"{verb}: expected {expect_positional} path argument(s)")
    return recursive, pattern, positional


def _join_relative(src_prefix: str, dst_key: str, key: str, recursive: bool) -> str:
    if not recursive:
        target = dst_key
        if not target or target.endswith("/"):
            target = f"{target}{key.rsplit('/', 1)[-1]}"
        return target
    rel = key[len(src_prefix) :].lstrip("/")
    return f"{dst_key.rstrip('/')}/{rel}" if rel else dst_key


def _join_relative_local(
    src_prefix: str, dst_path: Path, key: str, recursive: bool
) -> Path:
    if not recursive:
        if dst_path.is_dir() or str(dst_path).endswith("/"):
            return dst_path / key.rsplit("/", 1)[-1]
        return dst_path
    rel = key[len(src_prefix) :].lstrip("/")
    return dst_path / rel if rel else dst_path / key.rsplit("/", 1)[-1]


def _format_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{n} B"
            return f"{size:.1f} {unit} ({n:,} bytes)"
        size /= 1024
    return f"{n} B"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def _client_error_message(exc: ClientError) -> str:
    error = exc.response.get("Error", {})
    code = error.get("Code", "")
    message = error.get("Message", str(exc))
    return f"S3 error ({code}): {message}" if code else str(exc)
