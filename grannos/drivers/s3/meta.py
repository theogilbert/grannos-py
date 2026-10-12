"""Capabilities of the S3 driver, importable without its client package."""

from ...protocol import DriverParam, ParamType
from ..base import DriverMeta


class S3Meta(DriverMeta):
    """What the S3 driver advertises in ``capabilities`` and ``driver.help``."""

    REQUIRES = ("boto3", "yaml")

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
