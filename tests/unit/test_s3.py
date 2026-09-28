"""Unit tests for S3Driver — no live AWS account required."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from botocore.exceptions import ClientError, EndpointConnectionError

from grannos.drivers.base import ConnectionLostError, DriverError, DriverSettings
from grannos.drivers.s3 import S3Driver, _parse_s3_uri, _try_parse_s3_uri
from grannos.protocol import (
    ExploreItem,
    GenericRecordDescription,
    RawDocument,
    ReadResult,
    WriteResult,
)


def _client_error(code: str, message: str = "boom") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Operation")


def _paginated(*pages: dict) -> MagicMock:
    """A get_paginator().paginate(...) mock returning the given pages."""
    paginator = MagicMock()
    paginator.paginate.return_value = list(pages)
    return paginator


def _make_driver(client: MagicMock) -> S3Driver:
    return S3Driver({}, client, DriverSettings())


class TestS3UriParsing:
    def test_parses_bucket_and_key(self) -> None:
        assert _try_parse_s3_uri("s3://my-bucket/logs/a.log") == (
            "my-bucket",
            "logs/a.log",
        )

    def test_parses_bucket_only(self) -> None:
        assert _try_parse_s3_uri("s3://my-bucket") == ("my-bucket", "")
        assert _try_parse_s3_uri("s3://my-bucket/") == ("my-bucket", "")

    def test_non_s3_uri_returns_none(self) -> None:
        assert _try_parse_s3_uri("./local/path") is None
        assert _try_parse_s3_uri("/abs/local/path") is None

    def test_parse_s3_uri_raises_on_non_s3(self) -> None:
        with pytest.raises(DriverError):
            _parse_s3_uri("not-s3")


class TestExploreList:
    async def test_root_lists_buckets(self) -> None:
        client = MagicMock()
        client.list_buckets.return_value = {
            "Buckets": [
                {"Name": "b1", "CreationDate": datetime(2024, 1, 1, tzinfo=UTC)},
                {"Name": "b2", "CreationDate": datetime(2024, 2, 1, tzinfo=UTC)},
            ]
        }
        driver = _make_driver(client)
        items = await driver.explore_list([])
        assert items == [
            ExploreItem(name="b1", type="bucket", expandable=True),
            ExploreItem(name="b2", type="bucket", expandable=True),
        ]

    async def test_bucket_lists_prefixes_and_objects(self) -> None:
        client = MagicMock()
        client.get_paginator.return_value = _paginated(
            {
                "CommonPrefixes": [{"Prefix": "logs/"}],
                "Contents": [{"Key": "readme.txt"}],
            }
        )
        driver = _make_driver(client)
        items = await driver.explore_list(["my-bucket"])
        assert items == [
            ExploreItem(name="logs", type="prefix", expandable=True),
            ExploreItem(name="readme.txt", type="object", expandable=False),
        ]
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="my-bucket", Prefix="", Delimiter="/"
        )

    async def test_prefix_folder_marker_object_is_skipped(self) -> None:
        client = MagicMock()
        client.get_paginator.return_value = _paginated(
            {
                "CommonPrefixes": [],
                "Contents": [
                    {"Key": "logs/"},  # zero-byte folder marker, equals the prefix
                    {"Key": "logs/a.log"},
                ],
            }
        )
        driver = _make_driver(client)
        items = await driver.explore_list(["my-bucket", "logs"])
        assert items == [ExploreItem(name="a.log", type="object", expandable=False)]


class TestExploreDescribe:
    async def test_root_is_not_describable(self) -> None:
        driver = _make_driver(MagicMock())
        assert await driver.explore_describe([]) is None

    async def test_bucket_describe_returns_yaml_document(self) -> None:
        client = MagicMock()
        client.get_bucket_location.return_value = {"LocationConstraint": "eu-west-1"}
        client.get_bucket_versioning.return_value = {"Status": "Enabled"}
        client.get_bucket_lifecycle_configuration.return_value = {
            "Rules": [
                {
                    "ID": "expire-old-logs",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "logs/"},
                    "Expiration": {"Days": 30},
                },
                {
                    "ID": "archive-to-glacier",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "archive/"},
                    "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],
                },
            ]
        }
        client.list_buckets.return_value = {
            "Buckets": [
                {"Name": "my-bucket", "CreationDate": datetime(2023, 1, 1, tzinfo=UTC)}
            ]
        }
        driver = _make_driver(client)
        result = await driver.explore_describe(["my-bucket"])
        assert isinstance(result, RawDocument)
        assert result.filetype == "yaml"
        doc = yaml.safe_load(result.content)
        assert doc["region"] == "eu-west-1"
        assert doc["versioning"] == "enabled"
        assert doc["created"] == "2023-01-01T00:00:00+00:00"
        assert doc["lifecycle_rules"] == [
            {
                "id": "expire-old-logs",
                "status": "enabled",
                "filter": {"prefix": "logs/"},
                "expiration": {"days": 30},
            },
            {
                "id": "archive-to-glacier",
                "status": "enabled",
                "filter": {"prefix": "archive/"},
                "transitions": [{"days": 90, "storage_class": "GLACIER"}],
            },
        ]

    async def test_bucket_describe_no_lifecycle_configuration(self) -> None:
        client = MagicMock()
        client.get_bucket_location.return_value = {"LocationConstraint": "us-east-1"}
        client.get_bucket_versioning.return_value = {}
        client.get_bucket_lifecycle_configuration.side_effect = _client_error(
            "NoSuchLifecycleConfiguration"
        )
        client.list_buckets.return_value = {"Buckets": []}
        driver = _make_driver(client)
        result = await driver.explore_describe(["my-bucket"])
        assert isinstance(result, RawDocument)
        doc = yaml.safe_load(result.content)
        assert "lifecycle_rules" not in doc
        assert doc["versioning"] == "disabled"

    async def test_object_describe_returns_generic_record(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {
            "ContentLength": 13,
            "ContentType": "text/plain",
            "LastModified": datetime(2026, 7, 28, 14, 3, 11, tzinfo=UTC),
            "ETag": '"abc123"',
            "StorageClass": "STANDARD",
            "ServerSideEncryption": "aws:kms",
            "VersionId": "v1",
            "Metadata": {"uploaded-by": "ci-pipeline"},
        }
        client.get_object_tagging.return_value = {
            "TagSet": [{"Key": "environment", "Value": "prod"}]
        }
        driver = _make_driver(client)
        result = await driver.explore_describe(["my-bucket", "logs", "a.log"])
        assert isinstance(result, GenericRecordDescription)
        assert result.type == "generic_record"
        assert result.kind == "s3.object"
        assert result.name == "logs/a.log"
        labels = {f.label: f.value for f in result.fields}
        assert labels["Content-Type"] == "text/plain"
        assert labels["ETag"] == '"abc123"'
        assert labels["Server-Side Encryption"] == "aws:kms"
        assert labels["Version ID"] == "v1"
        assert labels["uploaded-by"] == "ci-pipeline"
        assert labels["Tag: environment"] == "prod"
        client.head_object.assert_called_once_with(Bucket="my-bucket", Key="logs/a.log")

    async def test_nonexistent_key_is_treated_as_prefix_not_describable(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _client_error("404", "Not Found")
        driver = _make_driver(client)
        result = await driver.explore_describe(["my-bucket", "logs"])
        assert result is None


class TestExploreDownload:
    async def test_downloads_object_content(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {"ContentLength": 5}
        body = MagicMock()
        body.read.return_value = b"hello"
        client.get_object.return_value = {"Body": body, "ContentType": "text/plain"}
        driver = _make_driver(client)
        result = await driver.explore_download(["my-bucket", "a.txt"], None)
        assert result.content_base64 == "aGVsbG8="
        assert result.written_to is None
        assert result.filename == "a.txt"
        assert result.content_type == "text/plain"
        assert result.size == 5

    async def test_rejects_object_above_size_limit(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {"ContentLength": 100 * 1024 * 1024}
        driver = _make_driver(client)
        with pytest.raises(DriverError):
            await driver.explore_download(["my-bucket", "huge.bin"], None)
        client.get_object.assert_not_called()

    async def test_downloads_directly_to_dest_path_bypassing_size_limit(self) -> None:
        client = MagicMock()
        client.head_object.return_value = {
            "ContentLength": 100 * 1024 * 1024,
            "ContentType": "application/zip",
        }
        driver = _make_driver(client)
        result = await driver.explore_download(
            ["my-bucket", "huge.zip"], "/tmp/out.zip"
        )
        assert result.written_to == "/tmp/out.zip"
        assert result.content_base64 is None
        assert result.size == 100 * 1024 * 1024
        client.download_file.assert_called_once_with(
            "my-bucket", "huge.zip", "/tmp/out.zip"
        )
        client.get_object.assert_not_called()


class TestExecuteLs:
    async def test_ls_no_args_lists_buckets(self) -> None:
        client = MagicMock()
        client.list_buckets.return_value = {
            "Buckets": [
                {"Name": "b1", "CreationDate": datetime(2024, 1, 1, tzinfo=UTC)}
            ]
        }
        driver = _make_driver(client)
        result = await driver.execute("ls", [])
        assert isinstance(result, ReadResult)
        assert result.rows == [["b1", None, "2024-01-01T00:00:00+00:00", None, None]]
        assert result.columns == [
            "key",
            "size",
            "last_modified",
            "storage_class",
            "etag",
        ]

    async def test_comments_are_ignored(self) -> None:
        client = MagicMock()
        client.list_buckets.return_value = {"Buckets": []}
        driver = _make_driver(client)
        result = await driver.execute("# every bucket\nls # none filtered", [])
        assert isinstance(result, ReadResult)
        client.list_buckets.assert_called_once()

    async def test_ls_bucket_lists_objects(self) -> None:
        client = MagicMock()
        client.get_paginator.return_value = _paginated(
            {
                "Contents": [
                    {
                        "Key": "a.log",
                        "Size": 42,
                        "LastModified": datetime(2024, 1, 1, tzinfo=UTC),
                        "StorageClass": "STANDARD",
                        "ETag": '"x"',
                    }
                ],
                "CommonPrefixes": [],
            }
        )
        driver = _make_driver(client)
        result = await driver.execute("ls s3://my-bucket", [])
        assert isinstance(result, ReadResult)
        assert result.rows == [
            ["a.log", 42, "2024-01-01T00:00:00+00:00", "STANDARD", '"x"']
        ]

    async def test_ls_pattern_filters_results(self) -> None:
        client = MagicMock()
        client.get_paginator.return_value = _paginated(
            {
                "Contents": [
                    {"Key": "a.log", "Size": 1, "LastModified": None, "ETag": ""},
                    {"Key": "b.txt", "Size": 1, "LastModified": None, "ETag": ""},
                ],
                "CommonPrefixes": [],
            }
        )
        driver = _make_driver(client)
        result = await driver.execute("ls s3://my-bucket --pattern *.log", [])
        assert isinstance(result, ReadResult)
        assert [r[0] for r in result.rows] == ["a.log"]


class TestExecuteTransfer:
    async def test_cp_upload_local_to_s3(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        client = MagicMock()
        driver = _make_driver(client)
        result = await driver.execute(f"cp {f} s3://my-bucket/uploaded.txt", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        client.upload_file.assert_called_once_with(str(f), "my-bucket", "uploaded.txt")

    async def test_cp_download_s3_to_local(self, tmp_path: Path) -> None:
        dst = tmp_path / "out.txt"
        client = MagicMock()
        driver = _make_driver(client)
        result = await driver.execute(f"cp s3://my-bucket/a.txt {dst}", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        client.download_file.assert_called_once_with("my-bucket", "a.txt", str(dst))

    async def test_cp_bucket_to_bucket_uses_server_side_copy(self) -> None:
        client = MagicMock()
        driver = _make_driver(client)
        result = await driver.execute(
            "cp s3://bucket-a/key.txt s3://bucket-b/key.txt", []
        )
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        client.copy_object.assert_called_once_with(
            Bucket="bucket-b",
            Key="key.txt",
            CopySource={"Bucket": "bucket-a", "Key": "key.txt"},
        )

    async def test_cp_local_to_local_raises(self, tmp_path: Path) -> None:
        driver = _make_driver(MagicMock())
        with pytest.raises(DriverError):
            await driver.execute(f"cp {tmp_path}/a.txt {tmp_path}/b.txt", [])

    async def test_mv_deletes_source_object_after_download(
        self, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.txt"
        client = MagicMock()
        driver = _make_driver(client)
        result = await driver.execute(f"mv s3://my-bucket/a.txt {dst}", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        client.download_file.assert_called_once_with("my-bucket", "a.txt", str(dst))
        client.delete_objects.assert_called_once_with(
            Bucket="my-bucket", Delete={"Objects": [{"Key": "a.txt"}]}
        )

    async def test_mv_deletes_local_file_after_upload(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hi")
        client = MagicMock()
        driver = _make_driver(client)
        await driver.execute(f"mv {f} s3://my-bucket/uploaded.txt", [])
        assert not f.exists()


class TestExecuteRm:
    async def test_rm_single_object(self) -> None:
        client = MagicMock()
        driver = _make_driver(client)
        result = await driver.execute("rm s3://my-bucket/a.txt", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1
        client.delete_objects.assert_called_once_with(
            Bucket="my-bucket", Delete={"Objects": [{"Key": "a.txt"}]}
        )

    async def test_rm_recursive_deletes_all_matching(self) -> None:
        client = MagicMock()
        client.get_paginator.return_value = _paginated(
            {"Contents": [{"Key": "logs/a.log"}, {"Key": "logs/b.log"}]}
        )
        driver = _make_driver(client)
        result = await driver.execute("rm s3://my-bucket/logs --recursive", [])
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 2


class TestExecutePresign:
    async def test_presign_returns_url_row(self) -> None:
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://example.com/signed"
        driver = _make_driver(client)
        result = await driver.execute(
            "presign s3://my-bucket/a.txt --expires-in 60", []
        )
        assert isinstance(result, ReadResult)
        assert result.rows == [["https://example.com/signed"]]
        client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "my-bucket", "Key": "a.txt"},
            ExpiresIn=60,
        )

    async def test_presign_requires_object_key(self) -> None:
        driver = _make_driver(MagicMock())
        with pytest.raises(DriverError):
            await driver.execute("presign s3://my-bucket", [])


class TestExecuteErrors:
    async def test_unknown_command_raises(self) -> None:
        driver = _make_driver(MagicMock())
        with pytest.raises(DriverError):
            await driver.execute("frobnicate s3://my-bucket", [])

    async def test_empty_command_raises(self) -> None:
        driver = _make_driver(MagicMock())
        with pytest.raises(DriverError):
            await driver.execute("", [])


class TestConnectionHandling:
    async def test_endpoint_connection_error_raises_connection_lost(self) -> None:
        client = MagicMock()
        client.list_buckets.side_effect = EndpointConnectionError(
            endpoint_url="https://s3.amazonaws.com"
        )
        driver = _make_driver(client)
        with pytest.raises(ConnectionLostError):
            await driver.explore_list([])

    async def test_client_error_wrapped_as_driver_error(self) -> None:
        client = MagicMock()
        client.list_buckets.side_effect = _client_error("AccessDenied", "nope")
        driver = _make_driver(client)
        with pytest.raises(DriverError, match="AccessDenied"):
            await driver.explore_list([])
