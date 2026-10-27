import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from uuid import UUID

import pytest

from grannos.protocol import (
    DecodeError,
    ExploreItem,
    Method,
    Progress,
    ProgressDetail,
    Request,
    Result,
    SpecialFloat,
    decode,
    encode,
)


class TestDecode:
    def test_should_return_request_dataclass_when_json_is_valid(self) -> None:
        msg = decode(b'{"id":1,"method":"connect","params":{"driver":"sqlite"}}\n')
        assert msg == Request(id=1, method=Method.CONNECT, params={"driver": "sqlite"})

    def test_should_raise_when_json_is_malformed(self) -> None:
        with pytest.raises(DecodeError):
            decode(b"not json\n")

    def test_should_raise_when_id_field_is_missing(self) -> None:
        with pytest.raises(DecodeError):
            decode(b'{"method": "connect", "params": {}}\n')

    def test_should_raise_when_id_field_is_invalid(self) -> None:
        with pytest.raises(DecodeError):
            decode(b'{"id": "foo", "method": "connect", "params": {}}\n')

    def test_should_raise_when_method_field_is_missing(self) -> None:
        with pytest.raises(DecodeError):
            decode(b'{"id":1, "params": {}}\n')

    def test_should_raise_when_method_field_is_invalid(self) -> None:
        with pytest.raises(DecodeError):
            decode(b'{"id":1, "method": "invalid", "params": {}}\n')

    def test_should_raise_when_params_field_is_missing(self) -> None:
        with pytest.raises(DecodeError):
            decode(b'{"id":1, "method": "connect"}\n')

    def test_should_raise_when_params_is_not_an_object(self) -> None:
        with pytest.raises(DecodeError, match="params must be a JSON object"):
            decode(b'{"id":1,"method":"connect","params":[1,2,3]}\n')


class TestEncode:
    def test_should_serialise_result_to_json_bytes(self) -> None:
        data = encode(Result(id=1, result={"rows": [[1, 2]]}, error=None))
        assert json.loads(data) == {
            "id": 1,
            "result": {"rows": [[1, 2]]},
            "error": None,
        }

    def test_should_serialise_error_result_with_null_id(self) -> None:
        data = encode(Result(id=None, result=None, error="oops"))
        assert json.loads(data) == {"id": None, "result": None, "error": "oops"}

    def test_should_serialise_progress_with_nested_detail(self) -> None:
        data = encode(
            Progress(id=2, progress=ProgressDetail(status="running", message="..."))
        )
        assert json.loads(data) == {
            "id": 2,
            "progress": {"status": "running", "message": "..."},
        }

    def test_should_serialise_explore_items_nested_in_result(self) -> None:
        result = Result(
            id=1,
            result={"items": [ExploreItem(name="t", type="table", expandable=True)]},
            error=None,
        )
        assert json.loads(encode(result))["result"]["items"] == [
            {"name": "t", "type": "table", "expandable": True}
        ]

    def test_output_ends_with_newline(self) -> None:
        assert encode(Result(id=1, result=None, error=None)).endswith(b"\n")

    def test_should_serialise_uuid_as_string(self) -> None:
        uid = UUID("12345678-1234-5678-1234-567812345678")
        data = encode(Result(id=1, result=str(uid), error=None))
        assert json.loads(data)["result"] == "12345678-1234-5678-1234-567812345678"

    def test_should_serialise_datetime_as_iso_string(self) -> None:
        dt = datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC)
        data = encode(Result(id=1, result={"ts": dt}, error=None))
        assert json.loads(data)["result"]["ts"] == "2024-03-15T10:30:00+00:00"

    def test_should_serialise_date_as_iso_string(self) -> None:
        data = encode(Result(id=1, result={"d": date(2024, 3, 15)}, error=None))
        assert json.loads(data)["result"]["d"] == "2024-03-15"

    def test_should_serialise_time_as_iso_string(self) -> None:
        data = encode(Result(id=1, result={"t": time(10, 30, 0)}, error=None))
        assert json.loads(data)["result"]["t"] == "10:30:00"

    def test_should_serialise_decimal_as_float(self) -> None:
        data = encode(Result(id=1, result={"v": Decimal("3.14")}, error=None))
        assert json.loads(data)["result"]["v"] == pytest.approx(3.14)

    def test_should_raise_for_unhandled_type(self) -> None:
        with pytest.raises(TypeError, match="not JSON serializable"):
            encode(Result(id=1, result={"v": object()}, error=None))

    def test_should_serialise_special_float_as_tagged_object(self) -> None:
        data = encode(Result(id=1, result={"v": SpecialFloat(text="NaN")}, error=None))

        def reject_constant(name: str) -> float:
            raise AssertionError(f"encoded non-standard JSON constant: {name}")

        parsed = json.loads(data, parse_constant=reject_constant)
        assert parsed["result"]["v"] == {"text": "NaN", "type": "special_float"}
