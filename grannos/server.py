import asyncio
import json
import logging
import pathlib
import re
import sys
from typing import Any, BinaryIO

from . import log
from .dispatcher import Dispatcher, DispatchError
from .drivers import SENSITIVE_PARAM_KEYS
from .drivers.base import DriverError, DriverSettings
from .log import truncate
from .protocol import (
    DecodeError,
    Method,
    Progress,
    ProgressDetail,
    Request,
    Response,
    Result,
    decode,
    encode,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
"""Default cap on a single request line. Well above asyncio's 64 KiB default,
which one long SQL statement can exceed, and well above any hand-written query;
a line past the cap is dropped rather than buffered without bound. Raise it with
``--max-request-bytes`` if a client legitimately sends more."""

_ID_HEAD_BYTES = 256
"""How much of an oversized line is kept to recover its request id. The id is the
first key a client writes, so the head is enough — and a client that writes it
later just gets an id-less error."""

_ID_HEAD_RE = re.compile(rb'\s*\{\s*"id"\s*:\s*(\d+)')
"""Matches the id only as the object's first key, so an 'id' appearing later —
inside the query text, say — can never be reported as some other request's id."""


class Server:
    """Stdio JSON server — reads requests from stdin, writes responses to stdout.

    Args:
        cache_dir: Directory for persisting explore caches between sessions.
        max_concurrency: Maximum concurrent requests allowed per connection.
        max_request_bytes: Largest single request line accepted; longer lines are
            discarded with an error rather than buffered.
        column_sample_size: Number of distinct non-null values sampled per column in describe results.
        stdin: Binary stream to read requests from (typically ``sys.stdin.buffer``).
        stdout: Binary stream to write responses to (typically ``sys.stdout.buffer``).
    """

    def __init__(
        self,
        cache_dir: pathlib.Path,
        driver_settings: DriverSettings,
        max_concurrency: int = 1,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        stdin: BinaryIO = sys.stdin.buffer,
        stdout: BinaryIO = sys.stdout.buffer,
    ) -> None:
        self._dispatcher = Dispatcher(
            max_concurrency=max_concurrency,
            cache_dir=cache_dir,
            driver_settings=driver_settings,
        )
        self._out = stdout
        self._stdin = stdin
        self._max_request_bytes = max_request_bytes
        self._lock = asyncio.Lock()
        """Serializes concurrent response writes to stdout."""
        self._tasks: dict[int, asyncio.Task[None]] = {}
        """Maps request id to its in-flight task, enabling cancellation by id."""

    async def run(self) -> None:
        """Start the read loop.

        Reads newline-delimited JSON requests from stdin and dispatches each as
        a concurrent task. Returns when stdin is closed.
        """
        reader = asyncio.StreamReader(limit=self._max_request_bytes)
        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            self._stdin,
        )
        logger.info("Server ready")
        while True:
            try:
                line = await _read_line(reader)
            except _OversizedRequest as e:
                logger.warning(
                    f"Discarded oversized request {e.request_id}: over {self._max_request_bytes} bytes"
                )
                asyncio.create_task(
                    self._send(
                        Result(
                            id=e.request_id,
                            result=None,
                            error=(
                                f"Request is too large: it exceeds the "
                                f"{_human_size(self._max_request_bytes)} limit on a single "
                                f"request (raise it with --max-request-bytes)"
                            ),
                        )
                    )
                )
                continue
            if not line:
                logger.info("Server exiting")
                break
            try:
                req: Request = decode(line)
                logger.debug(
                    f"Received {truncate(json.dumps({'id': req.id, 'method': req.method, 'params': _redact(req.params)}))}"
                )
                if req.method == Method.CANCEL:
                    asyncio.create_task(self._handle_cancel(req))
                else:
                    task = asyncio.create_task(self._handle(req))
                    self._tasks[req.id] = task
                    task.add_done_callback(lambda _: self._tasks.pop(req.id, None))
            except DecodeError as e:
                logger.warning(f"Received invalid request: {e}")
                asyncio.create_task(
                    self._send(Result(id=e.id, result=None, error=str(e)))
                )

    async def _handle(self, req: Request) -> None:
        log.request_id_var.set(req.id)

        async def send_progress(status: str, message: str) -> None:
            await self._send(
                Progress(
                    id=req.id,
                    progress=ProgressDetail(status=status, message=message),
                )
            )

        response: Result
        try:
            result = await self._dispatcher.dispatch(
                req.method, req.params, send_progress
            )
            response = Result(id=req.id, result=result, error=None)
        except (DispatchError, DriverError) as exc:
            response = Result(id=req.id, result=None, error=str(exc))
        except asyncio.CancelledError:
            response = Result(id=req.id, result=None, error="cancelled")
        except Exception:
            logger.exception(f"Unhandled error for request {req.id}")
            response = Result(id=req.id, result=None, error="internal error")
        await self._send(response)

    async def _handle_cancel(self, req: Request) -> None:
        target_id = req.params.get("request_id")
        if not isinstance(target_id, int):
            await self._send(
                Result(
                    id=req.id, result=None, error="Missing required param: 'request_id'"
                )
            )
            return
        task = self._tasks.get(target_id)
        if task is not None:
            task.cancel()
        await self._send(Result(id=req.id, result={"ok": True}, error=None))

    async def _send(self, msg: Response) -> None:
        data = encode(msg)
        async with self._lock:
            self._out.write(data)
            self._out.flush()
            logger.debug(f"Sent {truncate(data.decode(errors='replace').rstrip())}")


class _OversizedRequest(Exception):
    """Raised when a request line is longer than the reader will buffer."""

    def __init__(self, request_id: int | None) -> None:
        super().__init__("Request exceeds the maximum line length")
        self.request_id = request_id
        """Id read off the head of the line, or None if it was not the first key."""


async def _read_line(reader: asyncio.StreamReader) -> bytes:
    """Read one newline-terminated request line, or b"" at EOF.

    Raises:
        _OversizedRequest: The line is longer than the reader's limit. The whole
            line, tail included, is consumed first, so the next call starts on
            the following request and one error is reported rather than two.
    """
    try:
        return await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as e:
        return e.partial
    except asyncio.LimitOverrunError as e:
        head = await reader.readexactly(min(e.consumed, _ID_HEAD_BYTES))
        await _skip_line(reader)
        match = _ID_HEAD_RE.match(head)
        raise _OversizedRequest(int(match.group(1)) if match else None) from None


async def _skip_line(reader: asyncio.StreamReader) -> None:
    """Consume and discard bytes up to and including the next newline, or to EOF."""
    while True:
        try:
            await reader.readuntil(b"\n")
            return
        except asyncio.LimitOverrunError as e:
            # `consumed` stops at the separator when there is one, so this never
            # eats into the request that follows.
            await reader.readexactly(e.consumed)
        except asyncio.IncompleteReadError:
            return


def _human_size(num_bytes: int) -> str:
    """Render a byte count as the largest unit it divides into evenly."""
    for unit, size in (("MiB", 1024 * 1024), ("KiB", 1024)):
        if num_bytes >= size and num_bytes % size == 0:
            return f"{num_bytes // size} {unit}"
    return f"{num_bytes} bytes"


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    return {k: "***" if k in SENSITIVE_PARAM_KEYS else v for k, v in params.items()}
