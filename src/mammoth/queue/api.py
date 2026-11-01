"""Versioned JSON over HTTP on an owner-only Unix socket, and its client.

Submitting a job executes a command, so no TCP listener may reach this API. The
filesystem permissions on the socket are its only authentication.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Union
from urllib.parse import parse_qs, urlencode, urlsplit

from mammoth.queue.model import (
    Job,
    JobNotFoundError,
    JobStateError,
    JobValidationError,
    QueueError,
    format_size,
)
from mammoth.queue.service import MAX_WAIT_SECONDS, QueueService

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 4 * 1024 * 1024
_CLIENT_TIMEOUT_SECONDS = 30.0


class ServiceUnavailableError(QueueError):
    """Raised when no queue service answers on the socket."""


def job_document(job: Job, wait_reason: Union[str, None] = None) -> dict[str, Any]:
    """Return one job for clients; environment values stay in the service."""
    warnings = []
    if job.spec.vram is not None and (job.peak_vram or 0) > job.spec.vram:
        warnings.append(
            f"VRAM peak {format_size(job.peak_vram)} exceeds declared {format_size(job.spec.vram)}"
        )
    if job.spec.ram is not None and (job.peak_ram or 0) > job.spec.ram:
        warnings.append(
            f"RAM peak {format_size(job.peak_ram)} exceeds declared {format_size(job.spec.ram)}"
        )
    return {
        "job_id": job.job_id,
        "sequence": job.sequence,
        "status": job.status,
        "priority": job.priority,
        "argv": list(job.spec.argv),
        "cwd": job.spec.cwd,
        "env_keys": sorted(job.spec.env),
        "gpus": job.spec.gpus,
        "pinned_gpus": list(job.spec.gpu),
        "group": job.spec.group,
        "run_dir": job.spec.run_dir,
        "after": [{"job_id": item.job_id, "mode": item.mode} for item in job.spec.after],
        "declared_vram": job.spec.vram,
        "declared_ram": job.spec.ram,
        "gpu_indices": list(job.gpu_indices),
        "reserved_vram": job.reserved_vram,
        "reserved_ram": job.reserved_ram,
        "peak_vram": job.peak_vram,
        "peak_ram": job.peak_ram,
        "peak_partial": job.peak_partial,
        "tx_id": job.tx_id,
        "submitted_at": job.submitted_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "label": job.label,
        "message": job.message,
        "cancel_requested": job.cancel_requested,
        "wait_reason": wait_reason,
        "warnings": warnings,
    }


class _Handler(BaseHTTPRequestHandler):
    server: QueueApiServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """Route one read."""
        self._dispatch(self._read)

    def do_POST(self) -> None:
        """Route one write."""
        self._dispatch(self._write)

    def _dispatch(self, route: Any) -> None:
        try:
            status, document = route()
        except JobValidationError as error:
            status, document = 400, {"error": str(error), "kind": "validation"}
        except JobNotFoundError as error:
            status, document = 404, {"error": str(error), "kind": "not_found"}
        except JobStateError as error:
            status, document = 409, {"error": str(error), "kind": "state"}
        except Exception:
            logger.exception("Could not answer %s %s", self.command, self.path)
            status, document = 500, {"error": "internal error", "kind": "internal"}
        body = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> tuple[int, dict[str, Any]]:
        service = self.server.service
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        if url.path == "/v1/jobs":
            return 200, {"jobs": [self._document(job) for job in service.list_jobs()]}
        if url.path == "/v1/resources":
            return 200, service.resources()
        if url.path == "/v1/wait":
            query = parse_qs(url.query)
            done, jobs = service.wait(
                query.get("job", []),
                query.get("group", [None])[0],
                _seconds(query.get("timeout", [str(MAX_WAIT_SECONDS)])[0]),
            )
            return 200, {"done": done, "jobs": [self._document(job) for job in jobs]}
        if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
            return 200, self._document(service.get(parts[2]))
        return 404, {"error": "unknown route", "kind": "route"}

    def _write(self) -> tuple[int, dict[str, Any]]:
        service = self.server.service
        parts = urlsplit(self.path).path.strip("/").split("/")
        body = self._body()
        if parts == ["v1", "jobs"]:
            return 201, self._document(service.submit(body))
        if len(parts) == 4 and parts[:2] == ["v1", "jobs"]:
            if parts[3] == "cancel":
                return 200, self._document(service.cancel(parts[2]))
            if parts[3] == "priority":
                return 200, self._document(service.set_priority(parts[2], body.get("priority")))
        return 404, {"error": "unknown route", "kind": "route"}

    def _document(self, job: Job) -> dict[str, Any]:
        return job_document(job, self.server.service.wait_reason(job.job_id))

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= _MAX_BODY_BYTES:
            self.close_connection = True
            raise JobValidationError("Request body length is invalid or too large")
        try:
            document = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as error:
            raise JobValidationError(f"Request body is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise JobValidationError("Request body must be a JSON object")
        return document

    def log_message(self, format: str, *args: Any) -> None:
        """Keep routine requests out of the service log."""
        logger.debug(format, *args)


class QueueApiServer(ThreadingHTTPServer):
    """Serve one :class:`QueueService` on an owner-only Unix socket."""

    daemon_threads = True

    def __init__(self, path: Path, service: QueueService) -> None:
        if sys.platform == "win32":
            raise QueueError("The queue service requires Linux")
        self.address_family = socket.AF_UNIX
        self.service = service
        self.socket_path = path
        if path.is_socket():
            # The caller holds the state lock, so this socket has no live owner.
            path.unlink()
        previous = os.umask(0o177)
        try:
            super().__init__(str(path), _Handler)  # type: ignore[arg-type]
        finally:
            os.umask(previous)

    def server_bind(self) -> None:
        """Bind the path; a Unix socket has no host name to resolve."""
        socketserver.TCPServer.server_bind(self)
        self.server_name = "queue"
        self.server_port = 0

    def serve_in_thread(self) -> threading.Thread:
        """Accept requests on a daemon thread."""
        thread = threading.Thread(target=self.serve_forever, name="mammoth-queue-api")
        thread.daemon = True
        thread.start()
        return thread

    def close(self) -> None:
        """Stop accepting and remove the socket file."""
        self.shutdown()
        self.server_close()
        if self.socket_path.is_socket():
            self.socket_path.unlink()


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float) -> None:
        super().__init__("queue", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        if sys.platform == "win32":
            raise OSError("The queue service requires Linux")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))


class QueueClient:
    """Talk to the running service, surfacing a down service as a clear error."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit one job; nothing is kept when the service is down."""
        return self._request("POST", "/v1/jobs", payload)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return active and recent terminal jobs."""
        jobs: list[dict[str, Any]] = self._request("GET", "/v1/jobs")["jobs"]
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return one job."""
        return self._request("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel one pending or running job."""
        return self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    def set_priority(self, job_id: str, priority: Any) -> dict[str, Any]:
        """Change one pending job's priority."""
        return self._request("POST", f"/v1/jobs/{job_id}/priority", {"priority": priority})

    def resources(self) -> dict[str, Any]:
        """Return the service's resource accounting."""
        return self._request("GET", "/v1/resources")

    def wait_once(
        self,
        job_ids: Sequence[str],
        group: Union[str, None],
        timeout_seconds: float = MAX_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Block on the socket for one bounded wait."""
        query: list[tuple[str, Any]] = [("job", job_id) for job_id in job_ids]
        if group is not None:
            query.append(("group", group))
        query.append(("timeout", timeout_seconds))
        return self._request(
            "GET",
            f"/v1/wait?{urlencode(query)}",
            timeout=timeout_seconds + _CLIENT_TIMEOUT_SECONDS,
        )

    def wait(
        self,
        job_ids: Sequence[str],
        group: Union[str, None],
        *,
        reconnect_seconds: float,
        retry_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Block until the jobs end, reconnecting across service restarts."""
        unreachable_since = None
        while True:
            try:
                result = self.wait_once(job_ids, group)
            except ServiceUnavailableError as error:
                now = time.monotonic()
                unreachable_since = now if unreachable_since is None else unreachable_since
                if now - unreachable_since >= reconnect_seconds:
                    raise ServiceUnavailableError(
                        f"Gave up after {reconnect_seconds:g}s; job state is unknown: {error}"
                    ) from error
                time.sleep(retry_seconds)
                continue
            unreachable_since = None
            if result["done"]:
                jobs: list[dict[str, Any]] = result["jobs"]
                return jobs

    def _request(
        self,
        method: str,
        path: str,
        body: Union[dict[str, Any], None] = None,
        *,
        timeout: float = _CLIENT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        connection = _UnixConnection(self.socket_path, timeout)
        try:
            payload = None if body is None else json.dumps(body).encode("utf-8")
            headers = {"Content-Type": "application/json"} if payload is not None else {}
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            document: dict[str, Any] = json.loads(response.read())
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise ServiceUnavailableError(
                f"Queue service is not answering on {self.socket_path}: {error}. "
                "Start it with `systemctl --user start mammoth-queue`."
            ) from error
        finally:
            connection.close()
        if response.status < 300:
            return document
        message = document.get("error", f"HTTP {response.status}")
        error_type = {
            "validation": JobValidationError,
            "not_found": JobNotFoundError,
            "state": JobStateError,
        }.get(document.get("kind", ""), QueueError)
        raise error_type(message)


def _seconds(text: str) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise JobValidationError(f"Invalid timeout: {text!r}") from error
