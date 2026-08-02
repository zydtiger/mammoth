"""Framework-neutral run identity, artifact, provenance, and event APIs.

Higher Mammoth layers depend on this package; it imports only the Python
standard library and never imports model, dataset, or project code.
"""

from __future__ import annotations

from mammoth.core.artifacts import (
    atomic_publish,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from mammoth.core.events import (
    ExecutionEvent,
    ExecutionEventReadError,
    ExecutionEventTailReader,
    ExecutionEventWriter,
    iter_execution_events,
    read_execution_events,
)
from mammoth.core.execution import (
    ExecutionContext,
    ExecutionMetadata,
    LogicalRunLease,
    claim_logical_run_lease,
    create_execution_context,
    execution_id_from_environment,
    generate_execution_id,
    join_execution_context,
    latest_execution_id,
    sanitize_command,
    sanitize_metadata_fields,
    sanitize_reference,
)
from mammoth.core.identity import validate_execution_id, validate_run_name
from mammoth.core.layout import RunLayout

__all__ = [
    "ExecutionContext",
    "ExecutionEvent",
    "ExecutionEventReadError",
    "ExecutionEventTailReader",
    "ExecutionEventWriter",
    "ExecutionMetadata",
    "LogicalRunLease",
    "RunLayout",
    "atomic_publish",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "claim_logical_run_lease",
    "create_execution_context",
    "execution_id_from_environment",
    "generate_execution_id",
    "iter_execution_events",
    "join_execution_context",
    "latest_execution_id",
    "read_execution_events",
    "sanitize_command",
    "sanitize_metadata_fields",
    "sanitize_reference",
    "validate_execution_id",
    "validate_run_name",
]
