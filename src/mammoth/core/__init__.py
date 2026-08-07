"""Framework-neutral run identity, artifact, provenance, and event APIs.

Higher Mammoth layers depend on this package; it imports only the Python
standard library and never imports model, dataset, or project code.
"""

from __future__ import annotations

from mammoth.core.artifacts import (
    PreparedArtifact,
    atomic_publish,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    discard_prepared_artifact,
    prepare_artifact,
    publish_prepared_artifact,
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
    normalize_execution_id_environment_aliases,
    sanitize_command,
    sanitize_metadata_fields,
    sanitize_reference,
)
from mammoth.core.identity import validate_execution_id, validate_run_name
from mammoth.core.layout import RunLayout
from mammoth.core.pipeline import (
    BackgroundPipelineError,
    BackgroundPipelineResult,
    BackgroundPipelineSubmission,
    BoundedBackgroundPipeline,
)

__all__ = [
    "BackgroundPipelineError",
    "BackgroundPipelineResult",
    "BackgroundPipelineSubmission",
    "BoundedBackgroundPipeline",
    "ExecutionContext",
    "ExecutionEvent",
    "ExecutionEventReadError",
    "ExecutionEventTailReader",
    "ExecutionEventWriter",
    "ExecutionMetadata",
    "LogicalRunLease",
    "PreparedArtifact",
    "RunLayout",
    "atomic_publish",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "claim_logical_run_lease",
    "create_execution_context",
    "discard_prepared_artifact",
    "execution_id_from_environment",
    "generate_execution_id",
    "iter_execution_events",
    "join_execution_context",
    "latest_execution_id",
    "normalize_execution_id_environment_aliases",
    "prepare_artifact",
    "publish_prepared_artifact",
    "read_execution_events",
    "sanitize_command",
    "sanitize_metadata_fields",
    "sanitize_reference",
    "validate_execution_id",
    "validate_run_name",
]
