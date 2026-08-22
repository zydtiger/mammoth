"""Local device-aware job queue: opaque spooled jobs and a FIFO device runner."""

from __future__ import annotations

from mammoth.queue.serve import (
    DeviceLease,
    DeviceLeaseConflictError,
    claim_device_lease,
    reconcile_interrupted_jobs,
    run_serve_loop,
    serve_once,
)
from mammoth.queue.spool import (
    QUEUE_JOURNAL_SCHEMA_VERSION,
    QUEUE_SCHEMA_VERSION,
    Job,
    JobAlreadyClaimedError,
    JobNotFoundError,
    JobOutcome,
    JobStatus,
    QueueError,
    QueueSnapshot,
    append_job_outcome,
    cancel_job,
    job_filename,
    journal_job_ids,
    list_jobs,
    load_job,
    read_job_journal,
    submit_job,
)

__all__ = [
    "QUEUE_JOURNAL_SCHEMA_VERSION",
    "QUEUE_SCHEMA_VERSION",
    "DeviceLease",
    "DeviceLeaseConflictError",
    "Job",
    "JobAlreadyClaimedError",
    "JobNotFoundError",
    "JobOutcome",
    "JobStatus",
    "QueueError",
    "QueueSnapshot",
    "append_job_outcome",
    "cancel_job",
    "claim_device_lease",
    "job_filename",
    "journal_job_ids",
    "list_jobs",
    "load_job",
    "read_job_journal",
    "reconcile_interrupted_jobs",
    "run_serve_loop",
    "serve_once",
    "submit_job",
]
