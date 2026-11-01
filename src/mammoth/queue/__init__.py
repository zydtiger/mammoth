"""Host-wide GPU job queue: a scheduler service, its client, and durable execution."""

from __future__ import annotations

from mammoth.queue.api import QueueApiServer, QueueClient, ServiceUnavailableError, job_document
from mammoth.queue.daemon import ServiceAlreadyRunningError, run_service
from mammoth.queue.executor import ActlExecutor, ExecutionRecord, Executor, ExecutorError
from mammoth.queue.model import (
    Dependency,
    Job,
    JobNotFoundError,
    JobSpec,
    JobStateError,
    JobStatus,
    JobValidationError,
    QueueError,
    default_socket_path,
    default_state_dir,
    format_size,
    parse_duration,
    parse_size,
)
from mammoth.queue.sampling import (
    ExecutionMemory,
    GpuSample,
    HostSample,
    HostSampler,
    Sampler,
    SamplingError,
)
from mammoth.queue.scheduler import GpuState, Placement, Request, Reservation, schedule
from mammoth.queue.service import QueueService, ServiceConfig
from mammoth.queue.store import JobStore

__all__ = [
    "ActlExecutor",
    "Dependency",
    "ExecutionMemory",
    "ExecutionRecord",
    "Executor",
    "ExecutorError",
    "GpuSample",
    "GpuState",
    "HostSample",
    "HostSampler",
    "Job",
    "JobNotFoundError",
    "JobSpec",
    "JobStateError",
    "JobStatus",
    "JobStore",
    "JobValidationError",
    "Placement",
    "QueueApiServer",
    "QueueClient",
    "QueueError",
    "QueueService",
    "Request",
    "Reservation",
    "Sampler",
    "SamplingError",
    "ServiceAlreadyRunningError",
    "ServiceConfig",
    "ServiceUnavailableError",
    "default_socket_path",
    "default_state_dir",
    "format_size",
    "job_document",
    "parse_duration",
    "parse_size",
    "run_service",
    "schedule",
]
