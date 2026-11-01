"""SQLite persistence for queue jobs, written only by the running service."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from mammoth.queue.model import ACTIVE_STATUSES, Job, JobNotFoundError, JobSpec

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE jobs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    spec TEXT NOT NULL,
    group_name TEXT,
    run_dir TEXT,
    run_name TEXT,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    submitted_at REAL NOT NULL,
    gpu_indices TEXT NOT NULL DEFAULT '[]',
    reserved_vram INTEGER,
    reserved_ram INTEGER,
    tx_id TEXT,
    launching_at REAL,
    started_at REAL,
    finished_at REAL,
    exit_code INTEGER,
    label TEXT,
    message TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    peak_vram INTEGER,
    peak_ram INTEGER,
    peak_partial INTEGER NOT NULL DEFAULT 0,
    last_sampled_at REAL
);
CREATE INDEX jobs_status ON jobs (status);
CREATE INDEX jobs_group ON jobs (group_name);
CREATE INDEX jobs_run_dir ON jobs (run_dir);
CREATE INDEX jobs_run_name ON jobs (run_name);
"""
_UPDATABLE = frozenset(
    {
        "status",
        "priority",
        "gpu_indices",
        "reserved_vram",
        "reserved_ram",
        "tx_id",
        "launching_at",
        "started_at",
        "finished_at",
        "exit_code",
        "label",
        "message",
        "cancel_requested",
        "peak_vram",
        "peak_ram",
        "peak_partial",
        "last_sampled_at",
    }
)


class JobStore:
    """Own one state database; callers serialize access through the service lock."""

    def __init__(self, path: Path) -> None:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(path, 0o600)
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA synchronous=FULL")
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._connection.executescript(
                f"BEGIN; {_SCHEMA} PRAGMA user_version={_SCHEMA_VERSION}; COMMIT;"
            )
        elif version != _SCHEMA_VERSION:
            self._connection.close()
            raise sqlite3.DatabaseError(f"Unsupported queue state schema: {version}")

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def insert(self, job_id: str, spec: JobSpec, submitted_at: float) -> Job:
        """Durably add one pending job."""
        self._connection.execute(
            "INSERT INTO jobs (job_id, spec, group_name, run_dir, run_name, status, priority, "
            "submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                json.dumps(spec.to_dict()),
                spec.group,
                spec.run_dir,
                os.path.basename(spec.run_dir) if spec.run_dir else None,
                "pending",
                spec.priority,
                submitted_at,
            ),
        )
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        """Return one job, or raise when it does not exist."""
        row = self._connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def update(self, job_id: str, **fields: Any) -> Job:
        """Apply scheduler-owned column changes atomically and return the new record."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Not updatable: {sorted(unknown)}")
        if "gpu_indices" in fields:
            fields["gpu_indices"] = json.dumps(list(fields["gpu_indices"]))
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self._connection.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*fields.values(), job_id)
        )
        return self.get(job_id)

    def active(self) -> list[Job]:
        """Return every pending, launching, and running job in submission order."""
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        rows = self._connection.execute(
            f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY sequence",
            tuple(sorted(ACTIVE_STATUSES)),
        )
        return [_job_from_row(row) for row in rows]

    def recent(self, limit: int) -> list[Job]:
        """Return active jobs plus the newest terminal jobs, newest last."""
        marks = ",".join("?" for _ in ACTIVE_STATUSES)
        rows = self._connection.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({marks}) ORDER BY sequence DESC LIMIT ?",
            (*sorted(ACTIVE_STATUSES), limit),
        ).fetchall()
        return sorted(
            [*self.active(), *(_job_from_row(row) for row in rows)], key=lambda job: job.sequence
        )

    def group(self, group: str) -> list[Job]:
        """Return every job in one group in submission order."""
        rows = self._connection.execute(
            "SELECT * FROM jobs WHERE group_name = ? ORDER BY sequence", (group,)
        )
        return [_job_from_row(row) for row in rows]

    def matching(self, reference: str) -> list[Job]:
        """Return jobs a ``like`` reference names, newest first.

        A reference is a job ID, an absolute run directory, or a run directory's
        final path component.
        """
        rows = self._connection.execute(
            "SELECT * FROM jobs WHERE job_id = ? OR run_dir = ? OR run_name = ? "
            "ORDER BY sequence DESC",
            (reference, reference, reference),
        )
        return [_job_from_row(row) for row in rows]

    def run_dirs(self) -> list[str]:
        """Return every run directory any job ever named, kept after the job ends."""
        rows = self._connection.execute(
            "SELECT DISTINCT run_dir FROM jobs WHERE run_dir IS NOT NULL ORDER BY run_dir"
        )
        return [row[0] for row in rows]

    def by_run_dir(self, run_dir: str) -> list[Job]:
        """Return every job for one run directory, newest first."""
        rows = self._connection.execute(
            "SELECT * FROM jobs WHERE run_dir = ? ORDER BY sequence DESC", (run_dir,)
        )
        return [_job_from_row(row) for row in rows]


def _job_from_row(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        sequence=row["sequence"],
        spec=JobSpec.from_dict(json.loads(row["spec"])),
        status=row["status"],
        priority=row["priority"],
        submitted_at=row["submitted_at"],
        gpu_indices=tuple(json.loads(row["gpu_indices"])),
        reserved_vram=row["reserved_vram"],
        reserved_ram=row["reserved_ram"],
        tx_id=row["tx_id"],
        launching_at=row["launching_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        label=row["label"],
        message=row["message"],
        cancel_requested=bool(row["cancel_requested"]),
        peak_vram=row["peak_vram"],
        peak_ram=row["peak_ram"],
        peak_partial=bool(row["peak_partial"]),
        last_sampled_at=row["last_sampled_at"],
    )
