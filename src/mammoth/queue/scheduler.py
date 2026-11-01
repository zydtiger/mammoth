"""Pure admission decisions: which ready jobs start now, where, and why others wait.

A reservation holds from launch, so a job still loading cannot make its GPU look
free. A job with no known VRAM holds its whole GPU. Later jobs may pass a blocked
job only on GPUs that job could never use, and never while it is blocked on RAM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union

from mammoth.compat import DATACLASS_SLOTS
from mammoth.queue.model import format_size


@dataclass(frozen=True, **DATACLASS_SLOTS)
class GpuState:
    """One sampled GPU, in bytes."""

    index: int
    total: int
    used: int


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Reservation:
    """What one launching or running job holds, with its latest observed use."""

    job_id: str
    gpu_indices: tuple[int, ...]
    vram: Union[int, None]
    ram: Union[int, None]
    observed_vram: Mapping[int, int] = field(default_factory=dict)
    anon_ram: Union[int, None] = None
    ram_stable: bool = False


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Request:
    """One ready job with its effective reservations, in queue order."""

    job_id: str
    gpus: int
    pinned: tuple[int, ...]
    vram: Union[int, None]
    ram: Union[int, None]


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Placement:
    """One admitted job and what it reserves."""

    job_id: str
    gpu_indices: tuple[int, ...]
    vram: Union[int, None]
    ram: Union[int, None]


@dataclass(frozen=True, **DATACLASS_SLOTS)
class GpuFree:
    """One GPU's memory accounting after reservations."""

    index: int
    total: int
    external: int
    free: int
    capacity: int
    exclusive: bool
    jobs: tuple[str, ...]


@dataclass(frozen=True, **DATACLASS_SLOTS)
class Decision:
    """The outcome of one scheduling pass."""

    placements: tuple[Placement, ...]
    reasons: Mapping[str, str]


def gpu_accounting(gpus: Sequence[GpuState], held: Sequence[Reservation]) -> list[GpuFree]:
    """Apply ``free = total - external - sum(max(declared, observed))`` to each GPU."""
    accounting = []
    for gpu in gpus:
        members = [item for item in held if gpu.index in item.gpu_indices]
        observed = sum(item.observed_vram.get(gpu.index, 0) for item in held)
        external = max(0, gpu.used - observed)
        exclusive = any(item.vram is None for item in members)
        reserved = sum(
            max(item.vram or 0, item.observed_vram.get(gpu.index, 0)) for item in members
        )
        capacity = gpu.total - external
        accounting.append(
            GpuFree(
                index=gpu.index,
                total=gpu.total,
                external=external,
                free=0 if exclusive else max(0, capacity - reserved),
                capacity=capacity,
                exclusive=exclusive,
                jobs=tuple(item.job_id for item in members),
            )
        )
    return accounting


def ram_available(
    memory_available: int, headroom: int, held: Sequence[Reservation]
) -> tuple[int, Union[str, None]]:
    """Return admissible RAM, and the job whose unknown use has not yet stabilized."""
    available = memory_available - headroom
    growing = None
    for item in held:
        if item.ram is not None:
            available -= max(0, item.ram - (item.anon_ram or 0))
        elif not item.ram_stable and growing is None:
            growing = item.job_id
    return available, growing


def schedule(
    requests: Sequence[Request],
    gpus: Sequence[GpuState],
    memory_available: int,
    headroom: int,
    held: Sequence[Reservation],
    ram_capacity: Union[int, None] = None,
) -> Decision:
    """Admit jobs in queue order under the rules in this module's docstring."""
    state = {item.index: item for item in gpu_accounting(gpus, held)}
    free = {index: item.free for index, item in state.items()}
    exclusive = {index: item.exclusive for index, item in state.items()}
    occupied = {index: bool(item.jobs) for index, item in state.items()}
    ram_left, growing = ram_available(memory_available, headroom, held)
    held_for: dict[int, str] = {}
    ram_blocker: Union[str, None] = None
    placements: list[Placement] = []
    reasons: dict[str, str] = {}
    for request in requests:
        if ram_blocker is not None:
            reasons[request.job_id] = f"waiting: behind {ram_blocker}, which is blocked on RAM"
            continue
        needed_ram = request.ram or 0
        if ram_capacity is not None and needed_ram > ram_capacity:
            # Nothing finishing can ever admit this, so it must not stall the queue.
            reasons[request.job_id] = (
                f"waiting: RAM needs {format_size(needed_ram)}, "
                f"host can admit at most {format_size(ram_capacity)}"
            )
            continue
        if request.ram is None and growing is not None:
            reasons[request.job_id] = f"waiting: RAM of {growing} is still growing"
            ram_blocker = request.job_id
            continue
        if needed_ram > ram_left:
            reasons[request.job_id] = (
                f"waiting: RAM needs {format_size(needed_ram)}, has {format_size(max(0, ram_left))}"
            )
            ram_blocker = request.job_id
            continue
        if request.gpus == 0:
            chosen: list[int] = []
        else:
            eligible = [
                item
                for item in state.values()
                if (not request.pinned or item.index in request.pinned)
                and (request.vram is None or item.capacity >= request.vram)
            ]
            usable = [
                item.index
                for item in eligible
                if item.index not in held_for
                and (
                    not occupied[item.index]
                    if request.vram is None
                    else not exclusive[item.index] and free[item.index] >= request.vram
                )
            ]
            if request.vram is None:
                usable.sort(key=lambda index: (-free[index], index))
            else:
                usable.sort(key=lambda index: (free[index], index))
            if len(usable) < request.gpus:
                reasons[request.job_id] = _gpu_reason(request, eligible, free, held_for)
                if len(eligible) >= request.gpus:
                    for item in eligible:
                        held_for.setdefault(item.index, request.job_id)
                    ram_left -= needed_ram
                continue
            chosen = usable[: request.gpus]
        for index in chosen:
            occupied[index] = True
            if request.vram is None:
                exclusive[index] = True
                free[index] = 0
            else:
                free[index] -= request.vram
        ram_left -= needed_ram
        if request.ram is None:
            growing = request.job_id
        placements.append(Placement(request.job_id, tuple(chosen), request.vram, request.ram))
    return Decision(tuple(placements), reasons)


def _gpu_reason(
    request: Request,
    eligible: Sequence[GpuFree],
    free: Mapping[int, int],
    held_for: Mapping[int, str],
) -> str:
    if len(eligible) < request.gpus:
        need = format_size(request.vram) if request.vram is not None else "a GPU"
        return f"waiting: {request.gpus} GPU(s) cannot fit {need} beside external use"
    open_gpus = [item for item in eligible if item.index not in held_for]
    if not open_gpus:
        blocker = held_for[eligible[0].index]
        return f"waiting: GPU{eligible[0].index} is held for earlier job {blocker}"
    best = max(open_gpus, key=lambda item: free[item.index])
    if request.vram is None:
        return f"waiting: needs a whole GPU; GPU{best.index} is in use"
    return (
        f"waiting: GPU{best.index} needs {format_size(request.vram)}, "
        f"has {format_size(free[best.index])}"
    )
