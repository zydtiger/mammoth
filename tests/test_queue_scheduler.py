"""Exercise pure queue admission decisions."""

from __future__ import annotations

from mammoth.queue.scheduler import (
    GpuState,
    Request,
    Reservation,
    gpu_accounting,
    ram_available,
    schedule,
)

G = 1024**3
BIG = GpuState(0, 95 * G, 0)
SMALL = GpuState(1, 31 * G, 0)
PLENTY = 500 * G


def request(job_id: str, *, gpus: int = 1, pinned=(), vram=None, ram=None) -> Request:
    return Request(job_id, gpus, tuple(pinned), vram, ram)


def holding(job_id: str, gpus, *, vram=None, ram=10 * G, observed=None, anon=None, stable=False):
    return Reservation(job_id, tuple(gpus), vram, ram, observed or {}, anon, stable)


def run(requests, gpus=(BIG, SMALL), held=(), available=PLENTY, headroom=0):
    return schedule(requests, gpus, available, headroom, held)


def test_unknown_vram_takes_the_gpu_with_the_most_free_memory_exclusively() -> None:
    decision = run([request("a", ram=G), request("b", ram=G), request("c", ram=G)])
    assert [item.gpu_indices for item in decision.placements] == [(0,), (1,)]
    assert decision.reasons["c"] == "waiting: GPU0 is held for earlier job c" or decision.reasons[
        "c"
    ].startswith("waiting: needs a whole GPU")


def test_two_unknown_jobs_never_share_even_with_vast_free_memory() -> None:
    decision = run([request("a", ram=G), request("b", ram=G)], gpus=(BIG,))
    assert [item.job_id for item in decision.placements] == ["a"]
    assert "b" in decision.reasons


def test_known_reservations_share_only_when_the_formula_allows() -> None:
    gpu = GpuState(0, 40 * G, 0)
    decision = run(
        [request("a", vram=20 * G, ram=G), request("b", vram=15 * G, ram=G)],
        gpus=(gpu,),
    )
    assert [item.job_id for item in decision.placements] == ["a", "b"]
    decision = run(
        [request("a", vram=20 * G, ram=G), request("b", vram=25 * G, ram=G)],
        gpus=(gpu,),
    )
    assert [item.job_id for item in decision.placements] == ["a"]
    assert decision.reasons["b"] == "waiting: GPU0 needs 25.0G, has 20.0G"


def test_known_reservation_prefers_the_tightest_gpu_that_fits() -> None:
    decision = run([request("a", vram=20 * G, ram=G)])
    assert decision.placements[0].gpu_indices == (1,)


def test_known_reservation_cannot_join_an_exclusive_gpu() -> None:
    held = [holding("x", [0], vram=None, observed={0: G})]
    decision = run([request("a", pinned=[0], vram=G, ram=G)], gpus=(BIG,), held=held)
    assert not decision.placements
    assert decision.reasons["a"] == "waiting: GPU0 needs 1.0G, has 0B"


def test_reservation_holds_before_memory_grows_and_external_tenants_reduce_free() -> None:
    # Only 48 G of the 95 G GPU is external use; the just-launched job has observed nothing.
    gpu = GpuState(0, 95 * G, 48 * G)
    held = [holding("x", [0], vram=30 * G, observed={})]
    state = gpu_accounting([gpu], held)[0]
    assert state.external == 48 * G
    assert state.free == 17 * G
    decision = run([request("a", vram=20 * G, ram=G)], gpus=(gpu,), held=held)
    assert not decision.placements


def test_observed_use_above_a_declaration_raises_the_reservation() -> None:
    gpu = GpuState(0, 95 * G, 40 * G)
    held = [holding("x", [0], vram=10 * G, observed={0: 40 * G})]
    state = gpu_accounting([gpu], held)[0]
    assert state.external == 0
    assert state.free == 55 * G


def test_ram_formula_counts_only_the_unreached_part_of_a_reservation() -> None:
    held = [holding("x", [], ram=40 * G, anon=30 * G)]
    assert ram_available(100 * G, 10 * G, held) == (80 * G, None)


def test_ram_shortfall_blocks_the_head_and_everything_behind_it() -> None:
    decision = run(
        [request("a", ram=90 * G), request("b", gpus=0, ram=G)],
        available=80 * G,
        headroom=8 * G,
    )
    assert not decision.placements
    assert decision.reasons["a"] == "waiting: RAM needs 90.0G, has 72.0G"
    assert decision.reasons["b"] == "waiting: behind a, which is blocked on RAM"


def test_unknown_ram_jobs_launch_one_at_a_time_until_stable() -> None:
    decision = run([request("a"), request("b")])
    assert [item.job_id for item in decision.placements] == ["a"]
    assert decision.reasons["b"] == "waiting: RAM of a is still growing"
    held = [holding("x", [], ram=None, stable=False)]
    assert "a" in run([request("a")], held=held).reasons
    held = [holding("x", [], ram=None, stable=True)]
    assert run([request("a")], held=held).placements


def test_backfill_passes_a_blocked_head_only_on_a_gpu_it_can_never_use() -> None:
    # The head needs 60 G, which only GPU0 can ever hold, and GPU0 is busy.
    held = [holding("x", [0], vram=50 * G, ram=G)]
    decision = run(
        [
            request("head", vram=60 * G, ram=G),
            request("small", vram=10 * G, ram=G),
            request("onbig", pinned=[0], vram=G, ram=G),
        ],
        held=held,
    )
    assert [(item.job_id, item.gpu_indices) for item in decision.placements] == [("small", (1,))]
    assert decision.reasons["onbig"] == "waiting: GPU0 is held for earlier job head"


def test_backfill_cannot_use_any_gpu_the_head_could_use() -> None:
    held = [holding("x", [0], ram=G), holding("y", [1], ram=G)]
    decision = run([request("head", ram=G), request("later", vram=G, ram=G)], held=held)
    assert not decision.placements


def test_later_job_cannot_spend_ram_a_gpu_blocked_head_needs() -> None:
    held = [holding("x", [0, 1], ram=G)]
    decision = run(
        [request("head", ram=30 * G), request("cpu", gpus=0, ram=25 * G)],
        held=held,
        available=50 * G,
    )
    assert not decision.placements
    assert decision.reasons["cpu"] == "waiting: RAM needs 25.0G, has 19.0G"


def test_a_head_that_no_gpu_can_hold_does_not_stall_the_queue() -> None:
    gpu = GpuState(0, 95 * G, 48 * G)
    decision = run(
        [request("huge", vram=60 * G, ram=G), request("fine", vram=G, ram=G)],
        gpus=(gpu,),
    )
    assert [item.job_id for item in decision.placements] == ["fine"]
    assert "cannot fit" in decision.reasons["huge"]


def test_a_ram_request_the_host_can_never_admit_does_not_stall_the_queue() -> None:
    decision = schedule(
        [request("huge", gpus=0, ram=900 * G), request("fine", gpus=0, ram=G)],
        (BIG,),
        PLENTY,
        0,
        (),
        ram_capacity=100 * G,
    )
    assert [item.job_id for item in decision.placements] == ["fine"]
    assert "host can admit at most 100.0G" in decision.reasons["huge"]


def test_multi_gpu_and_pinned_placement() -> None:
    decision = run([request("a", gpus=2, ram=G)])
    assert decision.placements[0].gpu_indices == (0, 1)
    decision = run([request("a", pinned=[1], ram=G)])
    assert decision.placements[0].gpu_indices == (1,)


def test_cpu_only_job_ignores_gpus() -> None:
    held = [holding("x", [0, 1], ram=G)]
    decision = run([request("a", gpus=0, ram=G)], held=held)
    assert decision.placements[0].gpu_indices == ()
