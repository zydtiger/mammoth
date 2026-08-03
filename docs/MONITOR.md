# Monitor Contract

This document owns Mammoth's passive-monitor reconstruction, interactive
presentation, telemetry, and interaction behavior. `docs/ARCHITECTURE.md` owns
the package boundaries, artifact roles, runtime concepts, and compatibility
policy that constrain this behavior.

The monitor is project-neutral. Phase names, task identifiers, coordinates,
metric names, and extension fields remain opaque producer data except for the
small presentation conventions stated below.

## Passive State Reconstruction

- Monitoring reads immutable execution metadata and append-only JSONL streams.
  It never writes execution artifacts, contacts producers, or controls work.
- A malformed or temporarily unreadable stream produces a warning while the
  last valid state from that stream and all other streams remain usable.
- Direct process activity is sufficient to infer a running attempt when no
  runner lifecycle exists. When no runner terminal exists, process terminals
  infer completed, failed, or interrupted attempt state.
- A new `process_started` event begins a new producer generation and clears
  stale tasks previously folded for that producer.
- A running producer becomes stale after the configured observation horizon;
  the public CLI default is 90 seconds.

## Attempts, Identity, And Resume State

- The run monitor retains every valid immutable attempt in creation order and
  selects the newest attempt unless an exact execution ID is requested.
- Overview identity uses the final eight characters of an execution ID.
  Exact detail is the only dashboard view that prints the complete immutable
  ID and full lineage/provenance fields.
- `parent_execution_id` defines resume continuity. `previous_execution_id`
  records adjacency but does not make metrics or coordinates continuous.
- Logical run coordinates are folded chronologically across the resolved
  parent lineage. Parent metric samples at or beyond a child's declared
  starting global step, or starting epoch when no global-step coordinate is
  available, are removed before child samples are appended.
- Run progress presents epoch, optimizer step, ETA, and the short source
  execution ID. Global step is retained in reconstructed state but is not
  displayed because optimizer step is the canonical progress coordinate.
- Run ETA uses the median of up to 32 recent positive optimizer-step intervals
  and the remaining optimizer horizon. Until samples exist, it is marked as
  calibrating. Task ETA remains producer-local and uses reported throughput or
  observed completed work.

## Hierarchical And Distributed Progress

- The selected-attempt overview prefers the highest active task ancestor that
  has progress. Rank rows show the deepest active leaf for each producer.
- In the wide rank table, rank is a fixed four-character field and progress has
  protected width for comma-formatted production counters. Progress receives
  expansion space before rank or status fields.
- Distributed progress is reconciled only among matching process-rank tasks.
  Identical counters are treated as replicated global progress and shown once.
  Distinct compatible counters are summed. Missing expected ranks produce an
  aggregation-pending state instead of a partial total.
- Count units are preserved exactly. Throughput abbreviates both `batch/s` and
  `microbatch/s` as `b/s`; one displayed batch represents one model forward
  pass.

## Dashboard Information Hierarchy

The wide dashboard uses this stable order:

1. Run header and refresh state.
2. `RUN PROGRESS`.
3. `TRAINING TRENDS`, when loss or learning-rate history exists.
4. `HOST RESOURCES`, when viewer telemetry is enabled.
5. Selected attempt and its overall progress.
6. Per-rank state and active leaf progress.
7. Attempt history and lineage.
8. Stream warnings and interaction help.

Compact mode preserves the same order but stacks charts and resource blocks,
uses textual overall progress instead of a narrow progress bar, and keeps each
rendered line within the available terminal width. The layout switches below
80 available columns.

An exact execution selection opens in exact detail, cannot navigate to another
attempt, and cannot toggle back to overview. Run-level selection supports arrow
keys or `j`/`k`, Enter toggles exact detail, `r` refreshes, and `q` quits.

## Training Charts

- The interactive dashboard renders at most one conventional loss history and
  one learning-rate history. A loss name is `loss` or ends in `_loss`; a
  learning-rate name is `learning_rate` or `lr`.
- Other folded metrics remain available to programmatic consumers and the
  stable plain snapshot, but do not appear in the interactive dashboard.
- Loss and learning rate use axis-free Braille line charts. Wide mode renders
  them side by side at four terminal rows; compact mode stacks them at three
  rows. Each chart includes latest value, range, and sample count.
- The dashboard does not add a secondary metric/latest/range/trend table.

## Viewer-Host Resources

Telemetry describes the machine running the monitor, not the machine that ran
the workload. It is live viewer state, is cached only in memory, and is never
written to execution artifacts.

`HOST RESOURCES · <hostname>` renders resources in CPU, RAM, then ascending GPU
index order. Every resource occupies exactly two rows: a full-width identity
row followed by edge-aligned live metrics.

- CPU identity includes the processor model when available. Its metric row
  shows aggregate utilization, zenpower package power, and average current
  reporting-core frequency. CPU power is read only from `sensors -j
  zenpower-*`; other CPU power drivers are outside the current contract. Load
  average and sample timestamp are not displayed.
- RAM shows used/total GiB and utilization on the left, with DIMM generation
  and configured speed on the right. If configured speed is unavailable, rated
  speed is used.
- DIMM identity is sampled once per monitor process. The interactive monitor
  first runs `sudo -n dmidecode --type memory`. If cached sudo authentication is
  unavailable, it permits sudo to request a password from the local terminal
  once before the Textual app starts. A failed probe leaves hardware identity
  unavailable without blocking later refreshes.
- NVIDIA telemetry queries `nvidia-smi` on every refresh. Every reported GPU
  receives its own block whose identity includes index and model name and whose
  metric row shows utilization, power draw, and current graphics-core
  frequency. Non-NVIDIA GPU telemetry is outside the current contract. Absence
  or failure is represented as unavailable telemetry rather than a monitor
  error.

## Refresh And Failure Isolation

- Interactive refresh defaults to two seconds. File polling and telemetry run
  outside the Textual event loop and publish only the newest completed refresh.
- Static CPU and DIMM identity is reused for the process lifetime. CPU, memory,
  and GPU utilization, power, and frequency are sampled on each refresh.
- Optional telemetry failures must not prevent execution state from rendering.
  The most recent valid dashboard state remains visible if a refresh fails.

## Implementation And Validation Owners

- `src/mammoth/monitor/model.py` owns discovery and state folding.
- `src/mammoth/monitor/dashboard.py` owns Rich presentation.
- `src/mammoth/monitor/textual_ui.py` owns refresh and interaction behavior.
- `src/mammoth/monitor/psutil_telemetry.py` owns optional hardware sampling.
- `src/mammoth/monitor/render.py` owns the stable ANSI-free plain snapshot.
- `tests/test_monitor.py` pins reconstruction, layout, telemetry, responsive
  width, and interaction behavior.

Any change to the behavior above updates this document and the owning tests in
the same change. Presentation changes should also be smoke-tested at wide and
80-column widths against a realistic long run name and production-length
execution ID.
