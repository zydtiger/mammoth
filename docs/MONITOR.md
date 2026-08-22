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
- A caller-supplied `parent_execution_id` defines resume continuity.
  `resume_checkpoint` remains an independent artifact reference and never
  infers a parent. `previous_execution_id` records adjacency but does not make
  metrics or coordinates continuous.
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
- In the wide rank table, rank is a fixed six-character field and progress has
  an 18-character protected minimum for production counters. Progress receives
  expansion space before rank or status fields.
- Distributed progress is reconciled only among matching process-rank tasks.
  Identical counters are treated as replicated global progress and shown once.
  Distinct compatible counters are summed. Missing expected ranks produce an
  aggregation-pending state instead of a partial total.
- Progress counts are unitless producer-owned numbers and render without a
  suffix. Within a task, producers must use one logical work quantity for
  `completed` and `total`, and report `throughput` for that same quantity per
  second or omit it. Every present throughput renders as `b/s`; this compact
  label does not define or convert the producer's work batch.

## View Hierarchy And Navigation

The monitor presents three nested Textual views above the same passive
reconstruction:

1. **Fleet** (`mammoth monitor --entry <root>` with no run name). Lists every
   published group under `<entry>/.mammoth/groups/` plus every run directory
   under `<entry>` not claimed by a group ("loose" runs). Each group row
   shows members completed/failed/total, the currently active member, the
   newest member heartbeat's recency and staleness, and the terminal group
   status where one was recorded. An entry with no `.mammoth/` subtree, or no
   runs at all, renders an empty-but-valid fleet rather than an error.
2. **Group** (selecting one group from the fleet view, or `--group <id>`
   directly). Shows one row per member run in the group manifest's recorded
   schedule order: each declared step's folded status, the member's overall
   run status, its own task progress, and its newest heartbeat. Failed,
   interrupted, and blocked members are highlighted. An aggregate ETA is
   shown only when every not-yet-terminal member currently reports both a
   total and a positive throughput; when a member's rate is unknown, the view
   omits the synthetic aggregate rather than guessing and lets each member's
   own progress speak for itself — member durations can diverge widely across
   a group, and a summed estimate would otherwise mislead.
3. **Run** (selecting one member or loose run). Pushes the existing
   single-run dashboard described in the sections below, unchanged. Back
   navigation (`Esc`/`Backspace`) pops the Textual screen stack and returns to
   the fleet or group view the operator drilled in from.

`mammoth monitor <run_name>` without `--group` or `--match` is unaffected by
any of this: it keeps its original single-run behavior exactly, entering
directly at the run view with no fleet or group screen beneath it.

### Passive Folding Sources Per Level

- **Fleet and group rows** fold from three passive sources, mirroring how a
  single run folds from its own execution streams:
  - The group's immutable manifest (`manifest.json`), read once and cached
    for the monitor's lifetime.
  - The group's append-only event stream (`events.jsonl`), tailed
    incrementally with `mammoth.core.groups.GroupEventTailReader` — never
    re-read in full on a later poll. Group-scoped terminal events
    (`group_completed`/`group_failed`/`group_interrupted`) and run-scoped
    terminal events (`run_completed`/`run_failed`/`run_blocked`/
    `run_interrupted`) are authoritative for that scope's status; step-scoped
    events (`step_started`/`step_completed`/`step_failed`/
    `step_interrupted`) are the sole source for each declared step's status.
  - A cheap tail of each member or loose run's *newest* execution only,
    using the same `ExecutionMonitor` a single run's overview uses for one
    attempt — never a full `RunMonitor` across a run's whole lineage. This
    supplies live task progress, heartbeat recency, and staleness detection
    for runs (or run-major segments) the group event stream has not yet
    recorded a terminal outcome for, including a crashed workflow that never
    wrote one.
- **Run view**, once drilled into, folds exactly as described in the rest of
  this document: a full `RunMonitor` reconstructs every valid attempt in that
  run's lineage. This is the only point in the hierarchy where a full
  `RunMonitor` is constructed, and it happens lazily, only for the run an
  operator actually opens.
- **`--match <glob>`** provides ad-hoc grouping for entries that predate
  group manifests, or for runs launched outside `mammoth.workflow.Workflow`:
  loose runs whose name matches the glob are folded into one synthetic group
  with no manifest, no group event stream, and no declared step schedule —
  its members' statuses come from their own execution tails alone.
- The same passivity contract governs every level: no writes, no producer
  contact, no control actions. A group directory whose manifest failed to
  publish (or fails to parse) is skipped with one warning rather than
  breaking the fleet view.

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
  once before the Textual app starts. This holds at every interactive entry
  point, including the fleet view (which has no `HOST RESOURCES` panel of its
  own): the initial telemetry sample is always taken before any Textual app
  is constructed or run, never from inside a running screen stack, so the
  prompt never erupts mid-navigation. Drilling from the fleet into a run
  reuses that same pre-UI sample for the first run screen instead of
  sampling again; later drill-ins and periodic refreshes sample again
  through the same sampler, which never reprompts since it retries cached
  sudo credentials first. A failed probe leaves hardware identity
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

- `src/mammoth/monitor/model.py` owns single-run discovery and state folding.
- `src/mammoth/monitor/fleet.py` owns fleet and group discovery and folding
  from group manifests, group event streams, and run tails.
- `src/mammoth/monitor/dashboard.py` owns Rich presentation for the run,
  fleet, and group views.
- `src/mammoth/monitor/textual_ui.py` owns refresh, the Fleet -> Group -> Run
  Textual screen stack, and interaction behavior at every level.
- `src/mammoth/monitor/psutil_telemetry.py` owns optional hardware sampling.
- `src/mammoth/monitor/render.py` owns the stable ANSI-free plain snapshot for
  the run, fleet, and group views.
- `tests/test_monitor.py` pins single-run reconstruction, layout, telemetry,
  responsive width, and interaction behavior, plus fleet/group presentation
  and navigation.
- `tests/test_fleet.py` pins fleet and group folding, ad-hoc `--match`
  grouping, and their plain-mode rendering.

Any change to the behavior above updates this document and the owning tests in
the same change. Presentation changes should also be smoke-tested at wide and
80-column widths against a realistic long run name and production-length
execution ID.
