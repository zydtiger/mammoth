# Monitor Contract

This document owns Mammoth's passive-monitor reconstruction, interactive
presentation, telemetry, and interaction behavior. `docs/ARCHITECTURE.md` owns
the package boundaries, artifact roles, runtime concepts, and compatibility
policy that constrain this behavior.

The monitor is project-neutral. Phase names, task identifiers, coordinates,
metric names, and extension fields remain opaque producer data except for the
small presentation conventions stated below.

## Commands

- `mammoth monitor tui` opens the interactive terminal dashboard.
- `mammoth monitor web` serves the read-only web monitor.
- `mammoth monitor show` prints a stable plain-text snapshot or, with `--json`,
  one JSON document per snapshot. `--watch` repeats until a run reaches a terminal
  status. Fleet and group snapshots repeat until interrupted.
- `mammoth monitor` alone prints help.

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
- `ExecutionMonitor` folds incrementally: each poll applies only newly read
  events to one persistent folded state instead of re-folding the complete
  accumulated history, and it never retains the complete raw event list.
  This is an internal optimization only — its observable snapshots are
  exactly equivalent to a full re-fold of the same event sequence, pinned by
  a property test in `tests/test_monitor.py`.

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

The terminal and web monitors present three nested views above the same passive
reconstruction. In the terminal they are Textual screens:

1. **Fleet** (`mammoth monitor tui --entry <root>` with no run name). Lists every
   published group under `<entry>/.mammoth/groups/` plus every run directory
   under `<entry>` not claimed by a group ("loose" runs). Each group row
   shows members completed/failed/total, the currently active member, the
   newest member heartbeat's recency and staleness, and the terminal group
   status where one was recorded. An entry with no `.mammoth/` subtree, or no
   runs at all, renders an empty-but-valid fleet rather than an error.
   Groups and loose runs are each independently ordered most-recent-first:
   a group's or run's sort timestamp is its terminal event time (the group's
   own terminal event, or the run's newest execution's terminal event) when
   one is recorded, otherwise its newest observed activity (member activity
   for a group, heartbeat for a run) — so a still-active entry sorts by its
   most recent activity rather than dropping to the bottom. An entry with no
   usable timestamp at all sorts last; ties break on name so a repeated poll
   never reshuffles otherwise-equal rows. Plain-mode rendering shows the same
   order. A group's member rows (see level 2 below) keep their manifest
   schedule order regardless of this fleet-level ordering.
2. **Group** (selecting one group from the fleet view, or `--group <id>`
   directly). Shows one row per member run in the group manifest's recorded
   schedule order: each declared step's folded status, the member's overall
   run status, its own task progress, and its newest heartbeat. Failed,
   interrupted, and blocked members are highlighted. An aggregate ETA is
   shown only when every not-yet-terminal member currently reports both a
   total and a positive throughput; when a member's rate is unknown, the view
   omits the synthetic aggregate rather than guessing and lets each member's
   own progress speak for itself — member durations can diverge widely across
   a group, and a summed estimate would otherwise mislead. The same honesty
   rule applies to one member's own ETA (and a loose run's): it is shown only
   when that member's task currently reports throughput; otherwise it is
   withheld rather than falling back to an elapsed-time estimate, because
   under the bounded tail window (see below) a task's recorded start is only
   the first event the window happened to capture, not its true start, and
   an elapsed-time ETA derived from it can be wrong by an arbitrary factor.
3. **Run** (selecting one member or loose run). Pushes the existing
   single-run dashboard described in the sections below, unchanged. Back
   navigation (`Esc`/`Backspace`) pops the Textual screen stack and returns to
   the fleet or group view the operator drilled in from.

Pushing the run screen never blocks the Textual UI thread on that run's
first full-history read. The screen is pushed immediately, showing a loading
state, while `RunMonitor` construction and its first `poll()` run in the same
exclusive background-worker pattern (`@work(thread=True)`, generation-stamped)
the fleet, group, and run screens already use for their periodic refreshes;
the loading state is replaced once that first snapshot arrives. That first
poll also only fully folds the selected (by default, newest) execution —
every other attempt in the run's history gets a cheap unfolded placeholder
(immutable metadata only, no event-stream read) for that one call, so a run
with many historical attempts does not make the first paint wait on all of
them. Every poll after that first one folds every attempt exactly as
`mammoth monitor tui <run_name>` always has, so the attempt-history panel is
complete again from the very next scheduled refresh (or immediately once an
operator navigates to a not-yet-folded attempt, at which point it becomes the
selected execution for that poll). Standalone single-run monitoring
(`mammoth monitor tui <run_name>`, with no fleet or group screen beneath it)
never uses this lazy first poll, so its behavior and output are unaffected.

Both the fleet screen (its groups table and its loose-runs table
independently) and the group screen window their row table to the terminal's
actual height instead of rendering every row unconditionally: rather than
scrolling, each render computes and shows only a band of rows around the
`j`/`k`-selected one, with "... N more above ..." / "... N more below ..."
markers in place of the rows that do not fit. `j`/`k` still walk the entire
row set — every group and loose run on the fleet screen, every member on the
group screen — not only the currently visible band. Plain-mode snapshot
rendering is unaffected: it keeps printing every row unconditionally.

The selected row is guaranteed visible whenever the viewport can physically
fit its chrome (headers, section labels, the table's own header row) plus at
least one data row, and the total rendered height never exceeds the
viewport. The budget available to each table is the real, wrap-aware
rendered height of everything around it at the terminal's actual width, not
a fixed line-count guess or piecewise arithmetic that can drift from what a
real, assembled render actually produces — a summary line wrapping at a
narrow width, or a table's own header row wrapping (for example the fleet
group table's "Members (done/failed/total)" column), is accounted for
exactly.

The table region is sized first and has priority, since it carries the
selected-row guarantee; the trailing warnings and footer, which render
*after* it, are fitted to whatever room genuinely remains once the table
region's real height is known, rather than reserved at a fixed guessed
height and trusted to fit. On an ultra-small viewport the guarantee can
still floor the table region taller than a first guess at that remaining
room predicted, so trailing content is measured again against the table
region's actual height and, if it no longer fits, drops least-essential
content first (warnings, then the footer) — disappearing entirely rather
than ever pushing the total past the viewport. The row's priority over the
footer always holds.

An unfocused table (the unselected one, whether it renders before or after
the focused table) uses as many rows as genuinely fit, not an artificial
cap: a small guaranteed floor exists only to protect the focused table's
budget on a genuinely scarce viewport, so a large viewport with only a
handful of groups and loose runs shows every row of both tables with no
overflow markers, instead of an unfocused table stopping early while
surplus rows of screen space sit empty.

On a viewport too small even for the selected row's own minimum, optional
chrome is dropped before the selected row ever would be, richest to
sparsest:

1. The group screen drops its summary line, then its "MEMBERS" label.
2. The fleet screen drops the *unfocused* table's whole section (its label
   and its table together, never just one) when that section renders
   *before* the focused table — content that renders after the focused table
   never threatens the guarantee, so it is only capped, never force-dropped.
   If still tight, the focused table's own section label goes too.
3. As an absolute last resort, if even the focused table's own header row
   cannot fit alongside its one guaranteed data row, the header row itself
   is suppressed (the table renders with no column headers) rather than the
   row.

`mammoth monitor tui <run_name>` without `--group` or `--match` is unaffected by
any of this: it keeps its original single-run behavior exactly, entering
directly at the run view with no fleet or group screen beneath it.

### Passive Folding Sources Per Level

- **Fleet and group rows** fold from three passive sources, mirroring how a
  single run folds from its own execution streams:
  - The group's immutable manifest (`manifest.json`), read once and cached
    for the monitor's lifetime.
  - The group's append-only event stream (`events.jsonl.zst`, or historical `events.jsonl`), tailed
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
    wrote one. This tail is *bounded*: see "Bounded Fleet-Level Folding"
    below. No fleet- or group-level source ever performs a full-history read
    of a run's event streams.
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

### Bounded Fleet-Level Folding

A fleet or group roll-up only ever needs a run's *current* state — status,
newest heartbeat, terminal outcome, displayed progress — not its full
history, which the run view alone reconstructs. Reading and folding each
member's complete event streams on every poll is what made an entry with
hundreds of runs and large streams take tens of seconds to open and several
seconds per steady-state refresh. Instead, the newest-execution tail each
fleet or group row folds from (see above) bounds its *first* read of every
underlying stream to the last `FLEET_TAIL_WINDOW_BYTES` (128 KiB) bytes,
discarding the partial line straddling the seek point, instead of reading
from byte 0; every read after that first one is unaffected and reads only
newly appended bytes exactly like the unbounded reader `mammoth monitor tui
<run_name>` uses. 128 KiB comfortably covers ordinary tails: a schema-v1
JSONL record is on the order of a few hundred bytes, so the window holds
hundreds of records per stream — far more than one heartbeat interval (30s
default) or one terminal record's worth of trailing history. A stream
smaller than the window is read from its true start, identically to the
unbounded reader. A window too small to contain even one complete line (an
oversized single record) falls back to a full read from the true start for
that stream, so the bound never silently starves a roll-up of events; this
is expected to be unreachable at the configured window size in practice.

For compressed streams, the window counts **uncompressed** JSONL bytes.
A closed stream uses its standard seek table; a live stream uses its optional
file-identity-bound `.idx` cache. Invalid embedded seek metadata also falls back
to that cache or sequential decoding, preserving independently readable frames. The reader starts at a frame boundary, decodes
the remaining suffix (including an unsealed live frame), then discards records
before the requested window. Indexed reads may decode an additional boundary
frame and the current live frame. Index metadata itself grows with frame count.
Absent, invalid, or stale cache metadata may require scanning a larger prefix;
this never authorizes a monitor write. Steady-state polling retains decompressor
state and reads only appended compressed bytes, with the same file identity,
truncation, and consumed-prefix checks as plain streams. Each event reader
serializes overlapping polls so shared fleet/group refreshes cannot advance
its decoder or sequence state concurrently. An incomplete compressed
append is retained for later polling; malformed compressed data reports a stream
warning, preserving events decoded before the damaged frame.

**The approximation this bound introduces**: a fleet or group row cannot see
task or metric history older than the window, only the *current* value of
each field once fully written. This costs nothing observable, because every
roll-up field a fleet or group row displays is itself a "what is true right
now" value, not a history: `TaskState.completed`/`total`/`throughput` are
overwritten (not accumulated) by each new `progress` event, so the latest one
within the window is the same value the field would hold under a full read.
The only roll-up field that *is* order-sensitive across producers rather than
current-value-only is a group's own `run_started`/`run_completed`/... status
from its group event stream — and that stream is read via
`GroupEventTailReader`, never bounded (see below), so it is unaffected.

**Multi-rank all-ranks-terminal resolution under the window**: a run's
overall status only reports `completed` once every expected rank's own
process has completed (`_finalize_run_status` in `model.py`), which could
seem to need each rank's *entire* history to confirm none is still pending.
It does not, because each rank (and the runner) writes to its own reserved
stream file (`rank-N.jsonl.zst`, `runner.jsonl.zst`, or historical plain equivalents), and a bounded read always seeks
from the *current end* of that specific file backward — so whatever that
rank's most recently written record is (its own terminal `process_completed`
if it has one, or its latest heartbeat/progress if it is still running) is
always inside the window, regardless of window size, as long as the window
holds at least one complete line (guaranteed by the fallback above). A rank's
terminal record is therefore never pushed out of view by bounding: the bound
only ever discards that rank's *earlier* history, which the roll-up does not
need. This means fleet-level status resolution is exact, not approximate,
under the window; only task/metric *history depth* (not needed by any
roll-up field) is bounded.

**Group event streams remain unbounded.** They are already fully
incremental (`GroupEventTailReader` never re-reads history), and they carry
only control-plane run/step lifecycle records rather than per-step training
telemetry, so they stay orders of magnitude smaller than an execution's own
streams even for a long-running group; bounding their first read was not
worth the added complexity. If a future producer ever made a single group's
event stream large, this choice should be revisited.

## Stage And Display Events

Both events live in `rank-N.jsonl.zst` or `runner.jsonl.zst`, need no phase, and
stay opaque to Mammoth apart from the conventions below. A schema-v1 reader older
than these events stops at the first one and reports a stream warning, so monitors
must be at least as new as the producers they read.

- `stage_changed` records that a named `lane` entered a named `stage`. Folding keeps
  a chronological history for every lane, independent of poll batching, and joins
  histories across the selected resume lineage. A stage lasts until the next
  transition on the same producer and lane within its attempt. An attempt's last
  stage ends at that attempt's last observation, so a resume gap is not stage time.
  The last stage of a live attempt stays open.
- `display` carries a `kind`, a `name`, and JSON `data` of at most
  `MAX_DISPLAY_DATA_BYTES` (16 KiB) and `MAX_DISPLAY_DEPTH` (32) nested levels. The newest item of each `kind` and `name` wins,
  across the resume lineage. Like every extension field, `data` passes the metadata
  credential redactor, so a column or key named like a credential is stored as
  `<redacted>`. A stage, lane, kind, or display name that is one plain word, such as
  `tokens/s`, is stored as written; anything richer is sanitized.
- `RunObserver.stage` and `RunObserver.display` drop anything that cannot be stored
  instead of raising or disabling the sink, logging a warning through the Python
  logger, and do nothing after close. `kind` and `name` are positional-only, so a
  payload can use those field names.
- `table` is the built-in display kind, with `columns` and `rows` in `data`. Web,
  terminal, and plain output render it. For any other kind the web page shows the
  raw JSON, and the terminal and plain output skip it. A new kind needs a renderer
  and no schema change.
- Stage names, lane names, display kinds, and display names are limited to
  `MAX_PRESENTATION_NAME_CHARS` (256) and must be encodable as UTF-8.
- Frequent stage events can push every task out of a fleet tail's first 128 KiB. When
  a bounded first read skipped records and folded no task, that one run's newest
  execution is read fully, once, so its row keeps its progress.

## Execution Telemetry

`ExecutionSession` samples at a default 1 Hz, on rank 0, into `telemetry.jsonl.zst`
in the execution directory, starting with the first phase. Project code passes
nothing; `ExecutionSpec(telemetry=False)` opts out. Each line is one JSON record:

- `{"kind": "gpu", "time", "gpu", "utilization", "power_watts", "memory_mib"}` for
  each GPU from one long-lived `nvidia-smi` loop, limited to `CUDA_VISIBLE_DEVICES`
  or all reported GPUs when unset. An unavailable field is `null`.
- `{"kind": "cpu", "time", "main_cores", "worker_cores"}` from `/proc`. Workers are
  direct children of any thread in the process, excluding the sampler's own child.
- `{"kind": "status", "time", "source", "status": "unavailable", "detail"}` when a
  source cannot be sampled or stops. The other source continues.

`event_stream_paths` does not return this file, so telemetry cannot refresh
producer liveness, change event sequence numbers, or consume fleet tail windows.
The child stays in the process group so supervised cleanup reaches it. After its
parent dies it exits through `SIGPIPE` once every process holding the pipe has ended,
which includes forked workers. Forked workers do not own the recorder. Stopping waits
at most a few seconds, escalating to `SIGKILL`.

## Web Monitor

`mammoth monitor web [--root DIR]... [--host] [--port] [--interval] [--stale-after]`
serves one long-lived, read-only process.

- Discovery walks each root up to `MAX_ENTRY_DEPTH` directories every
  `DISCOVERY_INTERVAL_SECONDS`, skipping hidden directories, symbolic links, and the
  inside of run directories. A directory is an entry when a child has
  `logs/executions` or it has `.mammoth/groups`. A new entry can take that long to
  appear; a new run in a known entry appears on the next sweep.
- One background sweep polls every entry's bounded `FleetMonitor`. The fleet page
  lists only runs with recorded executions, active runs first, and shows the 100 most
  recent matching rows.
- A full `RunMonitor` and telemetry readers exist only for an opened run, are shared
  by its viewers, and are released after `RUN_IDLE_SECONDS` without a request.
- `GET /api/fleet`, `/api/group`, `/api/run`, `/api/series`, and `/api/stages` return
  JSON. `/api/series` takes one `after` per `name`, so each series resumes from its
  own newest point and a refresh costs only new points. Its downsampling keeps each
  bucket's minimum and maximum. `/api/stages` returns stage spans overlapping the time
  after `since`, including a span already in progress, and beyond `max_points` shows
  one stage per time bucket in proportion to each stage's time. A run whose artifacts cannot be read right now
  returns 503, and the page keeps retrying through that and a temporary 404. Every
  other method returns 501, and only the packaged static files are served.
- `/api/run` holds at most `MAX_DOCUMENT_ROWS` (100) stage rows, stage names, and
  display items, newest display items first. The stage summary is recomputed at most
  every `SUMMARY_SECONDS` (10). An opened run keeps its telemetry in memory until it
  is released, and its first load decodes the whole stream.
- The page shows progress, status, tasks, telemetry charts with stage shading,
  per-stage summaries, one chart per metric, display items, producers, attempts, and
  warnings. Metrics are grouped by leading namespace. The x-axis is the `epoch`
  coordinate when every point has one and time otherwise. A run with stages and no
  telemetry gets a stage timeline chart. Runs still reporting open on a five-minute window and other runs on their
  whole history. Pages use `/?entry=<id>&run=<name>[&execution=<id>]` and
  `/?entry=<id>&group=<id>`.
- The server binds to `127.0.0.1` unless `--host` says otherwise and has no
  authentication. While bound to loopback it refuses requests whose `Host` header
  names anything else, which blocks DNS rebinding from a browser.

## Dashboard Information Hierarchy

The wide dashboard uses this stable order:

1. Run header and refresh state.
2. `RUN PROGRESS`.
3. `TRAINING TRENDS`, when loss or learning-rate history exists.
4. `TABLES`, when `table` display items exist.
5. `HOST RESOURCES`, when viewer telemetry is enabled.
6. Selected attempt and its overall progress.
7. Per-rank state and active leaf progress.
8. Attempt history and lineage.
9. Stream warnings and interaction help.

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
- `src/mammoth/monitor/documents.py` owns JSON documents for `show --json` and the
  web monitor.
- `src/mammoth/monitor/web.py` and `src/mammoth/monitor/web_static/` own entry
  discovery, the read-only HTTP API, and the browser page.
- `src/mammoth/core/telemetry.py` owns execution-host sampling and reading.
- `tests/test_monitor.py` pins single-run reconstruction, layout, telemetry,
  responsive width, and interaction behavior, plus fleet/group presentation
  and navigation; also pins `ExecutionMonitor`'s incremental-fold/full-fold
  equivalence and the fleet drill-down's non-blocking loading state.
- `tests/test_web_monitor.py` pins entry discovery, routes, cursors, downsampling,
  lazy views, and read-only behavior. `tests/test_telemetry.py` pins execution-host
  sampling, failure isolation, session ownership, and liveness separation.
- `tests/test_fleet.py` pins fleet and group folding, ad-hoc `--match`
  grouping, and their plain-mode rendering; also pins bounded-tail-window
  status/heartbeat/terminal/progress extraction and bounded initial-poll
  read cost at scale.

Any change to the behavior above updates this document and the owning tests in
the same change. Presentation changes should also be smoke-tested at wide and
80-column widths against a realistic long run name and production-length
execution ID.
