# Mammoth Architecture

## Boundary

Mammoth owns operational infrastructure. A consuming project owns the meaning
of its computation.

This boundary is deliberately stricter than merely avoiding imports from one
originating repository. Mammoth must not encode assumptions about:

- model architectures or model factories;
- dataset classes, paths, transforms, labels, or sampling policies;
- batch shapes or model call signatures;
- losses or scientific metric calculations;
- project configuration models;
- checkpoint compatibility or inference export formats; or
- domain phases such as segmentation, plotting, or result refresh.

The optional PyTorch layer may understand generic training concepts such as an
epoch, optimizer step, precision policy, and gradient accumulation. It must
receive already constructed framework objects and project-supplied step
functions.

## Dependency layers

Dependencies point downward only:

```text
consuming project
    │
    ├── mammoth.torch       optional generic nn.Module/DataLoader training
    ├── mammoth.workflow    programmatic serial steps and supervision
    ├── mammoth.monitor     replay, terminal UI, and telemetry
    ├── mammoth.execution   direct process lifecycle composition
    ├── mammoth.logging     JSONL, text, and TensorBoard sinks
    └── mammoth.core        identities, layout, events, artifacts, provenance
```

`mammoth.core` should use the Python standard library. TensorBoard, Rich,
Textual, psutil, and PyTorch belong in optional dependency groups and must not
be imported by the core package.

`mammoth.execution` is the framework-neutral composition layer above core and
logging. It may import both packages, while neither lower layer imports it.
This keeps direct sessions available to CPU-only consumers without adding a
core-to-logging dependency or importing an optional framework.

`mammoth.core` also owns framework-neutral identity for raw local-file bytes.
`inspect_artifact(path)` reads one regular file through a single open descriptor
and returns an immutable `ArtifactReceipt` containing the caller-supplied path,
byte size, and lowercase SHA-256 digest. `verify_artifact(receipt)` repeats that
inspection and succeeds only when the visible path still names exactly those
bytes. Receipts do not encode roles, schemas, content meaning, or a promise
that bytes remain unchanged after verification. The default rejects final-path
symlinks, directories, and special files; missing paths retain
`FileNotFoundError`, observed replacement or mutation raises
`ArtifactChangedError`, and a stable but different regular file raises
`ArtifactVerificationError`.

When a caller must parse one artifact, `open_artifact_session(path)` creates a
one-use `ArtifactReadSession`. Session entry records the receipt on a private
no-follow descriptor. Callers obtain binary readers only through nested,
serial `session.open_reader()` contexts: every reader starts at offset zero,
may seek or rewind freely, and is closed by its context. Mammoth never exposes a file
descriptor or transient filesystem identity. It validates the descriptor and
visible path before and after each successful reader, and on a successful outer
exit rechecks the exact bytes and receipt. Readers cannot nest or run
concurrently because they share the session's file object. Parser exceptions
propagate unchanged after cleanup, so callers must discard parse results unless
the outer session exits successfully. This establishes entry/exit exact-byte
identity and visible-path binding, not an immutable snapshot for arbitrary
seek-heavy parsing or a guarantee against a hostile writer changing bytes
between checks. Ancestor symlinks and root containment remain caller policy;
the core default rejects only a final-component symlink. Lower layers may
create the same generic receipt from a publication descriptor before atomic
rename, while callers own portable path serialization and all payload meaning.
After a reader context closes, its binary handle is closed; after the one-use
session closes, its immutable receipt remains available but opening another
reader raises `RuntimeError`.

For one logical result that spans several local paths, `mammoth.core` owns a
separate `ArtifactTransactionPlan` protocol. A caller supplies at least two
prepared `TransactionArtifact` values and a pre-existing coordinator
`lease_root`, which owns the journal. This low-level contract remains stable
for existing callers and journals. Consumers that only know artifact keys,
targets, kinds, validators, a namespace, and create-or-replace intent use
`TransactionArtifactSpec` with `build_artifact_transaction_plan()`. The planner
derives the established deterministic transaction ID, sibling stage names,
coordinator root, non-overlapping artifact roots, and recovery policy. Its only
filesystem side effect is safely provisioning missing target-parent directories
through no-follow descriptor walks; it never creates a target, stage, or
journal. The namespace identifies the logical publication and must distinguish
otherwise identical create and replacement operations.

By default every artifact stays below one root. A plan may instead declare
non-overlapping `artifact_roots`; every target and its reserved sibling stage
must belong to exactly one declared root, and that root owns the artifact's
backups and retired cleanup objects. `stage_transaction_file()` exclusively
creates, writes, and synchronizes one file stage. `move_directory_into_transaction_stage()`
exclusively reserves a directory stage, atomically adopts a rendered local tree,
and preserves stage/source evidence on ambiguous failure. Files are identified
through `ArtifactReceipt`; ordinary directory trees are sealed, synchronized,
and recorded with a transaction-local tree identity. Callers may add validators
that authenticate their own marker or payload semantics, while Mammoth never
interprets them. The plan explicitly selects `create_only` plus roll-forward
recovery or `replace` plus rollback-before-commit recovery.

Publication first acquires deterministic advisory leases for all target paths
and their ancestors within their individual roots, anchors the coordinator and
all declared roots with no-follow directory FDs, then writes a strict
no-clobber schema-v2 journal below
`<lease_root>/.mammoth-transactions/`. The journal records the canonical set of
artifact roots; schema-v1 journals remain recoverable only with the original
one-root plan. Mammoth synchronizes every staged object before the journal and
synchronizes each parent directory after a local backup move, target rename,
journal mutation, or cleanup mutation. Replacement retains an authenticated
original at its transaction-specific backup path until every new target is
visible and revalidated, at which point the journal's durable `committed` state
is the recovery boundary. Cleanup first moves remnants to deterministic private
retired paths on the artifact's own root, so a crash never leaves an unrecorded
random quarantine name; it also retires the coordinator journal at a
deterministic name before its final removal. `recover_artifact_transaction()`
binds either journal name back to the complete caller-supplied plan; it rejects
target substitution, malformed state, identities that no longer match,
symlinks, and special files without speculative cleanup. Before any uncommitted
recovery path can rename a stage to a target or restore a replacement target,
it authenticates and runs the current-plan validator for every still-visible
stage. A visible target is never mistaken for a stage: an old replacement
generation and a target-only new generation retain their normal state-specific
recovery handling. Existing journals always require that explicit recovery
call.

This is a Linux local-filesystem protocol: it requires no-follow opens,
advisory `flock`, `renameat2(RENAME_NOREPLACE | RENAME_EXCHANGE)`,
same-filesystem renames, and successful file/directory `fsync`. It rejects
unsupported or unsafe topology rather than weakening its durability claim.
The cleanup protocol assumes local writers cooperate through Mammoth's target
leases. It does not attempt a hostile same-UID filesystem-security model in
which another process deliberately replaces private cleanup entries between
identity verification and unlinking them. Several final paths may be visible
one by one during ordered publication; Mammoth never claims simultaneous
filesystem-atomic visibility. Network filesystems, object stores,
distributed coordination, reader-side generation selection, and cross-host
coordination remain outside this contract. The protocol never moves or copies
an artifact between declared roots: each publication rename is local to one
root, while the coordinator journal makes interrupted multi-root publication
recoverable.

`mammoth.core.workstore` owns a separate, simpler recoverable primitive for one
long-running chunked producer: exclusive store leasing, an append-only
hash-chained completion journal, durable creation, and fail-closed
classification of prior state. A consumer identifies its store with an opaque
JSON-compatible identity payload and its own chunk-ID space. Mammoth derives
a discrimination digest, `identity_digest`, from the caller's *raw* payload
to detect a returning identity: two payloads differing only in a field whose
name happens to match the sensitive-name heuristics `sanitize_metadata_fields`
uses for execution `runtime` and event `extensions` fields (for example two
different `session_id` values) would otherwise redact to the same value and
collide as one identity, so deriving the digest from the raw form is what
keeps them discriminated. Mammoth separately sanitizes that same payload the
same way and persists and exposes only that redacted copy; the raw form is
used only to derive the digest and is never itself persisted. Otherwise
Mammoth never interprets the payload. A commit's chunk marker is likewise an
opaque caller-supplied string recorded verbatim. Mammoth does not read,
write, or verify the consumer's actual chunk payloads; those live and stay
durable outside this contract, and a commit only records that the caller
already made them durable. `WorkStoreInspection.completed_chunks` and
`WorkStoreSession.completed_chunks` read those markers back as a chunk-ID
keyed mapping so a consumer can re-verify a durable chunk payload against
its own marker at resume time; the mapping is populated only from a journal
that already passed hash-chain verification, so a `damaged` classification
still exposes no markers, and `completed_chunk_ids` is unchanged.

`identity_digest` is not a plain hash: it is `hashlib.scrypt` over the raw
payload's canonical JSON, with a fresh random 16-byte salt generated per
store (`os.urandom`) and fixed, deliberately memory-hard cost parameters
(`n=2**14, r=8, p=1, dklen=32`). The salt and parameters are persisted
alongside the digest (as `identity_kdf`) and reused, never regenerated, to
recompute and compare the digest on every later load. A plain unsalted hash
would let anyone holding the persisted metadata file mount an offline
dictionary attack straight from a wordlist to recover a low-entropy raw
payload (a short session ID, for example); scrypt's memory-hardness and a
random per-store salt (which also defeats any precomputed/rainbow-table
attack shared across stores) raise the cost of that attack substantially.
**They do not eliminate it.** An attacker who holds the metadata file can
still spend the same per-guess cost a legitimate load pays to brute-force a
sufficiently low-entropy identity payload offline; consumers must not treat
identity redaction as protection for high-value secrets with few plausible
values, only as discrimination between otherwise-opaque returning
identities. A genuinely secret, high-entropy credential does not belong in
the identity payload at all.

A `WorkStoreSession` is confined to one thread at a time within its owning
process: the exclusive lease is a process-level `flock`, not a per-thread
lock, so two threads sharing one session are not mutually excluded, and
`commit()`/`verify()` hold no internal lock against concurrent callers. Two
independent processes are always safely excluded by the lease itself.

Mammoth defines its own versioned store format,
`mammoth-work-store-jsonl-v1`, structurally mirroring the transaction
journal's proven hash-chained-JSONL approach rather than adopting any
consumer's pre-existing on-disk format. `claim_work_store_lease()` acquires a
non-blocking advisory lease at a deterministic sibling path next to the store
directory; it authenticates ownership and permissions on the already-opened
lease descriptor itself, not a separate path re-stat, so a lease path
replaced between open and validation cannot slip past the check. A second
claim fails closed as `WorkStoreConflictError` instead of racing writers.
Any other `OSError` from creating the lease's parent, opening the lease
path (for example a pre-existing symlink defeating `O_NOFOLLOW`), or
locking it translates into the `WorkStoreError` family rather than
escaping as a raw `OSError`.
`inspect_work_store()` claims that lease only long enough to classify
existing state as `absent`, `resumable`, `legacy` (no Mammoth metadata and
no Mammoth journal at the path), `incompatible` (Mammoth's own metadata
names a different identity), `relocated` (Mammoth's own metadata matches
this identity but was durably recorded for a different canonical store
path), `damaged` (structurally invalid metadata, a missing or broken
completion journal, a broken or foreign-seeded hash chain, or Mammoth's own
journal present without Mammoth's own metadata — debris that must never be
mistaken for adoptable-or-preservable legacy content), or
`concurrently_owned`, and never mutates anything it inspects.
`WorkStoreSession.open_or_create()` performs the same classification under a
lease it retains: `absent` durably creates a new owner-only store (`0700`
directory — with the new directory's own dentry made durable by fsyncing
its parent, matching the transaction protocol's mkdir-then-parent-fsync
precedent — `0600` files, fsynced metadata, an eagerly created empty
completion journal, and their directory entries) and `resumable` reopens the
authenticated match, while every other classification raises the matching
`WorkStoreStatusError` instead of silently adopting or overwriting ambiguous
state. A creation failure after some but not all of that durable state
exists rolls back best-effort: it first attempts the same
quarantine-rename-then-remove path as every other deletion (below), and if
that cannot even be attempted, falls back to removing the live path
directly. Both are best-effort and their failures are suppressed, not a
guarantee: a further failure can still leave a remnant at the live store
path, and the next `inspect_work_store()` or `open_or_create()` there
classifies whatever it finds — `absent` if fully removed, or `damaged` if
partial Mammoth debris such as a bare journal file survives.

Store metadata durably records the creating call's canonical
(`Path.resolve()`) store path alongside the identity digest, but that field
is plain and unauthenticated by itself: it is a cheap first check, not a
security boundary. The real binding is cryptographic. Every store's
completion-journal chain is seeded — its first record's `previous_sha256` —
with a digest over the format identifier, the identity digest, and
metadata's own recorded store path, computed fresh from that same metadata
every time the journal is loaded. An honest whole-directory copy or move
carries its unedited metadata and untouched journal together, so the chain
still validates against the recomputed seed; the plain path comparison then
classifies it `relocated`, and copying or moving a store directory after
creation costs its resume state by design, the same way a legacy or
genuinely incompatible store is preserved but never adopted. A journal
grafted from a different store's directory, or a metadata `store_path` field
edited without regenerating the chain it seeds, desynchronizes the
recomputed seed from what is actually baked into the journal's first record:
both fail the very first record's check and classify `damaged` rather than
being silently adopted or merely degrading to `relocated`. A validly created
store's completion journal file always exists, even before the first
commit; its later absence is therefore always `damaged`; the same guarantee
is what makes `commit()`'s later `os.O_CREAT` a no-op except after
intentional repair. `commit()` writes, flushes, and fsyncs each completion
record under one shared failure guard, updates the session's cached state
only from that already-durable point, and only then fsyncs the directory
entry (load-bearing solely for a brand-new journal file's own creation), so
a crash can never observe a completion that is not durable and can never
lose one that was.

After any exception from `commit()`, the session is unusable — every
further `commit()` or `verify()` raises `WorkStoreDamagedError` — but the
durable outcome differs by failure site. If the write, the flush, or the
record's own fsync fails, including a short write, some or all of the
record's bytes may already be visible to a later reader (a deferred ENOSPC
or EIO can surface only at flush, after bytes already reached the
descriptor), so Mammoth best-effort truncates the journal back to its
pre-record length, even if that truncation itself fails, before poisoning
the session and raising. If only the trailing directory fsync fails
afterward, the record's own fsync had already succeeded — the commit is
durable and nothing is truncated — but the session is still poisoned, since
it just observed a filesystem failure; the caller must close it, and a
reopen afterward may legitimately show the chunk as already completed. An
interrupted commit's incomplete trailing journal line is recoverable, not
damage: resuming truncates that torn tail before further commits are
appended, the same way a torn write is tolerated elsewhere in Mammoth's
append-only formats. `verify()` re-reads the durable journal under the same
seed and rejects any divergence from the session's observed state, including
a hash chain that was tampered with, or correctly re-hashed, to resurrect an
already-completed chunk ID.

`close()` releases ownership and preserves the store for a later resume;
`remove_after_publication()` and the cross-invocation `cleanup_work_store()`
delete a store only after the caller confirms its own final publication,
never inferring publication from committed chunk markers. Deletion — for
both an explicit cleanup and a failed creation's rollback — first renames
the store directory to a deterministic retired sibling path and fsyncs the
parent directory, mirroring the transaction protocol's documented
retired-path approach, before removing the retired tree; a crash between the
rename and the removal leaves the live path simply absent, never a
half-deleted store that could be misread as a legitimate empty one or as
foreign legacy content, and a stale retired remnant from an earlier
interrupted deletion is reclaimed deterministically by the next call that
retires a store at the same path.

This module shares the transaction protocol's exact scope: local POSIX
filesystems, no network filesystems, no cross-host coordination, and no
hostile-writer model. A crashed or killed cooperating writer is recoverable;
a writer that bypasses this module's lease is not.

The workflow layer owns project-neutral local process supervision and serial
multi-run orchestration through Python values only. `Workflow`, `Run`, `Step`,
and `Execution` are immutable caller inputs; step commands are already-final
argv, step names are canonical phases, and workflow/run roots derive each
`RunLayout`. `Workflow` freezes inputs, validates every workflow-wide identity
and ordering invariant, and derives one immutable canonical dispatch during
construction without touching the filesystem, claiming a lease, resolving
execution inputs, or emitting lifecycle records. `Workflow.plan()` is a
side-effect-free projection of that dispatch to command plans. Run-major order
preserves declared runs and steps. Step-major order interleaves runs through
one explicit canonical `step_order`; each run must declare a duplicate-free
subsequence and may omit canonical phases.

`Workflow.run()` requires the Python main thread and POSIX
`pthread_sigmask` support because it owns process signal handlers. Unsupported
invocations fail before creating artifacts or children. Mammoth blocks SIGINT
and SIGTERM while replacing or restoring both handlers, so a signal in either
transition remains pending until the complete new or restored handler set
exists. After handler installation, it owns setup and execution interruption,
prepared layouts, logical-run leases, immutable execution contexts, observers,
lifecycle transitions, child supervision, and cleanup. SIGINT and SIGTERM
received under Mammoth's installed handlers before any run activates return a
structured interruption result with all runs blocked and no artifacts. For each activated run,
`resolve_execution()` may replace the
baseline `Execution` after layout preparation and lease acquisition but before
metadata publication. Mammoth records the workflow invocation from `sys.argv`,
derives intended phases from the run's steps, and resolves the previous attempt
under the lease. `before_first_step()` runs at most once after
`execution_started` and before the first `phase_started`; its failure therefore
produces only execution-level failure lifecycle. These are the only caller
lifecycle boundaries. Callers retain command construction, phase meanings,
configuration schemas, and all domain policy.

Static workflow, run, and step environments are copied and frozen. Child
launches strip inherited `MAMMOTH_*` values and inject only the canonical
execution ID, run name, invocation kind, and current phase. The first ordinary
failure stops dispatch and determines the structured `WorkflowResult` exit
code. Runs that never activate become blocked in memory without artifacts.
A post-lease resolver exception remains exceptional: Mammoth finalizes active
runs, terminates or reaps any child, closes observers, releases every lease,
and then re-raises the original error without allowing cleanup failures to
replace it.

`Workflow.run()` also publishes the optional entry-level group described
under Artifact layout. Publication is lazy and happens at most once per
invocation, the first time any run activates (inside that first run's own
blocked setup window, alongside its layout preparation and lease claim), so a
signal caught during earlier setup still returns a structured interruption
result with no artifacts at all, group included. Once published, the same
`GroupEventWriter` records every member run's and step's lifecycle
transitions for the rest of that invocation, and `WorkflowResult.group_id`
names it. `Workflow`'s `group_metadata` field is the caller's opaque,
JSON-compatible attachment; Mammoth validates only its JSON compatibility,
never interprets or redacts it, and it round-trips unchanged through the
manifest. Group event emission is best-effort and mirrors the JSONL
durability contract below: a write failure disables only that writer and
never the workflow it observes, and `_finalize_group()` records the terminal
group status from the outer `finally` block so it runs whether
`Workflow.run()` is about to return a result or re-raise a setup or cleanup
exception.

The ordinary workflow launcher inherits the parent terminal streams; callers
that explicitly need captured output use `run_captured_process()` for separately
drained text stdout and stderr, timeout facts, and the same bounded
launcher/descendant cleanup. Mammoth does not persist child environments or
inspect commands beyond executing their already-tokenized argument arrays,
and does not assign outcome policy or persist captured output.

`Workflow`'s optional `launcher` field is the supported dependency-injection
seam for step-level process creation, typed by the structural `Launcher`
protocol in `mammoth.workflow.launch`. `None`, the default, resolves to
`launch_process` at dispatch time, so default behavior is byte-for-byte
identical to a workflow with no `launcher` field. A caller-supplied launcher
only replaces the single `launch_process` call per step; signal handling,
lease ownership, lifecycle-JSONL emission, and cleanup remain owned by the
runner regardless of which launcher is active. This is the intended seam for
substituting subprocess creation in tests without importing or monkeypatching
`mammoth.workflow.runner` internals.

Framework-neutral callers may submit typed inputs to one
`BoundedBackgroundPipeline`. Its single worker preserves acceptance order,
applies a caller-selected bound across queued and active work, and associates
each result or failure with the accepted submission. Accepted work is never
cancelled. Results and failures remain owned until explicit acknowledgment;
interruption after queue acceptance is deferred for the submitter to
propagate after recording the handoff. Exact input-identity ownership checks
cover the caller return boundary, and interruption after acknowledgment removal
is deferred. Later flushes or closes therefore retain unacknowledged outcomes,
while cleanup remains idempotent and does not replace an active workload
exception. Optional done callbacks run on a pipeline-owned
dispatcher only after the submission state is final, so reentrant waits cannot
block the sole work thread and callback failures cannot replace worker outcomes.

`AsyncCheckpointPublisher` is a compatibility adapter over this pipeline. It
retains checkpoint-specific snapshot, plan, receipt, and `Future` APIs while
delegating ordered execution, pending bounds, failure ownership, and shutdown
to the framework-neutral lifecycle. Its generic mapping APIs default to an
`auto` capture policy: a conservative pre-allocation CUDA-memory check may
capture one immutable GPU snapshot and transfer it in the worker, otherwise
they retain the synchronous CPU snapshot path. The `cpu` policy always takes
the compatibility path. Project-owned ordered publication plans continue to
require caller-owned immutable captures.

## Runtime model

Mammoth uses seven nested concepts:

1. **Entry**: an arbitrary artifact root such as `runs` or `experiments`.
2. **Group**: an optional, entry-level record naming the runs one
   `mammoth.workflow.Workflow` invocation dispatched together. Groups are
   produced only by the workflow executor and consumed passively; an entry
   with no group ever published, and every run launched outside a `Workflow`,
   remain fully valid without one.
3. **Run**: a stable logical identity beneath one entry.
4. **Execution**: one immutable attempt to produce or continue a run.
5. **Producer**: a runner or process/rank that exclusively owns one stream.
6. **Phase and task**: arbitrary project-named work scopes.
7. **Observation**: lifecycle, progress, heartbeat, metric, or terminal state.

Core event consumers treat phase names, task names, coordinates, metric names,
and artifact extensions as opaque validated data.

`mammoth.core.derive_run_name(prefix, target_path)` derives a stable Run
identity from an opaque caller-supplied prefix and a target path: the target's
basename is sanitized into the name and an 8-hex-character SHA-256 digest of
its absolute path is appended, so same-basename targets at different absolute
paths stay distinguishable while the absolute path itself is never embedded
in the returned name. The result always satisfies `validate_run_name`.

Execution metadata may contain an optional sanitized `runtime` object. It
records allowlisted framework facts such as strategy, backend, and device type;
it never captures a complete process environment. Omitting this extension keeps
historical schema-version-1 records unchanged.

## Artifact layout

`RunLayout(entry, run_name)` resolves the stable run directory and its
operational paths. The default contract is:

```text
<entry>/<run-name>/
├── manifest.json
├── checkpoints/                 project-owned contents
├── logs/
│   ├── .logical-run.lock        Mammoth producer lease
│   ├── events.out.tfevents.*    Mammoth TensorBoard sink
│   └── executions/
│       └── <execution-id>/
│           ├── execution.json   immutable attempt identity
│           ├── runner.jsonl     optional workflow producer
│           ├── rank-N.jsonl     process/rank producer
│           └── rank-N.log       human-readable diagnostics
├── results/                     project-owned contents
└── vis/                         project-owned contents
```

The entry path is supplied by the caller. Mammoth does not assign semantic
meaning to entry names.

`GroupLayout(entry, group_id)` resolves the optional, entry-level group
contract published only by `Workflow.run()`. The default contract is:

```text
<entry>/.mammoth/groups/<group-id>/
├── manifest.json                immutable group manifest
└── events.jsonl                 append-only group event stream
```

`manifest.json` records the group ID, creation time, declared schedule order
(`run-major`/`step-major`), the ordered member run names and their planned
step names, and the caller-supplied opaque metadata block, published
atomically with `mammoth.core.artifacts.atomic_write_json` before the first
step dispatches. `events.jsonl` reuses the schema-v1 JSONL conventions from
`mammoth.core.events` at group scope: a monotonically sequenced, flush-per-
record append-only stream recording member run and step lifecycle
transitions (`run_started`/`run_completed`/`run_failed`/`run_blocked`/
`run_interrupted`, `step_started`/`step_completed`/`step_failed`/
`step_interrupted`) and exactly one terminal group status
(`group_completed`/`group_failed`/`group_interrupted`) recorded on every exit
path from `Workflow.run()`, including signal-driven interruption. A crashed
workflow leaves a group without that terminal record; consumers infer
staleness from event recency, exactly as they do for a run's own execution
events. The `.mammoth/` subtree is entirely optional: an entry that never
hosted a `Workflow` invocation, and every run launched outside one, remain
fully valid without it.

`mammoth.core.is_immutable_log_entry(log_dir, child)` classifies whether one
child of a run's `logs/` directory is Mammoth-owned immutable state that a
consumer log reset must preserve: the `executions/` container (and everything
nested beneath it) and the `.logical-run.lock` lease file answer `True`; every
other entry, including TensorBoard's own `logs/` contents, answers `False`.
Classification is by `child`'s path identity relative to `log_dir`, derived
from the same `EXECUTIONS_RELATIVE_DIR` and `LOGICAL_RUN_LEASE_FILENAME`
constants the layout already uses, never by inspecting `child`'s content or
filesystem state, so the answer is identical whether or not `child` exists and
whether it is a file, directory, or symlink. `child` must resolve inside
`log_dir`; a path that is `log_dir` itself or lies outside it raises
`ValueError` instead of guessing. Extend this classification in the same
change as any future addition to the immutable `logs/` layout.

## Logging responsibilities

### JSONL

JSONL is the live operational source of truth. Each producer exclusively owns
one append-only file. Records contain stable identity and lifecycle fields plus
opaque project coordinates and metrics. Progress values are unitless
producer-owned numbers: within one task, `completed`, `total`, and any
`throughput` must describe the same logical work quantity, while a producer
that cannot provide that relationship omits throughput. Mammoth neither
validates nor converts that domain meaning. Progress may be throttled and
replaced; lifecycle and terminal records flush immediately. A writer failure
disables only that writer and must not terminate the workload.

`RunObserver` asynchronously dispatches CPU-owned scalar observations. It owns
one bounded ordered worker per sink, so a slow TensorBoard writer cannot stall
JSONL dispatch. JSONL may coalesce unprocessed
non-final task progress; TensorBoard retains dense scalar history and applies
backpressure instead. Lifecycle, terminal, explicit flush, and shutdown
operations are per-sink barriers. Asynchronous mode currently rejects media
until it has an explicit immutable CPU snapshot contract. Its per-sink pending
bound also limits the number of retained JSONL progress scopes.

### TensorBoard

TensorBoard stores dense numerical and media history. Mammoth manages writer
ownership, logical steps, flushing, and shutdown. Projects choose metric names
and compute every value. An observation may select its dense-history logical
step explicitly; otherwise the sink infers it from generic coordinates and then
falls back to its local sequence. The primary process writes by default; other
ranks use a no-op sink unless explicitly configured otherwise.

### Text logs

Per-producer text logs contain diagnostics and tracebacks. Monitoring never
parses them for state; it consumes immutable metadata and JSONL events. Each
process holds an exclusive append lease for its rank log until logging closes.
`ExecutionLogging` composes that handler with the rank's JSONL sink and any
caller-selected sinks, including rank-aware TensorBoard. Consumers that own a
different text presentation use `ExecutionObservability` to obtain the same
JSONL-first observer lifecycle without claiming the rank log; the structured
bundle remains independently composable with Mammoth's text handler.

`RunObserver` owns optional periodic-heartbeat scopes above its sink fan-out.
Heartbeat scheduling stays process-local, observes the configured idle interval,
and retains sink failure isolation. Projects continue to choose phase, task, and
message policy.

## Monitoring

The generic monitor reconstructs executions, producer liveness, task trees,
progress, throughput, arbitrary metrics, failures, and lineage. Generic ETA
uses only producer-reported completed work, total work, and throughput or
observed time. A run-level monitor keeps every valid immutable execution
available for navigation while selected resume-lineage histories provide
continuous project-neutral metric state. Historical schema-version-1 `unit`
fields remain readable but do not affect reconstructed state or presentation.
Callers record that continuity explicitly with `parent_execution_id`; a
`resume_checkpoint` is an independent sanitized artifact reference and never
causes Mammoth to infer a parent from its location, name, phase, or timestamp.
When a PyTorch execution request resumes and joins an existing attempt, Mammoth
requires its caller-provided lowercase SHA-256 and starting epoch, then compares
the sanitized checkpoint reference, digest, parent ID, epoch, and optional
global step with immutable metadata before project work begins. Mammoth does
not inspect checkpoint contents or infer any of these facts.

An interactive terminal launches the optional Textual dashboard by default.
Textual owns refresh workers, keyboard navigation, scrolling, and responsive
layout; Rich renderables provide progress bars, execution panels, task tables,
and terminal-width-aware training charts. Redirected output remains one stable
ANSI-free snapshot. Conventional loss and learning-rate names affect
interactive presentation only and do not alter folding or assign scientific
meaning. `docs/MONITOR.md` owns the detailed reconstruction and presentation
contract.

The optional PyTorch integration may provide a training view for Mammoth's own
trainer. Project-specific pipeline projections are supplied as data or plugins,
not embedded in the monitor.

Mammoth owns the canonical names for standard resumable trainer checkpoints and
can discover matching direct children as ephemeral, filename-derived candidates.
Discovery neither opens artifacts nor establishes regular-file, symlink,
readability, payload, or compatibility guarantees; consuming projects retain
those validations and their resume-selection policy.

Viewer-host telemetry and execution-host telemetry are distinct. Local resource
sampling must never be presented as historical or remote execution provenance.
The Textual dashboard samples optional viewer CPU, memory, and GPU state,
labels it as viewer-host data, and isolates sampling failures. The monitor
remains passive: neither rendering nor telemetry writes artifacts, contacts
producers, or controls an execution.

## Direct execution sessions and PyTorch runtime

`mammoth.execution` owns the public `ExecutionSpec` and context-managed
`ExecutionSession` for one direct single-process execution. `create(spec)`
rejects a canonical inherited execution ID, claims the logical-run lease,
derives adjacency while it is held, publishes the immutable context, and opens
rank-zero JSONL and text logging. `attach(expected)` instead requires the four
canonical workflow-child variables, performs the same exact immutable identity,
topology, configuration, runtime, and resume-fact checks, and never claims the
lease. The session owns process and phase success, failure, interruption, and
skip records; observers and background pipelines created through its factories;
periodic heartbeats through the observer; deterministic cleanup; and one
terminal process outcome. Its resource order is pipeline, observer, execution
logging, then caller-supplied finalizers such as the direct lease. Cleanup is
idempotent and retains cleanup failures as notes on an active workload error.

`ExecutionSpec` deliberately remains outside `mammoth.core` because its direct
session composition depends on logging. Core continues to own only immutable
metadata, artifact paths, event schemas, leases, sanitization, and strict
context join/publication primitives.

`mammoth.execution` also owns `ExecutionObserver`, a detached no-op
phase/task/progress/heartbeat call surface, and `SessionExecutionObserver`,
which forwards the identical calls onto one live `ExecutionSession`. A
consumer writes workflow code once against this shared shape and passes
either the shared `NULL_EXECUTION_OBSERVER` instance or a session-bound
observer depending on whether monitoring is active, without branching at each
call site. The detached observer never creates artifacts, files, or events.

`Runtime` owns framework-level single-process or standard DDP
state. It resolves rank, local rank, world size, and device; initializes an
uninitialized default process group; exposes common object and tensor
collectives; validates optional caller-selected launch constraints; applies
caller-supplied rank weights to generic count and index partitions; and destroys
only a process group that it created. Execution establishment is available
separately from rank-logging startup so projects can attach presentation sinks
without recreating the runtime. The runtime does
not encode GPU models, concrete workload weights, or project topology rules.

Torch execution establishment is explicit and strict. `create_execution(spec)`
rejects a populated canonical execution-ID environment, claims the logical-run
lease on rank zero, resolves the previous attempt while holding that lease, and
publishes a new immutable execution using the invocation snapshot agreed by all
ranks. `attach_execution(expected)` requires the four canonical workflow-child
variables, joins their exact execution ID, and never creates metadata or claims
the producer lease. It validates the resolved run directory, run name,
invocation kind, phase membership, execution mode, world size, config reference,
runtime metadata, and all five nullable resume facts. Both operations propagate
rank-local preparation or validation failures through distributed startup
consensus before workload construction. After either operation succeeds, every
rank may open its own JSONL and text streams. TensorBoard's rank-aware sink and
trainer checkpoints default to rank zero. `mammoth.torch.ExecutionSpec` remains
a compatibility re-export of `mammoth.execution.ExecutionSpec`.

`mammoth.torch.ExecutionSession` is a compatibility adapter that composes the
neutral session and retains only Torch-specific trainer ownership. Factory
inputs such as models, optimizers, schedulers, loaders, policies, serializers,
metrics, and directly supplied observers remain borrowed. Owned trainers close
before neutral pipelines and observers; the neutral session then closes
execution logging and invokes the runtime lease and owned-process-group
finalizers. Projects may attach presentation cleanup through the session close
hook. No framework inheritance reaches from core back into PyTorch.

A workflow execution is owned by its single runner. Projects are responsible
for constructing any `torchrun` argv and for declaring execution topology that
matches the child runtime exactly; strict attachment does not relax workflow
metadata topology.

## Generic PyTorch trainer

The trainer accepts constructed objects:

- `torch.nn.Module`;
- optimizer and optional scheduler;
- training and optional validation `DataLoader` objects; and
- project functions that interpret batches and return scalar loss/metrics.

The project may also provide an accumulation policy, scalar reduction specs,
additive stateful metrics, metric sink routes, and a checkpoint policy. These
contracts describe mechanics only: names and update values remain opaque to
Mammoth, while project code owns every metric calculation and checkpoint
serializer.

Mammoth may own the ordinary loop mechanics: mode switching, device transfer,
precision, backward, accumulation, clipping, optimizer/scheduler steps,
standard DDP, callbacks, logging, interruption handling, and registered-state
checkpoint publication. A trainer may consume a `Runtime` for
device/rank identity and its active execution observer; constructing the
trainer without a runtime remains supported for callers that already own their
process group.

For the built-in batch mover, a CUDA trainer may prefetch one fully pinned CPU
batch on a dedicated copy stream while the current compute stream consumes the
previous batch. The compute stream waits only for that batch's copy-stream
transfer before invoking the project step and records the stream against its
CUDA tensor storage. The pipeline never changes DataLoader policy or runs a
project-supplied batch mover on a Mammoth-owned stream; ineligible batches use
the ordinary mover.

For DDP, Mammoth suppresses reducer communication only for non-final
microbatches in an accumulation window. Every rank reaches a Python-status
consensus before its final backward, then lets PyTorch's native DDP reducer
average gradient buckets during that backward. This preserves one reduction
boundary per logical optimizer step without trainer-owned CUDA scalar
materialization or per-parameter reductions. A rank-local error before that
final backward is reported coherently; a failure during native backward follows
the DDP launcher's fail-fast behavior because another rank may already be in a
reducer collective. The trainer requires standard native-reducer invariants;
projects with dynamic gradient participation must configure or own a compatible
DDP training loop.

`WarmupLinearLR` provides the reusable BERT-style zero-to-base warmup followed
by linear decay to zero. Projects choose its warmup ratio and optimizer-step
horizon. Checkpoint restore preserves the saved cursor, accepts an unchanged or
extended active horizon, rebases every optimizer parameter group onto an
extended curve, and rejects horizon shrinkage.

The caller-supplied module remains the canonical `base_model`. Mammoth derives
an `execution_model` by applying device placement, then optional DDP wrapping,
then optional caller-configured `torch.compile`. Step functions receive the
execution model. Mammoth retains the underlying DDP wrapper privately so
native gradient accumulation can use `no_sync()` only before its final
microbatch even when the execution path is compiled. Projects keep
architecture-specific methods on the canonical model when those methods are
not part of the ordinary forward contract.

Mammoth's validation-metric early-stopping callback owns the generic best-value
and consecutive-failure state machine and stops on the `patience`-th failed
check. `TrainerState.stopped_early` is the sole persisted terminal decision;
`fit()` returns without touching loaders, callbacks, or optimizer state when a
restored state is already terminal. The trainer reads the callback's transient
`improved` signal after validation to select best-model publication, while the
project continues to choose the metric, mode, patience, minimum delta, and
serializer.

`StepOutput` carries the optional loss, already-computed scalar metrics, and
opaque updates for registered stateful metrics. Mammoth reduces configured
distributed training-window summaries and train/validation epoch summaries.
Training applies batch routes at its configured logical-batch observation
cadence and epoch routes at epoch end. Validation DataLoader batches update
device-resident scalar accumulators and stateful metrics while their progress
observations omit metric values; validation reduces, materializes, and applies
epoch routes only at the validation epoch boundary. Validation routes therefore
support `epoch_name` only. Metrics configured with `distributed=False` remain
rank-local. `StepOutput.weight` controls weighted scalar means, with the default
weight of one treating each DataLoader batch equally; callers supply a batch
size when sample-weighted means are required. The trainer emits generic phase,
task, progress, heartbeat, completion, and failure observations; projects
select phase names, metric names, and display fields.
When a surrounding command already owns the outer training phase, it disables
the trainer's fit-level phase records while retaining Mammoth's nested task,
progress, validation-phase, heartbeat, and metric observations.

An accumulation policy receives rank identity and the local loader length, then
returns the local microbatch count and loss scale for each shared optimizer
window. Every rank must produce the same number of optimizer windows. Explicit
per-window scales cover unequal partial windows without assigning workload
meaning to Mammoth. Native DDP reaches failure consensus before each final
backward and lets the reducer average that window's gradient buckets. The
persisted global-step cursor counts all ranks' microbatches at completed
windows; step callbacks receive a deterministic rank-ordered position inside
the active window.
Consumers may supply post-optimizer metric providers for values, such as the
current scheduler rate, that only become authoritative after Mammoth completes
the optimizer and scheduler boundary.
The completed optimizer cursor remains checkpoint state, while consumers may
select a completed-step or zero-based logical clock for routed sink history.

The generic weighted policy converts arbitrary positive caller-supplied rank
weights into integer local microbatch counts and the DDP loss scale for one
global window. The matching batch sampler partitions opaque dataset indices,
drops incomplete global windows, and exposes `set_epoch()` for deterministic
reshuffling. Contiguous weighted index ranges support validation or inference
partitioning without constructing or interpreting a dataset. Consuming
projects retain concrete weights, device eligibility, DataLoader construction,
and every model and dataset policy.

For coarse independently executable work, the weighted task allocator accepts
only opaque string IDs, nonnegative numeric costs, and positive rank weights.
It considers tasks largest-cost-first, breaks equal-cost ties by ID, minimizes
each rank's projected cost divided by its weight, and breaks rank ties by lower
rank. Projects retain task discovery, cost estimation, skip/resume policy,
capacity checks, execution, artifacts, and failure reporting.

Complex algorithms with several optimizers, alternating updates, reinforcement
learning control flow, or custom collectives keep their loop in the consuming
project and use Mammoth only for orchestration, logging, monitoring, and
artifact mechanics.

## Checkpoint publication

Mammoth's optional PyTorch layer provides two checkpoint paths. The generic
trainer may use Mammoth's versioned registered-state payload and restore it
directly. Projects with established checkpoint schemas instead capture one
immutable snapshot and provide resumable and best-model serializers.

A trainer checkpoint policy captures project state when Mammoth's
`CheckpointSavePolicy` selects resumable or best-model publication and
translates its own format into Mammoth's typed restore contract. The trainer
first calls the policy once on rank zero through `inspect_checkpoint()`, then
shares its `CheckpointInspection`, including the immutable `RestoreOptions`
selection, with every rank. `load_checkpoint()` restores the project-owned
model representation and lets Mammoth restore or reset generic optimizer,
scheduler, callback, and terminal early-stop state while restoring trainer coordinates. Its
returned `TrainerCheckpointRestore` reports restored and reset components plus
opaque project metadata. Mammoth infers absent cursors from the active
accumulation plan and requires coordinates, terminal state, component reports,
and metadata to agree across DDP ranks before training resumes. Projects may
also request a synchronized manual publication. Rank zero publishes an
interruption snapshot without entering collectives that could deadlock when a
signal reaches only one process. The checkpoint context identifies the reason
and exposes the prior restore report to the project serializer.

`Trainer.fit()` publishes that interruption plan automatically when
`KeyboardInterrupt` escapes the loop, then preserves and re-raises the original
interruption. Callers may disable this behavior explicitly when an outer system
owns interruption persistence. Under DDP, failure consensus preserves an
interrupt as `KeyboardInterrupt` on every rank instead of converting it to an
ordinary failure; rank zero can therefore publish while peers perform only
local checkpoint shutdown.

The trainer applies publisher backpressure before asking a project checkpoint
policy to capture state. `CheckpointSavePolicy` selects `all` or `latest`
resumable retention, periodic cadence, and optional best-model publication.
Mammoth names zero-based `epoch_<N>.pt`, `latest_epoch_<N>.pt`, and
`best.safetensors`; best publication follows `EarlyStopping.improved` on every
validation epoch and is independent of resumable cadence. Manual and
interrupted saves publish resumable state only. The publisher receives
caller-owned immutable state before background work, bounds pending
publications, prepares and syncs every artifact before the first commit,
replaces best before the resumable commit marker, and retires old latest files
only after every commit succeeds. Before each atomic rename, Mammoth hashes the
completed temporary artifact and records its exact size. Successful plans yield
typed `PublishedCheckpoint` receipts with path, role, epoch, size, and SHA-256;
the trainer retains their futures and delivers receipts through observers and
callbacks during checkpoint flush. Paths are confined to the declared
checkpoint root. Projects retain serializers, payload fields, compatibility
rules, receipt consumers, and restore policy. Resume discovery remains a
filesystem concern; publication does not maintain a persistent checkpoint
catalog.

The lower-level ordered-plan API remains available for non-trainer artifact
publication where callers need custom names and retirement targets.

Atomic replacement applies to each file independently. An interruption between
ordered replacements may expose a prefix of the plan, so the caller places its
commit-marker artifact last. True multi-file transactions require a different
generation-directory and atomic-index layout and are not implied by this API.

## Generic PyTorch backend configuration

The optional PyTorch layer applies caller-selected float32 matmul precision,
CUDA matmul and cuDNN TF32 policy, cuDNN benchmarking, cuDNN determinism, and
deterministic-algorithm mode as process-global backend configuration. The same
API captures effective state and provides reversible overrides. A separate seed
policy selects Python, Torch CPU, and available Torch CUDA generators without
assigning a concrete seed or dataset-worker policy. Consuming projects retain
all chosen values, environment variables, logging filters, and DataLoader seed
construction.

Compile backends use `inductor` only when the caller leaves the backend unset;
an explicitly empty backend is invalid. Deterministic warning-only behavior is
likewise valid only when the caller explicitly selects deterministic-algorithm
mode, so an omitted mode cannot silently inherit a warning policy.

When a caller supplies both the newer matmul-precision control and the legacy
CUDA TF32 boolean, Mammoth expresses the legacy boolean's precedence through
the new API so PyTorch never enters an unreadable mixed-API state.

## Generic PyTorch profiling

The optional PyTorch layer profiles arbitrary caller-owned zero-argument
callables. The caller constructs the model and inputs, captures positional or
keyword arguments, selects inference, autocast, or autograd contexts, and owns
the semantic interpretation of every output. Mammoth owns synchronized cold
and steady-state timing, caller-labelled throughput, CUDA allocator evidence,
explicit caller-selected component ranges, normalized operation summaries,
optional traces, reversible Torch runtime settings, and versioned report
publication. Shape recording groups operation rows by input shape and delegates
its temporary runtime settings to the generic backend configuration API.

The default output summary records only project-neutral tensor and container
metadata. Callers provide semantic summarizers for predicted classes, masks,
tokens, numerical tolerances, or scientific metrics. Mammoth does not discover
architecture components, generate inputs, load checkpoints, select compile
scopes, or define what one work unit means. Instrumented profiler time remains
separate from the uninstrumented latency distribution.

For compound workflows, `NamedPhaseProfiler` measures caller-owned dependent
zero-argument regions under arbitrary caller-selected names and returns each
region result unchanged. It aggregates only completed caller-selected samples,
so callers can time warmup or diagnostic regions without reporting them.
Mammoth also exposes process-local device synchronization, CUDA allocator peak
reset and snapshots, and profiler-row normalization. Callers retain phase
names and order, profiler lifecycles, workload semantics, report schemas, and
any distributed aggregation policy.

## Compatibility policy

Artifact readers preserve compatibility with schema-version-1 execution and
event records from the originating project. Package and environment names may
change, but historical artifact paths and wire records remain readable. A
schema version changes only for a wire-contract change, not for a module
rename.
