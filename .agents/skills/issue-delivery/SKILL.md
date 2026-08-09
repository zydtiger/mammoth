---
name: issue-delivery
description: Pick up an existing GitHub or Gitea issue and deliver it through an isolated worktree, iterative implementation steps, validation, clean-context audits, focused commits, pull-request drafting, approval-gated merge, verification, and cleanup. Use when the user explicitly asks to implement, pick up, complete, or ship an existing forge issue with disciplined review and Git lifecycle management.
---

# Issue Delivery

Deliver an existing issue through explicit planning, implementation, review, publication, merge, and cleanup gates.

## Invariants

- Read and obey repository instructions; they override this generic workflow.
- Require an existing issue and an explicit pickup request. Issue creation alone is not implementation authority.
- Treat a worktree as task-owned, not session-owned. Reuse it when work resumes, allow only one active writer, and give concurrent writers separate branches and worktrees.
- Preserve unrelated changes and unexplained commits or files. Never reset, clean, force-remove, or silently stash them.
- Validate and commit each implementation step before starting the next, and independently audit it unless the approved handoff assigns that gate to an outer orchestrator.
- Apply `$issue-pr-body-contract` to every pull-request or issue body write; keep all forge-body mechanics in that skill.
- When an approved managed-worker handoff explicitly assigns the independent-review gate to an outer orchestrator and forbids producer review delegation, let that recorded policy override only the producer's audit launch. Keep validation, focused commits, raw evidence, and every other delivery gate unchanged. Never infer outer ownership from delegation alone.
- Obtain explicit approval before creating lifecycle state, publishing, merging, or cleaning up unless the user already authorized that action. At the ready-to-merge gate, apply the compound authorization defined below rather than requiring the user to enumerate its routine closeout actions.

## Workflow

### 1. Orient

Read the complete issue and discussion, repository instructions, linked work, remotes, current branch and worktree state, base branch, and any existing pull request. Confirm the issue is open, in scope, and not already implemented.

Choose the forge adapter from the remote: use GitHub capabilities or `gh` for GitHub and authenticated `tea` for Gitea. Verify the account and repository before any forge write.

### 2. Plan and approve

Follow repository policy for direct-base eligibility, branch names, and worktree locations. Propose any lifecycle detail not already authorized: branch, base, worktree, implementation steps, validation, compatibility concerns, pull-request plan, anticipated acceptance-criterion body amendments and checkbox edits, issue-state reconciliation, eligible milestone closure, and expected cleanup targets. Include how `$issue-pr-body-contract` will be applied and the strongest available body-revision precondition for every anticipated pull-request or issue body write. Prepare the normal post-merge closeout and cleanup actions so they can be covered by one later merge approval; do not require the user to enumerate routine acceptance-criterion marker edits, issue close or reopen actions, worktree removal, or expected branch deletion. Record only the acceptance-criteria marker, body, and count reconciliation as not applicable when no acceptance criteria exist; issue-state verification remains required. Obtain approval before creating a branch, worktree, commit, or forge state.

### 3. Prepare or resume the task worktree

Read [references/worktree-reconciliation.md](references/worktree-reconciliation.md). Reuse a safe task-owned worktree when resuming. Do not create a worktree per session. If another session must write concurrently, use a separate approved branch and worktree.

Before editing existing work, reconcile the base, local feature head, remote feature head, and pull-request head. Stop on unexplained divergence, commits, renames, dirty files, or uncertain ownership. Preserve an unsafe or dirty original worktree; when approved, continue PR follow-up from a separate clean worktree at the exact remote PR head.

When no reusable worktree exists, fetch the approved base and create the approved branch in a separate worktree without changing a dirty main checkout.

### 4. Implement in focused steps

For each approved step:

1. Implement only that step and inspect the diff for unrelated changes.
2. Run formatting, targeted tests, and proportionate regression checks.
3. Unless the approved handoff explicitly records an outer-orchestrator review owner and forbids producer review delegation, launch a fresh read-only independent reviewer through an applicable repository-selected audit capability or host-native review mechanism. Supply the requirements, current-step scope, raw diff, and validation evidence without inherited implementation context or preferred conclusions.
4. For a producer under that explicit outer review contract, do not launch audit, overview, critique, or certification subagents; instead preserve the raw diff and validation evidence for the orchestrator's fresh reviewer. Otherwise verify each audit finding, fix valid findings, and re-audit after material corrections.
5. Stage only intended files, create one focused commit using repository conventions, and verify a clean step boundary.

Do not defer every audit or commit until the end.

### 5. Publish the pull request

Run the full relevant validation suite and review the complete issue diff and commit sequence. Read [references/pr-body.md](references/pr-body.md) and build the body from the current audited head. Invoke `$issue-pr-body-contract` directly and require its successful publication verification before reporting success.

Follow the reconciliation reference for final conflict preflight and publication. Immediately before pushing, verify the exact local commit, clean worktree, destination, and expected remote commit. Push the exact audited commit and use an exact `--force-with-lease` expectation for any rewrite. Immediately fetch and verify that the remote branch and pull-request head equal the audited commit.

Verify the PR base, title, issue linkage, displayed commits, checks, and review state. Require `$issue-pr-body-contract` to complete successfully for the body. Distinguish a real Git content conflict from draft status, missing checks, or other forge workflow gates.

### 6. Merge gate

PR creation is not merge authorization. Re-read the PR and verify its exact head, current base, checks, reviews, mergeability, and requested strategy. If the base moved, rerun conflict preflight and any affected validation or audit before establishing closeout readiness.

Read [references/forge-closeout.md](references/forge-closeout.md) and complete its pre-merge closeout-readiness gate before invoking merge. For every linked delivery issue, apply and round-trip any authorized scope amendment, re-read the issue, and only then map each declared acceptance criterion to current evidence. Require every criterion to be satisfied and unambiguous, prepare the issue's intended post-merge body with only the authorized marker changes, and retain its raw body and discussion as a separate readiness snapshot for the compound merge authorization. Do not check criteria before verifying merged delivery. Block the merge for a pending scope amendment, an unmet or ambiguous criterion, a body edit that cannot preserve unrelated content, an unapproved forge-precondition limitation, or missing closeout authority. When a linked issue has no acceptance criteria, record only its marker, body, and count reconciliation as not applicable and still verify its expected post-merge state.

Only after establishing the ready snapshot, request or accept merge authorization for it. Do not consume an imperative merge instruction received before readiness as authorization for a snapshot established later; report the completed readiness result and obtain a gate-time instruction instead. At this ready-to-merge gate, treat an unambiguous instruction to merge the identified pull request, such as "merge the PR" or "go ahead and merge," as compound authorization to merge the exact verified head with the established strategy; verify the resulting delivery; reconcile only evidence-backed acceptance-criterion markers and the linked issues' required close or reopen states; fast-forward a safe local base when applicable; and, only after closeout succeeds, remove the clean task-owned worktree and expected local and policy-eligible remote feature branches. Own these routine lifecycle details without requiring the user to list them separately. Do not treat a hypothetical question, status request, readiness check, or discussion of a future merge as authorization.

Apply compound authorization only to the ready snapshot. Obtain fresh approval when the target or operation changes materially, including changed scope or acceptance-criterion wording, a different pull-request head or merge strategy, a different linked-issue set, an unapproved adapter-precondition limitation, milestone closure or item disposition, dirty or unexplained cleanup state, forced cleanup, discarded work, reset, or unprotected deletion. Continue under the original compound authorization after unrelated issue-body drift only when a fresh read proves that scope and acceptance criteria are unchanged and `$issue-pr-body-contract` validates the newly authorized marker update from that fresh body.

Use the strongest commit and linked-issue readiness preconditions the forge adapter supports. Immediately before invoking merge, re-read the PR, base, head, and every linked issue's raw body and complete discussion; restart the affected validation or closeout-readiness gate on any movement. When the adapter cannot bind the approved commit or issue snapshots to the merge operation, proceed only if the user has approved that limitation and the immediately repeated reads still match the approved state. Fetch and re-query immediately after the merge.

### 7. Verify and close out

Verify the merged state, resulting base, remote and PR head identity, delivered content, required checks, and linked-issue closure. For an ancestry-preserving merge, require the published head to be an ancestor of the base. For squash or rebase, verify equivalent delivered content instead.

Resume the closeout reference at its per-issue reconciliation section immediately after merge verification. Re-read each final issue and discussion, process any newly emerged approved scope amendment before evaluation, reconfirm every acceptance criterion against merged evidence, and reconcile only the compound-authorized acceptance-criteria markers and issue state. Invoke `$issue-pr-body-contract` directly for every issue-body edit while applying the strongest available body-revision precondition. When acceptance criteria exist, require the final checked count to equal their total before accepting automated closure; otherwise report marker, body, and count reconciliation as not applicable while still verifying issue state. Never treat issue or pull-request closure as evidence that a criterion passed. If any linked-issue reconciliation fails after merge, keep or reopen that issue and report the whole delivery as incomplete until every failure is corrected and reverified.

For every milestone associated with a linked issue, inspect its description, completion conditions, and complete issue and pull-request roster. Close it only when every item has verified delivery or a disposition explicitly approved and recorded on the forge and every declared condition is satisfied under the approved closeout gate; closed state or zero open items alone is insufficient. Re-read each issue after every body or state write and re-read each milestone after closure. Report per-issue criterion evidence and checkbox totals, issue states, per-milestone roster evidence, states and item counts, performed writes, pending actions, and exact blockers.

### 8. Synchronize and clean up

Synchronize a local base only by fast-forward. If its worktree is dirty, diverged, owned by another writer, or otherwise uncertain, leave it untouched and report the deferral.

When merge proceeded under the compound authorization above, its cleanup authority remains current through successful verified closeout and no second approval prompt is required. Otherwise clean up only with current approval. Inventory tracked, untracked, ignored, and submodule state; preserve anything unexplained; remove a worktree without `--force`; and delete branches only after verifying delivery and their expected OIDs. Stop before cleanup and obtain fresh approval for any material deviation from the ready snapshot or safe cleanup envelope. Report every retained worktree, branch, or artifact.

Never force cleanup merely to make the repository look finished.
