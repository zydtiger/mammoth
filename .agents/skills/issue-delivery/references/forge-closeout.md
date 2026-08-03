# Forge Closeout

Use this two-phase procedure: complete the closeout-readiness gate before invoking merge, then complete reconciliation after merged delivery is verified and before declaring it complete. Apply consuming-repository instructions and stricter approval rules first.

## Establish authority

- Include anticipated acceptance-criterion wording or body amendments, checkbox edits, issue closing or reopening, and eligible milestone closure in the delivery plan and merge approval.
- Treat each body, issue-state, and milestone-state mutation as a forge write. Perform only writes already authorized; otherwise remain read-only and report the exact pending actions.
- Follow [the forge-body round-trip procedure](forge-body-round-trip.md) for every issue-body mutation. Do not report an edit as successful until its intended UTF-8 Markdown source matches the raw forge read-back.
- If a scope change or required body amendment emerges after approval, amend the delivery plan and obtain explicit authority for its issue-body writes before applying it.
- Record every approved milestone-item disposition on the forge issue, pull request, or milestone; chat, local plans, and final reports are not durable disposition records.
- Do not treat a closed issue, merged pull request, or zero-open-item milestone as proof that the underlying requirements passed.

## Gate the merge on closeout readiness

1. Before any evidence mapping, re-read every linked delivery issue's raw body and complete discussion as its readiness snapshot. Resolve any authorized scope amendment: record the decision and reason on the issue, round-trip the amended wording, and restart this gate. Block merge while an amendment is pending. Keep acceptance-criterion marker publication deferred until merged delivery is verified.
2. Identify each issue's acceptance criteria separately from examples and unrelated task lists, then map every criterion to current implementation, test, documentation, or other evidence. Block merge for any unmet or ambiguous criterion.
3. For each issue, prepare its intended post-merge body with only satisfied acceptance-criterion markers changed, and retain its raw body as the separate pre-merge source base. Preserve all wording, ordering, formatting, examples, and unrelated checkboxes. Do not publish it or check markers before merged delivery is verified.
4. Require merge approval to include per-issue authority for every exact body write and any issue closing or reopening needed to reconcile automation. Record the strongest body-revision and merge-time linked-issue-snapshot preconditions supported by the adapter. When either conditional protection is unavailable, block merge unless approval explicitly accepts that base-race limitation and its required immediate re-read. Block merge when any other closeout authority is missing.
5. When a linked issue declares no acceptance criteria, record only its marker, body, and count reconciliation as not applicable. Still record its expected post-merge state and require authority for any close or reopen write needed to reach that state.
6. Immediately before invoking merge, re-read every linked issue's raw body and complete discussion. Restart this gate at step 1 on any movement. If the merge operation cannot bind these readiness snapshots, proceed only under the explicitly approved limitation and report that linked-issue movement cannot be prevented between this final read and merge.

## Reconcile each delivery issue

Complete these steps separately for every linked delivery issue; retain distinct source bases, intended bodies, authority, evidence, and results.

1. Re-read the issue's raw body and complete discussion from the forge after merge verification and retain them as the current evaluation snapshot.
2. Before evaluating criteria or generating marker changes, resolve any changed scope: obtain explicit approval, record the decision and reason in the issue, round-trip the authorized wording amendment, then re-read the raw body and restart at step 1. Never use a discussion-only decision to check unchanged unmet wording, and never silently delete, rewrite, weaken, or check an unmet criterion.
3. Identify the acceptance-criteria set separately from examples and unrelated task lists, and reconfirm each criterion against merged evidence.
4. If no acceptance criteria exist, preserve the marker/body/count not-applicable result, perform no body edit, and skip directly to issue-state verification in step 10.
5. Compare the evaluation snapshot's raw body with this issue's current source base. If content drifted before evaluation, do not write the stale source: rebuild the intended source from the evaluated body, reconfirm that its diff changes only authorized acceptance-criterion markers, renew authority for the changed exact body, adopt the evaluated body as the new source base, and repeat this step. If the rebuilt intended source already equals the evaluated body, skip publication and continue at step 8.
6. Immediately before publishing, re-read the raw body, complete discussion, and body revision identifier when available. If any moved from the evaluation snapshot, restart at step 1 so scope and evidence are reevaluated. Apply the strongest conditional body-update or revision precondition supported by the adapter. If a conditional update rejects because the issue moved after this read, never retry the stale source: re-read the issue and restart at step 1 for scope processing, evidence evaluation, marker-only rebuilding, exact-diff confirmation, renewed per-issue authority, and retry from a new safe source base. When no conditional update exists, proceed only under the explicitly approved base-race limitation and only if this immediate re-read still exactly matches the evaluation snapshot; report that the adapter could not prevent a concurrent write rather than claiming overwrite protection.
7. Publish the authorized intended body by changing only satisfied acceptance-criterion markers. Preserve its wording, ordering, formatting, and every unrelated checkbox or body section.
8. Re-read the raw issue body and complete discussion immediately after every body update, verify the body against the intended source, and confirm that the intended markers changed without collateral edits. On any unexpected movement or mismatch, do not accept or retry the stale source: distinguish a concurrent edit from unexplained transport or forge damage, establish a safe fresh raw source, and restart at step 1 for scope processing, evidence evaluation, marker-only rebuilding, exact-diff confirmation, renewed authority, and conditional retry. Never adopt unexplained damaged content as a new source base.
9. Require the checked acceptance-criteria count to equal the total before closing or accepting automated closure. If any criterion remains unmet, keep the issue open; if automation closed it, handle reopening in step 10.
10. Immediately before verifying issue state, re-read the raw body and complete discussion. Restart at step 1 if either contains movement not produced and verified by this reconciliation; this freshness check also prevents a no-criteria issue from skipping newly added criteria. Then verify the expected state and perform only an authorized close or reopen write. After a state write, re-read the raw body, complete discussion, and state; compare them with the last verified snapshot while allowing only this reconciliation's state change, and restart at step 1 on any other movement. Reopen or keep the issue open as needed until the restarted reconciliation succeeds. Require a closed issue with acceptance criteria to contain no unchecked acceptance-criteria boxes. Without required authority, report the inconsistent state and pending write.
11. If the body write, raw read-back, count verification, or state reconciliation fails after merge, keep or reopen the affected issue and report the whole delivery as incomplete. Continue remediation until every linked issue's intended body and state are verified; merge success does not clear this failure.

Forge APIs differ in whether a closed issue body can be edited and in the order of linked-issue automation. Use the authenticated GitHub or Gitea adapter selected for the repository, inspect the current state before each write, and reconcile the final body and state rather than assuming one universal sequence.

## Reconcile associated milestones

Complete this section once for every distinct milestone associated with a linked issue. Skip it when none of the issues has a milestone.

1. Re-read the associated milestone, including its title, description, state, declared completion conditions, and reported open and closed counts.
2. Retrieve the complete milestone roster with pagination, including both issues and pull requests. For every item, verify both its closed state and its delivery evidence or a disposition explicitly approved and recorded on the forge; inspect delivery issues for unresolved acceptance criteria, and require a pull request to be merged unless its supersession or removal from scope was explicitly approved and recorded on the forge.
3. Treat a closed-unmerged pull request, abandoned issue, or other closed item without verified delivery or a disposition approved and recorded on the forge as a blocker. Approval in chat, a local plan, or the final report does not satisfy this record. Evaluate descriptive completion conditions against current evidence; do not substitute item counts or closed state for intended outcomes.
4. Close the milestone only when every roster item is closed and verified complete or has a disposition explicitly approved and recorded on the forge, every delivery issue has resolved acceptance criteria, every declared condition is satisfied, and milestone closure was authorized.
5. Otherwise leave the milestone open and report each blocking item, unresolved criterion, unmerged pull request, disposition lacking approval or a forge record, unmet condition, incomplete roster query, or missing approval.
6. After an authorized closure, re-read the milestone and verify its closed state and final open and closed counts.

## Verify representative paths

Exercise the closeout logic against representative GitHub and Gitea states before relying on revised guidance:

- a merged issue with satisfied but unchecked criteria;
- an issue whose unrelated body content changes between merge approval and closeout, proving that no stale body is written and renewed authority is obtained for a rebuilt marker-only source;
- an issue that changes after the immediate pre-write comparison, proving that a conditional update rejects the stale write and recovery performs a fresh read, marker-only rebuild, exact-diff confirmation, renewed authority, and conditional retry;
- an adapter without conditional body updates, proving that merge is blocked before invocation until its base-race limitation and immediate pre-write re-read are explicitly approved and later reported without claiming overwrite protection;
- a linked issue amended after readiness approval but before merge, proving that the final freshness check restarts the entire readiness gate;
- an open or automatically closed issue with one unmet criterion;
- a linked issue with no acceptance criteria, including criteria added before state verification, proving explicit marker/body/count N/A reporting only while fresh, no body edit, continued state verification, and authority for any close or reopen write;
- a criterion changed during a conditional retry, proving that recovery restarts scope processing and evidence evaluation before rebuilding markers;
- multiple linked delivery issues, verifying separate source bases, authority, writes, re-reads, and all-issues completion;
- an explicitly approved scope amendment recorded and round-tripped before pre-merge evidence mapping, with the gate restarted and no merge or marker publication allowed first;
- the final completed item in an eligible milestone;
- a milestone containing a closed-unmerged pull request without approved disposition;
- a milestone containing an abandoned issue approved for disposition in chat but lacking a forge record;
- a zero-open-item milestone with an unmet descriptive condition;
- an issue with no milestone; and
- a closeout without approval for forge mutations.

For every path, verify the expected writes or read-only outcome, the required re-reads, preservation of unrelated task lists, and the reported blockers.

## Report final state

Report:

- for each linked delivery issue, its pre-merge closeout-readiness result, evidence mapping, authority, body-precondition support or approved limitation, and marker/body/count not-applicable status when relevant;
- for each issue, every acceptance criterion and the evidence or blocker used to evaluate it;
- per-issue acceptance criteria checked, total, and still unresolved, or explicit marker/body/count N/A;
- each final issue state and whether it was edited, reopened, or closed;
- each associated milestone's identity or the absence of milestones, final state, and open and closed item counts;
- each milestone roster item's delivery evidence or approved disposition, including the verified forge artifact that records it;
- per-milestone evidence used for descriptive completion conditions;
- every performed write and its verified re-read; and
- every pending action or blocker that prevents complete delivery.
