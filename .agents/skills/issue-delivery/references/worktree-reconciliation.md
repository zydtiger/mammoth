# Worktree and Head Reconciliation

Use this procedure before resuming task work, rewriting or publishing a branch, merging, or cleaning up.

## Own the worktree by task

- Reuse one worktree across sessions for the same task.
- Allow one active writer. Read-only inspection may share it; concurrent writers need separate approved branches and worktrees.
- At a pause or handoff, stop writers and record the worktree path, branch, `HEAD`, status, and any unfinished Git operation. A resumed writer must verify that record before editing.
- If ownership or state is uncertain, do not take over the checkout. Preserve it and use a separate approved worktree.
- Leave a dirty main checkout untouched. Create a separate task worktree directly from the fetched base commit when isolation is safe and approved.

## Reconcile the four heads

Before reusing a branch or pull request:

1. Fetch the intended base and feature refs without pruning or changing the current checkout.
2. Re-read the pull request, including its base and head repositories, refs, commits, draft state, checks, reviews, and mergeability.
3. Record the exact base, local feature, remote feature, and pull-request head commits. Treat a missing ref as a distinct state.
4. Compare ahead/behind counts and commits unique to each side. Inspect the patches, not just the counts.
5. Inventory tracked, untracked, and ignored paths. Compare changed paths with rename detection so published renames and deletions are visible.

Do not assume any head is authoritative. Stop until every divergence, remote-only commit, rename, deletion, and dirty path is understood.

## Choose what to preserve

- Preserve remote-only work unless the user explicitly approves discarding it.
- Before a rebase, replacement, or other rewrite, show the old local, remote, and PR heads; the proposed result; and which commits or renames will be preserved or dropped.
- After material reconciliation, rerun affected validation and a clean-context audit.
- Obtain publication approval for the exact resulting commit and expected remote head. Use `--force-with-lease=<remote-ref>:<expected-oid>` for a non-fast-forward update; never use an unleased force push.

If the original task worktree is dirty or unsafe for PR follow-up, preserve it. With approval, create a clean follow-up worktree at the exact fetched remote PR head and keep the original read-only.

## Preflight and publish

1. Fetch again and stop if the base, remote feature ref, or PR head moved.
2. Run `git merge-tree --write-tree <base> <head>`, or a documented worktree-neutral equivalent, to distinguish content conflicts from forge workflow gates. Inspect custom merge drivers before relying on the result.
3. Validate and audit the final head. Record the base and head commits, validation results, audit result, and the exact diff reviewed.
4. Immediately before pushing, verify the exact audited commit, clean worktree, destination ref, and expected remote commit.
5. Push the audited commit itself as `<audited-oid>:<remote-feature-ref>`. For a rewrite, use the approved exact force-with-lease expectation.
6. Fetch immediately after the push. Verify the remote feature ref and existing PR head equal the audited commit. For a new PR, verify the remote ref first, create the approved PR, then read it back and verify its head.
7. Refresh the PR body and validation evidence whenever the published head changes. Use an explicit intended Markdown source, a newline-preserving transport, and raw forge read-back comparison for every refresh.

Do not merge unless the audited local head, remote feature head, and PR head are identical.

## Merge and verify

Immediately before merging, re-read the PR and compare its base and head with the audited state. If the base moved, rebuild conflict-preflight and affected validation or audit evidence.

Use every commit precondition the adapter supports. If it cannot bind the base or head, require approval for that limitation and repeat the full state read immediately before invoking it. Immediately afterward:

1. Fetch the base and feature refs and re-query the PR and linked issue.
2. Verify the remote feature ref and PR head still equal the audited commit, or verify their expected deletion from recorded pre-merge values.
3. Verify merged state, strategy, resulting base commit, checks, reviews, and expected issue closure.
4. For a merge or fast-forward, verify the published head is an ancestor of the base. For squash, rebase, or resolved-conflict merges, verify patch or tree equivalence instead.
5. Stop if delivered content or forge state is uncertain.

## Synchronize and clean up

- Update a local base only when it is the expected branch, its worktree is clean and not in use, and the fetched base is a fast-forward. Use fast-forward-only behavior with autostash disabled. Otherwise leave it untouched and report why.
- Before removing a feature worktree, inventory tracked, untracked, ignored, and initialized-submodule state. Preserve anything dirty or unexplained and never use forced removal.
- Delete a local feature branch only after verifying delivery, confirming no worktree uses it, and confirming it still names the expected commit. Remove its repository-local `branch.<name>.*` configuration as part of the same approved cleanup.
- Delete a remote feature branch only when policy and approval allow it, and protect deletion with the expected remote OID.
- Report every deferred update and retained worktree, branch, ref, or artifact.
