---
name: issue-discovery
description: Inspect a codebase and a user's feature request or bug report, search for existing work, draft a complete issue, obtain approval, and create the approved issue on GitHub or Gitea. Use when a request should be analyzed and turned into a forge issue, when the user asks to draft or file an issue, or when issue scope and acceptance criteria must be established before implementation.
---

# Issue Discovery

Turn an initial request into one evidence-backed GitHub or Gitea issue. Creating an issue never authorizes implementation.

## Rules

- Read and obey repository instructions before inspecting or writing.
- Treat code, tests, docs, history, and existing issues as evidence; separate facts from inference.
- Remain read-only until the user approves the complete issue draft.
- Preserve approved detail. Never create a duplicate or invent a missing label.
- Never expose credentials or private content outside the selected forge.

## Workflow

1. **Resolve the target.** Identify the repository from the request or local remote. Inspect the remote host and repository instructions.
2. **Select the adapter.**
   - GitHub: prefer an available GitHub connector; use authenticated `gh` for unsupported operations.
   - Gitea: use authenticated `tea`.
   Verify the account and repository, and consult current `--help` output. Never use `gh` against Gitea or `tea` against GitHub. Ask if the target remains ambiguous.
3. **Investigate.** Inspect relevant implementation, tests, docs, configuration, and history. Establish current behavior, impact, affected components, constraints, and material compatibility, security, migration, or data-loss risks. Do not implement a fix.
4. **Search existing work.** Search open and closed issues for duplicates or substantial overlap. If one exists, report it and stop unless the user explicitly wants a distinct follow-up.
5. **Query labels.** Read the repository's current labels and select only applicable existing labels.
6. **Draft.** Read [references/issue-template.md](references/issue-template.md). Present the exact forge, repository, title, labels, complete body, acceptance criteria, tests, and exclusions.
7. **Approve.** Obtain explicit approval to create that exact issue. Revise and re-present after feedback; material post-approval changes require renewed approval. Approval to investigate or draft is not approval to create.
8. **Write and verify.** Immediately restate the target, title, labels, and single intended write. Create through the selected adapter, read it back, and verify number, URL, title, labels, and body.
9. **Stop.** Report the created issue. Do not create a branch, worktree, commit, or pull request without a separate pickup request.
