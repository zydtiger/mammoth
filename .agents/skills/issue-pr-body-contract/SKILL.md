---
name: issue-pr-body-contract
description: Validate, publish, and verify Markdown bodies for forge issues and pull requests. Use when creating or updating an issue or PR body, verifying raw body round trips, or reconciling acceptance-criterion markers without rewriting existing content.
---

# Issue and PR Body Contract

Use this package as the single authority for forge-body source validation, raw read-back verification, and closeout marker-only updates.

## Source and publication

1. Write the complete intended body to an explicit UTF-8 Markdown source file.
2. Keep each logical prose paragraph on one uninterrupted Markdown line. Use exactly one blank line between headings, prose paragraphs, lists, checklists, code fences, blockquotes, tables, and other Markdown blocks. Preserve line breaks that are part of Markdown structure. Treat list-item capitalization as optional editorial guidance rather than a publication gate, and never rewrite substantive content to satisfy that guidance.
3. Resolve this skill directory to an absolute path in `ISSUE_PR_BODY_CONTRACT_DIR`, then validate a newly authored body before publication:

   ```sh
   python3 "$ISSUE_PR_BODY_CONTRACT_DIR/scripts/forge_body.py" check-source intended-body.md
   ```

4. Publish through a newline-preserving body-file argument when the adapter supports one. Otherwise send a correctly encoded JSON API payload; let the encoder represent real line breaks instead of manually inserting textual `\n` sequences. Do not pass multiline body text through an ambiguous shell argument. For Gitea AGit, use a supported base64 description option only after confirming server support; otherwise choose another authenticated adapter.
5. Immediately read the raw body through the authenticated forge adapter into a separate UTF-8 file without wrapping or interpreting it. Do not verify rendered HTML or command-formatted output.
6. Verify the returned body:

   ```sh
   python3 "$ISSUE_PR_BODY_CONTRACT_DIR/scripts/forge_body.py" verify intended-body.md returned-body.md
   ```

The verifier permits only CRLF-to-LF normalization and the presence or absence of one terminal newline. Structural source violations, transport-generated literal `\n` sequences, damaged Markdown, body mismatches, and read-back failures prevent successful publication reporting. An intentionally authored literal `\n` remains valid when the returned body contains it identically.

## Closeout preservation

For an existing issue whose wording must be preserved, validate only the base-to-intended marker diff. This permits unchecked-to-checked checklist markers and rejects every other content change:

Marker-only validation uses `markdown-it-py` for CommonMark structure and fails closed when that package is unavailable. Ensure it is installed in the Python environment that runs the validator.

```sh
python3 "$ISSUE_PR_BODY_CONTRACT_DIR/scripts/forge_body.py" check-marker-only base-body.md intended-body.md
```

After publication, read the raw body and verify it without source-style lint:

```sh
python3 "$ISSUE_PR_BODY_CONTRACT_DIR/scripts/forge_body.py" verify-marker-only base-body.md intended-body.md returned-body.md
```

Do not use marker-only validation for a newly authored body. Do not use the ordinary `verify` command or source-style failures to justify rewriting legacy body content during closeout.

## Failed publication

Treat a failed validation, transport, read-back, or verification as incomplete publication. Before retrying any write, re-read the target identity, raw body, and revision when available, then classify the outcome:

- For source validation that failed before a write, correct only the source-format defect and validate again.
- For a create request that may have been accepted, never create a second object. Resolve the created object's identity and verify or repair that exact object; stop when its existence cannot be established safely.
- For an update whose current body or revision moved away from both the expected base and intended result, never replay the stale source. Return the movement to the calling workflow for a fresh snapshot, evidence evaluation, and authorization decision.
- Retry publication only when evidence shows that the previous write was not accepted or the exact current target still satisfies the caller's approved write precondition. Verify a fresh raw read-back after any retry.

Preserve the intended source throughout recovery. Do not convert a transport or verification failure into authority to overwrite concurrent content.

Return a clear result to the calling workflow:

- `verified`: the intended body matches the fresh raw read-back.
- `source-invalid`: validation failed before publication.
- `target-moved`: the current body or revision no longer matches the approved write base or intended result.
- `publication-incomplete`: acceptance, target identity, or verification cannot be established safely.

Only `verified` completes the body operation. The caller owns lifecycle decisions after a result, including rebuilding evidence or authorization after `target-moved` and retaining state after `publication-incomplete`.
