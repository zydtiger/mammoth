# Forge Body Round Trip

Use this procedure for every pull-request or issue body creation or update, including pull-request refreshes, acceptance-criterion edits, and closeout changes.

## Prepare the intended source

1. Write the complete intended body to an explicit UTF-8 Markdown source file. Treat that file as the publication authority for the operation.
2. Keep each logical prose paragraph on one uninterrupted Markdown line. Use exactly one blank line to separate headings, prose paragraphs, lists, checklists, code fences, blockquotes, tables, and other Markdown blocks; do not add blank lines merely for visual wrapping.
3. Preserve line breaks that are part of Markdown structure, including adjacent list items, table rows, blockquote lines, and fenced code contents.
4. Begin every unordered-list item with a capitalized lexical word after its marker. Begin every checklist item with a capitalized lexical word after its checkbox marker.
5. Add a capitalized introductory word before inline code, a link, a command, or a lowercase identifier that would otherwise begin a bullet or checklist item. Ordered lists, fenced code contents, and non-list Markdown are outside this capitalization rule.
6. Resolve the selected `issue-delivery` skill directory to an absolute path, record it in `ISSUE_DELIVERY_SKILL_DIR`, and validate the source before publication:

   ```sh
   python3 "$ISSUE_DELIVERY_SKILL_DIR/scripts/forge_body.py" check-source intended-body.md
   ```

Do not rewrite substantive content while correcting formatting.

## Select a newline-preserving transport

- Prefer a forge adapter's native body-file argument when it reads the file contents without shell reinterpretation.
- Otherwise send a correctly encoded JSON request body through an authenticated API adapter. Let the JSON encoder represent the source string; do not manually replace real line breaks with textual `\n` sequences.
- For GitHub, use an authenticated adapter path that accepts the source file or a correctly encoded API payload.
- For Gitea, use an authenticated `tea` body-file capability or a correctly encoded API payload.
- Do not place multiline body text directly in shell arguments or ordinary Git push options when their newline behavior is ambiguous.
- If Gitea AGit is unavoidable, use its supported base64 description push option only after confirming that the target server supports it. Otherwise stop and select another authenticated adapter.

Keep shell argument quoting, JSON escaping, and stored Markdown distinct. A JSON payload may contain encoded newline escapes in transit, but the forge body returned by the API must contain real newline characters.

## Read back and compare

1. Immediately after every create or update, read the raw body from the forge through the authenticated adapter. Do not verify only rendered HTML or command-formatted output.
2. Save the returned UTF-8 body bytes to a separate file without adding, deleting, wrapping, or interpreting lines.
3. Compare the source and returned body:

   ```sh
   python3 "$ISSUE_DELIVERY_SKILL_DIR/scripts/forge_body.py" verify intended-body.md returned-body.md
   ```

The verifier permits only CRLF-to-LF normalization and the presence or absence of one terminal newline. Everything else must match. Source-format violations, transport-generated literal `\n` sequences, damaged fences, body mismatches, and read-back failures prevent successful publication reporting.

Exact source comparison is authoritative for literal `\n` text: an intentionally authored example is accepted when it appears identically in the intended source and returned body. Do not reject a literal escape solely because it exists.

## Remediate a failed publication

Treat failure as an incomplete publication, even when the forge created the issue or pull request. Preserve the intended source, correct only the transport or formatting defect, repeat the body write through a newline-preserving adapter, then repeat raw read-back and verification. Report success only after the latest returned body passes.
