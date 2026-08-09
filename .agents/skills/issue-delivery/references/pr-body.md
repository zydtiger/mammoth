# Pull Request Body

Build the body from the actual issue, commits, diff, and latest validation. Never reuse stale counts or claim checks that were not run.

Construct the complete intended body, then return it to the main delivery workflow for publication through `$issue-pr-body-contract`.

## Summary

Explain the user-visible outcome and main changes.

## Implementation details

Explain important behavior, design choices, security boundaries, persistence, APIs, and failure handling where relevant.

## Step and commit structure

List phased implementation commits and material follow-up fixes.

## Validation

List exact commands and results. Distinguish automated, manual, browser, integration, destructive, skipped, and ignored tests.

## Compatibility, risks, and limitations

State migrations, platform constraints, external limits, network requirements, private API risks, fallbacks, and remaining limitations.

## Issue linkage

Use linkage syntax supported by the selected forge. Verify closure after merge instead of assuming it.

## After follow-up commits

Refresh the summary, design details, commit structure, validation, limitations, and linkage in a new authoritative source file, then return it to the main delivery workflow for verified publication.
