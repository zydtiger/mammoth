import base64
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("forge_body.py")
SPEC = importlib.util.spec_from_file_location("forge_body", MODULE_PATH)
assert SPEC and SPEC.loader
forge_body = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forge_body)


class JsonApiAdapterFixture:
    def publish(self, body):
        self.request_bytes = json.dumps(
            {"body": body}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        stored_body = json.loads(self.request_bytes.decode("utf-8"))["body"]
        self.response_bytes = json.dumps(
            {"body": stored_body}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

    def read_raw_body(self):
        return json.loads(self.response_bytes.decode("utf-8"))["body"]


class AgitBase64AdapterFixture:
    def publish(self, body):
        encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
        self.push_option = f"description-base64={encoded}"

    def read_raw_body(self):
        encoded = self.push_option.split("=", 1)[1]
        return base64.b64decode(encoded).decode("utf-8")


class GitHubBodyFileAdapterFixture:
    def publish(self, source):
        self.command = ("gh", "pr", "create", "--body-file", str(source))
        with source.open("r", encoding="utf-8", newline="") as handle:
            stored_body = handle.read()
        self.response_bytes = json.dumps({"body": stored_body}).encode("utf-8")

    def read_raw_body(self):
        return json.loads(self.response_bytes.decode("utf-8"))["body"]


VALID_BODY = """## Summary

One contiguous prose paragraph explains the change without hard wrapping.

- The `tisam` command supports the workflow.
- Review the [Gitea documentation](https://docs.gitea.com/).
- [ ] Verify the published body.

```sh
tisam publish \\
  --body-file body.md
```
"""


class MarkdownFormatTests(unittest.TestCase):
    def assertInvalid(self, body, message):
        self.assertTrue(
            any(message in error for error in forge_body.validate_markdown(body)),
            forge_body.validate_markdown(body),
        )

    def test_complete_multiline_body_is_valid(self):
        self.assertEqual([], forge_body.validate_markdown(VALID_BODY))

    def test_consecutive_blank_lines_are_rejected(self):
        self.assertInvalid(
            "First paragraph.\n\n\nSecond paragraph.\n", "consecutive blank lines"
        )

    def test_hard_wrapped_prose_is_rejected(self):
        self.assertInvalid(
            "This paragraph was\nwrapped unnecessarily.\n",
            "one uninterrupted Markdown line",
        )

    def test_required_block_separator_is_enforced(self):
        self.assertInvalid(
            "Paragraph before a list.\n- Capitalized item.\n",
            "separate prose and unordered-list blocks",
        )

    def test_unbalanced_fence_is_rejected(self):
        self.assertInvalid("```sh\nprintf ok\n", "unbalanced fenced code block")

    def test_nested_list_fence_balance_and_blank_lines_are_preserved(self):
        balanced = (
            "- Run the example.\n\n"
            "    ```sh\n"
            "    printf first\n"
            "\n"
            "\n"
            "    printf second\n"
            "    ```\n"
        )
        unbalanced = "- Run the example.\n\n    ```sh\n    printf ok\n"
        self.assertEqual([], forge_body.validate_markdown(balanced))
        self.assertInvalid(unbalanced, "unbalanced fenced code block")

    def test_nested_list_fence_after_continuation_is_balanced(self):
        balanced = (
            "- Parent item.\n\n"
            "  Continuation paragraph.\n\n"
            "    ```sh\n"
            "    printf ok\n"
            "    ```\n"
        )
        unbalanced = (
            "- Parent item.\n\n  Continuation paragraph.\n\n    ```sh\n    printf ok\n"
        )
        self.assertEqual([], forge_body.validate_markdown(balanced))
        self.assertInvalid(unbalanced, "unbalanced fenced code block")

    def test_lowercase_regular_bullet_is_rejected(self):
        self.assertInvalid("- review the documentation.\n", "must be capitalized")

    def test_empty_unordered_item_is_rejected(self):
        self.assertInvalid("-\n", "must be capitalized")
        self.assertInvalid("- \n", "must be capitalized")

    def test_lowercase_checklist_item_is_rejected(self):
        self.assertInvalid("- [ ] verify the body.\n", "must be capitalized")

    def test_inline_code_cannot_begin_a_bullet(self):
        self.assertInvalid(
            "- `tisam` supports the workflow.\n", "capitalized introductory word"
        )

    def test_source_validation_does_not_gate_editorial_style(self):
        self.assertEqual(
            [], forge_body.validate_markdown("- [ ] `legacy` criterion.\n", check_editorial_style=False)
        )

    def test_link_cannot_begin_a_bullet(self):
        self.assertInvalid(
            "- [Gitea](https://gitea.com/) documents the API.\n",
            "capitalized introductory word",
        )

    def test_command_cannot_begin_a_bullet(self):
        self.assertInvalid(
            "- ./tisam publishes the body.\n", "capitalized introductory word"
        )

    def test_capitalized_introduction_before_inline_markup_is_valid(self):
        body = (
            "- The `tisam` command supports the workflow.\n"
            "- Review the [Gitea documentation](https://docs.gitea.com/).\n"
        )
        self.assertEqual([], forge_body.validate_markdown(body))

    def test_acronym_and_proper_noun_are_valid(self):
        self.assertEqual(
            [],
            forge_body.validate_markdown(
                "- API behavior remains stable.\n- GitHub preserves the body.\n"
            ),
        )

    def test_non_bullet_contexts_are_not_capitalization_checked(self):
        body = """1. ordered items are outside the capitalization contract.

```text
- lowercase code content
```

A non-list line may begin with lowercase inline code such as `tisam`.
"""
        self.assertEqual([], forge_body.validate_markdown(body))

    def test_spaced_thematic_break_and_indented_code_are_not_bullets(self):
        body = """Paragraph before a break.

* * *

    - lowercase code content
"""
        self.assertEqual([], forge_body.validate_markdown(body))

    def test_lowercase_blockquote_bullet_and_checklist_are_rejected(self):
        self.assertInvalid("> - lowercase quoted list item.\n", "must be capitalized")
        self.assertInvalid("> > - [ ] verify the quoted item.\n", "must be capitalized")

    def test_capitalized_blockquote_bullet_is_valid(self):
        self.assertEqual(
            [], forge_body.validate_markdown("> - Capitalized quoted list item.\n")
        )

    def test_unicode_capitalization_is_supported(self):
        self.assertEqual(
            [], forge_body.validate_markdown("- Élan remains valid UTF-8 prose.\n")
        )
        self.assertInvalid("- élan is lowercase.\n", "must be capitalized")

    def test_lowercase_nested_bullet_and_checklist_are_rejected(self):
        self.assertInvalid(
            "- Parent item.\n\n    - lowercase nested item.\n",
            "must be capitalized",
        )
        self.assertInvalid(
            "- Parent item.\n\n    - [ ] verify the nested item.\n",
            "must be capitalized",
        )
        self.assertInvalid(
            "> - Parent quoted item.\n>\n>     - lowercase nested quoted item.\n",
            "must be capitalized",
        )

    def test_list_context_does_not_skip_regular_bullet_capitalization(self):
        self.assertInvalid(
            "- Parent item.\n\n- lowercase second list.\n", "must be capitalized"
        )
        self.assertInvalid(
            "1. Ordered item.\n\n- lowercase unordered list.\n",
            "must be capitalized",
        )
        self.assertInvalid(
            "- Parent item.\n  - lowercase nested item.\n", "must be capitalized"
        )

    def test_capitalized_nested_bullet_and_indented_code_are_distinguished(self):
        nested = "- Parent item.\n\n    - Capitalized nested item.\n"
        indented_code = "Paragraph before code.\n\n    - lowercase code content\n"
        self.assertEqual([], forge_body.validate_markdown(nested))
        self.assertEqual([], forge_body.validate_markdown(indented_code))

    def test_blockquote_fences_are_balanced(self):
        balanced = "> ```sh\n> printf ok\n> ```\n"
        self.assertEqual([], forge_body.validate_markdown(balanced))
        self.assertInvalid("> ```sh\n> printf ok\n", "unbalanced fenced code block")

    def test_setext_headings_and_tables_without_outer_pipes_are_valid(self):
        body = (
            "Heading\n"
            + "=======\n\n"
            + "Paragraph.\n\n"
            + "Name | Result\n"
            + "--- | ---\n"
            + "Check | Pass\n"
        )
        self.assertEqual([], forge_body.validate_markdown(body))

    def test_leading_blank_line_is_rejected(self):
        self.assertInvalid("\nParagraph.\n", "leading blank lines")

    def test_structural_markdown_lines_are_preserved(self):
        body = """## Structure

Prose remains contiguous.

- First list item.
- Second list item.

| Name | Result |
| --- | --- |
| Check | Pass |

> A blockquote keeps its lines.
> Another quoted line remains structural.

```sh
printf '%s\\n' ok
```
"""
        self.assertEqual([], forge_body.validate_markdown(body))


class RoundTripTests(unittest.TestCase):
    def test_multiline_body_round_trips_exactly(self):
        self.assertEqual([], forge_body.verify_round_trip(VALID_BODY, VALID_BODY))

    def test_gitea_literal_newline_transport_fails(self):
        returned = VALID_BODY.replace("\n", "\\n")
        errors = forge_body.verify_round_trip(VALID_BODY, returned)
        self.assertTrue(any("transport-generated" in error for error in errors))
        self.assertTrue(any("does not match" in error for error in errors))

    def test_gitea_json_transport_round_trips(self):
        adapter = JsonApiAdapterFixture()
        adapter.publish(VALID_BODY)
        returned = adapter.read_raw_body()
        self.assertIn(b'"body":"## Summary\\n', adapter.request_bytes)
        self.assertEqual([], forge_body.verify_round_trip(VALID_BODY, returned))

    def test_gitea_base64_transport_round_trips(self):
        adapter = AgitBase64AdapterFixture()
        adapter.publish(VALID_BODY)
        returned = adapter.read_raw_body()
        self.assertTrue(adapter.push_option.startswith("description-base64="))
        self.assertEqual([], forge_body.verify_round_trip(VALID_BODY, returned))

    def test_github_file_transport_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            intended_path = Path(directory, "intended.md")
            intended_path.write_text(VALID_BODY, encoding="utf-8")
            adapter = GitHubBodyFileAdapterFixture()
            adapter.publish(intended_path)
            returned = adapter.read_raw_body()
        self.assertEqual(
            ("gh", "pr", "create", "--body-file", str(intended_path)),
            adapter.command,
        )
        self.assertEqual([], forge_body.verify_round_trip(VALID_BODY, returned))

    def test_intentional_literal_newline_example_is_accepted(self):
        body = "The literal `\\\\n` example is intentional.\n"
        self.assertEqual([], forge_body.verify_round_trip(body, body))

    def test_crlf_and_one_terminal_newline_are_normalized(self):
        intended = VALID_BODY.replace("\n", "\r\n").rstrip("\r\n")
        returned = VALID_BODY
        self.assertEqual([], forge_body.verify_round_trip(intended, returned))

    def test_more_than_one_terminal_newline_is_not_normalized(self):
        errors = forge_body.verify_round_trip(VALID_BODY, VALID_BODY + "\n")
        self.assertTrue(any("does not match" in error for error in errors))

    def test_lone_carriage_return_is_not_normalized_by_cli_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            intended_path = Path(directory, "intended.md")
            returned_path = Path(directory, "returned.md")
            intended_path.write_bytes(b"## Heading\r\rParagraph.\r")
            returned_path.write_bytes(b"## Heading\n\nParagraph.\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "verify",
                    str(intended_path),
                    str(returned_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("does not match", result.stderr)

    def test_body_mismatch_fails(self):
        errors = forge_body.verify_round_trip(
            VALID_BODY, VALID_BODY.replace("supports", "breaks")
        )
        self.assertTrue(any("does not match" in error for error in errors))

    def test_transport_damaged_fence_fails(self):
        returned = VALID_BODY.rsplit("```", 1)[0]
        errors = forge_body.verify_round_trip(VALID_BODY, returned)
        self.assertTrue(any("transport-damaged fence" in error for error in errors))
        self.assertTrue(any("does not match" in error for error in errors))

    def test_transport_damaged_blockquote_fence_fails(self):
        intended = "> ```sh\n> printf ok\n> ```\n"
        returned = "> ```sh\n> printf ok\n"
        errors = forge_body.verify_round_trip(intended, returned)
        self.assertTrue(any("transport-damaged fence" in error for error in errors))

    def test_cli_runs_from_an_arbitrary_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "intended.md")
            source.write_text(VALID_BODY, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH.resolve()),
                    "check-source",
                    str(source),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_read_back_failure_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            intended_path = Path(directory, "intended.md")
            intended_path.write_text(VALID_BODY, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "verify",
                    str(intended_path),
                    str(Path(directory, "missing.md")),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("publication verification incomplete", result.stderr)


class MarkerOnlyUpdateTests(unittest.TestCase):
    def test_legacy_inline_code_checklist_can_be_checked(self):
        base = "## Acceptance criteria\n\n- [ ] `issue-discovery` renders Markdown.\n"
        intended = "## Acceptance criteria\n\n- [x] `issue-discovery` renders Markdown.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_marker_only_update_rejects_wording_changes(self):
        base = "- [ ] Preserve wording.\n"
        intended = "- [x] Rewrite wording.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))


class AdditionalMarkerOnlyUpdateTests(unittest.TestCase):
    def test_marker_only_update_rejects_reopening(self):
        base = "- [x] Completed criterion.\n"
        intended = "- [ ] Completed criterion.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_ordered_checklist_can_be_checked(self):
        self.assertEqual([], forge_body.validate_marker_only_update("## Acceptance criteria\n\n1. [ ] Criterion.\n", "## Acceptance criteria\n\n1. [x] Criterion.\n"))

    def test_legacy_body_round_trips_without_source_style_lint(self):
        body = "## Acceptance criteria\n\n- [x] `legacy` criterion.\n"
        self.assertEqual([], forge_body.verify_marker_only_round_trip("## Acceptance criteria\n\n- [ ] `legacy` criterion.\n", body, body))

    def test_marker_only_update_rejects_non_acceptance_checklist(self):
        base = "## Acceptance criteria\n\n- [ ] Criterion.\n\n## Operator checklist\n\n- [ ] Deploy production.\n"
        intended = "## Acceptance criteria\n\n- [ ] Criterion.\n\n## Operator checklist\n\n- [x] Deploy production.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_marker_only_update_rejects_checkbox_like_code(self):
        base = "## Acceptance criteria\n\n```text\n- [ ] Example only.\n```\n"
        intended = "## Acceptance criteria\n\n```text\n- [x] Example only.\n```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_marker_only_update_rejects_nested_fenced_checkbox(self):
        base = "## Acceptance criteria\n\n- Example:\n\n    ```text\n    - [ ] Example only.\n    ```\n"
        intended = "## Acceptance criteria\n\n- Example:\n\n    ```text\n    - [x] Example only.\n    ```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_fenced_heading_does_not_open_acceptance_criteria(self):
        base = "```text\n# Acceptance criteria\n```\n\n- [ ] Unrelated task.\n"
        intended = "```text\n# Acceptance criteria\n```\n\n- [x] Unrelated task.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_fence_like_content_does_not_close_before_checkbox(self):
        base = "## Acceptance criteria\n\n```text\n````still code\n- [ ] Example only.\n```\n"
        intended = "## Acceptance criteria\n\n```text\n````still code\n- [x] Example only.\n```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_nested_fence_like_content_does_not_close_before_checkbox(self):
        base = "## Acceptance criteria\n\n    ```text\n    ````still code\n    - [ ] Example only.\n    ```\n"
        intended = "## Acceptance criteria\n\n    ```text\n    ````still code\n    - [x] Example only.\n    ```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_indented_heading_closes_acceptance_criteria_section(self):
        base = "## Acceptance criteria\n\n- [ ] Criterion.\n\n  ## Operator checklist\n\n- [ ] Deploy.\n"
        intended = "## Acceptance criteria\n\n- [ ] Criterion.\n\n  ## Operator checklist\n\n- [x] Deploy.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_acceptance_heading_with_closing_hashes_allows_marker_update(self):
        base = "## Acceptance criteria ##\n\n- [ ] Criterion.\n"
        intended = "## Acceptance criteria ##\n\n- [x] Criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_over_indented_closing_fence_does_not_end_top_level_fence(self):
        base = "## Acceptance criteria\n\n```text\n    ```\n- [ ] Example only.\n```\n"
        intended = "## Acceptance criteria\n\n```text\n    ```\n- [x] Example only.\n```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_over_indented_closing_fence_does_not_end_nested_fence(self):
        base = "## Acceptance criteria\n\n- Example:\n\n    ```text\n        ```\n    - [ ] Example only.\n    ```\n"
        intended = "## Acceptance criteria\n\n- Example:\n\n    ```text\n        ```\n    - [x] Example only.\n    ```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_standalone_indented_code_checkbox_is_rejected(self):
        base = "## Acceptance criteria\n\n    - [ ] Example only.\n"
        intended = "## Acceptance criteria\n\n    - [x] Example only.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_standalone_tab_indented_code_checkbox_is_rejected(self):
        base = "## Acceptance criteria\n\n\t- [ ] Example only.\n"
        intended = "## Acceptance criteria\n\n\t- [x] Example only.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_nested_list_checkbox_is_allowed(self):
        base = "## Acceptance criteria\n\n- [ ] Parent criterion.\n    - [ ] Child criterion.\n"
        intended = "## Acceptance criteria\n\n- [ ] Parent criterion.\n    - [x] Child criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_optional_opening_indent_does_not_raise_closing_indent_limit(self):
        base = "## Acceptance criteria\n\n   ```text\n      ```\n- [ ] Example only.\n```\n"
        intended = "## Acceptance criteria\n\n   ```text\n      ```\n- [x] Example only.\n```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_less_indented_closer_ends_optionally_indented_fence(self):
        base = "## Acceptance criteria\n\n   ```text\nExample.\n```\n\n- [ ] Criterion.\n"
        intended = "## Acceptance criteria\n\n   ```text\nExample.\n```\n\n- [x] Criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_parent_marker_content_indent_blocks_false_nested_checkbox(self):
        base = "## Acceptance criteria\n\n   - Context item.\n\n    - [ ] Example only.\n"
        intended = "## Acceptance criteria\n\n   - Context item.\n\n    - [x] Example only.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_top_level_fence_clears_previous_list_container(self):
        base = "## Acceptance criteria\n\n- Parent.\n\n```text\nCode.\n```\n\n    - [ ] Example only.\n"
        intended = "## Acceptance criteria\n\n- Parent.\n\n```text\nCode.\n```\n\n    - [x] Example only.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_over_indented_nested_pseudo_fence_does_not_open(self):
        base = "## Acceptance criteria\n\n- Parent.\n\n      ```\n    ```\n    - [ ] Example only.\n    ```\n"
        intended = "## Acceptance criteria\n\n- Parent.\n\n      ```\n    ```\n    - [x] Example only.\n    ```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_backtick_in_info_string_does_not_open_fence(self):
        base = "## Acceptance criteria\n\n```bad`tick\n```\n- [ ] Example only.\n```\n"
        intended = "## Acceptance criteria\n\n```bad`tick\n```\n- [x] Example only.\n```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_continuation_preserves_parent_for_nested_checkbox(self):
        base = "## Acceptance criteria\n\n- Parent.\n  Continuation.\n\n  - [ ] Child criterion.\n"
        intended = "## Acceptance criteria\n\n- Parent.\n  Continuation.\n\n  - [x] Child criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_tab_content_indent_preserves_nested_checkbox(self):
        base = "## Acceptance criteria\n\n-\tParent.\n\t- [ ] Child criterion.\n"
        intended = "## Acceptance criteria\n\n-\tParent.\n\t- [x] Child criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_list_item_inline_fence_rejects_checkbox_update(self):
        base = "## Acceptance criteria\n\n- ```text\n  - [ ] Example only.\n  ```\n"
        intended = "## Acceptance criteria\n\n- ```text\n  - [x] Example only.\n  ```\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_ordered_list_inline_tilde_fence_rejects_checkbox_update(self):
        base = "## Acceptance criteria\n\n1. ~~~text\n   - [ ] Example only.\n   ~~~\n"
        intended = "## Acceptance criteria\n\n1. ~~~text\n   - [x] Example only.\n   ~~~\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_ten_digit_ordered_pseudo_list_is_rejected(self):
        base = "## Acceptance criteria\n\n1234567890. [ ] Not a list.\n"
        intended = "## Acceptance criteria\n\n1234567890. [x] Not a list.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_list_item_heading_closes_acceptance_section(self):
        base = "## Acceptance criteria\n\n- ## Operator checklist\n  - [ ] Deploy.\n"
        intended = "## Acceptance criteria\n\n- ## Operator checklist\n  - [x] Deploy.\n"
        self.assertTrue(forge_body.validate_marker_only_update(base, intended))

    def test_list_item_heading_opens_acceptance_section(self):
        base = "## Context\n\n- ## Acceptance criteria\n  - [ ] Criterion.\n"
        intended = "## Context\n\n- ## Acceptance criteria\n  - [x] Criterion.\n"
        self.assertEqual([], forge_body.validate_marker_only_update(base, intended))

    def test_marker_only_validation_fails_closed_without_commonmark_parser(self):
        parser = forge_body.MarkdownIt
        forge_body.MarkdownIt = None
        try:
            errors = forge_body.validate_marker_only_update(
                "## Acceptance criteria\n\n- [ ] Criterion.\n",
                "## Acceptance criteria\n\n- [x] Criterion.\n",
            )
        finally:
            forge_body.MarkdownIt = parser
        self.assertIn("requires markdown-it-py", errors[0])


if __name__ == "__main__":
    unittest.main()
