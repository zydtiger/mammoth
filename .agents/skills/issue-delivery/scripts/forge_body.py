#!/usr/bin/env python3
"""Validate intended forge Markdown and verify its raw read-back."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
NESTED_FENCE_RE = re.compile(r"^ {4,}(`{3,}|~{3,})(.*)$")
BULLET_RE = re.compile(r"^ {0,3}[-+*](?:[ \t]+(.*))?$")
NESTED_BULLET_RE = re.compile(r"^ {4,}[-+*](?:[ \t]+(.*))?$")
CHECKBOX_RE = re.compile(r"^\[(?: |x|X)\]\s+(.+)$")
ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+|$)")
SETEXT_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
THEMATIC_RE = re.compile(
    r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
TABLE_DELIMITER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


class BodyReadError(Exception):
    """The intended or returned body could not be read as UTF-8."""


def normalize_for_comparison(body: str) -> str:
    """Apply the only allowed comparison normalizations."""

    normalized = body.replace("\r\n", "\n")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    return normalized


def _line_kind(
    line: str, *, is_table_line: bool = False, nested_list_context: bool = False
) -> str:
    stripped = line.lstrip()
    if HEADING_RE.match(line):
        return "heading"
    if nested_list_context and NESTED_BULLET_RE.match(line):
        return "unordered-list"
    if line.startswith("    ") or line.startswith("\t"):
        return "indented-code"
    if THEMATIC_RE.match(line):
        return "thematic-break"
    if BULLET_RE.match(line):
        return "unordered-list"
    if ORDERED_RE.match(line):
        return "ordered-list"
    if stripped.startswith(">"):
        quoted = _strip_blockquote_prefix(line)
        if nested_list_context and NESTED_BULLET_RE.match(quoted):
            return "unordered-list"
        if BULLET_RE.match(quoted):
            return "unordered-list"
        if ORDERED_RE.match(quoted):
            return "ordered-list"
        return "blockquote"
    if is_table_line or (
        "|" in line
        and (
            stripped.startswith("|")
            or stripped.endswith("|")
            or TABLE_DELIMITER_RE.match(line)
        )
    ):
        return "table"
    if stripped.startswith("<") or re.match(r"^\[[^\]]+\]:\s+", stripped):
        return "definition"
    return "prose"


def _needs_blank_line(previous_kind: str, current_kind: str) -> bool:
    if previous_kind == current_kind:
        return previous_kind in {"heading", "prose"}
    return True


def _strip_blockquote_prefix(line: str) -> str:
    remaining = line
    while True:
        match = re.match(r"^ {0,3}>[ \t]?", remaining)
        if match is None:
            return remaining
        remaining = remaining[match.end() :]


def _first_lexical_word(content: str) -> Optional[str]:
    start = next(
        (index for index, character in enumerate(content) if character.isalpha()),
        None,
    )
    if start is None:
        return None
    end = start + 1
    while end < len(content):
        character = content[end]
        if not (character.isalpha() or character.isdigit() or character in "'-"):
            break
        end += 1
    return content[start:end]


def _capitalization_error(
    line_number: int, line: str, *, nested_list_context: bool = False
) -> Optional[str]:
    candidate = _strip_blockquote_prefix(line)
    if (
        candidate.startswith("\t")
        or (candidate.startswith("    ") and not nested_list_context)
        or THEMATIC_RE.match(candidate) is not None
    ):
        return None
    match = NESTED_BULLET_RE.match(candidate) if nested_list_context else None
    if match is None:
        match = BULLET_RE.match(candidate)
    if not match:
        return None
    content = match.group(1) or ""
    checkbox = CHECKBOX_RE.match(content)
    if checkbox:
        content = checkbox.group(1)
    if content.startswith(("`", "[")):
        return (
            f"line {line_number}: list item needs a capitalized introductory "
            "word before inline code or a link"
        )
    if content.startswith(("$", "./", "/", "-")):
        return (
            f"line {line_number}: list item needs a capitalized introductory "
            "word before a command"
        )
    word = _first_lexical_word(content)
    if word is None or not (word[0].isupper() or word[0].istitle()):
        return (
            f"line {line_number}: first lexical word in list item must be capitalized"
        )
    return None


def _table_line_numbers(lines: Sequence[str]) -> set:
    table_lines = set()
    for index, line in enumerate(lines):
        if (
            index == 0
            or TABLE_DELIMITER_RE.match(line) is None
            or "|" not in lines[index - 1]
        ):
            continue
        table_lines.update((index, index + 1))
        following = index + 1
        while (
            following < len(lines)
            and lines[following].strip()
            and "|" in lines[following]
        ):
            table_lines.add(following + 1)
            following += 1
    return table_lines


def validate_markdown(body: str) -> List[str]:
    """Return source-format violations without altering substantive content."""

    normalized = body.replace("\r\n", "\n")
    lines = normalized.split("\n")
    table_line_numbers = _table_line_numbers(lines)
    errors: List[str] = []
    fence: Optional[Tuple[str, int, int, bool]] = None
    previous_nonblank: Optional[Tuple[int, str]] = None
    blank_since_previous = True
    previous_blank = False
    list_container_active = False

    for line_number, line in enumerate(lines, start=1):
        candidate = _strip_blockquote_prefix(line)
        previous_kind = previous_nonblank[1] if previous_nonblank is not None else None
        nested_list_context = list_container_active
        standard_fence_match = FENCE_RE.match(candidate)
        nested_fence_match = NESTED_FENCE_RE.match(candidate)
        if fence is not None:
            marker, length, opening_line, nested_fence = fence
            fence_match = nested_fence_match if nested_fence else standard_fence_match
            if (
                fence_match
                and fence_match.group(1)[0] == marker
                and len(fence_match.group(1)) >= length
                and not fence_match.group(2).strip()
            ):
                fence = None
                previous_nonblank = (line_number, "fence")
                blank_since_previous = False
            continue

        fence_match = standard_fence_match
        if fence_match is None and nested_list_context:
            fence_match = nested_fence_match
        if fence_match:
            marker_text = fence_match.group(1)
            if previous_nonblank is not None and not blank_since_previous:
                errors.append(
                    f"line {line_number}: fenced code block must be separated "
                    "from the preceding block by one blank line"
                )
            nested_fence = NESTED_FENCE_RE.match(candidate) is not None
            fence = (
                marker_text[0],
                len(marker_text),
                line_number,
                nested_fence,
            )
            list_container_active = nested_fence
            previous_nonblank = (line_number, "fence")
            blank_since_previous = False
            previous_blank = False
            continue

        if not _strip_blockquote_prefix(line).strip():
            if previous_nonblank is None:
                errors.append(
                    f"line {line_number}: leading blank lines are not allowed"
                )
            if previous_blank:
                errors.append(
                    f"line {line_number}: consecutive blank lines are not "
                    "allowed between Markdown blocks"
                )
            previous_blank = True
            blank_since_previous = True
            continue

        previous_blank = False
        is_setext_heading = (
            previous_kind == "prose"
            and not blank_since_previous
            and SETEXT_RE.match(line) is not None
        )
        kind = (
            "heading"
            if is_setext_heading
            else _line_kind(
                line,
                is_table_line=line_number in table_line_numbers,
                nested_list_context=nested_list_context,
            )
        )
        if previous_nonblank is not None and not blank_since_previous:
            previous_line, previous_kind = previous_nonblank
            if not is_setext_heading and _needs_blank_line(previous_kind, kind):
                if previous_kind == kind == "prose":
                    errors.append(
                        f"lines {previous_line}-{line_number}: prose paragraph "
                        "must remain on one uninterrupted Markdown line"
                    )
                else:
                    errors.append(
                        f"line {line_number}: separate {previous_kind} and "
                        f"{kind} blocks with one blank line"
                    )

        capitalization_error = _capitalization_error(
            line_number, line, nested_list_context=nested_list_context
        )
        if capitalization_error:
            errors.append(capitalization_error)

        previous_nonblank = (line_number, kind)
        blank_since_previous = False
        if kind in {"unordered-list", "ordered-list"}:
            list_container_active = True
        elif not (
            list_container_active
            and (candidate.startswith(("  ", "\t")) or kind == "indented-code")
        ):
            list_container_active = False

    if fence is not None:
        marker, length, opening_line, _ = fence
        errors.append(
            f"line {opening_line}: unbalanced fenced code block "
            f"(expected {marker * length} closing fence)"
        )
    return errors


def verify_round_trip(intended: str, returned: str) -> List[str]:
    """Validate the source and compare it with the raw forge body."""

    errors = validate_markdown(intended)
    if normalize_for_comparison(intended) == normalize_for_comparison(returned):
        return errors

    intended_escapes = intended.count("\\n")
    returned_escapes = returned.count("\\n")
    if returned_escapes > intended_escapes:
        errors.append(
            "returned body contains transport-generated literal \\\\n sequences"
        )

    returned_fence_errors = [
        error for error in validate_markdown(returned) if "fenced code block" in error
    ]
    if returned_fence_errors:
        errors.append("returned body has an unbalanced or transport-damaged fence")

    errors.append(
        "returned body does not match the intended Markdown after CRLF and "
        "single trailing-newline normalization"
    )
    return errors


def read_utf8(path: Path, role: str) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise BodyReadError(f"{role} body could not be read as UTF-8: {exc}") from exc


def _print_errors(errors: Sequence[str]) -> int:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate intended forge Markdown and verify raw read-back."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser(
        "check-source", help="validate an intended Markdown source file"
    )
    source_parser.add_argument("source", type=Path)

    verify_parser = subparsers.add_parser(
        "verify", help="compare an intended source with a raw forge body file"
    )
    verify_parser.add_argument("source", type=Path)
    verify_parser.add_argument("returned", type=Path)

    args = parser.parse_args(argv)
    try:
        intended = read_utf8(args.source, "intended")
        if args.command == "check-source":
            errors = validate_markdown(intended)
            if errors:
                return _print_errors(errors)
            print("forge body source is valid")
            return 0

        returned = read_utf8(args.returned, "returned")
        errors = verify_round_trip(intended, returned)
        if errors:
            return _print_errors(errors)
        print("forge body round trip verified")
        return 0
    except BodyReadError as exc:
        print(f"error: publication verification incomplete: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
