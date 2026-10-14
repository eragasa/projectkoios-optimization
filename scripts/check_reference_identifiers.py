#!/usr/bin/env python3
"""Audit DOI- or ISBN-backed academic references in Markdown documents.

Every bullet under a heading named ``Academic references`` must contain at least one
explicit persistent identifier:

- a canonical ``https://doi.org/...`` URL; or
- a labelled ``ISBN``/``ISBN-10``/``ISBN-13`` value with a valid checksum.

The script rejects malformed identifiers and duplicate DOI or ISBN entries across the
supplied documents. It is an offline structural check. Use
``inspect_reference_metadata.py`` to retrieve metadata for human source inspection.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

_DOI_URL = re.compile(r"https://doi\.org/([^\s\]]+)", flags=re.IGNORECASE)
_DOI_VALUE = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    flags=re.IGNORECASE,
)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_ISBN_LABEL = re.compile(
    r"\bISBN(?:-1[03])?\s*:?[ \t]*([0-9X][0-9X -]{8,20}[0-9X])\b",
    flags=re.IGNORECASE,
)
_REFERENCE_HEADING = re.compile(r"^(#{1,6})\s+Academic references\s*$")
_ANY_HEADING = re.compile(r"^(#{1,6})\s+")


@dataclass(frozen=True, slots=True)
class ReferenceIdentifier:
    """One normalized persistent academic-reference identifier."""

    kind: str
    value: str


def _normalized_doi(value: str, /) -> str:
    """Return one URL-decoded, case-normalized DOI without Markdown punctuation."""
    normalized = unquote(value).rstrip(".,;:")
    while normalized.endswith(")") and normalized.count(")") > normalized.count("("):
        normalized = normalized[:-1]
    return normalized.lower()


def _normalized_isbn(value: str, /) -> str:
    """Return one separator-free uppercase ISBN candidate."""
    return "".join(character for character in value.upper() if character.isalnum())


def _valid_isbn10(value: str, /) -> bool:
    """Return whether a normalized candidate satisfies the ISBN-10 checksum."""
    if len(value) != 10 or not value[:9].isdigit():
        return False
    if not (value[9].isdigit() or value[9] == "X"):
        return False
    digits = [int(character) for character in value[:9]]
    digits.append(10 if value[9] == "X" else int(value[9]))
    return sum((10 - index) * digit for index, digit in enumerate(digits)) % 11 == 0


def _valid_isbn13(value: str, /) -> bool:
    """Return whether a normalized candidate satisfies the ISBN-13 checksum."""
    if len(value) != 13 or not value.isdigit():
        return False
    weighted = sum(
        digit * (1 if index % 2 == 0 else 3)
        for index, digit in enumerate(map(int, value[:12]))
    )
    check_digit = (10 - weighted % 10) % 10
    return check_digit == int(value[12])


def extract_identifiers(text: str, /) -> tuple[ReferenceIdentifier, ...]:
    """Extract normalized DOI and valid ISBN identifiers from Markdown text."""
    identifiers: list[ReferenceIdentifier] = []
    for match in _DOI_URL.findall(text):
        if _INVALID_PERCENT_ESCAPE.search(match):
            continue
        doi = _normalized_doi(match)
        if _DOI_VALUE.fullmatch(doi):
            identifiers.append(ReferenceIdentifier("doi", doi))
    for match in _ISBN_LABEL.findall(text):
        isbn = _normalized_isbn(match)
        if _valid_isbn10(isbn) or _valid_isbn13(isbn):
            identifiers.append(ReferenceIdentifier("isbn", isbn))
    return tuple(identifiers)


def extract_invalid_dois(text: str, /) -> tuple[str, ...]:
    """Return DOI URL suffixes with invalid encoding or canonical value syntax."""
    invalid: list[str] = []
    for match in _DOI_URL.findall(text):
        if _INVALID_PERCENT_ESCAPE.search(match):
            invalid.append(match)
            continue
        if _DOI_VALUE.fullmatch(_normalized_doi(match)) is None:
            invalid.append(match)
    return tuple(invalid)


def extract_invalid_isbns(text: str, /) -> tuple[str, ...]:
    """Return labelled ISBN candidates that fail both standard checksums."""
    invalid: list[str] = []
    for match in _ISBN_LABEL.findall(text):
        isbn = _normalized_isbn(match)
        if not (_valid_isbn10(isbn) or _valid_isbn13(isbn)):
            invalid.append(isbn)
    return tuple(invalid)


def academic_reference_bullets(text: str, /) -> tuple[tuple[int, str], ...]:
    """Return line-numbered, continuation-joined academic-reference bullets."""
    bullets: list[tuple[int, str]] = []
    active_level: int | None = None
    current_line: int | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_line is not None:
            bullets.append((current_line, " ".join(current_parts)))

    for line_number, line in enumerate(text.splitlines(), start=1):
        reference_heading = _REFERENCE_HEADING.match(line)
        if reference_heading is not None:
            flush()
            current_line = None
            current_parts = []
            active_level = len(reference_heading.group(1))
            continue
        heading = _ANY_HEADING.match(line)
        if (
            heading is not None
            and active_level is not None
            and len(heading.group(1)) <= active_level
        ):
            flush()
            current_line = None
            current_parts = []
            active_level = None
            continue
        if active_level is None:
            continue
        if line.startswith("- "):
            flush()
            current_line = line_number
            current_parts = [line]
        elif current_line is not None and line.strip():
            current_parts.append(line.strip())

    flush()
    return tuple(bullets)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "require a valid DOI URL or labelled ISBN on Markdown academic references"
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--minimum", type=int, default=1)
    return parser


def main() -> int:
    """Run the offline Markdown reference-identifier audit."""
    arguments = _parser().parse_args()
    occurrences: dict[ReferenceIdentifier, list[str]] = {}
    errors: list[str] = []
    reference_count = 0

    for path in arguments.paths:
        text = path.read_text(encoding="utf-8")
        bullets = academic_reference_bullets(text)
        if not bullets:
            errors.append(f"{path}: no Academic references bullets found")
            continue
        reference_count += len(bullets)
        for line_number, bullet in bullets:
            invalid_dois = extract_invalid_dois(bullet)
            for doi in invalid_dois:
                errors.append(f"{path}:{line_number}: invalid DOI URL suffix: {doi}")
            invalid_isbns = extract_invalid_isbns(bullet)
            for isbn in invalid_isbns:
                errors.append(f"{path}:{line_number}: invalid ISBN checksum: {isbn}")
            identifiers = extract_identifiers(bullet)
            if not identifiers:
                errors.append(
                    f"{path}:{line_number}: expected a DOI URL or valid labelled ISBN"
                )
                continue
            location = f"{path}:{line_number}"
            for identifier in identifiers:
                occurrences.setdefault(identifier, []).append(location)

    for identifier, locations in sorted(
        occurrences.items(), key=lambda item: (item[0].kind, item[0].value)
    ):
        if len(locations) > 1:
            errors.append(
                f"duplicate {identifier.kind.upper()} {identifier.value}: "
                f"{', '.join(locations)}"
            )
    if reference_count < arguments.minimum:
        errors.append(
            f"expected at least {arguments.minimum} academic references, found "
            f"{reference_count}"
        )

    if errors:
        for error in errors:
            print(error)
        return 1

    for identifier, locations in sorted(
        occurrences.items(), key=lambda item: (item[0].kind, item[0].value)
    ):
        print(f"{identifier.kind.upper()}\t{identifier.value}\t{locations[0]}")
    print(f"verified {reference_count} references with {len(occurrences)} identifiers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
