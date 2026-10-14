#!/usr/bin/env python3
"""Retrieve metadata for DOI and ISBN references found in Markdown files.

DOI records come from Crossref. ISBN records come from the Open Library Books API. The
output is intended for human inspection before architectural claims are written; it does
not treat metadata lookup as a substitute for reading available abstracts, tables of
contents, or publisher descriptions.

Supply ``--output`` to retain stable JSON evidence. No retrieval timestamp is inserted,
so repeated output is diffable. Network access occurs only when this script is invoked;
it is not part of package runtime, builds, or tests.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from check_reference_identifiers import extract_identifiers

_CROSSREF_ENDPOINT = "https://api.crossref.org/works/"
_OPEN_LIBRARY_ENDPOINT = "https://openlibrary.org/api/books"
_HTML_TAG = re.compile(r"<[^>]+>")
_USER_AGENT = (
    "projectkoios-optimization-reference-inspector/1 "
    "(https://github.com/eragasa/projectkoios-optimization)"
)


def _plain_text(value: str, /) -> str:
    """Remove Crossref JATS/HTML tags and collapse whitespace."""
    return " ".join(html.unescape(_HTML_TAG.sub(" ", value)).split())


def _first(value: object, /) -> str | None:
    """Return the first nonempty string from a metadata list-like field."""
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    return first if isinstance(first, str) and first else None


def _publication_year(message: dict[str, Any], /) -> int | None:
    """Extract the first deposited publication year from common Crossref fields."""
    for field in ("published", "published-print", "published-online", "issued"):
        candidate = message.get(field)
        if not isinstance(candidate, dict):
            continue
        date_parts = candidate.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None


def _crossref_authors(message: dict[str, Any], /) -> list[str]:
    """Normalize deposited Crossref author names without inventing components."""
    result: list[str] = []
    candidates = message.get("author", [])
    if not isinstance(candidates, list):
        return result
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        given = candidate.get("given")
        family = candidate.get("family")
        parts = [part for part in (given, family) if isinstance(part, str) and part]
        if parts:
            result.append(" ".join(parts))
    return result


def _json_request(url: str, timeout: float, /) -> object:
    """Fetch one JSON resource with the project reference-inspection identity."""
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"could not retrieve reference metadata from {url}"
        ) from error


def _fetch_doi(doi: str, timeout: float, /) -> dict[str, object]:
    """Fetch and normalize one Crossref work record."""
    payload = _json_request(_CROSSREF_ENDPOINT + quote(doi, safe=""), timeout)
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"Crossref returned an invalid response for {doi}")
    message = payload.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Crossref omitted work metadata for {doi}")
    typed_message = cast(dict[str, Any], message)
    returned_doi = typed_message.get("DOI")
    if not isinstance(returned_doi, str) or returned_doi.lower() != doi.lower():
        raise RuntimeError(f"Crossref DOI mismatch for {doi}: {returned_doi!r}")

    title = _first(typed_message.get("title"))
    if title is None:
        raise RuntimeError(f"Crossref omitted the title for {doi}")
    abstract = typed_message.get("abstract")
    container = _first(typed_message.get("container-title"))
    return {
        "identifier_kind": "doi",
        "identifier": returned_doi.lower(),
        "url": f"https://doi.org/{returned_doi.lower()}",
        "title": _plain_text(title),
        "authors": _crossref_authors(typed_message),
        "year": _publication_year(typed_message),
        "type": typed_message.get("type"),
        "publisher": typed_message.get("publisher"),
        "container": _plain_text(container) if container is not None else None,
        "abstract": _plain_text(abstract) if isinstance(abstract, str) else None,
    }


def _open_library_names(value: object, /) -> list[str]:
    """Extract named contributors from an Open Library list field."""
    if not isinstance(value, list):
        return []
    return [
        name
        for item in value
        if isinstance(item, dict)
        and isinstance((name := item.get("name")), str)
        and name
    ]


def _fetch_isbn(isbn: str, timeout: float, /) -> dict[str, object]:
    """Fetch and normalize one Open Library ISBN record."""
    key = f"ISBN:{isbn}"
    query = urlencode({"bibkeys": key, "format": "json", "jscmd": "data"})
    payload = _json_request(f"{_OPEN_LIBRARY_ENDPOINT}?{query}", timeout)
    if not isinstance(payload, dict) or not isinstance(payload.get(key), dict):
        raise RuntimeError(f"Open Library returned no metadata for ISBN {isbn}")
    message = cast(dict[str, Any], payload[key])
    title = message.get("title")
    if not isinstance(title, str) or not title:
        raise RuntimeError(f"Open Library omitted the title for ISBN {isbn}")
    publishers = _open_library_names(message.get("publishers"))
    publish_date = message.get("publish_date")
    return {
        "identifier_kind": "isbn",
        "identifier": isbn,
        "url": message.get("url"),
        "title": title,
        "authors": _open_library_names(message.get("authors")),
        "year": publish_date if isinstance(publish_date, str) else None,
        "type": "book",
        "publisher": publishers,
        "container": None,
        "abstract": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="inspect metadata for DOI or ISBN identifiers in Markdown files"
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main() -> int:
    """Fetch, display, and optionally persist normalized reference metadata."""
    arguments = _parser().parse_args()
    identifiers = sorted(
        {
            identifier
            for path in arguments.paths
            for identifier in extract_identifiers(path.read_text(encoding="utf-8"))
        },
        key=lambda item: (item.kind, item.value),
    )
    if not identifiers:
        raise SystemExit("no DOI or ISBN identifiers found")

    records = [
        _fetch_doi(identifier.value, arguments.timeout)
        if identifier.kind == "doi"
        else _fetch_isbn(identifier.value, arguments.timeout)
        for identifier in identifiers
    ]
    for record in records:
        print(f"{cast(str, record['identifier_kind']).upper()}: {record['identifier']}")
        print(f"Title: {record['title']}")
        print(f"Authors: {', '.join(cast(list[str], record['authors']))}")
        print(f"Year: {record['year']}")
        print(f"Publisher/container: {record['publisher']} / {record['container']}")
        abstract = record["abstract"]
        if isinstance(abstract, str):
            print(f"Abstract: {abstract}")
        print()

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
