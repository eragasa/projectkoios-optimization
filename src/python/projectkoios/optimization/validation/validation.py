"""Validate text and identifiers shared by optimization capabilities.

Dependency-free numerical representation checks live in
:mod:`projectkoios.numerics.validation`. This module owns Optimization-specific text,
measurement-name, run-identifier, and candidate-identifier boundaries only.
"""

from __future__ import annotations

import re

MAX_IDENTIFIER_LENGTH = 256
MAX_OPTIMIZER_FAMILY_LENGTH = 128
OPTIMIZER_FAMILY_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z", re.ASCII)

_CANDIDATE_INFIX = ".candidate."
_MIN_SEQUENCE_DIGITS = 8


def text(value: object, name: str, *, maximum: int = MAX_IDENTIFIER_LENGTH) -> str:
    """Validate nonempty trimmed built-in text within an explicit maximum length.

    Parameters
    ----------
    value
        Candidate text value.
    name
        Public field name included in failure messages.
    maximum
        Inclusive positive maximum character count.

    Returns
    -------
    str
        The original validated built-in string.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in string.
    ValueError
        If ``value`` is empty, has leading or trailing whitespace, or exceeds
        ``maximum`` characters.
    """
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty trimmed text")
    if len(value) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    return value


def measurement_name(value: object, name: str = "measurement name") -> str:
    """Validate a measurement or weighted-loss term name.

    Parameters
    ----------
    value
        Candidate measurement-name value.
    name
        Public field name included in failure messages.

    Returns
    -------
    str
        Validated built-in name using the common identifier length boundary.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in string.
    ValueError
        If the name is empty, untrimmed, or too long.
    """
    return text(value, name)


def candidate_identifier(run_id: str, sequence: int) -> str:
    """Create and validate a non-truncated sequence-based candidate identifier.

    Parameters
    ----------
    run_id
        Already validated optimizer run identifier.
    sequence
        Nonnegative proposal sequence represented as a built-in integer.

    Returns
    -------
    str
        Identifier of the form ``<run_id>.candidate.<sequence>`` with at least eight
        sequence digits.

    Raises
    ------
    TypeError
        If the constructed identifier is not built-in text.
    ValueError
        If the resulting identifier exceeds :data:`MAX_IDENTIFIER_LENGTH`.

    Notes
    -----
    Engine construction uses :func:`run_identifier` to reserve suffix capacity before
    this function is called. Identifiers are never truncated.
    """
    identifier = f"{run_id}{_CANDIDATE_INFIX}{sequence:0{_MIN_SEQUENCE_DIGITS}d}"
    return text(identifier, "candidate_id")


def run_identifier(value: object, maximum_sequence: int) -> str:
    """Validate a run identifier while reserving its largest candidate suffix.

    Parameters
    ----------
    value
        Candidate run identifier.
    maximum_sequence
        Largest proposal sequence the owning optimizer can emit.

    Returns
    -------
    str
        Validated built-in run identifier whose future candidate identifiers fit the
        common maximum length without truncation.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in string.
    ValueError
        If the identifier is empty, untrimmed, or too long after suffix reservation.
    """
    suffix = f"{_CANDIDATE_INFIX}{maximum_sequence:0{_MIN_SEQUENCE_DIGITS}d}"
    maximum = MAX_IDENTIFIER_LENGTH - len(suffix)
    return text(value, "run_id", maximum=maximum)


__all__ = [
    "MAX_IDENTIFIER_LENGTH",
    "MAX_OPTIMIZER_FAMILY_LENGTH",
    "OPTIMIZER_FAMILY_PATTERN",
    "candidate_identifier",
    "measurement_name",
    "run_identifier",
    "text",
]
