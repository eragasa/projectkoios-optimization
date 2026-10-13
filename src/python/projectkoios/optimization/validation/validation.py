"""Public validation primitives shared by optimization capabilities.

The functions in this module define portable input boundaries used by maintained
optimization APIs. They validate and normalize values only; they do not assign domain
meaning, execute workflows, or establish scientific acceptance.
"""

from __future__ import annotations

import math
import re
from numbers import Real

MAX_IDENTIFIER_LENGTH = 256
MAX_OPTIMIZER_FAMILY_LENGTH = 128
MAX_U32 = 4_294_967_295
MAX_U64 = 18_446_744_073_709_551_615

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


def finite_float(value: object, name: str) -> float:
    """Normalize a non-Boolean real value to finite built-in binary64.

    Parameters
    ----------
    value
        Candidate real numeric value. Booleans, strings, complex numbers, and non-real
        objects are not accepted.
    name
        Public field name included in failure messages.

    Returns
    -------
    float
        Finite built-in binary64 representation of ``value``.

    Raises
    ------
    TypeError
        If ``value`` is Boolean or is not an instance of :class:`numbers.Real`.
    ValueError
        If conversion fails, overflows, or produces NaN or infinity.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number excluding bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if type(result) is not float or not math.isfinite(result):
        raise ValueError(f"{name} must be representable as a finite float")
    return result


def integer(value: object, name: str) -> int:
    """Validate an exact built-in integer while excluding Boolean values.

    Parameters
    ----------
    value
        Candidate integer value.
    name
        Public field name included in failure messages.

    Returns
    -------
    int
        The original exact built-in integer.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in integer.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer excluding bool")
    return value


def u32(value: object, name: str, *, allow_zero: bool) -> int:
    """Validate a built-in integer in a requested unsigned 32-bit interval.

    Parameters
    ----------
    value
        Candidate exact built-in integer.
    name
        Public field name included in failure messages.
    allow_zero
        If true, accept ``0 <= value <= MAX_U32``; otherwise require
        ``1 <= value <= MAX_U32``.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in integer.
    ValueError
        If ``value`` lies outside the selected interval.
    """
    result = integer(value, name)
    minimum = 0 if allow_zero else 1
    if not minimum <= result <= MAX_U32:
        interval = "unsigned u32" if allow_zero else "positive u32"
        raise ValueError(f"{name} must be in the {interval} range")
    return result


def u64(value: object, name: str) -> int:
    """Validate a built-in integer in the unsigned 64-bit interval.

    Parameters
    ----------
    value
        Candidate exact built-in integer.
    name
        Public field name included in failure messages.

    Returns
    -------
    int
        Validated integer in ``0 <= value <= MAX_U64``.

    Raises
    ------
    TypeError
        If ``value`` is not exactly a built-in integer.
    ValueError
        If ``value`` lies outside the unsigned 64-bit interval.
    """
    result = integer(value, name)
    if not 0 <= result <= MAX_U64:
        raise ValueError(f"{name} must be in the unsigned u64 range")
    return result


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
    "MAX_U32",
    "MAX_U64",
    "OPTIMIZER_FAMILY_PATTERN",
    "candidate_identifier",
    "finite_float",
    "integer",
    "measurement_name",
    "run_identifier",
    "text",
    "u32",
    "u64",
]
