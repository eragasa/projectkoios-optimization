"""Internal shared validation for numeric values and identifiers."""

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
    """Return trimmed built-in text within an explicit maximum length."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty trimmed text")
    if len(value) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    return value


def measurement_name(value: object, name: str = "measurement name") -> str:
    """Validate the name shared by measurements and weighted-loss terms."""
    return text(value, name)


def finite_float(value: object, name: str) -> float:
    """Coerce a non-boolean real to a finite built-in float.

    Invalid numeric kinds are type errors. Real values that cannot be represented as
    finite binary64 values are value errors, including integer overflow during
    conversion.
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
    """Return an exact built-in integer, excluding bool."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer excluding bool")
    return value


def u32(value: object, name: str, *, allow_zero: bool) -> int:
    """Return a built-in integer in the requested unsigned u32 interval."""
    result = integer(value, name)
    minimum = 0 if allow_zero else 1
    if not minimum <= result <= MAX_U32:
        interval = "unsigned u32" if allow_zero else "positive u32"
        raise ValueError(f"{name} must be in the {interval} range")
    return result


def u64(value: object, name: str) -> int:
    """Return a built-in integer in the unsigned u64 interval."""
    result = integer(value, name)
    if not 0 <= result <= MAX_U64:
        raise ValueError(f"{name} must be in the unsigned u64 range")
    return result


def candidate_identifier(run_id: str, sequence: int) -> str:
    """Create one validated, non-truncated sequence-based candidate identifier."""
    identifier = f"{run_id}{_CANDIDATE_INFIX}{sequence:0{_MIN_SEQUENCE_DIGITS}d}"
    return text(identifier, "candidate_id")


def run_identifier(value: object, maximum_sequence: int) -> str:
    """Validate a run identifier against its largest possible candidate suffix."""
    suffix = f"{_CANDIDATE_INFIX}{maximum_sequence:0{_MIN_SEQUENCE_DIGITS}d}"
    maximum = MAX_IDENTIFIER_LENGTH - len(suffix)
    return text(value, "run_id", maximum=maximum)
