"""Unit tests for the dependency-free matrix-norm nominal contract."""

from __future__ import annotations

from enum import Enum
from inspect import isabstract

import pytest

from projectkoios.numerics.linalg.matrix_norm import (
    MatrixNormABC,
    MatrixNormEnum,
)

pytestmark = pytest.mark.unit


class _ConfiguredNorm(MatrixNormABC[int]):
    __slots__ = ("input", "output")

    def __init__(self, output: float) -> None:
        self.input: int | None = None
        self.output = output

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: int, /) -> float:
        self.input = matrix
        return self.output


class _FailingNorm(MatrixNormABC[int]):
    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        return MatrixNormEnum.spectral

    def _evaluate(self, matrix: int, /) -> float:
        del matrix
        raise LookupError("implementation failure")


def test_enum_is_pure_semantic_enum_with_lower_case_members() -> None:
    assert issubclass(MatrixNormEnum, Enum)
    assert not issubclass(MatrixNormEnum, str)
    assert tuple(MatrixNormEnum.__members__) == (
        "frobenius",
        "spectral",
        "maximum_absolute_entry",
    )


def test_base_is_abstract() -> None:
    assert isabstract(MatrixNormABC)


def test_public_call_passes_input_and_returns_exact_float() -> None:
    evaluator = _ConfiguredNorm(2.5)

    result = evaluator(7)

    assert evaluator.input == 7
    assert evaluator.matrix_norm is MatrixNormEnum.frobenius
    assert result == 2.5
    assert type(result) is float


@pytest.mark.parametrize("output", [float("nan"), float("inf"), -float("inf")])
def test_public_call_rejects_nonfinite_implementation_output(output: float) -> None:
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _ConfiguredNorm(output)(1)


def test_public_call_rejects_negative_implementation_output() -> None:
    with pytest.raises(FloatingPointError, match="negative"):
        _ConfiguredNorm(-1.0)(1)


def test_public_call_propagates_implementation_failure() -> None:
    with pytest.raises(LookupError, match="implementation failure"):
        _FailingNorm()(1)
