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


class _ConfiguredNorm(MatrixNormABC[float]):
    __slots__ = ()

    @property
    def matrix_norm(self) -> MatrixNormEnum:
        return MatrixNormEnum.frobenius

    def _evaluate(self, matrix: float, /) -> float:
        return matrix


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


def test_result_returns_exact_float_for_retained_input() -> None:
    evaluation = _ConfiguredNorm(2.5)

    result = evaluation.result()

    assert evaluation.matrix_norm is MatrixNormEnum.frobenius
    assert result == 2.5
    assert type(result) is float


@pytest.mark.parametrize("output", [float("nan"), float("inf"), -float("inf")])
def test_result_rejects_nonfinite_implementation_output(output: float) -> None:
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _ConfiguredNorm(output).result()


def test_result_rejects_negative_implementation_output() -> None:
    with pytest.raises(FloatingPointError, match="negative"):
        _ConfiguredNorm(-1.0).result()


def test_result_propagates_implementation_failure() -> None:
    with pytest.raises(LookupError, match="implementation failure"):
        _FailingNorm(1).result()
