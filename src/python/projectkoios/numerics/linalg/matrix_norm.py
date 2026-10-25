"""Nominal callable contract and semantic enumeration for matrix norms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import final


class MatrixNormEnum(Enum):
    """Identify supported matrix-norm semantics independently of implementations."""

    frobenius = auto()
    spectral = auto()
    maximum_absolute_entry = auto()


class MatrixNormABC[MatrixT](ABC):
    """Define one nominal matrix-norm evaluator for a specific representation.

    Concrete subclasses select their accepted matrix representation through ``MatrixT``
    and implement :meth:`_evaluate`. The public callable boundary validates a unit-free
    scalar result without coercing backend scalar types.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def matrix_norm(self) -> MatrixNormEnum:
        """Return the mathematical norm evaluated by this implementation."""
        raise NotImplementedError

    @final
    def __call__(self, matrix: MatrixT, /) -> float:
        """Evaluate one matrix and validate the implementation result.

        Parameters
        ----------
        matrix
            Matrix representation accepted by the concrete subclass.

        Returns
        -------
        float
            Exact built-in finite nonnegative unit-free norm.

        Raises
        ------
        TypeError
            If the implementation does not return an exact built-in float.
        FloatingPointError
            If the implementation returns a nonfinite or negative value.
        """
        result = self._evaluate(matrix)
        if type(result) is not float:
            raise TypeError(
                "matrix norm implementation must return an exact built-in float"
            )
        if not math.isfinite(result):
            raise FloatingPointError(
                "matrix norm implementation returned a nonfinite value"
            )
        if result < 0.0:
            raise FloatingPointError(
                "matrix norm implementation returned a negative value"
            )
        return result

    @abstractmethod
    def _evaluate(self, matrix: MatrixT, /) -> float:
        """Evaluate a matrix after representation checks owned by the subclass."""
        raise NotImplementedError


__all__ = ["MatrixNormABC", "MatrixNormEnum"]
