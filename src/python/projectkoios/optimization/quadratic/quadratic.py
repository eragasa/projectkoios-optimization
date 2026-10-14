"""Optional backend-composed positive-definite quadratic loss."""

from __future__ import annotations

from abc import ABC, abstractmethod


class _QuadraticLossBackend(ABC):
    """Nominal internal boundary for quadratic storage and evaluation backends."""

    @property
    @abstractmethod
    def center(self) -> tuple[float, ...]:
        """Return normalized center coordinates."""
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> object:
        """Return the backend matrix representation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def minimum_squared_loss(self) -> float:
        """Return the loss at the center."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the positive coordinate dimension."""
        raise NotImplementedError

    @abstractmethod
    def squared_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the completed-square loss."""
        raise NotImplementedError

    @abstractmethod
    def root_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the square root of the completed-square loss."""
        raise NotImplementedError


class PositiveDefiniteQuadraticLoss:
    """Represent a unit-free completed-square loss.

    Parameters
    ----------
    center
        Nonempty finite coordinate center.
    matrix
        Finite exactly symmetric positive-definite matrix. Dense construction requires
        a tuple of tuples. Sparse construction also accepts a SciPy sparse object.
    minimum_squared_loss
        Finite nonnegative loss at ``center``.
    sparse
        Select the SciPy CSR backend instead of the default NumPy dense backend.

    Notes
    -----
    The object chooses numerical storage and evaluation only. It does not assign units,
    normalize residuals, or define scientific acceptance policy.
    """

    __slots__ = ("_implementation", "_sparse")

    def __init__(
        self,
        center: tuple[float, ...],
        matrix: object,
        minimum_squared_loss: float,
        *,
        sparse: bool = False,
    ) -> None:
        if type(sparse) is not bool:
            raise TypeError("sparse must be a built-in bool")
        implementation: _QuadraticLossBackend
        if sparse:
            # Import lazily so the facade itself requires no optional backend.
            from .scipy.quadratic import ScipySparsePositiveDefiniteQuadraticLoss

            implementation = ScipySparsePositiveDefiniteQuadraticLoss(
                center,
                matrix,
                minimum_squared_loss,
            )
        else:
            # The default path needs only the lighter NumPy optional dependency.
            from .numpy.quadratic import NumpyPositiveDefiniteQuadraticLoss

            if type(matrix) is not tuple:
                raise TypeError("dense matrix must be a tuple of tuples")
            implementation = NumpyPositiveDefiniteQuadraticLoss(
                center,
                matrix,
                minimum_squared_loss,
            )
        self._implementation = implementation
        self._sparse = sparse

    @property
    def center(self) -> tuple[float, ...]:
        """Return the immutable normalized center.

        Returns
        -------
        tuple[float, ...]
            Built-in binary64 coordinates.
        """
        return self._implementation.center

    @property
    def matrix(self) -> object:
        """Return the backend matrix representation.

        Returns
        -------
        object
            Immutable dense tuples or a defensive SciPy CSR copy.
        """
        return self._implementation.matrix

    @property
    def minimum_squared_loss(self) -> float:
        """Return the completed-square minimum.

        Returns
        -------
        float
            Finite nonnegative loss at the center.
        """
        return self._implementation.minimum_squared_loss

    @property
    def dimension(self) -> int:
        """Return the coordinate dimension.

        Returns
        -------
        int
            Positive matrix dimension.
        """
        return self._implementation.dimension

    @property
    def sparse(self) -> bool:
        """Report whether the SciPy sparse backend is active.

        Returns
        -------
        bool
            ``True`` for SciPy CSR storage; otherwise ``False``.
        """
        return self._sparse

    def squared_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the completed-square loss.

        Parameters
        ----------
        coordinates
            Finite coordinates matching :attr:`dimension`.

        Returns
        -------
        float
            Finite nonnegative squared loss.
        """
        return self._implementation.squared_loss(coordinates)

    def root_loss(self, coordinates: tuple[float, ...], /) -> float:
        """Evaluate the root loss.

        Parameters
        ----------
        coordinates
            Finite coordinates matching :attr:`dimension`.

        Returns
        -------
        float
            Nonnegative square root of :meth:`squared_loss`.
        """
        return self._implementation.root_loss(coordinates)


__all__ = ["PositiveDefiniteQuadraticLoss"]
