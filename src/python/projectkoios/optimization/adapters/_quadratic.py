"""Shared validation for optional quadratic-loss backends."""

from __future__ import annotations

from .._validation import finite_float


def finite_vector(value: object, name: str) -> tuple[float, ...]:
    """Normalize a nonempty tuple of finite real values.

    Parameters
    ----------
    value
        Candidate tuple.
    name
        Public field name used in failure messages.

    Returns
    -------
    tuple[float, ...]
        Values normalized to built-in binary64 floats.

    Raises
    ------
    TypeError
        If ``value`` is not an exact tuple or an entry is not real.
    ValueError
        If the tuple is empty or an entry is nonfinite or unrepresentable.
    """
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return tuple(
        finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)
    )


def finite_square_matrix(
    value: object,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    """Normalize a square tuple matrix matching a nonzero dimension.

    Parameters
    ----------
    value
        Candidate tuple-of-tuples matrix.
    dimension
        Required row and column count.

    Returns
    -------
    tuple[tuple[float, ...], ...]
        Matrix normalized to built-in binary64 floats.

    Raises
    ------
    TypeError
        If the outer object or any row is not an exact tuple.
    ValueError
        If the shape is wrong or an entry is nonfinite or unrepresentable.
    """
    if type(value) is not tuple:
        raise TypeError("matrix must be a tuple of tuples")
    if len(value) != dimension:
        raise ValueError("matrix row count must match center dimension")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        normalized = finite_vector(row, f"matrix[{row_index}]")
        if len(normalized) != dimension:
            raise ValueError("matrix must be square with center dimension")
        rows.append(normalized)
    return tuple(rows)


def symmetric(matrix: tuple[tuple[float, ...], ...]) -> bool:
    """Test exact represented symmetry.

    Parameters
    ----------
    matrix
        Normalized square matrix.

    Returns
    -------
    bool
        ``True`` when mirrored entries compare exactly equal.
    """
    return all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(row)
    )


__all__: list[str] = []
