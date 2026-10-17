"""Unit tests for NumPy batched complex-unitary Procrustes definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesTolerance,
)

pytestmark = pytest.mark.unit


def test_module_exports_only_public_batched_names() -> None:
    import projectkoios.optimization.alignment.numpy.batched_procrustes as module

    assert module.__all__ == [
        "BatchedComplexUnitaryProcrustesSolver",
        "NumpyBatchedComplexUnitaryProcrustesDefinition",
        "NumpyBatchedComplexUnitaryProcrustesResult",
        "NumpyBatchedComplexUnitaryProcrustesSolver",
    ]


def test_package_facade_preserves_explicit_implementation_ownership() -> None:
    import projectkoios.optimization.alignment.numpy.batched_procrustes as facade
    from projectkoios.optimization.alignment.numpy.batched_procrustes import (
        BatchedComplexUnitaryProcrustesSolver,
        NumpyBatchedComplexUnitaryProcrustesResult,
        NumpyBatchedComplexUnitaryProcrustesSolver,
    )

    assert BatchedComplexUnitaryProcrustesSolver.__module__.endswith(".contract")
    assert NumpyBatchedComplexUnitaryProcrustesDefinition.__module__.endswith(
        ".contract"
    )
    assert NumpyBatchedComplexUnitaryProcrustesResult.__module__.endswith(".contract")
    assert NumpyBatchedComplexUnitaryProcrustesSolver.__module__.endswith(".solver")
    assert not hasattr(facade, "ResidualReconstruction")
    assert not hasattr(facade, "CompletionEvidence")
    assert not hasattr(facade, "ResultArrays")


def test_definition_preserves_shape_and_owns_read_only_copies() -> None:
    source = np.arange(48, dtype=np.float64).reshape(2, 3, 4, 2).astype(np.complex128)
    target = source + (1.0 + 2.0j)
    definition = NumpyBatchedComplexUnitaryProcrustesDefinition(source, target)
    source[0, 0, 0, 0] = 999.0 + 0.0j
    target[0, 0, 0, 0] = 999.0 + 0.0j

    assert definition.batch_shape == (2, 3)
    assert definition.batch_size == 6
    assert definition.row_count == 4
    assert definition.column_count == 2
    assert definition.source_matrices[0, 0, 0, 0] == 0.0 + 0.0j
    assert definition.target_matrices[0, 0, 0, 0] == 1.0 + 2.0j
    assert definition.source_matrices.flags.c_contiguous
    assert not definition.source_matrices.flags.writeable
    assert not definition.target_matrices.flags.writeable
    with pytest.raises(ValueError):
        definition.source_matrices.flags.writeable = True
    with pytest.raises(ValueError):
        definition.target_matrices.flags.writeable = True


def test_definition_retains_exact_nominal_tolerance() -> None:
    tolerance = ComplexUnitaryProcrustesTolerance(1.0e-14, 1.0e-9, 1.0e-12, 1.0e-10)
    matrices = np.ones((2, 3, 1, 1), dtype=np.complex128)

    definition = NumpyBatchedComplexUnitaryProcrustesDefinition(
        matrices,
        matrices,
        tolerance,
    )

    assert definition.tolerance is tolerance


@pytest.mark.parametrize(
    "value",
    (
        [[[1.0 + 0.0j]]],
        np.ones((2, 1, 1), dtype=np.float64),
        np.ones((2, 1, 1), dtype=np.complex64),
        np.ones((2, 1, 1), dtype=np.bool_),
        np.asarray([[[1.0 + 0.0j]]], dtype=object),
    ),
)
def test_definition_rejects_non_ndarray_or_non_complex128_inputs(value: object) -> None:
    valid = np.ones((2, 1, 1), dtype=np.complex128)

    with pytest.raises(TypeError):
        NumpyBatchedComplexUnitaryProcrustesDefinition(value, valid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NumpyBatchedComplexUnitaryProcrustesDefinition(valid, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_shape", "target_shape", "message"),
    (
        ((2, 2), (2, 2), "leading batch"),
        ((2, 3, 2), (2, 4, 2), "shape"),
        ((2, 0, 2), (2, 0, 2), "positive"),
        ((2, 2, 0), (2, 2, 0), "positive"),
    ),
)
def test_definition_rejects_invalid_shapes(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    message: str,
) -> None:
    source = np.empty(source_shape, dtype=np.complex128)
    target = np.empty(target_shape, dtype=np.complex128)

    with pytest.raises(ValueError, match=message):
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, target)


def test_definition_reports_first_nonfinite_full_index() -> None:
    source = np.ones((2, 3, 2, 2), dtype=np.complex128)
    source[0, 2, 1, 0] = complex(float("nan"), 0.0)
    source[1, 0, 0, 0] = complex(float("inf"), 0.0)

    with pytest.raises(ValueError, match=r"\(0, 2, 1, 0\)"):
        NumpyBatchedComplexUnitaryProcrustesDefinition(source, source.copy())


def test_definition_permits_shape_preserving_empty_batch() -> None:
    source = np.empty((4, 0, 3, 2), dtype=np.complex128)

    definition = NumpyBatchedComplexUnitaryProcrustesDefinition(source, source)

    assert definition.batch_shape == (4, 0)
    assert definition.batch_size == 0
    assert definition.row_count == 3
    assert definition.column_count == 2


def test_definition_rejects_wrong_tolerance_nominal_type() -> None:
    source = np.ones((2, 1, 1), dtype=np.complex128)

    with pytest.raises(TypeError, match="tolerance"):
        NumpyBatchedComplexUnitaryProcrustesDefinition(
            source,
            source,
            object(),  # type: ignore[arg-type]
        )


def test_definition_is_frozen() -> None:
    source = np.ones((2, 1, 1), dtype=np.complex128)
    definition = NumpyBatchedComplexUnitaryProcrustesDefinition(source, source)

    with pytest.raises(FrozenInstanceError):
        definition.tolerance = ComplexUnitaryProcrustesTolerance()  # type: ignore[misc]
    with pytest.raises(ValueError):
        definition.source_matrices[0, 0, 0] = 0.0 + 0.0j
