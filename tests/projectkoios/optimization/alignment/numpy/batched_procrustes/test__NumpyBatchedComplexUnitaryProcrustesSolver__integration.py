"""Integration tests for NumPy batched complex-unitary Procrustes solving."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from projectkoios.optimization.alignment.numpy.batched_procrustes import (
    NumpyBatchedComplexUnitaryProcrustesDefinition,
    NumpyBatchedComplexUnitaryProcrustesSolver,
)
from projectkoios.optimization.alignment.procrustes import (
    ComplexUnitaryProcrustesTolerance,
)

pytestmark = pytest.mark.integration

_VALIDATION_ABSOLUTE = 1.0e-12
_VALIDATION_RELATIVE = 1.0e-10


@dataclass(frozen=True, slots=True)
class _RightActionCase:
    identifier: str
    seed: int
    batch_shape: tuple[int, ...]
    row_count: int
    column_count: int
    transformation_absolute_tolerance: float
    alignment_absolute_tolerance: float


@dataclass(frozen=True, slots=True)
class _RankCutoffCase:
    identifier: str
    singular_values: tuple[float, ...]
    rank_absolute: float
    rank_relative: float
    expected_cutoff: float
    expected_rank: int


@dataclass(frozen=True, slots=True)
class _EmptyBatchCase:
    identifier: str
    batch_shape: tuple[int, ...]
    row_count: int
    column_count: int


@dataclass(frozen=True, slots=True)
class _IndexedFailureCase:
    identifier: str
    batch_shape: tuple[int, ...]
    failing_indices: tuple[tuple[int, ...], ...]
    expected_first_index: tuple[int, ...]


_RIGHT_ACTION_CASES = (
    _RightActionCase(
        identifier="two-dimensional-grid",
        seed=20261017,
        batch_shape=(3, 4),
        row_count=4,
        column_count=2,
        transformation_absolute_tolerance=3.0e-14,
        alignment_absolute_tolerance=4.0e-14,
    ),
    _RightActionCase(
        identifier="three-dimensional-grid",
        seed=20261018,
        batch_shape=(2, 3, 4),
        row_count=5,
        column_count=2,
        transformation_absolute_tolerance=3.0e-14,
        alignment_absolute_tolerance=4.0e-14,
    ),
)

_RANK_CUTOFF_CASES = (
    _RankCutoffCase(
        identifier="additive-differs-from-maximum",
        singular_values=(1.0, 0.25),
        rank_absolute=0.1,
        rank_relative=0.2,
        expected_cutoff=0.3,
        expected_rank=1,
    ),
    _RankCutoffCase(
        identifier="equality-is-deficient",
        singular_values=(2.0, 1.0),
        rank_absolute=0.5,
        rank_relative=0.25,
        expected_cutoff=1.0,
        expected_rank=1,
    ),
    _RankCutoffCase(
        identifier="relative-only",
        singular_values=(4.0, 0.5),
        rank_absolute=0.0,
        rank_relative=0.2,
        expected_cutoff=0.8,
        expected_rank=1,
    ),
)

_EMPTY_BATCH_CASES = (
    _EmptyBatchCase(
        identifier="empty-one-dimensional-batch",
        batch_shape=(0,),
        row_count=3,
        column_count=2,
    ),
    _EmptyBatchCase(
        identifier="empty-two-dimensional-batch",
        batch_shape=(2, 0),
        row_count=4,
        column_count=2,
    ),
)

_INDEXED_FAILURE_CASES = (
    _IndexedFailureCase(
        identifier="first-c-order-item",
        batch_shape=(2, 2),
        failing_indices=((0, 1), (1, 0)),
        expected_first_index=(0, 1),
    ),
    _IndexedFailureCase(
        identifier="three-dimensional-grid-order",
        batch_shape=(2, 2, 2),
        failing_indices=((0, 1, 1), (1, 0, 0)),
        expected_first_index=(0, 1, 1),
    ),
)


def _unitary(random: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    matrices = random.normal(size=shape) + 1.0j * random.normal(size=shape)
    left, _, right_adjoint = np.linalg.svd(matrices)
    return np.asarray(left @ right_adjoint, dtype=np.complex128)


@pytest.mark.parametrize(
    "case",
    _RIGHT_ACTION_CASES,
    ids=lambda case: case.identifier,
)
def test_solver_preserves_tensor_shape_and_right_action_orientation(
    case: _RightActionCase,
) -> None:
    random = np.random.default_rng(case.seed)
    matrix_shape = case.batch_shape + (case.row_count, case.column_count)
    transformation_shape = case.batch_shape + (
        case.column_count,
        case.column_count,
    )
    source = np.asarray(
        random.normal(size=matrix_shape) + 1.0j * random.normal(size=matrix_shape),
        dtype=np.complex128,
    )
    expected = _unitary(random, transformation_shape)
    target = np.asarray(source @ expected, dtype=np.complex128)

    result = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(source, target)
    )

    assert result.batch_shape == case.batch_shape
    assert bool(result.full_rank.all())
    np.testing.assert_allclose(
        result.right_unitary_completions,
        expected,
        rtol=0.0,
        atol=case.transformation_absolute_tolerance,
    )
    np.testing.assert_allclose(
        source @ result.right_unitary_completions,
        target,
        rtol=0.0,
        atol=case.alignment_absolute_tolerance,
    )


def test_solver_uses_one_public_batched_svd_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import projectkoios.optimization.alignment.numpy.batched_procrustes as module

    batch_shape = (3, 4)
    matrix_dimension = 2
    expected_svd_shape = batch_shape + (matrix_dimension, matrix_dimension)
    original: Any = np.linalg.svd
    shapes: list[tuple[int, ...]] = []

    def recording_svd(
        matrix: Any,
        *,
        full_matrices: bool,
        compute_uv: bool,
    ) -> object:
        shapes.append(matrix.shape)
        return original(
            matrix,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    source = np.broadcast_to(
        np.eye(matrix_dimension, dtype=np.complex128),
        expected_svd_shape,
    ).copy()

    module.NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        module.NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(
            source, source
        )
    )

    assert shapes == [expected_svd_shape]


@pytest.mark.parametrize(
    "case",
    _RANK_CUTOFF_CASES,
    ids=lambda case: case.identifier,
)
def test_rank_cutoff_uses_owner_absolute_plus_relative_rule(
    case: _RankCutoffCase,
) -> None:
    dimension = len(case.singular_values)
    source = np.eye(dimension, dtype=np.complex128)[None, ...]
    target = np.diag(case.singular_values).astype(np.complex128)[None, ...]
    tolerance = ComplexUnitaryProcrustesTolerance(
        rank_absolute=case.rank_absolute,
        rank_relative=case.rank_relative,
        validation_absolute=_VALIDATION_ABSOLUTE,
        validation_relative=_VALIDATION_RELATIVE,
    )

    result = NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(
            source, target, tolerance
        )
    )

    assert result.applied_rank_cutoffs[0] == pytest.approx(
        case.expected_cutoff,
        rel=0.0,
        abs=np.finfo(np.float64).eps,
    )
    assert result.numerical_ranks[0] == case.expected_rank
    assert bool(result.full_rank[0]) is (case.expected_rank == dimension)
    np.testing.assert_allclose(
        result.right_unitary_completions[0],
        np.eye(dimension),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "case",
    _EMPTY_BATCH_CASES,
    ids=lambda case: case.identifier,
)
def test_empty_batch_short_circuits_without_svd(
    monkeypatch: pytest.MonkeyPatch,
    case: _EmptyBatchCase,
) -> None:
    import projectkoios.optimization.alignment.numpy.batched_procrustes as module

    def unexpected_svd(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("empty batch must not invoke SVD")

    monkeypatch.setattr(np.linalg, "svd", unexpected_svd)
    source_shape = case.batch_shape + (case.row_count, case.column_count)
    source = np.empty(source_shape, dtype=np.complex128)

    result = module.NumpyBatchedComplexUnitaryProcrustesSolver().solve(
        module.NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(
            source, source
        )
    )

    assert result.batch_shape == case.batch_shape
    assert result.right_unitary_completions.shape == case.batch_shape + (
        case.column_count,
        case.column_count,
    )
    assert result.cross_singular_values.shape == case.batch_shape + (case.column_count,)
    assert result.all_full_rank
    assert result.rank_deficient_count == 0
    assert result.first_rank_deficient_index is None
    assert (
        result.require_unique_right_unitary_transformations()
        is result.right_unitary_completions
    )
    assert result.backend_identifier is None
    assert result.backend_version is None


@pytest.mark.parametrize(
    "case",
    _INDEXED_FAILURE_CASES,
    ids=lambda case: case.identifier,
)
def test_batched_svd_failure_reports_first_reproducible_c_order_index(
    monkeypatch: pytest.MonkeyPatch,
    case: _IndexedFailureCase,
) -> None:
    import projectkoios.optimization.alignment.numpy.batched_procrustes as module

    matrix_dimension = 2
    failure_marker = 2.0
    original: Any = np.linalg.svd

    def failing_svd(
        matrix: Any,
        *,
        full_matrices: bool,
        compute_uv: bool,
    ) -> object:
        if matrix.ndim > 2:
            raise np.linalg.LinAlgError("batch failure")
        if matrix[0, 0].real == failure_marker:
            raise np.linalg.LinAlgError("item failure")
        return original(
            matrix,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )

    monkeypatch.setattr(np.linalg, "svd", failing_svd)
    shape = case.batch_shape + (matrix_dimension, matrix_dimension)
    source = np.broadcast_to(
        np.eye(matrix_dimension, dtype=np.complex128),
        shape,
    ).copy()
    target = source.copy()
    for index in case.failing_indices:
        target[index] *= failure_marker

    with pytest.raises(
        RuntimeError,
        match=re.escape(repr(case.expected_first_index)),
    ):
        module.NumpyBatchedComplexUnitaryProcrustesSolver().solve(
            module.NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(
                source, target
            )
        )


def test_cross_overflow_reports_first_failing_batch_index() -> None:
    safe_value = 1.0
    overflowing_value = 1.0e308
    expected_failure_index = (1,)
    source = np.asarray(
        [[[safe_value]], [[overflowing_value]]],
        dtype=np.complex128,
    )

    with pytest.raises(
        FloatingPointError,
        match=re.escape(repr(expected_failure_index)),
    ):
        NumpyBatchedComplexUnitaryProcrustesSolver().solve(
            NumpyBatchedComplexUnitaryProcrustesDefinition.from_arrays(source, source)
        )


def test_solver_rejects_wrong_definition_nominal_type() -> None:
    with pytest.raises(TypeError, match="definition"):
        NumpyBatchedComplexUnitaryProcrustesSolver().solve(object())  # type: ignore[arg-type]
