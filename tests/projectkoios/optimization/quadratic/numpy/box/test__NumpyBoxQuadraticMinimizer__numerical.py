"""Independent bounded numerical checks for dense box minimization."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

import pytest

mp: Any = pytest.importorskip("mpmath")
np: Any = pytest.importorskip("numpy")
scipy: Any = pytest.importorskip("scipy")
optimize: Any = pytest.importorskip("scipy.optimize")

from projectkoios.optimization.quadratic.numpy.box import (  # noqa: E402
    NumpyBoxQuadraticMinimizer,
)
from projectkoios.optimization.quadratic.quadratic import (  # noqa: E402
    PositiveDefiniteQuadraticLoss,
)
from projectkoios.optimization.spaces.parameters import (  # noqa: E402
    ParameterBound,
    ParameterSpace,
)

pytestmark = pytest.mark.numerical


def _space(lower: Any, upper: Any) -> ParameterSpace:
    return ParameterSpace(
        tuple(
            ParameterBound(f"x_{index}", float(low), float(high))
            for index, (low, high) in enumerate(zip(lower, upper, strict=True))
        )
    )


def test_seeded_and_consumer_dimensions_agree_with_scipy_reference() -> None:
    generator = np.random.default_rng(20261014)
    solver = NumpyBoxQuadraticMinimizer()
    maximum_coordinate_error = 0.0
    maximum_loss_error = 0.0
    maximum_condition_number = 0.0
    maximum_kkt_ratio = 0.0

    for dimension in range(1, 7):
        for _ in range(8):
            factor = generator.normal(size=(dimension, dimension))
            matrix_array = factor.T @ factor + 0.25 * np.eye(dimension)
            center_array = generator.normal(size=dimension) * 2.0
            lower = generator.uniform(-2.0, 0.0, size=dimension)
            upper = lower + generator.uniform(0.1, 3.0, size=dimension)
            matrix = tuple(tuple(float(value) for value in row) for row in matrix_array)
            center = tuple(float(value) for value in center_array)
            minimum_squared_loss = 0.125
            loss = PositiveDefiniteQuadraticLoss(
                center,
                matrix,
                minimum_squared_loss,
            )
            actual = solver.solve(loss, _space(lower, upper))

            def objective(
                point: Any,
                center_reference: Any = center_array,
                matrix_reference: Any = matrix_array,
                minimum_reference: float = minimum_squared_loss,
            ) -> float:
                delta = np.asarray(point, dtype=np.float64) - center_reference
                return minimum_reference + float(delta @ matrix_reference @ delta)

            def gradient(
                point: Any,
                center_reference: Any = center_array,
                matrix_reference: Any = matrix_array,
            ) -> Any:
                delta = np.asarray(point, dtype=np.float64) - center_reference
                return 2.0 * matrix_reference @ delta

            reference = optimize.minimize(
                objective,
                np.clip(center_array, lower, upper),
                jac=gradient,
                method="L-BFGS-B",
                bounds=tuple(zip(lower, upper, strict=True)),
                options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 4000},
            )
            assert reference.success, reference.message
            coordinate_error = max(
                abs(actual_value - float(reference_value))
                for actual_value, reference_value in zip(
                    actual.coordinates,
                    reference.x,
                    strict=True,
                )
            )
            loss_error = abs(actual.squared_loss - float(reference.fun))
            maximum_coordinate_error = max(
                maximum_coordinate_error,
                coordinate_error,
            )
            maximum_loss_error = max(maximum_loss_error, loss_error)
            maximum_condition_number = max(
                maximum_condition_number,
                float(np.linalg.cond(matrix_array)),
            )
            maximum_kkt_ratio = max(maximum_kkt_ratio, actual.maximum_kkt_ratio)
            assert actual.coordinates == pytest.approx(
                tuple(float(value) for value in reference.x),
                rel=2.0e-7,
                abs=2.0e-8,
            )
            assert actual.squared_loss == pytest.approx(
                float(reference.fun),
                rel=2.0e-10,
                abs=2.0e-11,
            )
            assert actual.maximum_kkt_ratio <= 1.0

    consumer_bound = 0.11426128163399648
    for dimension in (1, 2, 3, 4, 6, 12, 27, 36):
        center_array = np.linspace(-0.3, 0.3, dimension)
        lower = np.full(dimension, -consumer_bound)
        upper = np.full(dimension, consumer_bound)
        vector = np.linspace(0.25, 1.0, dimension)
        for matrix_array in (
            np.eye(dimension) + 0.05 * np.outer(vector, vector),
            np.diag(np.linspace(1.0, 3.8, dimension)),
        ):
            matrix = tuple(tuple(float(value) for value in row) for row in matrix_array)
            center = tuple(float(value) for value in center_array)
            loss = PositiveDefiniteQuadraticLoss(center, matrix, 0.0)
            actual = solver.solve(loss, _space(lower, upper))

            def consumer_objective(
                point: Any,
                center_reference: Any = center_array,
                matrix_reference: Any = matrix_array,
            ) -> float:
                delta = np.asarray(point, dtype=np.float64) - center_reference
                return float(delta @ matrix_reference @ delta)

            def consumer_gradient(
                point: Any,
                center_reference: Any = center_array,
                matrix_reference: Any = matrix_array,
            ) -> Any:
                delta = np.asarray(point, dtype=np.float64) - center_reference
                return 2.0 * matrix_reference @ delta

            reference = optimize.minimize(
                consumer_objective,
                np.clip(center_array, lower, upper),
                jac=consumer_gradient,
                method="L-BFGS-B",
                bounds=tuple(zip(lower, upper, strict=True)),
                options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 4000},
            )
            assert reference.success, reference.message
            coordinate_error = max(
                abs(actual_value - float(reference_value))
                for actual_value, reference_value in zip(
                    actual.coordinates,
                    reference.x,
                    strict=True,
                )
            )
            loss_error = abs(actual.squared_loss - float(reference.fun))
            maximum_coordinate_error = max(
                maximum_coordinate_error,
                coordinate_error,
            )
            maximum_loss_error = max(maximum_loss_error, loss_error)
            maximum_condition_number = max(
                maximum_condition_number,
                float(np.linalg.cond(matrix_array)),
            )
            maximum_kkt_ratio = max(
                maximum_kkt_ratio,
                actual.maximum_kkt_ratio,
            )
            assert actual.coordinates == pytest.approx(
                tuple(float(value) for value in reference.x),
                rel=2.0e-7,
                abs=2.0e-8,
            )
            assert actual.squared_loss == pytest.approx(
                float(reference.fun),
                rel=2.0e-10,
                abs=2.0e-11,
            )
            assert actual.maximum_kkt_ratio <= 1.0

    high_precision_center = (2.0, -3.0, 0.25, 1.5)
    high_precision_matrix = (
        (1.0e-6, 2.0e-7, 0.0, 0.0),
        (2.0e-7, 2.0, 0.3, 0.0),
        (0.0, 0.3, 5.0e3, 10.0),
        (0.0, 0.0, 10.0, 1.0e7),
    )
    high_precision_bounds = (
        (-1.0, 1.0),
        (-2.0, 0.0),
        (-0.5, 0.5),
        (0.0, 2.0),
    )
    high_precision_point, high_precision_value = _mp_face_reference(
        high_precision_center,
        high_precision_matrix,
        high_precision_bounds,
    )
    high_precision_actual = solver.solve(
        PositiveDefiniteQuadraticLoss(
            high_precision_center,
            high_precision_matrix,
            0.0,
        ),
        _space(
            np.asarray([item[0] for item in high_precision_bounds]),
            np.asarray([item[1] for item in high_precision_bounds]),
        ),
    )
    high_precision_coordinate_error = max(
        abs(actual_value - float(reference_value))
        for actual_value, reference_value in zip(
            high_precision_actual.coordinates,
            high_precision_point,
            strict=True,
        )
    )
    high_precision_loss_error = abs(
        high_precision_actual.squared_loss - float(high_precision_value)
    )
    high_precision_condition_number = float(
        np.linalg.cond(np.asarray(high_precision_matrix, dtype=np.float64))
    )
    assert high_precision_coordinate_error <= 1.0e-10
    assert high_precision_loss_error <= 1.0e-10

    output = os.environ.get("PROJECTKOIOS_BOX_QUADRATIC_CALIBRATION_OUTPUT")
    if output is not None:
        repository_root = Path(__file__).resolve().parents[6]
        implementation_path = repository_root / (
            "src/python/projectkoios/optimization/quadratic/box.py"
        )
        numpy_backend_path = repository_root / (
            "src/python/projectkoios/optimization/quadratic/numpy/box.py"
        )
        numerical_test_path = Path(__file__).resolve()
        loss_facade_path = repository_root / (
            "src/python/projectkoios/optimization/quadratic/quadratic.py"
        )
        numpy_loss_path = repository_root / (
            "src/python/projectkoios/optimization/quadratic/numpy/quadratic.py"
        )
        tolerance_path = repository_root / (
            "src/python/projectkoios/optimization/quadratic/constrained.py"
        )
        parameter_space_path = repository_root / (
            "src/python/projectkoios/optimization/spaces/parameters.py"
        )
        numpy_configuration = cast(dict[str, Any], np.show_config(mode="dicts"))
        python_information = numpy_configuration.get("Python Information")
        if isinstance(python_information, dict):
            python_information.pop("path", None)
        evidence = {
            "calibration_schema": 1,
            "case_count": 64,
            "dimension_minimum": 1,
            "dimension_maximum": 36,
            "consumer_active_dimensions": [1, 2, 3, 4, 6, 12, 27, 36],
            "consumer_symmetric_bound_requested_decimal": "0.11426128163399648",
            "consumer_symmetric_bound_binary64": consumer_bound,
            "implementation_sha256": hashlib.sha256(
                implementation_path.read_bytes()
            ).hexdigest(),
            "loss_facade_sha256": hashlib.sha256(
                loss_facade_path.read_bytes()
            ).hexdigest(),
            "mpmath_decimal_precision": 100,
            "mpmath_version": mp.__version__,
            "numerical_test_sha256": hashlib.sha256(
                numerical_test_path.read_bytes()
            ).hexdigest(),
            "numpy_backend_sha256": hashlib.sha256(
                numpy_backend_path.read_bytes()
            ).hexdigest(),
            "numpy_configuration": numpy_configuration,
            "numpy_loss_sha256": hashlib.sha256(
                numpy_loss_path.read_bytes()
            ).hexdigest(),
            "numpy_version": np.__version__,
            "parameter_space_sha256": hashlib.sha256(
                parameter_space_path.read_bytes()
            ).hexdigest(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "scipy_version": scipy.__version__,
            "tolerance_contract_sha256": hashlib.sha256(
                tolerance_path.read_bytes()
            ).hexdigest(),
            "numpy_random_seed": 20261014,
            "observed": {
                "maximum_coordinate_absolute_error": maximum_coordinate_error,
                "maximum_loss_absolute_error": maximum_loss_error,
                "maximum_matrix_condition_number": maximum_condition_number,
                "maximum_kkt_ratio": maximum_kkt_ratio,
                "high_precision_active_set_change_count": (
                    high_precision_actual.active_set_change_count
                ),
                "high_precision_condition_number": high_precision_condition_number,
                "high_precision_coordinate_absolute_error": (
                    high_precision_coordinate_error
                ),
                "high_precision_iteration_count": (
                    high_precision_actual.iteration_count
                ),
                "high_precision_kkt_ratio": (high_precision_actual.maximum_kkt_ratio),
                "high_precision_linear_solve_count": (
                    high_precision_actual.linear_solve_count
                ),
                "high_precision_loss_absolute_error": high_precision_loss_error,
            },
        }
        Path(output).write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _mp_face_reference(
    center: tuple[float, ...],
    matrix: tuple[tuple[float, ...], ...],
    bounds: tuple[tuple[float, float], ...],
) -> tuple[tuple[Any, ...], Any]:
    mp.mp.dps = 100
    center_mp = tuple(
        mp.mpf(value.as_integer_ratio()[0]) / value.as_integer_ratio()[1]
        for value in center
    )
    matrix_mp = tuple(
        tuple(
            mp.mpf(value.as_integer_ratio()[0]) / value.as_integer_ratio()[1]
            for value in row
        )
        for row in matrix
    )
    bounds_mp = tuple(
        (
            mp.mpf(lower.as_integer_ratio()[0]) / lower.as_integer_ratio()[1],
            mp.mpf(upper.as_integer_ratio()[0]) / upper.as_integer_ratio()[1],
        )
        for lower, upper in bounds
    )
    best_point: tuple[Any, ...] | None = None
    best_value: Any = None
    for statuses in itertools.product((0, 1, 2), repeat=len(center)):
        point = list(center_mp)
        free = [index for index, status in enumerate(statuses) if status == 0]
        active = [index for index, status in enumerate(statuses) if status != 0]
        for index in active:
            point[index] = bounds_mp[index][0 if statuses[index] == 1 else 1]
        if free:
            principal = mp.matrix(
                [[matrix_mp[row][column] for column in free] for row in free]
            )
            right = mp.matrix(
                [
                    -sum(
                        matrix_mp[row][column] * (point[column] - center_mp[column])
                        for column in active
                    )
                    for row in free
                ]
            )
            solved = mp.lu_solve(principal, right)
            for offset, index in enumerate(free):
                point[index] = center_mp[index] + solved[offset]
        if any(
            point[index] < bounds_mp[index][0] or point[index] > bounds_mp[index][1]
            for index in free
        ):
            continue
        delta = tuple(point[index] - center_mp[index] for index in range(len(point)))
        value = sum(
            delta[row] * matrix_mp[row][column] * delta[column]
            for row in range(len(point))
            for column in range(len(point))
        )
        if best_value is None or value < best_value:
            best_point = tuple(point)
            best_value = value
    if best_point is None or best_value is None:
        raise AssertionError("high-precision enumeration found no candidate")
    return best_point, best_value


def test_ill_conditioned_four_dimensional_case_agrees_with_high_precision() -> None:
    center = (2.0, -3.0, 0.25, 1.5)
    matrix = (
        (1.0e-6, 2.0e-7, 0.0, 0.0),
        (2.0e-7, 2.0, 0.3, 0.0),
        (0.0, 0.3, 5.0e3, 10.0),
        (0.0, 0.0, 10.0, 1.0e7),
    )
    bounds = ((-1.0, 1.0), (-2.0, 0.0), (-0.5, 0.5), (0.0, 2.0))
    reference_point, reference_value = _mp_face_reference(center, matrix, bounds)
    actual = NumpyBoxQuadraticMinimizer().solve(
        PositiveDefiniteQuadraticLoss(center, matrix, 0.0),
        _space(
            np.asarray([item[0] for item in bounds]),
            np.asarray([item[1] for item in bounds]),
        ),
    )

    assert actual.coordinates == pytest.approx(
        tuple(float(value) for value in reference_point),
        rel=1.0e-10,
        abs=1.0e-11,
    )
    assert actual.squared_loss == pytest.approx(
        float(reference_value),
        rel=1.0e-10,
        abs=1.0e-11,
    )
