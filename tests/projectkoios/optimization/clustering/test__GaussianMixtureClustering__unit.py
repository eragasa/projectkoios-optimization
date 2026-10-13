import pytest

from projectkoios.optimization.clustering.gmm import (
    GaussianMixtureAssignment,
    GaussianMixtureClusteringCompleted,
    GaussianMixtureClusteringFailureCode,
    GaussianMixtureClusteringRequest,
    GaussianMixtureComponent,
    GaussianMixtureModelScore,
)

pytestmark = pytest.mark.unit


class TestGaussianMixtureClusteringContracts:
    def test_request_enforces_bounded_builtin_integers(self) -> None:
        assert GaussianMixtureClusteringRequest(4, 0) == (
            GaussianMixtureClusteringRequest(4, 0)
        )

        for value in (True, 1.0, "1"):
            with pytest.raises(TypeError):
                GaussianMixtureClusteringRequest(value, 0)  # type: ignore[arg-type]
        for value in (0, 65):
            with pytest.raises(ValueError):
                GaussianMixtureClusteringRequest(value, 0)
        with pytest.raises(ValueError):
            GaussianMixtureClusteringRequest(1, -1)

    def test_component_requires_matching_square_covariance(self) -> None:
        component = GaussianMixtureComponent(
            cluster_id=0,
            weight=1,
            mean=(1, 2.0),
            covariance=((2, 0), (0, 3)),
        )

        assert component.weight == 1.0
        assert component.mean == (1.0, 2.0)
        assert component.covariance == ((2.0, 0.0), (0.0, 3.0))
        with pytest.raises(ValueError):
            GaussianMixtureComponent(0, 1.0, (1.0, 2.0), ((1.0,),))

    def test_completed_result_requires_exact_ordered_pullback(self) -> None:
        score = GaussianMixtureModelScore(1, 0.0, 0.0)
        component = GaussianMixtureComponent(0, 1.0, (0.0,), ((1.0,),))

        with pytest.raises(ValueError, match="preserve every input row"):
            GaussianMixtureClusteringCompleted(
                feature_means=(0.0,),
                feature_scales=(1.0,),
                model_scores=(score,),
                selected_components=1,
                assignments=(
                    GaussianMixtureAssignment(1, 0, 1.0),
                    GaussianMixtureAssignment(0, 0, 1.0),
                ),
                components=(component,),
                converged=True,
                iterations=1,
                backend_version="1.8.0",
            )

    def test_failure_codes_are_numeric_capability_states(self) -> None:
        assert GaussianMixtureClusteringFailureCode.NO_OBSERVATIONS.value == (
            "no_observations"
        )
        assert GaussianMixtureClusteringFailureCode.FIT_FAILED.value == "fit_failed"
