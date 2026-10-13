from typing import Any

import pytest

np: Any = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.mixture import GaussianMixture  # noqa: E402

from projectkoios.optimization.clustering.gmm import (  # noqa: E402
    GaussianMixtureClusteringCompleted,
    GaussianMixtureClusteringFailed,
    GaussianMixtureClusteringFailureCode,
    GaussianMixtureClusteringRequest,
)
from projectkoios.optimization.clustering.scipy.gmm import (  # noqa: E402
    ScipyGaussianMixtureClusterer,
)

pytestmark = [pytest.mark.integration, pytest.mark.numerical]


class TestScipyGaussianMixtureClusterer:
    def test_cluster_selects_and_canonically_orders_two_separated_groups(self) -> None:
        observations = (
            (-5.0,),
            (-4.8,),
            (-5.2,),
            (-4.9,),
            (5.0,),
            (4.8,),
            (5.2,),
            (4.9,),
        )

        result = ScipyGaussianMixtureClusterer().cluster(
            observations,
            GaussianMixtureClusteringRequest(2, 0),
        )

        assert type(result) is GaussianMixtureClusteringCompleted
        assert result.selected_components == 2
        assert tuple(score.component_count for score in result.model_scores) == (1, 2)
        assert tuple(value.observation_index for value in result.assignments) == tuple(
            range(8)
        )
        assert tuple(value.cluster_id for value in result.assignments) == (
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        )
        assert all(value.probability > 0.99 for value in result.assignments)
        assert result.components[0].mean < result.components[1].mean

    def test_cluster_matches_recovery_backend_configuration(self) -> None:
        observations = (
            (-5.0,),
            (-4.8,),
            (-5.2,),
            (-4.9,),
            (5.0,),
            (4.8,),
            (5.2,),
            (4.9,),
        )
        matrix = np.asarray(observations, dtype=np.float64)
        standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        models = tuple(
            GaussianMixture(
                n_components=count,
                covariance_type="full",
                random_state=0,
                n_init=5,
                max_iter=500,
                reg_covar=1.0e-8,
            ).fit(standardized)
            for count in (1, 2)
        )
        selected = min(
            models,
            key=lambda model: (model.bic(standardized), model.n_components),
        )
        labels = selected.predict(standardized)
        probabilities = selected.predict_proba(standardized)
        order = tuple(
            sorted(
                range(selected.n_components),
                key=lambda index: (
                    tuple(float(value) for value in selected.means_[index]),
                    index,
                ),
            )
        )
        canonical = {original: new for new, original in enumerate(order)}

        result = ScipyGaussianMixtureClusterer().cluster(
            observations,
            GaussianMixtureClusteringRequest(2, 0),
        )

        assert type(result) is GaussianMixtureClusteringCompleted
        assert tuple(
            score.akaike_information_criterion for score in result.model_scores
        ) == pytest.approx(tuple(model.aic(standardized) for model in models))
        assert tuple(
            score.bayesian_information_criterion for score in result.model_scores
        ) == pytest.approx(tuple(model.bic(standardized) for model in models))
        assert tuple(value.cluster_id for value in result.assignments) == tuple(
            canonical[int(label)] for label in labels
        )
        observed_probabilities = tuple(
            value.probability for value in result.assignments
        )
        expected_probabilities = tuple(
            probabilities[index, int(labels[index])] for index in range(len(labels))
        )
        assert observed_probabilities == pytest.approx(expected_probabilities)
        assert tuple(value.weight for value in result.components) == pytest.approx(
            tuple(selected.weights_[index] for index in order)
        )
        assert np.allclose(
            np.asarray(tuple(value.mean for value in result.components)),
            selected.means_[list(order)],
        )
        assert np.allclose(
            np.asarray(tuple(value.covariance for value in result.components)),
            selected.covariances_[list(order)],
        )

    def test_cluster_is_deterministic_with_constant_feature(self) -> None:
        observations = tuple(
            (value, 7.0) for value in (-5.0, -4.8, -5.2, -4.9, 5.0, 4.8, 5.2, 4.9)
        )
        request = GaussianMixtureClusteringRequest(2, 23)
        clusterer = ScipyGaussianMixtureClusterer()

        first = clusterer.cluster(observations, request)
        second = clusterer.cluster(observations, request)

        assert first == second
        assert type(first) is GaussianMixtureClusteringCompleted
        assert first.feature_means == pytest.approx((0.0, 7.0))
        assert first.feature_scales[1] == 1.0

    def test_single_observation_has_explicit_zero_covariance_result(self) -> None:
        result = ScipyGaussianMixtureClusterer().cluster(
            ((2.0, 5.0),),
            GaussianMixtureClusteringRequest(64, 4_294_967_295),
        )

        assert type(result) is GaussianMixtureClusteringCompleted
        assert result.feature_means == (2.0, 5.0)
        assert result.feature_scales == (1.0, 1.0)
        assert result.selected_components == 1
        assert result.assignments[0].observation_index == 0
        assert result.assignments[0].probability == 1.0
        assert result.components[0].mean == (0.0, 0.0)
        assert result.components[0].covariance == ((0.0, 0.0), (0.0, 0.0))
        assert result.iterations == 0

    def test_empty_observations_return_closed_failure(self) -> None:
        result = ScipyGaussianMixtureClusterer().cluster(
            (),
            GaussianMixtureClusteringRequest(2, 0),
        )

        assert type(result) is GaussianMixtureClusteringFailed
        assert result.code is GaussianMixtureClusteringFailureCode.NO_OBSERVATIONS

    def test_input_matrix_rejects_awkward_numeric_boundaries(self) -> None:
        clusterer = ScipyGaussianMixtureClusterer()
        request = GaussianMixtureClusteringRequest(1, 0)

        with pytest.raises(ValueError, match="rectangular"):
            clusterer.cluster(((1.0,), (2.0, 3.0)), request)
        with pytest.raises(TypeError, match="excluding bool"):
            clusterer.cluster(((True,),), request)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="real number"):
            clusterer.cluster((("1",),), request)  # type: ignore[arg-type]
