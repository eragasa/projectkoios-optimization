"""Unit verification for centered uniform finite-periodic meshes."""

from __future__ import annotations

import pytest

from projectkoios.numerics.finite_periodic.meshes import CenteredUniformMesh

pytestmark = pytest.mark.unit


def test_one_dimensional_mesh_retains_offset_and_derived_origin() -> None:
    mesh = CenteredUniformMesh(1, 6, (0.5,))

    assert mesh.point_count == 6
    assert mesh.coordinate_origin == pytest.approx((-5.0 / 12.0,), abs=0.0, rel=0.0)


def test_three_dimensional_mesh_has_tensor_product_point_count() -> None:
    mesh = CenteredUniformMesh(3, 4, (0.0, 0.5, 1.0))

    assert mesh.point_count == 64
    assert mesh.coordinate_origin == (-0.5, -0.375, -0.25)


def test_mesh_rejects_dimensions_outside_owned_range() -> None:
    with pytest.raises(ValueError):
        CenteredUniformMesh(4, 4, (0.0, 0.0, 0.0, 0.0))


def test_mesh_rejects_odd_size_without_ambiguous_nyquist_inventory() -> None:
    with pytest.raises(ValueError):
        CenteredUniformMesh(2, 5, (0.5, 0.5))


def test_mesh_rejects_noncanonical_offsets() -> None:
    with pytest.raises(TypeError):
        CenteredUniformMesh(2, 4, (0, 0.5))


def test_mesh_rejects_nonfinite_offset() -> None:
    with pytest.raises(ValueError):
        CenteredUniformMesh(2, 4, (0.5, float("inf")))
