# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import pytest
from ugants.tests._mesh_generator import cubedsphere_mesh


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_number_of_faces(n):
    """Test that the expected number of faces are present on the mesh."""
    expected_faces = 6 * n * n
    mesh = cubedsphere_mesh(n)
    actual_face_x_shape = mesh.face_coords.face_x.shape
    actual_face_y_shape = mesh.face_coords.face_y.shape

    assert actual_face_x_shape == (expected_faces,)
    assert actual_face_y_shape == (expected_faces,)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_number_of_nodes(n):
    """Test that the expected number of nodes are present on the mesh."""
    expected_nodes = 6 * n * n + 2
    mesh = cubedsphere_mesh(n)
    actual_node_x_shape = mesh.node_coords.node_x.shape
    actual_node_y_shape = mesh.node_coords.node_y.shape

    assert actual_node_x_shape == (expected_nodes,)
    assert actual_node_y_shape == (expected_nodes,)
