# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import pytest
from iris.experimental.ugrid import Mesh
from ugants.tests.stock import panel_mesh


@pytest.mark.parametrize("panel_id", range(6))
def test_mesh_size(panel_id):
    """Mesh should have c*c faces, and (c+1)*(c+1) nodes."""
    mesh = panel_mesh(4, panel_id)
    assert isinstance(mesh, Mesh)

    expected_faces = 16
    expected_nodes = 25

    actual_faces = len(mesh.face_coords.face_x.points)
    assert actual_faces == expected_faces

    actual_nodes = len(mesh.node_coords.node_x.points)
    assert actual_nodes == expected_nodes

    # Both face-face and face-node connectivities are expected to be of shape
    # (nfaces, 4), since each face is connected to 4 other faces, and 4 nodes
    expected_connectivity_shape = (16, 4)

    actual_face_face_connectivity_shape = mesh.face_face_connectivity.shape
    assert actual_face_face_connectivity_shape == expected_connectivity_shape

    actual_face_node_connectivity_shape = mesh.face_node_connectivity.shape
    assert actual_face_node_connectivity_shape == expected_connectivity_shape
