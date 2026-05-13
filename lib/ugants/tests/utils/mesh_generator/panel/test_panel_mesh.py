import numpy as np
import pytest
from ugants.tests import assert_masked_array_equal
from ugants.utils.mesh_generator.panel import panel_mesh


class PanelTestBase:
    """Base class for building tests for single panel meshes.

    Subclasses must provide the following:

    - c (int)
    - panel_id (int)
    - expected_nodes (tuple of arrays: expected_node_x, expected_node_y)
    - expected_faces (tuple of arrays: expected_face_x, expected_face_y)
    - expected_face_node_connectivity (masked array)
    - expected_face_face_connectivity (masked array)
    """

    @pytest.fixture()
    def mesh(self):
        mesh = panel_mesh(c=self.c, panel_id=self.panel_id)
        return mesh

    def test_node_coords(self, mesh, expected_nodes):
        expected_node_x, expected_node_y = expected_nodes

        actual_node_x = mesh.node_coords.node_x.points
        actual_node_y = mesh.node_coords.node_y.points

        np.testing.assert_array_almost_equal(actual_node_x, expected_node_x)
        np.testing.assert_array_almost_equal(actual_node_y, expected_node_y)

    def test_face_coords(self, mesh, expected_faces):
        expected_face_x, expected_face_y = expected_faces

        actual_face_x = mesh.face_coords.face_x.points
        actual_face_y = mesh.face_coords.face_y.points

        np.testing.assert_array_equal(actual_face_x, expected_face_x)
        np.testing.assert_array_equal(actual_face_y, expected_face_y)

    def test_face_node_connectivity(self, mesh, expected_face_node_connectivity):
        actual_face_node_connectivity = mesh.face_node_connectivity.indices

        np.testing.assert_array_equal(
            actual_face_node_connectivity, expected_face_node_connectivity
        )

    def test_face_face_connectivity(self, mesh, expected_face_face_connectivity):
        actual_face_face_connectivity = mesh.face_face_connectivity.indices

        assert_masked_array_equal(
            actual_face_face_connectivity, expected_face_face_connectivity
        )


class TestPanelMeshC1panel0(PanelTestBase):
    c = 1
    panel_id = 0

    @pytest.fixture()
    def expected_nodes(self):
        expected_node_x = np.array([-45.0, 45.0, -45.0, 45.0])
        expected_node_y = np.array(
            [35.26438968, 35.26438968, -35.26438968, -35.26438968]
        )
        return expected_node_x, expected_node_y

    @pytest.fixture()
    def expected_faces(self):
        expected_face_x = np.array([0.0])
        expected_face_y = np.array([0.0])
        return expected_face_x, expected_face_y

    @pytest.fixture()
    def expected_face_node_connectivity(self):
        expected_face_node_connectivity = np.ma.array([[0, 2, 3, 1]])
        return expected_face_node_connectivity

    @pytest.fixture()
    def expected_face_face_connectivity(self):
        expected_face_face_connectivity = np.ma.array(
            [[0, 0, 0, 0]], mask=[[True, True, True, True]]
        )
        return expected_face_face_connectivity
