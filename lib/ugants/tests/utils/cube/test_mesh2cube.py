# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
from iris.cube import Cube
from iris.experimental.ugrid import Mesh
from ugants.tests.stock import panel_mesh
from ugants.utils.cube import mesh2cube


def test_single_panel_no_data():
    mesh = panel_mesh(4)
    assert isinstance(mesh, Mesh)
    cube = mesh2cube(mesh)
    assert isinstance(cube, Cube)
    assert cube.mesh is mesh


def test_single_panel_with_data():
    mesh = panel_mesh(4)
    assert isinstance(mesh, Mesh)
    data = np.ones(16)
    cube = mesh2cube(mesh, data)
    assert isinstance(cube, Cube)
    assert cube.mesh is mesh
    np.testing.assert_array_equal(cube.data, data)
