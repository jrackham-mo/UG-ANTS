# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office Github Copilot Enterprise.
import numpy as np
import pytest
from ugants.utils.mesh_generator.panel import CartesianPoints, rotate


@pytest.fixture()
def cube_corners():
    """Requarter turn a set of points defining the corners of a cube."""
    x = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    y = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    z = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    points = CartesianPoints(x, y, z)
    return points


def test_invalid_panel_id_exception(cube_corners):
    expected_msg = "Invalid panel id: 6"
    with pytest.raises(ValueError, match=expected_msg):
        rotate(cube_corners, 6)


def test_rotate_0(cube_corners):
    """Test the null operation."""
    expected_x = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    expected_y = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    expected_z = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    actual = rotate(cube_corners, 0)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)


def test_rotate_1(cube_corners):
    """Test rotation one quarter turn anticlockwise about the z axis."""
    expected_x = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    expected_y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    expected_z = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    actual = rotate(cube_corners, 1)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)


def test_rotate_2(cube_corners):
    """Test rotation two quarter turns anticlockwise about the z axis."""
    expected_x = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    expected_y = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    expected_z = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    actual = rotate(cube_corners, 2)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)


def test_rotate_3(cube_corners):
    """Test rotation three quarter turns anticlockwise about the z axis."""
    expected_x = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    expected_y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    expected_z = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    actual = rotate(cube_corners, 3)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)


def test_rotate_4(cube_corners):
    """Test rotation one quarter quarter turn clockwise about the y axis."""
    expected_x = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    expected_y = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    expected_z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    actual = rotate(cube_corners, 4)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)


def test_rotate_5(cube_corners):
    """Test rotation one quarter turn anticlockwise about the y axis."""
    expected_x = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    expected_y = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    expected_z = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    actual = rotate(cube_corners, 5)

    np.testing.assert_array_equal(actual.x, expected_x)
    np.testing.assert_array_equal(actual.y, expected_y)
    np.testing.assert_array_equal(actual.z, expected_z)
