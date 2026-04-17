# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office Github Copilot Enterprise.
import numpy as np
import pytest
from ugants.utils._mesh import _face_indices


@pytest.fixture()
def expected_c4_panel_0():
    return np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    )


@pytest.fixture()
def expected_c4_panel_1():
    return np.array(
        [
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
        ]
    )


@pytest.fixture()
def expected_c4_panel_2():
    return np.array(
        [
            [32, 33, 34, 35],
            [36, 37, 38, 39],
            [40, 41, 42, 43],
            [44, 45, 46, 47],
        ]
    )


@pytest.fixture()
def expected_c4_panel_3():
    return np.array(
        [
            [48, 49, 50, 51],
            [52, 53, 54, 55],
            [56, 57, 58, 59],
            [60, 61, 62, 63],
        ]
    )


@pytest.fixture()
def expected_c4_panel_4():
    return np.array(
        [
            [64, 65, 66, 67],
            [68, 69, 70, 71],
            [72, 73, 74, 75],
            [76, 77, 78, 79],
        ]
    )


@pytest.fixture()
def expected_c4_panel_5():
    return np.array(
        [
            [80, 81, 82, 83],
            [84, 85, 86, 87],
            [88, 89, 90, 91],
            [92, 93, 94, 95],
        ]
    )


class TestC4:
    N = 4

    def test_panel_0(self, expected_c4_panel_0):
        actual = _face_indices(n=self.N, panel_id=0)
        np.testing.assert_array_equal(actual, expected_c4_panel_0)

    def test_panel_1(self, expected_c4_panel_1):
        actual = _face_indices(n=self.N, panel_id=1)
        np.testing.assert_array_equal(actual, expected_c4_panel_1)

    def test_panel_2(self, expected_c4_panel_2):
        actual = _face_indices(n=self.N, panel_id=2)
        np.testing.assert_array_equal(actual, expected_c4_panel_2)

    def test_panel_3(self, expected_c4_panel_3):
        actual = _face_indices(n=self.N, panel_id=3)
        np.testing.assert_array_equal(actual, expected_c4_panel_3)

    def test_panel_4(self, expected_c4_panel_4):
        actual = _face_indices(n=self.N, panel_id=4)
        np.testing.assert_array_equal(actual, expected_c4_panel_4)

    def test_panel_5(self, expected_c4_panel_5):
        actual = _face_indices(n=self.N, panel_id=5)
        np.testing.assert_array_equal(actual, expected_c4_panel_5)
