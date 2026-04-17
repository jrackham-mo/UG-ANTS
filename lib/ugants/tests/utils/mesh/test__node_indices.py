# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pytest
from ugants.utils._mesh import (
    gen_node_indices_panel_0,
    gen_node_indices_panel_1,
    gen_node_indices_panel_2,
    gen_node_indices_panel_3,
)


@pytest.fixture()
def node_indices_panel_0():
    node_indices = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    )
    return node_indices


@pytest.fixture()
def node_indices_panel_1():
    node_indices = np.array(
        [
            [4, 25, 26, 27, 28],
            [9, 29, 30, 31, 32],
            [14, 33, 34, 35, 36],
            [19, 37, 38, 39, 40],
            [24, 41, 42, 43, 44],
        ]
    )
    return node_indices


@pytest.fixture()
def node_indices_panel_2():
    node_indices = np.array(
        [
            [28, 45, 46, 47, 48],
            [32, 49, 50, 51, 52],
            [36, 53, 54, 55, 56],
            [40, 57, 58, 59, 60],
            [44, 61, 62, 63, 64],
        ]
    )
    return node_indices


@pytest.fixture()
def node_indices_panel_3():
    node_indices = np.array(
        [
            [48, 65, 66, 67, 0],
            [52, 68, 69, 70, 5],
            [56, 71, 72, 73, 10],
            [60, 74, 75, 76, 15],
            [64, 77, 78, 79, 20],
        ]
    )
    return node_indices


@pytest.fixture()
def node_indices_panel_4():
    node_indices = np.array([[]])
    return node_indices


class TestC4:
    def test_panel_0(self, node_indices_panel_0):
        actual = gen_node_indices_panel_0(n=4)
        np.testing.assert_array_equal(actual, node_indices_panel_0)

    def test_panel_1(self, node_indices_panel_0, node_indices_panel_1):
        actual = gen_node_indices_panel_1(node_indices_panel_0)
        np.testing.assert_array_equal(actual, node_indices_panel_1)

    def test_panel_2(self, node_indices_panel_1, node_indices_panel_2):
        actual = gen_node_indices_panel_2(node_indices_panel_1)
        np.testing.assert_array_equal(actual, node_indices_panel_2)

    def test_panel_3(
        self, node_indices_panel_0, node_indices_panel_2, node_indices_panel_3
    ):
        actual = gen_node_indices_panel_3(node_indices_panel_0, node_indices_panel_2)
        np.testing.assert_array_equal(actual, node_indices_panel_3)

    def test_panel_4(
        self,
        node_indices_panel_0,
        node_indices_panel_1,
        node_indices_panel_2,
        node_indices_panel_3,
        node_indices_panel_4,
    ):
        pass

    # def test_panel_0(self):
    #     expected = np.array(
    #         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    #     )
    #     actual = _node_indices(n=self.N, panel_id=0)
    #     np.testing.assert_array_equal(actual, expected)
