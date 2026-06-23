# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
from ugants.tests import assert_masked_array_equal
from ugants.utils.mesh_generator.panel import generate_face_face_connectivity_array


def test_2by2():
    connectivity = generate_face_face_connectivity_array(2)
    assert isinstance(connectivity, np.ma.MaskedArray)

    expected = np.ma.masked_array(
        data=[
            [-1, -1, 2, 1],
            [-1, 0, 3, -1],
            [0, -1, -1, 3],
            [1, 2, -1, -1],
        ],
        mask=[
            [True, True, False, False],
            [True, False, False, True],
            [False, True, True, False],
            [False, False, True, True],
        ],
        fill_value=999999,
    )
    assert_masked_array_equal(connectivity, expected)


def test_4by4():
    connectivity = generate_face_face_connectivity_array(4)
    assert isinstance(connectivity, np.ma.MaskedArray)

    data = np.array(
        [
            [-1, -1, 4, 1],
            [-1, 0, 5, 2],
            [-1, 1, 6, 3],
            [-1, 2, 7, -1],
            [0, -1, 8, 5],
            [1, 4, 9, 6],
            [2, 5, 10, 7],
            [3, 6, 11, -1],
            [4, -1, 12, 9],
            [5, 8, 13, 10],
            [6, 9, 14, 11],
            [7, 10, 15, -1],
            [8, -1, -1, 13],
            [9, 12, -1, 14],
            [10, 13, -1, 15],
            [11, 14, -1, -1],
        ]
    )
    expected = np.ma.masked_array(
        data=data,
        mask=data == -1,
        fill_value=999999,
    )
    assert_masked_array_equal(connectivity, expected)
