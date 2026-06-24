# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
from ugants.tests import assert_masked_array_equal
from ugants.utils.mesh_generator.panel import generate_face_node_connectivity_array


def test_2by2():
    connectivity = generate_face_node_connectivity_array(2)
    assert isinstance(connectivity, np.ma.MaskedArray)

    expected = np.ma.masked_array(
        data=[
            [0, 3, 4, 1],
            [1, 4, 5, 2],
            [3, 6, 7, 4],
            [4, 7, 8, 5],
        ],
        mask=False,
        fill_value=999999,
    )
    assert_masked_array_equal(connectivity, expected)


def test_4by4():
    connectivity = generate_face_node_connectivity_array(4)
    assert isinstance(connectivity, np.ma.MaskedArray)

    data = np.array(
        [
            [0, 5, 6, 1],
            [1, 6, 7, 2],
            [2, 7, 8, 3],
            [3, 8, 9, 4],
            [5, 10, 11, 6],
            [6, 11, 12, 7],
            [7, 12, 13, 8],
            [8, 13, 14, 9],
            [10, 15, 16, 11],
            [11, 16, 17, 12],
            [12, 17, 18, 13],
            [13, 18, 19, 14],
            [15, 20, 21, 16],
            [16, 21, 22, 17],
            [17, 22, 23, 18],
            [18, 23, 24, 19],
        ]
    )
    expected = np.ma.masked_array(
        data=data,
        mask=False,
        fill_value=999999,
    )
    assert_masked_array_equal(connectivity, expected)
