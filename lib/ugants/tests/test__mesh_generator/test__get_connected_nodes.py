# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
from ugants.tests._mesh_generator import _get_connected_nodes


def test_n_1():
    expected = [0, 2, 3, 1]
    actual = _get_connected_nodes(face_id=0, n=1)
    np.testing.assert_array_equal(actual, expected)


def test_n_2():
    n = 2
    expected = np.array(
        [
            [0, 3, 4, 1],
            [1, 4, 5, 2],
            [3, 6, 7, 4],
            [4, 7, 8, 5],
        ]
    )
    actual = _get_connected_nodes(face_id=np.arange(n * n), n=n).T
    np.testing.assert_array_equal(actual, expected)
