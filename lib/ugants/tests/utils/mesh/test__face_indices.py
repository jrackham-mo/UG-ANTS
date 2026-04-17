# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
from ugants.utils.mesh import _face_indices


class TestC4:
    N = 4

    def test_panel_0(self):
        expected = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        actual = _face_indices(n=self.N, panel_id=0)
        np.testing.assert_array_equal(actual, expected)

    def test_panel_1(self):
        expected = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        actual = _face_indices(n=self.N, panel_id=0)
        np.testing.assert_array_equal(actual, expected)
