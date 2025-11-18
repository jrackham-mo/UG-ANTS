# (C) Crown Copyright, Met Office. All rights reserved.  # noqa: D100
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from collections import UserDict

import ugants.io.save


class Outputs(UserDict):  # noqa: D101
    pass


class NetCDFOutput:  # noqa: D101
    def save(self, cube, filepath):  # noqa: D102
        ugants.io.save.ugrid(cube, filepath)
