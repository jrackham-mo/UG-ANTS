# (C) Crown Copyright, Met Office. All rights reserved.  # noqa: D100
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import abc
from collections import UserDict

import iris
import ugants
import ugants.io.load


class Sources(UserDict):  # noqa: D101
    pass


class Source(abc.ABC):
    """Base class for source inputs."""

    @abc.abstractmethod
    def load(self, filepath, constraint):
        """Load a source from file, with a constraint."""
        pass


class NetCDFSource(Source):  # noqa: D101
    def __init__(self, ugrid=True, constraint=None):
        self.ugrid = ugrid
        self.constraint = constraint
        if constraint is None:
            self.constraint = iris.Constraint()

    def load(self, filepath, constraint):
        """Load an iris CubeList from a netCDF file."""
        constraint = iris.Constraint(constraint) & self.constraint
        if self.ugrid:
            source = ugants.io.load.ugrid(filepath, constraint)
        else:
            source = ugants.io.load.cf(filepath, constraints=constraint)
        return source


class MeshSource(Source):  # noqa: D101
    def load(self, filepath, constraint):  # noqa: D102
        return ugants.io.load.mesh(filepath, constraint)
