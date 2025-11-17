import abc  # noqa: D100
from collections import UserDict

import iris
import ugants.io.load
import ugants.io.save


class Sources(UserDict):  # noqa: D101
    pass


class Source(abc.ABC):
    """Base class for source inputs."""

    @abc.abstractmethod
    def load(self, filepath, constraint):
        """Load a source from file, with a constraint."""
        pass


class Settings(UserDict):  # noqa: D101
    pass


class Setting:
    """Generic setting class that passes the value directly from the config."""

    def parse(self, value):  # noqa: D102
        return value


class Outputs(UserDict):  # noqa: D101
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


class ChoiceOf(Setting):
    """Setting that takes a restricted set of choices."""

    def __init__(self, *args):
        self.choices = args

    def parse(self, choice):  # noqa: D102
        if choice not in self.choices:
            raise ValueError(
                f"Invalid choice: {choice}, expected one of {self.choices}"
            )
        return choice


class FloatSetting(Setting):
    """Setting that takes a floating point value."""

    def parse(self, value):
        """Cast config value to float."""
        return float(value)


class IntSetting(Setting):
    """Setting that takes an integer value."""

    def parse(self, value):
        """Cast config value to int."""
        return int(value)


class BoolSetting(Setting):
    """Setting that takes a boolean value."""

    def parse(self, value):
        """Cast config value to bool."""
        return bool(value)


class NetCDFOutput:  # noqa: D101
    def save(self, cube, filepath):  # noqa: D102
        ugants.io.save.ugrid(cube, filepath)


_latitude_constraint = iris.Constraint(latitude=lambda y: y > 0)
SOURCES = Sources(
    structured_source=NetCDFSource(ugrid=False, constraint=_latitude_constraint),
    ugrid_source=NetCDFSource(),
    target=MeshSource(),
)

SETTINGS = Settings(
    scheme=ChoiceOf("a", "b"), a_float=FloatSetting(), an_int=IntSetting()
)

OUTPUTS = Outputs(result=NetCDFOutput())


def main(  # noqa: D103
    structured_source, ugrid_source, target, scheme, a_float, an_int
):
    return Outputs(result=ugrid_source)
