from collections import UserDict  # noqa: D100

import ugants.io.load
import ugants.io.save


class Inputs(UserDict):  # noqa: D101
    pass


class Outputs(UserDict):  # noqa: D101
    pass


class NetCDFSource:  # noqa: D101
    def __init__(self, ugrid=True, fieldname=None):
        self.ugrid = ugrid

    def load(self, filepath, constraint):  # noqa: D102
        if self.ugrid:
            source = ugants.io.load.ugrid(filepath, constraint)
        else:
            source = ugants.io.load.cf(filepath, constraints=constraint)
        return source


class MeshSource:  # noqa: D101
    def load(self, filepath, constraint):  # noqa: D102
        return ugants.io.load.mesh(filepath, constraint)


class ChoiceOf:  # noqa: D101
    def __init__(self, *args):
        self.choices = args

    def load(self, choice, constraint):  # noqa: D102
        if choice not in self.choices:
            raise ValueError(
                f"Invalid choice: {choice}, expected one of {self.choices}"
            )
        return choice


class NetCDFOutput:  # noqa: D101
    def save(self, cube, filepath):  # noqa: D102
        ugants.io.save.ugrid(cube, filepath)


INPUTS = Inputs(
    structured_source=NetCDFSource(ugrid=False),
    ugrid_source=NetCDFSource(),
    target=MeshSource(),
    scheme=ChoiceOf("a", "b"),
)

OUTPUTS = Outputs(result=NetCDFOutput())


def main(structured_source, ugrid_source, target, scheme):  # noqa: D103
    return Outputs(result=ugrid_source)
