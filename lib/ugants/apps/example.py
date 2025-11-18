# (C) Crown Copyright, Met Office. All rights reserved.  # noqa: D100
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.

import iris
from ugants.application import outputs, settings, sources

_latitude_constraint = iris.Constraint(latitude=lambda y: y > 0)
SOURCES = sources.Sources(
    structured_source=sources.NetCDFSource(
        ugrid=False, constraint=_latitude_constraint
    ),
    ugrid_source=sources.NetCDFSource(),
    target=sources.MeshSource(),
)

SETTINGS = settings.Settings(
    scheme=settings.ChoiceOf("a", "b"),
    a_float=settings.FloatSetting(),
    an_int=settings.IntSetting(),
)

OUTPUTS = outputs.Outputs(result=outputs.NetCDFOutput())


def main(  # noqa: D103
    structured_source, ugrid_source, target, scheme, a_float, an_int
):
    return outputs.Outputs(result=ugrid_source)
