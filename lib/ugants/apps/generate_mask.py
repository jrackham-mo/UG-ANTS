# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the mask generation application."""

from ugants.application import outputs, settings, sources
from ugants.mask.command_line import derive_mask

SOURCES = sources.Sources(
    land_fraction=sources.NetCDFSource(
        constraint="land_area_fraction", single_cube=True
    )
)
SETTINGS = settings.Settings(mask_type=settings.ChoiceOf("land", "sea"))
OUTPUTS = outputs.Outputs(result=outputs.NetCDFOutput())


def main(land_fraction, mask_type):  # noqa: D103
    result = derive_mask(land_fraction, mask_type)
    return outputs.Outputs(result=result)
