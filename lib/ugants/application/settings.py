# (C) Crown Copyright, Met Office. All rights reserved.  # noqa: D100
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from collections import UserDict


class Settings(UserDict):  # noqa: D101
    pass


class Setting:
    """Generic setting class that passes the value directly from the config."""

    def parse(self, value):  # noqa: D102
        return value


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
