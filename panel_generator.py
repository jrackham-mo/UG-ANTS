#!/usr/bin/env python

import argparse

from ugants.io.save import ugrid
from ugants.utils.mesh_generator.panel import Panel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c_number", type=int)
    parser.add_argument("panel_number", type=int)
    parser.add_argument("output")
    parsed_args = parser.parse_args()
    panel = Panel(parsed_args.c_number, parsed_args.panel_number)
    cube = panel.to_iris_cube()
    ugrid(cube, parsed_args.output)
