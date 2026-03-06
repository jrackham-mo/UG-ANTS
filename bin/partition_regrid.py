#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Regrid using iris-esmf-regrid's partitioned regridder."""

import argparse

import iris


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to source file")
    parser.add_argument("--output", help="Path to output")
    parser.add_argument(
        "--num-src-chunks",
        help="Specify the number of blocks to use to divide up the cube. "
        "Dimensions are specified in y,x axis order. Each integer describes the"
        "number of blocks that dimension will be divided into.",
        nargs=2,
    )
    return parser


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()
    source_path = args.source
    output_path = args.output
    num_src_chunks = args.num_src_chunks
    source_cubes = iris.load(source_path)
