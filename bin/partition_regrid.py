#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office GitHub Copilot Enterprise.
"""Regrid using iris-esmf-regrid's partitioned regridder."""

import argparse

import esmf_regrid
import iris
import iris.mesh
from esmf_regrid.experimental.partition import Partition


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to source file", required=True)
    parser.add_argument("--target-mesh", help="Path to target mesh file", required=True)
    parser.add_argument("--output", help="Path to write output", required=True)
    parser.add_argument(
        "--num-src-chunks",
        help="Specify the number of blocks to use to divide up the cube. "
        "Dimensions are specified in y,x axis order. Each integer describes the"
        "number of blocks that dimension will be divided into.",
        nargs=2,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--temp-dir",
        help="Directory to write temporary partition files to.",
        required=True,
    )
    return parser


def load_source(args):
    """Load source cubes.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
        Must contain a `source` attribute with the path to the source file.

    Returns
    -------
    iris.cube.CubeList
        The loaded source cubes.
    """
    source_cubes = iris.load(args.source)
    return source_cubes


def load_mesh(args):
    """Load target mesh.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
        Must contain a `target_mesh` attribute with the path to the target mesh file.

    Returns
    -------
    iris.mesh.Mesh
        The loaded target mesh.
    """
    target_mesh = iris.mesh.load_mesh(args.target_mesh)
    return target_mesh


def regrid(source_cubes, target_mesh, num_src_chunks, partition_dir):
    """Regrid source cubes to target mesh.

    Parameters
    ----------
    source_cubes : iris.cube.CubeList
        The source cubes to regrid.
    target_mesh : iris.mesh.Mesh
        The target mesh to regrid to.
    num_src_chunks : list of int
        The number of blocks to use to divide up the cube.
        Dimensions are specified in y,x axis order.
        Each integer describes the number of blocks that dimension will be divided into.
    partition_dir : str
        Directory to write temporary partition files to.

    Returns
    -------
    iris.cube.CubeList
        The regridded cubes.
    """
    file_names = [
        f"{partition_dir}/partition_{n}.nc"
        for n in range(num_src_chunks[0] * num_src_chunks[1])
    ]
    scheme = esmf_regrid.ESMFAreaWeighted()
    partition = Partition(
        source_cubes[0],
        target_mesh,
        scheme,
        num_src_chunks=num_src_chunks,
        file_names=file_names,
    )
    partition.generate_files()

    results = iris.cube.CubeList(
        partition.apply_regridders(source) for source in source_cubes
    )
    return results


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()

    source_cubes = load_source(args)
    target_mesh = load_mesh(args)

    regridded_cubes = regrid(
        source_cubes, target_mesh, args.num_src_chunks, args.temp_dir
    )

    iris.save(regridded_cubes, args.output)
