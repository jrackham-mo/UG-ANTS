#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office GitHub Copilot Enterprise.
"""Regrid using iris-esmf-regrid's partitioned regridder."""

import argparse
import multiprocessing
import os

import dask.bag
import esmf_regrid
import iris
import iris.mesh
from esmf_regrid.experimental.partition import Partition


class MultiPartition(Partition):
    """Extend the default partition class to parallelise the weights calculation."""

    def __init__(
        self,
        src_file,
        tgt_file,
        scheme,
        file_names,
        use_dask_src_chunks=False,
        src_chunks=None,
        num_src_chunks=None,
        explicit_src_blocks=None,
        auto_generate=False,
        saved_files=None,
    ):
        """Class for breaking down regridding into manageable chunks.

        Parameters
        ----------
        src_file : str
            Path to source file.
        tgt_file : str
            Path to target file.
        scheme : regridding scheme
            Regridding scheme to generate regridders, either ESMFAreaWeighted or ESMFBilinear.
        file_names : iterable of str
            A list of file names to save/load parts of the regridder to/from.
        use_dask_src_chunks : bool, default=False
            If true, partition using the same chunks from the source cube.
        src_chunks : numpy array, tuple of int or tuple of tuple of int, default=None
            Specify the size of blocks to use to divide up the cube. Dimensions are specified
            in y,x axis order. If `src_chunks` is a tuple of int, each integer describes
            the maximum size of a block in that dimension. If `src_chunks` is a tuple of tuples,
            each sub-tuple describes the size of each successive block in that dimension. The sum
            of these block sizes in each of the sub-tuples should add up to the total size of that
            dimension or else an error is raised.
        num_src_chunks : tuple of int
            Specify the number of blocks to use to divide up the cube. Dimensions are specified
            in y,x axis order. Each integer describes the number of blocks that dimension will
            be divided into.
        explicit_src_blocks : arraylike NxMx2
            Explicitly specify the bounds of each block in the partition. Describes N blocks
            along M dimensions with a pair of upper and lower bounds. The upper and lower bounds
            describe a slice of an array, e.g. the bounds (3, 6) describe the indices 3, 4, 5 in
            a particular dimension.
        auto_generate : bool, default=False
            When true, start generating files on initialisation.
        saved_files : iterable of str
            A list of paths to previously saved files.

        Warnings
        --------
        This class is still experimental. While we aim to maintain backwards compatibility where
        possible, there is no guarantee that the structure of any generated files will remain
        consistent and compatible with future versions.

        Note
        ----
        The source is partitioned into blocks using one of the four mutually exclusive arguments,
        `use_dask_src_chunks`, `src_chunks`, `num_src_chunks`, or `explicit_src_blocks`. These
        describe a partition into a number of blocks which must equal the number of `file_names`.

        Currently, it is only possible to divide the source grid into chunks.
        Meshes are not yet supported as a source.
        """  # noqa: E501
        self.src_file = src_file
        self.tgt_file = tgt_file
        src = iris.load_cube(src_file)
        tgt = iris.mesh.load_mesh(tgt_file)
        super().__init__(
            src=src,
            tgt=tgt,
            scheme=scheme,
            file_names=file_names,
            use_dask_src_chunks=use_dask_src_chunks,
            src_chunks=src_chunks,
            num_src_chunks=num_src_chunks,
            explicit_src_blocks=explicit_src_blocks,
            auto_generate=auto_generate,
            saved_files=saved_files,
        )

    def generate_files(self, files_to_generate=None):
        """Generate files with regridding information.

        Parameters
        ----------
        files_to_generate : int, default=None
            Specify the number of files to generate,
            default behaviour is to generate all files.
        """
        if files_to_generate is None:
            files = self.unsaved_files
        else:
            if not isinstance(files_to_generate, int):
                msg = "`files_to_generate` must be an integer."
                raise ValueError(msg)
            files = self.unsaved_files[:files_to_generate]

        src_blocks = [self.file_block_dict[file] for file in files]
        cpu_count = multiprocessing.cpu_count()

        # .. or as given by slurm allocation.
        # Only relevant when using Slurm for job scheduling
        if "SLURM_NTASKS" in os.environ:
            cpu_count = int(os.environ["SLURM_NTASKS"])

        # Do not exceed the number of CPUs available, leaving 1 for the system.
        num_workers = cpu_count - 1
        print(f"Using {num_workers} workers from {cpu_count} CPUs...")

        # Now do the parallel regrid
        with dask.config.set(num_workers=num_workers):
            file_bag = dask.bag.from_sequence(zip(files, src_blocks, strict=True))
            self.saved_files = file_bag.starmap(
                _generate_regrid_weights,
                src_file=self.src_file,
                tgt_file=self.tgt_file,
                scheme=self.scheme,
            ).compute()


def _generate_regrid_weights(file, src_block, src_file, tgt_file, scheme):
    src = iris.load_cube(src_file)
    src = esmf_regrid.experimental.partition._get_chunk(src, src_block)
    tgt = iris.mesh.load_mesh(tgt_file)
    regridder = scheme.regridder(src, tgt)
    weights = regridder.regridder.weight_matrix
    regridder = esmf_regrid.experimental.partition.PartialRegridder(
        src, tgt, src_block, None, weights, scheme
    )
    esmf_regrid.experimental.partition.save_regridder(
        regridder, file, allow_partial=True
    )
    return file


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
    parser.add_argument(
        "--files-to-generate",
        type=int,
        help="Number of pieces to calculate",
        default=None,
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


def regrid(src_file, tgt_file, num_src_chunks, partition_dir, files_to_generate=None):
    """Regrid source to target mesh.

    Parameters
    ----------
    src_file : str
        Path to source file.
    tgt_file : str
        Path to target file.
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
    print(f"{num_src_chunks=}, num_files={len(file_names)}, {files_to_generate=}")
    partition = MultiPartition(
        src_file,
        tgt_file,
        scheme,
        num_src_chunks=num_src_chunks,
        file_names=file_names,
    )
    partition.generate_files(files_to_generate)
    # breakpoint()
    # partition.saved_files = set(os.listdir(partition_dir)).intersection(file_names)

    results = partition.apply_regridders(partition.src, allow_incomplete=True)
    return results


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()

    regridded_cubes = regrid(
        args.source,
        args.target_mesh,
        args.num_src_chunks,
        args.temp_dir,
        args.files_to_generate,
    )

    iris.save(regridded_cubes, args.output)
