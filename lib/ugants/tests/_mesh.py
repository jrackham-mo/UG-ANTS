# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np
import pyvista as pv
from iris.coords import AuxCoord
from iris.experimental.ugrid import Connectivity, Mesh, save_mesh


def cubedsphere_mesh(side_length):
    box_polydata = pv.Box(level=side_length + 1)
    cell_centres_polydata = box_polydata.cell_centers()

    node_lats, node_lons = calculate_lat_lon(box_polydata.points)
    face_lats, face_lons = calculate_lat_lon(cell_centres_polydata.points)

    face_node_indices = box_polydata.regular_faces

    node_x_aux = AuxCoord(
        points=node_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="node_x_coordinates",
    )
    node_y_aux = AuxCoord(
        points=node_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="node_y_coordinates",
    )
    face_x_aux = AuxCoord(
        points=face_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="face_x_coordinates",
    )
    face_y_aux = AuxCoord(
        points=face_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="face_y_coordinates",
    )
    face_node_connectivity = Connectivity(
        indices=face_node_indices, cf_role="face_node_connectivity", start_index=0
    )
    mesh = Mesh(
        long_name="my_mesh",
        topology_dimension=2,
        node_coords_and_axes=[(node_x_aux, "x"), (node_y_aux, "y")],
        connectivities=[face_node_connectivity],
        face_coords_and_axes=[(face_x_aux, "x"), (face_y_aux, "y")],
    )
    return mesh


def calculate_lat_lon(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    xy = np.hypot(x, y)
    lats = np.degrees(np.arctan2(z, xy))
    lons = np.degrees(np.arctan2(y, x))
    return lats, lons


if __name__ == "__main__":
    import os
    import pathlib

    path = pathlib.Path(os.environ["SCRATCH"]) / "C12_mesh.nc"
    mesh = cubedsphere_mesh(12)
    save_mesh(mesh, path)
