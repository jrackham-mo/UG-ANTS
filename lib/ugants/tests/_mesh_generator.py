# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office Github Copilot Enterprise.
"""Utility functions for creating test meshes."""

import argparse
from collections.abc import Iterable

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.experimental.ugrid import Connectivity, Mesh


def _panel_angles_to_lonlat(
    panel_id: int,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert panel-local equiangular angles to geographic coordinates.

    Uses an equiangular gnomonic projection to map the panel-local angular
    coordinates ``(alpha, beta)`` to geographic longitude and latitude.

    Parameters
    ----------
    panel_id : int
        Panel index in the range 0-5.
    alpha : :class:`numpy.ndarray`
        Panel-local angle in the along-x direction, in radians.
        Must lie in the range [-pi/4, pi/4].
    beta : :class:`numpy.ndarray`
        Panel-local angle in the along-y direction, in radians.
        Must lie in the range [-pi/4, pi/4].

    Returns
    -------
    lons : :class:`numpy.ndarray`
        Longitudes in degrees, in the range (-180, 180].
    lats : :class:`numpy.ndarray`
        Latitudes in degrees, in the range [-90, 90].
    """
    u = np.tan(alpha)
    v = np.tan(beta)
    r = np.sqrt(1.0 + u**2 + v**2)

    if panel_id == 0:  # +x face, centred at lon=0°, lat=0°
        x, y, z = 1.0 / r, u / r, v / r
    elif panel_id == 1:  # +y face, centred at lon=90°, lat=0°
        x, y, z = -u / r, 1.0 / r, v / r
    elif panel_id == 2:  # -x face, centred at lon=180°, lat=0°
        x, y, z = -1.0 / r, -u / r, v / r
    elif panel_id == 3:  # -y face, centred at lon=270°, lat=0°
        x, y, z = u / r, -1.0 / r, v / r
    elif panel_id == 4:  # +z face, north polar panel
        x, y, z = -v / r, u / r, 1.0 / r
    else:  # -z face (panel_id == 5), south polar panel
        x, y, z = v / r, u / r, -1.0 / r

    lons = np.degrees(np.arctan2(y, x))
    lats = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    return lons, lats


def cubedsphere_panel_coords(
    panel_id: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate lat/lon coordinates for a single panel of a cubed-sphere mesh.

    Uses an equiangular gnomonic projection. The six panels are arranged as
    follows:

    * Panel 0: equatorial panel centred at longitude 0°
    * Panel 1: equatorial panel centred at longitude 90°
    * Panel 2: equatorial panel centred at longitude 180°
    * Panel 3: equatorial panel centred at longitude 270°
    * Panel 4: north polar panel
    * Panel 5: south polar panel

    Adjacent equatorial panels share edges, as do each equatorial panel and
    both polar panels.

    Example
    -------
    For a C2 panel (n=2), the four equatorial panels cover roughly:

    >>> node_lons, node_lats, face_lons, face_lats = cubedsphere_panel_coords(0, 2)
    >>> node_lons.shape
    (3, 3)
    >>> face_lons.shape
    (2, 2)

    Parameters
    ----------
    panel_id : int
        Index of the panel to generate, in the range 0-5.
    n : int
        Panel resolution; each panel contains n x n faces and
        (n+1) x (n+1) nodes.

    Returns
    -------
    node_lons : :class:`numpy.ndarray` of shape (n+1, n+1)
        Longitude coordinates of the panel nodes, in degrees.
    node_lats : :class:`numpy.ndarray` of shape (n+1, n+1)
        Latitude coordinates of the panel nodes, in degrees.
    face_lons : :class:`numpy.ndarray` of shape (n, n)
        Longitude coordinates of the panel face centres, in degrees.
    face_lats : :class:`numpy.ndarray` of shape (n, n)
        Latitude coordinates of the panel face centres, in degrees.

    Raises
    ------
    ValueError
        If ``panel_id`` is not in the range 0-5.
    ValueError
        If ``n`` is not a positive integer.
    """
    if panel_id not in range(6):
        raise ValueError(f"panel_id must be in the range 0-5, got {panel_id}.")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")

    node_angles = np.linspace(-np.pi / 4, np.pi / 4, n + 1)
    face_angles = (node_angles[:-1] + node_angles[1:]) / 2

    node_alpha, node_beta = np.meshgrid(node_angles, node_angles)
    face_alpha, face_beta = np.meshgrid(face_angles, face_angles)

    node_lons, node_lats = _panel_angles_to_lonlat(panel_id, node_alpha, node_beta)
    face_lons, face_lats = _panel_angles_to_lonlat(panel_id, face_alpha, face_beta)

    node_lons, node_lats, face_lons, face_lats = (
        node_lons.reshape(-1),
        node_lats.reshape(-1),
        face_lons.reshape(-1),
        face_lats.reshape(-1),
    )

    return node_lons, node_lats, face_lons, face_lats


def cubedsphere_panel_face_node_connectivity(
    n: int,
    start_index: int = 0,
) -> np.ma.MaskedArray:
    """Derive the face-node connectivity for one cubed-sphere panel.

    The returned indices represent an ``n x n`` panel of quadrilateral faces
    connected to a ``(n+1) x (n+1)`` node grid. Node numbering matches the
    flattening of a 2D node array with shape ``(n+1, n+1)`` in C-order.

    Parameters
    ----------
    n : int
        Panel resolution; each panel contains ``n x n`` faces.
    start_index : int, default=0
        Offset to apply to all node indices, for compatibility with one-indexed
        or globally-indexed connectivities.

    Returns
    -------
    :class:`numpy.ma.MaskedArray`
        A masked array of shape ``(n*n, 4)`` containing face-node indices.
        The node order for each face is anti-clockwise:
        southwest, northwest, northeast, southeast.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    TypeError
        If ``start_index`` is not an integer.
    ValueError
        If ``start_index`` is negative.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}.")
    if not isinstance(start_index, int):
        raise TypeError(f"start_index must be an integer, got {type(start_index)}.")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}.")
    node_ids = np.arange(n * n, dtype=int)
    face_node_indices = _get_connected_nodes(node_ids, n).transpose()
    return np.ma.masked_array(face_node_indices + start_index)


def _get_connected_nodes(face_id: int, n: int):
    """Get the connected nodes for a given face ID."""
    row_number = face_id // n
    column_number = face_id % n
    top_left = ((n + 1) * row_number) + column_number
    top_right = top_left + 1
    bottom_left = top_left + n + 1
    bottom_right = bottom_left + 1

    connected_nodes = np.array([top_left, bottom_left, bottom_right, top_right])
    return connected_nodes


def cubedsphere_panel_mesh(panel_id: int, n: int) -> Mesh:
    """Construct a single panel of a cubedsphere mesh."""
    node_lons, node_lats, face_lons, face_lats = cubedsphere_panel_coords(panel_id, n)
    node_x = AuxCoord(
        points=node_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="node_x_coordinates",
    )
    node_y = AuxCoord(
        points=node_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="node_y_coordinates",
    )
    face_x = AuxCoord(
        points=face_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="face_x_coordinates",
    )
    face_y = AuxCoord(
        points=face_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="face_y_coordinates",
    )

    face_node_connectivity_indices = cubedsphere_panel_face_node_connectivity(n)
    face_node_connectivity = Connectivity(
        indices=face_node_connectivity_indices, cf_role="face_node_connectivity"
    )
    mesh = Mesh(
        long_name=f"Panel {panel_id}",
        topology_dimension=2,
        node_coords_and_axes=[(node_x, "x"), (node_y, "y")],
        face_coords_and_axes=[(face_x, "x"), (face_y, "y")],
        connectivities=[face_node_connectivity],
    )
    return mesh


def cubedsphere_panel_mesh_cube(panel_id: int, n: int, location: str = "face") -> Cube:
    """Construct a cube defined on a single panel of a cubedsphere mesh."""
    match location:
        case "face":
            n_points = n * n
        case "node":
            n_points = (n + 1) * (n + 1)
        case _:
            raise ValueError(f"Unsupported location: '{location}'")
    mesh = cubedsphere_panel_mesh(panel_id, n)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords(location)
    data = np.arange(n_points)
    cube = Cube(
        data=data,
        long_name=f"{location}_data",
        aux_coords_and_dims=[(mesh_coord_x, 0), (mesh_coord_y, 0)],
    )
    return cube


def cubedsphere_from_panels(meshes: Iterable[Mesh]) -> Mesh:
    """Construct a single cubedsphere mesh from individual panel meshes."""
    face_lons = np.concatenate([mesh.face_coords.face_x.points for mesh in meshes])
    face_lats = np.concatenate([mesh.face_coords.face_y.points for mesh in meshes])

    new_face_x = AuxCoord(face_lons, **meshes[0].face_coords.face_x.metadata._asdict())
    new_face_y = AuxCoord(face_lats, **meshes[0].face_coords.face_y.metadata._asdict())

    node_lons = np.concatenate([mesh.node_coords.node_x.points for mesh in meshes])
    node_lats = np.concatenate([mesh.node_coords.node_y.points for mesh in meshes])

    nodes_xy = np.stack([node_lons, node_lats])
    unique_nodes_xy, first_instances = np.unique(
        nodes_xy,
        return_inverse=True,
        axis=1,
    )
    new_node_x = AuxCoord(
        unique_nodes_xy[0], **meshes[0].node_coords.node_x.metadata._asdict()
    )
    new_node_y = AuxCoord(
        unique_nodes_xy[1], **meshes[0].node_coords.node_y.metadata._asdict()
    )
    original_face_node_indices = np.concatenate(
        [
            _remap_connectivity_indices(mesh, panel_id)
            for panel_id, mesh in enumerate(meshes)
        ]
    )
    new_face_node_indices = first_instances[original_face_node_indices]
    new_face_node_indices = np.ma.masked_array(new_face_node_indices)
    new_connectivity_kwargs = meshes[0].face_node_connectivity.metadata._asdict()
    new_face_node_connectivity = Connectivity(
        new_face_node_indices, **new_connectivity_kwargs
    )

    new_mesh = Mesh(
        long_name="UG-ANTS cubedsphere",
        topology_dimension=2,
        node_coords_and_axes=[(new_node_x, "x"), (new_node_y, "y")],
        face_coords_and_axes=[(new_face_x, "x"), (new_face_y, "y")],
        connectivities=[new_face_node_connectivity],
    )
    return new_mesh


def _remap_connectivity_indices(mesh: Mesh, panel_id: int) -> np.ndarray:
    original_face_node_connectivity = mesh.face_node_connectivity
    original_face_node_indices = (
        original_face_node_connectivity.indices_by_location()
        - original_face_node_connectivity.start_index
    )
    offset = panel_id * len(mesh.node_coords.node_x.points)
    face_node_indices = original_face_node_indices + offset
    return face_node_indices


def cubedsphere_mesh(n: int) -> Mesh:
    panels = [cubedsphere_panel_mesh(panel_id, n) for panel_id in range(6)]
    mesh = cubedsphere_from_panels(panels)
    return mesh


def cubedsphere_cube(n: int):
    mesh = cubedsphere_mesh(n)
    mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords("face")
    data = np.arange(n * n * 6)
    cube = Cube(
        data=data,
        long_name="face_data",
        aux_coords_and_dims=[(mesh_coord_x, 0), (mesh_coord_y, 0)],
    )
    return cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "n", type=int, help="Number of cells along a panel edge (the 'C-number')"
    )
    parser.add_argument(
        "output", help="Where to write the generated cubed sphere (must be a .nc file)"
    )
    args = parser.parse_args()
    n = args.n
    output = args.output
    cube = cubedsphere_cube(n)
    iris.save(cube, output)
