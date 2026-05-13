# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from dataclasses import dataclass

import iris.coords
import iris.cube
import numpy as np
from iris.experimental.ugrid import Connectivity, Mesh


def panel_mesh(c: int, panel_id: int):
    panel = Panel(c, panel_id)
    mesh = panel.to_iris_mesh()
    return mesh


class Panel:
    def __init__(self, c: int, panel_id: int):
        if panel_id not in range(6):
            raise ValueError(
                f"panel_id should be in range [0, 5], got unexpected value {panel_id}"
            )
        self.c = c
        self.panel_id = panel_id
        panel_0_node_points, panel_0_face_points = (
            self._generate_panel_0_plane_cartesian_coordinates()
        )
        if panel_id < 4:
            # Equatorial panels
            # Rotate panel 0 by a multiple of 90 degrees in longitude
            self.node_lats, self.node_lons = panel_0_node_points.to_lat_lon()
            self.face_lats, self.face_lons = panel_0_face_points.to_lat_lon()
            self.node_lons += panel_id * 90.0
            self.face_lons += panel_id * 90.0

        elif panel_id == 4:
            # North polar panel
            # Rotate panel 0 by 90 degrees "upwards"
            panel_4_node_points = rotate_panel_0_to_4(panel_0_node_points)
            panel_4_face_points = rotate_panel_0_to_4(panel_0_face_points)
            self.node_lats, self.node_lons = panel_4_node_points.to_lat_lon()
            self.face_lats, self.face_lons = panel_4_face_points.to_lat_lon()

        elif panel_id == 5:
            # South polar panel
            # Rotate panel 0 by 90 degrees "downwards"
            panel_5_node_points = rotate_panel_0_to_5(panel_0_node_points)
            panel_5_face_points = rotate_panel_0_to_5(panel_0_face_points)
            self.node_lats, self.node_lons = panel_5_node_points.to_lat_lon()
            self.face_lats, self.face_lons = panel_5_face_points.to_lat_lon()

    def to_iris_mesh(self):
        node_x_auxcoord = iris.coords.AuxCoord(
            points=self.node_lons.flatten(),
            standard_name="longitude",
            units="degrees_east",
            long_name="node_x_coordinates",
        )
        node_y_auxcoord = iris.coords.AuxCoord(
            points=self.node_lats.flatten(),
            standard_name="latitude",
            units="degrees_north",
            long_name="node_y_coordinates",
        )
        face_x_auxcoord = iris.coords.AuxCoord(
            points=self.face_lons.flatten(),
            standard_name="longitude",
            units="degrees_east",
            long_name="face_x_coordinates",
        )
        face_y_auxcoord = iris.coords.AuxCoord(
            points=self.face_lats.flatten(),
            standard_name="latitude",
            units="degrees_north",
            long_name="face_y_coordinates",
        )
        face_node_connectivity = Connectivity(
            generate_face_node_connectivity_array(self.c),
            cf_role="face_node_connectivity",
        )
        face_face_connectivity = Connectivity(
            generate_face_face_connectivity_array(self.c),
            cf_role="face_face_connectivity",
        )
        mesh = Mesh(
            long_name=f"UG-ANTS mesh: panel {self.panel_id}",
            topology_dimension=2,
            node_coords_and_axes=[(node_x_auxcoord, "x"), (node_y_auxcoord, "y")],
            connectivities=[face_node_connectivity, face_face_connectivity],
            face_coords_and_axes=[(face_x_auxcoord, "x"), (face_y_auxcoord, "y")],
        )
        return mesh

    def to_iris_cube(self, data=None):
        if data is None:
            data = np.arange(self.c**2)
        mesh = self.to_iris_mesh()
        location = "face"
        mesh_coord_x, mesh_coord_y = mesh.to_MeshCoords(location)
        cube = iris.cube.Cube(
            data=data,
            long_name=f"{location}_data",
            aux_coords_and_dims=[(mesh_coord_x, 0), (mesh_coord_y, 0)],
        )
        return cube

    def _generate_panel_0_plane_cartesian_coordinates(self):
        """Generate points on the x=1 plane representing the nodes and faces.

        Points are spaced equally in angle, so that they are more evenly distributed
        on the surface of the sphere.

        Parameters
        ----------
        c : int
            The number of faces along the panel edge

        Returns
        -------
        tuple[CartesianPoints, CartesianPoints]
        """
        # Alphas are angles about the z-axis
        # Betas are angles about the y-axis
        node_alphas = np.linspace(-np.pi / 4, np.pi / 4, self.c + 1)
        node_betas = np.linspace(np.pi / 4, -np.pi / 4, self.c + 1)
        face_alphas = 0.5 * (node_alphas[:-1] + node_alphas[1:])
        face_betas = 0.5 * (node_betas[:-1] + node_betas[1:])

        node_alphas_grid, node_betas_grid = np.meshgrid(node_alphas, node_betas)
        face_alphas_grid, face_betas_grid = np.meshgrid(face_alphas, face_betas)

        node_points = CartesianPoints.from_panel_angles(
            node_alphas_grid, node_betas_grid
        )
        face_points = CartesianPoints.from_panel_angles(
            face_alphas_grid, face_betas_grid
        )

        return node_points, face_points


@dataclass
class CartesianPoints:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def from_panel_angles(cls, alphas: np.ndarray, betas: np.ndarray):
        z = np.tan(betas)
        y = np.tan(alphas)
        x = np.ones_like(y)
        return CartesianPoints(x, y, z)

    @property
    def r(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    def project_to_unit_sphere(self):
        r = self.r
        x = self.x / r
        y = self.y / r
        z = self.z / r
        return CartesianPoints(x, y, z)

    def to_lat_lon(self):
        """Convert cartesian vectors to latitude-longitude coordinates in degrees.

        Longitude is defined as the angle between the x-axis and the projection
        of the vector in the x-y plane.

        Latitude is defined as the angle between the vector and the x-y plane.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays of latitude and longitude coordinates, respectively
        """
        xy_radius = np.hypot(self.x, self.y)
        lat_rad = np.arctan2(self.z, xy_radius)
        lon_rad = np.arctan2(self.y, self.x)

        lat_deg = np.rad2deg(lat_rad)
        lon_deg = np.rad2deg(lon_rad)
        return lat_deg, lon_deg


def generate_face_node_connectivity_array(c: int):
    """Generate a face-node connectivity array for a single panel.

    Each face is connected to 4 nodes. The connectivity array maps each face to
    its nodes in anticlockwise order, starting from the top left node.

    Example
    -------
    For c=2, there are 2 faces and 3 nodes along each edge of the panel.
    The arrangement looks like this:

    [0]---[1]---[2]
     | (0) | (1) |
    [3]---[4]---[5]
     | (2) | (3) |
    [6]---[7]---[8]

    Key:
    [i] = node i
    (j) = face j

    >>> generate_face_node_connectivity_array(c=2)
    masked_array(
        data=[
            [0, 3, 4, 1],
            [1, 4, 5, 2],
            [3, 6, 7, 4],
            [4, 7, 8, 5]
            ],
        mask=False,
        fill_value=999999)

    Parameters
    ----------
    c: int
        The number of faces along a panel edge

    Returns
    -------
    np.ma.array
        An array of shape (c**2, 4) mapping each face to its 4 nodes
    """
    node_indices = np.arange((c + 1) ** 2).reshape(c + 1, c + 1)
    top_left = node_indices[:-1, :-1].flatten()
    bottom_left = node_indices[1:, :-1].flatten()
    top_right = node_indices[:-1, 1:].flatten()
    bottom_right = node_indices[1:, 1:].flatten()
    connectivity_array = np.vstack([top_left, bottom_left, bottom_right, top_right]).T
    connectivity_array = np.ma.masked_array(connectivity_array)
    return connectivity_array


def generate_face_face_connectivity_array(c: int):
    """Generate a face-face connectivity array for a single panel.

    Each face is connected to a maximum of 4 faces:

    - faces internal to the panel have 4 neighbours
    - faces on the edge of a panel have 3 neighbours
    - faces on the corner of a panel have two neighbours

    Where a face has less than 4 neighbours (i.e. it is on an edge/corner of the
    panel), the connectivity array will contain masked elements.

    Connected faces are returned in anticlockwise order, starting from the top.

    Example
    -------
    For c=2, there are 2 faces and 3 nodes along each edge of the panel.
    The arrangement looks like this:

    [0]---[1]---[2]
     | (0) | (1) |
    [3]---[4]---[5]
     | (2) | (3) |
    [6]---[7]---[8]

    Key:
    [i] = node i
    (j) = face j

    >>> generate_face_face_connectivity_array(c=2)
    masked_array(
        data=[
            [--, --, 2, 1],
            [--, 0, 3, --],
            [0, --, --, 3],
            [1, 2, --, --]
            ],
        mask=[
            [True, True, False, False],
            [True, False, False, True],
            [False, True, True, False],
            [False, False, True, True]
            ],
        fill_value=999999)

    Parameters
    ----------
    c: int
        The number of faces along a panel edge

    Returns
    -------
    np.ma.array
        An array of shape (c**2, 4) mapping each face to its neighbouring faces
    """
    face_indices = np.arange(c**2).reshape(c, c)
    padded_face_indices = np.pad(face_indices, pad_width=1, constant_values=-1)
    top = padded_face_indices[:-2, 1:-1].flatten()
    bottom = padded_face_indices[2:, 1:-1].flatten()
    left = padded_face_indices[1:-1, :-2].flatten()
    right = padded_face_indices[1:-1, 2:].flatten()
    connectivity_array = np.vstack([top, left, bottom, right]).T
    mask = connectivity_array == -1
    connectivity_array = np.ma.masked_array(connectivity_array, mask=mask)
    return connectivity_array


def rotate_panel_0_to_1(cartesian_points: CartesianPoints) -> CartesianPoints:
    x = -cartesian_points.y
    y = cartesian_points.x
    z = cartesian_points.z
    return CartesianPoints(x, y, z)


def rotate_panel_0_to_2(cartesian_points: CartesianPoints) -> CartesianPoints:
    x = -cartesian_points.x
    y = -cartesian_points.y
    z = cartesian_points.z
    return CartesianPoints(x, y, z)


def rotate_panel_0_to_3(cartesian_points: CartesianPoints) -> CartesianPoints:
    x = cartesian_points.y
    y = -cartesian_points.x
    z = cartesian_points.z
    return CartesianPoints(x, y, z)


def rotate_panel_0_to_4(cartesian_points: CartesianPoints) -> CartesianPoints:
    x = -cartesian_points.z
    y = cartesian_points.y
    z = cartesian_points.x
    return CartesianPoints(x, y, z)


def rotate_panel_0_to_5(cartesian_points: CartesianPoints) -> CartesianPoints:
    x = cartesian_points.z
    y = cartesian_points.y
    z = -cartesian_points.x
    return CartesianPoints(x, y, z)
