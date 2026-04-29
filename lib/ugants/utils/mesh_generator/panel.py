# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
from dataclasses import dataclass

import iris.coords
import numpy as np
from iris.experimental.ugrid import Mesh


class Panel:
    def __init__(self, c: int, panel_id: int):
        if panel_id not in range(6):
            raise ValueError(
                f"panel_id should be in range [0, 5], got unexpected value {panel_id}"
            )
        self.c = c
        self.panel_id = panel_id
        node_points, face_points = self._generate_plane_cartesian_coordinates()
        self.node_lats, self.node_lons = node_points.to_lat_lon()
        self.face_lats, self.face_lons = face_points.to_lat_lon()

    def to_iris_mesh(self):
        node_x_auxcoord = iris.coords.AuxCoord(
            points=self.node_lons.flatten(),
            standard_name="longitude",
            units="degrees_east",
            long_name="node_x_coordinates",
        )
        node_y_auxcoord = iris.coords.AuxCoord(
            points=self.node_lats.flatten(),
            standard_name="longitude",
            units="degrees_east",
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
            standard_name="longitude",
            units="degrees_east",
            long_name="face_y_coordinates",
        )

        mesh = Mesh(
            long_name=f"UG-ANTS mesh: panel {self.panel_id}",
            topology_dimension=2,
            node_coords_and_axes=[(node_x_auxcoord, "x"), (node_y_auxcoord, "y")],
            # connectivities=[edge_node_c, face_node_c],
            face_coords_and_axes=[(face_x_auxcoord, "x"), (face_y_auxcoord, "y")],
        )

    def _generate_plane_cartesian_coordinates(self):
        """Generate points on the x=1 plane representing the nodes and faces.

        Parameters
        ----------
        c : int
            The number of faces along the panel edge

        Returns
        -------
        tuple[CartesianPoints, CartesianPoints]
        """
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
        xy_radius = np.sqrt(np.square(self.x) + np.square(self.y))
        lat_rad = np.arctan2(self.z, xy_radius)
        lon_rad = np.arctan2(self.y, self.x)

        lat_deg = np.rad2deg(lat_rad)
        lon_deg = np.rad2deg(lon_rad)
        return lat_deg, lon_deg


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
