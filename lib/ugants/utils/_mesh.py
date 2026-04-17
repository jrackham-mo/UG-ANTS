# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
import numpy as np

from ugants.tests._mesh_generator import _panel_angles_to_lonlat


class Panel:
    def __init__(self, n: int, panel_id: int):
        if panel_id not in range(6):
            raise ValueError(f"panel_id must be in the range 0-5, got {panel_id}.")
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n}.")
        self.n = n
        self.panel_id = panel_id

        node_angles = np.linspace(-np.pi / 4, np.pi / 4, n + 1)
        face_angles = (node_angles[:-1] + node_angles[1:]) / 2

        node_alpha, node_beta = np.meshgrid(node_angles, node_angles)
        face_alpha, face_beta = np.meshgrid(face_angles, face_angles)

        self.node_lons, self.node_lats = _panel_angles_to_lonlat(
            panel_id, node_alpha, node_beta
        )
        self.face_lons, self.face_lats = _panel_angles_to_lonlat(
            panel_id, face_alpha, face_beta
        )

        self.face_indices = _face_indices(n, panel_id)

        self._node_indices_set = False
        self._node_indices = np.empty(((n + 1), (n + 1)), dtype=np.int64)

    @property
    def node_indices(self):
        if not self._node_indices_set:
            raise ValueError(
                "node_indices have not been initialised, use set_node_indices method."
            )
        return self._node_indices

    @node_indices.setter
    def node_indices(self, value):
        self._node_indices = value
        self._node_indices_set = True

    def set_node_indices(self, *other_panels):
        match self.panel_id:
            case 0:
                self.node_indices = gen_node_indices_panel_0(self.n)
            case 1:
                panel_0 = get_panel_by_id(other_panels, 0)
                self.node_indices = gen_node_indices_panel_1(panel_0.node_indices)
            case 2:
                panel_1 = get_panel_by_id(other_panels, 1)
                self.node_indices = gen_node_indices_panel_2(panel_1.node_indices)
            case 3:
                panel_0 = get_panel_by_id(other_panels, 0)
                panel_2 = get_panel_by_id(other_panels, 2)
                self.node_indices = gen_node_indices_panel_3(
                    panel_0.node_indices, panel_2.node_indices
                )
            case _:
                raise ValueError(f"Invalid panel_id: {self.panel_id}")

    def __repr__(self):
        return f"Panel(n={self.n}, panel_id={self.panel_id})"


def get_panel_by_id(panels: list[Panel], panel_id: int) -> Panel:
    index = [panel.panel_id for panel in panels].index(panel_id)
    panel = panels[index]
    return panel


def _face_indices(n: int, panel_id: int):
    n_squared = n * n
    offset = panel_id * n_squared
    face_indices = np.arange(n_squared).reshape(n, n) + offset
    return face_indices


def get_square_array_size(x: np.ndarray):
    shape = x.shape
    if (len(shape) != 2) or (shape[0] != shape[1]):
        raise ValueError(
            f"The provided array is not square: it has shape {shape}. "
            "Expected shape of the form (N, N)."
        )
    n = shape[0]
    return n


def gen_node_indices_panel_0(n: int):
    node_indices_panel_0 = _face_indices(n=(n + 1), panel_id=0)
    return node_indices_panel_0


def gen_node_indices_panel_1(node_indices_panel_0: np.ndarray):
    n = get_square_array_size(node_indices_panel_0)
    last_column_of_panel_0 = node_indices_panel_0[:, -1]
    first_column_panel_1 = last_column_of_panel_0.reshape(n, 1)
    start = last_column_of_panel_0[-1] + 1
    stop = start + (n * (n - 1))
    rest_of_panel_1 = np.arange(start, stop).reshape(n, (n - 1))

    node_indices_panel_1 = np.hstack([first_column_panel_1, rest_of_panel_1])
    return node_indices_panel_1


def gen_node_indices_panel_2(node_indices_panel_1: np.ndarray):
    node_indices_panel_2 = gen_node_indices_panel_1(node_indices_panel_1)
    return node_indices_panel_2


def gen_node_indices_panel_3(
    node_indices_panel_0: np.ndarray, node_indices_panel_2: np.ndarray
):
    n = get_square_array_size(node_indices_panel_0)
    m = get_square_array_size(node_indices_panel_2)
    if n != m:
        raise ValueError(
            "Panels have inconsistent sizes. "
            f"Panel 0 has size {n}, Panel 2 has size {m}."
        )
    first_column_of_panel_0 = node_indices_panel_0[:, 0]
    last_column_of_panel_3 = first_column_of_panel_0.reshape(n, 1)

    last_column_of_panel_2 = node_indices_panel_2[:, -1]
    first_column_of_panel_3 = last_column_of_panel_2.reshape(n, 1)

    start = last_column_of_panel_2[-1] + 1
    stop = start + (n * (n - 2))
    rest_of_panel_3 = np.arange(start, stop).reshape(n, (n - 2))

    node_indices_panel_3 = np.hstack(
        [first_column_of_panel_3, rest_of_panel_3, last_column_of_panel_3]
    )
    return node_indices_panel_3
