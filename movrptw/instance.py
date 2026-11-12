"""Solomon VRPTW instance data model and parser.

A Solomon instance file looks like::

    C101

    VEHICLE
    NUMBER     CAPACITY
      25         200

    CUSTOMER
    CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
        0      40         50          0          0       1236          0
        1      45         68         10        912        967         90
        ...

Customer ``0`` is always the depot. All vehicles are homogeneous with the
same capacity ``Q`` and there are at most ``K`` of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class Node:
    """A single node (depot or customer)."""

    index: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float

    @property
    def is_depot(self) -> bool:
        return self.index == 0


@dataclass
class Instance:
    """A parsed Solomon VRPTW instance.

    Attributes
    ----------
    name:
        Instance name, e.g. ``"C101"``.
    nodes:
        All nodes including the depot at index 0.
    num_vehicles:
        Maximum fleet size ``K`` declared by the instance.
    capacity:
        Vehicle capacity ``Q``.
    """

    name: str
    nodes: List[Node]
    num_vehicles: int
    capacity: float
    _dist: np.ndarray | None = field(default=None, repr=False, compare=False)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def depot(self) -> Node:
        return self.nodes[0]

    @property
    def customers(self) -> List[Node]:
        return self.nodes[1:]

    @property
    def n_customers(self) -> int:
        return len(self.nodes) - 1

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def coords(self) -> np.ndarray:
        if "coords" not in self._cache:
            self._cache["coords"] = np.array([[n.x, n.y] for n in self.nodes], dtype=float)
        return self._cache["coords"]

    @property
    def demands(self) -> np.ndarray:
        if "demands" not in self._cache:
            self._cache["demands"] = np.array([n.demand for n in self.nodes], dtype=float)
        return self._cache["demands"]

    @property
    def ready(self) -> np.ndarray:
        if "ready" not in self._cache:
            self._cache["ready"] = np.array([n.ready_time for n in self.nodes], dtype=float)
        return self._cache["ready"]

    @property
    def due(self) -> np.ndarray:
        if "due" not in self._cache:
            self._cache["due"] = np.array([n.due_time for n in self.nodes], dtype=float)
        return self._cache["due"]

    @property
    def service(self) -> np.ndarray:
        if "service" not in self._cache:
            self._cache["service"] = np.array([n.service_time for n in self.nodes], dtype=float)
        return self._cache["service"]

    @property
    def horizon(self) -> float:
        """Planning horizon = the depot's due time."""
        return self.depot.due_time

    def distance_matrix(self, *, rounding: str | None = None) -> np.ndarray:
        """Return the full Euclidean distance matrix.

        Parameters
        ----------
        rounding:
            ``None`` for full float precision (default), ``"floor1"`` to
            truncate to one decimal (a common Solomon convention used by some
            BKS tables), or ``"round"`` for nearest integer.
        """
        if self._dist is None:
            c = self.coords
            diff = c[:, None, :] - c[None, :, :]
            d = np.sqrt((diff ** 2).sum(axis=-1))
            if rounding == "floor1":
                d = np.floor(d * 10) / 10
            elif rounding == "round":
                d = np.round(d)
            self._dist = d
        return self._dist

def parse_solomon(path: str | Path) -> Instance:
    """Parse a Solomon ``.txt`` instance file into an :class:`Instance`."""
    path = Path(path)
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    return _parse_lines(lines, default_name=path.stem)


def parse_solomon_str(text: str, default_name: str = "instance") -> Instance:
    """Parse a Solomon instance from raw text (useful for tests)."""
    return _parse_lines(text.splitlines(), default_name=default_name)


def _parse_lines(raw_lines: Sequence[str], default_name: str) -> Instance:
    lines = [ln for ln in raw_lines]
    # The first non-empty line is the instance name.
    name = default_name
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines):
        name = lines[idx].strip()

    # Locate the VEHICLE section; the next numeric line is "K  Q".
    num_vehicles, capacity = _read_vehicle(lines)

    # Locate the CUSTOMER section and read every numeric row after the header.
    nodes = _read_customers(lines)
    if not nodes:
        raise ValueError(f"No customer rows found in instance '{name}'.")

    return Instance(name=name, nodes=nodes, num_vehicles=num_vehicles, capacity=capacity)


def _read_vehicle(lines: Sequence[str]) -> tuple[int, float]:
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("VEHICLE"):
            # Scan forward for the first line with two numbers.
            for j in range(i + 1, min(i + 6, len(lines))):
                toks = lines[j].split()
                if len(toks) >= 2 and _is_number(toks[0]) and _is_number(toks[1]):
                    return int(float(toks[0])), float(toks[1])
    raise ValueError("Could not locate the VEHICLE / capacity section.")


def _read_customers(lines: Sequence[str]) -> List[Node]:
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("CUST"):
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not locate the CUSTOMER section.")

    nodes: List[Node] = []
    for ln in lines[start:]:
        toks = ln.split()
        if len(toks) < 7:
            continue
        if not all(_is_number(t) for t in toks[:7]):
            continue
        nodes.append(
            Node(
                index=int(float(toks[0])),
                x=float(toks[1]),
                y=float(toks[2]),
                demand=float(toks[3]),
                ready_time=float(toks[4]),
                due_time=float(toks[5]),
                service_time=float(toks[6]),
            )
        )
    # Sort by declared index so nodes[0] is the depot.
    nodes.sort(key=lambda n: n.index)
    return nodes


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False
