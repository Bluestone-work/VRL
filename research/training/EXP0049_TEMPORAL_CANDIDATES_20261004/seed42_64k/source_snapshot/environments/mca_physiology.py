"""Explicit-unit MCA flow calibration, isolated from historical RL environments.

This module is a reduced-order research model, not patient-specific CFD.
The resting inlet volume flow is calibrated from PC-MRI, not from a Doppler
spectral velocity mislabelled as a cross-sectional mean. No controller or
robot speed is increased here to make the navigation task solvable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


MCA_FLOW_ML_MIN = 146.0
MCA_FLOW_SD_ML_MIN = 31.0
MCA_FLOW_DOI = '10.1038/jcbfm.2014.241'


def _finite_scalar(value, name, *, positive=True):
    """Reject missing/nonfinite inputs before they contaminate a rollout."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a finite scalar') from None
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        bound = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {bound}')
    return value


def mean_speed_mm_s(flow_ml_min, diameter_mm):
    """Cross-sectional mean = volume flow / area; mL/min -> mm³/s."""
    flow = _finite_scalar(flow_ml_min, 'flow_ml_min', positive=False)
    diameter = _finite_scalar(diameter_mm, 'diameter_mm')
    return flow*1000/60/(np.pi*(diameter/2)**2)


@dataclass(frozen=True)
class PhysicalUnits:
    mm_per_unit: float
    control_dt_s: float
    robot_speed_mm_s: float
    provenance: str

    def __post_init__(self):
        for name in ('mm_per_unit', 'control_dt_s', 'robot_speed_mm_s'):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('Measured or literature-proxy provenance required')

    def displacement_per_step(self, speed_mm_s):
        return np.asarray(speed_mm_s)*self.control_dt_s/self.mm_per_unit

    def speed_from_step_displacement(self, displacement):
        return np.asarray(displacement)*self.mm_per_unit/self.control_dt_s

    @property
    def robot_action_scale(self):
        return float(self.displacement_per_step(self.robot_speed_mm_s))

    def required_substeps(self, maximum_speed_mm_s, minimum_radius_mm,
                          fraction_of_radius=.1):
        """Numerical resolution requirement, NOT a cap on the real flow speed."""
        maximum_speed_mm_s = _finite_scalar(maximum_speed_mm_s, 'maximum_speed_mm_s', positive=False)
        minimum_radius_mm = _finite_scalar(minimum_radius_mm, 'minimum_radius_mm')
        fraction_of_radius = _finite_scalar(fraction_of_radius, 'fraction_of_radius')
        if fraction_of_radius > 1:
            raise ValueError('Invalid spatial resolution')
        return max(1, int(np.ceil(maximum_speed_mm_s*self.control_dt_s/
                                 (minimum_radius_mm*fraction_of_radius))))


class PressureDrivenTreeFlow:
    """Quasi-steady resistance tree, calibrated to an unobstructed inlet Q.

    Edge resistance is proportional to integral(ds/r**4), approximated with
    endpoint trapezoids. A common 8*viscosity/pi factor cancels when boundary
    pressure is calibrated to the specified healthy inlet volume flow.
    Inlet driving pressure is then HELD FIXED as occlusion changes resistance.
    Thus obstruction can reduce total Q, rather than demanding the original
    Q through an almost closed vessel and hiding that with a velocity cap.

    Assumptions: rigid walls, circular laminar profile, all terminal reference
    pressures equal, no autoregulation/pulsatility/stenotic jet losses. Optional
    terminal resistors are a sensitivity assumption, not a measured vascular
    bed. Calibration to healthy Q alone does not identify patient-specific
    occluded flow. Use for sensitivity and unit/conservation tests until
    measured outlet conditions are available.
    """

    def __init__(self, tree, mm_per_unit, inlet_flow_ml_min=MCA_FLOW_ML_MIN,
                 distal_resistance_ratio=0.0):
        self.tree = tree
        self.mm_per_unit = _finite_scalar(mm_per_unit, 'mm_per_unit')
        self.inlet_flow_ml_min = _finite_scalar(inlet_flow_ml_min, 'inlet_flow_ml_min')
        self.distal_resistance_ratio = _finite_scalar(
            distal_resistance_ratio, 'distal_resistance_ratio', positive=False)
        self.root = int(tree.inlet_station)
        n = tree.n_stations
        if n < 2 or not 0 <= self.root < n or len(tree.station_graph) != n:
            raise ValueError('A connected tree with at least two stations is required')
        self.parent = np.full(n, -1, np.int32)
        self.children = [[] for _ in range(n)]
        self.length = np.zeros(n, np.float64)
        self.order = [self.root]
        self.parent[self.root] = self.root
        for node in self.order:
            for neighbour, length in tree.station_graph[node]:
                if not isinstance(neighbour, (int, np.integer)) or not 0 <= neighbour < n:
                    raise ValueError('Invalid graph station index')
                length = _finite_scalar(length, 'graph edge length', positive=False)
                if neighbour == self.parent[node]:
                    continue
                if self.parent[neighbour] != -1:
                    raise ValueError('Resistance tree does not support anastomotic cycles')
                self.parent[neighbour] = node
                self.children[node].append(neighbour)
                self.length[neighbour] = length*self.mm_per_unit
                self.order.append(neighbour)
        if len(self.order) != n:
            raise ValueError('Disconnected geometry')
        self.healthy_radius_mm = np.asarray(tree.radii, np.float64)*self.mm_per_unit
        if self.healthy_radius_mm.shape != (n,) or not np.isfinite(self.healthy_radius_mm).all() or np.any(self.healthy_radius_mm <= 0):
            raise ValueError('Healthy geometry must have one positive finite radius per station')
        self.terminal_resistance = np.zeros(n, np.float64)
        _, resistance = self._resistance(self.healthy_radius_mm)
        self.reference_resistance = resistance[self.root]
        if not np.isfinite(self.reference_resistance) or self.reference_resistance <= 0:
            raise ValueError('Degenerate zero/infinite-resistance healthy tree')
        # Pressure in normalized resistance × mm³/s units, not mmHg.
        self.driving_pressure = self.inlet_flow_ml_min*1000/60*self.reference_resistance
        # Optional *sensitivity assumption*, not a measured vascular bed.
        # Give each leaf R_i = ratio * R_tree / healthy_flow_fraction_i.
        # This adds equal healthy outlet pressure drops, preserving the
        # original healthy flow split while exposing occluded-flow ambiguity.
        if self.distal_resistance_ratio:
            healthy = self.solve()['station_inflow_mm3_s']
            leaves = [node for node in self.order if not self.children[node]]
            fractions = healthy[leaves]/healthy[self.root]
            self.terminal_resistance[leaves] = (
                self.distal_resistance_ratio*self.reference_resistance/fractions)
            _, resistance = self._resistance(self.healthy_radius_mm)
            self.driving_pressure = self.inlet_flow_ml_min*1000/60*resistance[self.root]

    def _resistance(self, radii_mm):
        radius = np.asarray(radii_mm, np.float64)
        if radius.shape != self.healthy_radius_mm.shape or not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError('Provide one finite nonnegative physical radius per station')
        edge = np.zeros(len(radius), np.float64)
        equivalent = self.terminal_resistance.copy()
        for node in self.order[1:]:
            a, b = radius[self.parent[node]], radius[node]
            # Coincident branch-junction nodes have no hydraulic length.
            edge[node] = (0.0 if self.length[node] == 0 else np.inf if min(a,b) == 0 else
                          self.length[node]*.5*(a**-4+b**-4))
        for node in reversed(self.order):
            children = self.children[node]
            if children:
                path = edge[children]+equivalent[children]
                if np.any(path == 0):
                    raise ValueError('Zero resistance from an internal node to outlet')
                conductance = np.sum(1/path)
                equivalent[node] = 1/conductance if conductance > 0 else np.inf
        return edge, equivalent

    def solve(self, radius_mm=None):
        radius = self.healthy_radius_mm if radius_mm is None else np.asarray(radius_mm, np.float64)
        edge, equivalent = self._resistance(radius)
        flow = np.zeros(len(radius), np.float64)  # entering each node, mm³/s
        flow[self.root] = self.driving_pressure/equivalent[self.root]
        for node in self.order:
            children = self.children[node]
            if not children:
                continue
            conductance = 1/(edge[children]+equivalent[children])
            total = conductance.sum()
            if total:
                flow[children] = flow[node]*conductance/total
        speed = np.divide(flow, np.pi*radius**2, out=np.zeros_like(flow), where=radius>0)
        residual = [abs(flow[node]-flow[self.children[node]].sum())
                    for node in self.order if self.children[node]]
        return dict(station_inflow_mm3_s=flow, mean_speed_mm_s=speed,
                    inlet_flow_ml_min=float(flow[self.root]*60/1000),
                    radius_mm=radius.copy(),
                    equivalent_inlet_resistance=float(equivalent[self.root]),
                    maximum_conservation_residual_mm3_s=float(max(residual,default=0)))

    def velocity_mm_s(self, positions, station, solution, radius_mm=None):
        """Local quasi-steady parabola, with no artificial navigation speed cap."""
        positions = np.asarray(positions, dtype=np.float64)
        station = np.asarray(station)
        if (positions.ndim != 2 or positions.shape[1] != 3 or
                station.shape != (len(positions),) or
                not np.issubdtype(station.dtype, np.integer) or
                not np.isfinite(positions).all() or
                np.any((station < 0) | (station >= len(self.order)))):
            raise ValueError('Expected finite [N,3] positions and valid integer [N] stations')
        radii = np.asarray(solution['radius_mm'], dtype=np.float64)
        if radius_mm is not None and not np.array_equal(np.asarray(radius_mm), radii):
            raise ValueError('Velocity profile radius must match the solved flow field')
        axis, _ = self.tree._axis_point(positions, station)
        radial_mm = np.linalg.norm(positions-axis,axis=-1)*self.mm_per_unit
        radius = radii[station]
        normalized = np.divide(radial_mm,radius,out=np.ones_like(radial_mm),where=radius>0)
        profile = 2*np.clip(1-normalized**2,0,1)
        return self.tree.tangents[station]*(solution['mean_speed_mm_s'][station]*profile)[:,None]


def optimistic_wall_upstream_margin(mean_speed, lumen_radius_mm, robot_radius_mm, robot_speed_mm_s):
    """Best accessible no-slip profile point for a finite-radius robot.

    Ignores lubrication loss, so a negative value already excludes upstream
    motion in this straight-lumen model; not a proof for arbitrary 3D flows.
    """
    mean_speed = _finite_scalar(mean_speed, 'mean_speed', positive=False)
    lumen_radius_mm = _finite_scalar(lumen_radius_mm, 'lumen_radius_mm')
    robot_radius_mm = _finite_scalar(robot_radius_mm, 'robot_radius_mm')
    robot_speed_mm_s = _finite_scalar(robot_speed_mm_s, 'robot_speed_mm_s')
    if robot_radius_mm >= lumen_radius_mm:
        raise ValueError('Robot must fit inside the lumen')
    wall_flow = 2*mean_speed*(1-(1-robot_radius_mm/lumen_radius_mm)**2)
    return float(robot_speed_mm_s-wall_flow)
