"""Gym interface for explicit-unit MCA engineering dynamics (EXP22B).

New observation schema: 36 features per robot; never load old-policy weights
as though their observation/physics semantics were unchanged. The actions
passed to step are world-frame unit commands, held for control_dt_s.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.mca_physical_dynamics import PhysicalTubeTransport
from environments.mca_physiology import PhysicalUnits, PressureDrivenTreeFlow, _finite_scalar
from environments.vessel_anatomy import build_territory
from environments.mca_obstacle_forecast import (forecast_particles, trajectory_features,
    linear_particle_predictions, bound_trajectory_features)


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/experiments/EXP_0022B_MCA_DYNAMICS.json'

# Appended routed-target block: per-clot slot features plus a per-robot summary.
TARGET_SLOT_DIMS = 8
TARGET_SUMMARY_DIMS = 4
# Route lookahead mirrors the diagnostic path follower; a robot standing on its
# routing node steps to the next hop instead of reporting a degenerate bearing.
ROUTE_LOOKAHEAD_MM = .06
# Inside this geodesic range the route collapses onto the target point itself.
ROUTE_DIRECT_MM = .7


def bound_command(action, mode='bounded'):
    """Clip to the unit ball; 'unit' then runs every nonzero direction at full speed.

    'unit' models a constant-speed actuator: the policy still chooses the
    direction (and can stop with an exactly zero command), but not a slower
    speed. No route, target or flow information is read here.
    """
    action = np.clip(action, -1, 1)
    norm = np.linalg.norm(action, axis=1, keepdims=True)
    if mode == 'unit':
        return np.where(norm > 1e-12, action/np.maximum(norm, 1e-12), 0.)
    return action/np.maximum(norm, 1)


@dataclass(frozen=True)
class DynamicsConfig:
    control_dt_s: float = .05
    episode_duration_s: float = 1.
    robot_speed_mm_s: float = 1.
    robot_radius_mm: float = .08
    num_robots: int = 5
    inlet_flow_ml_min: float = 146.
    distal_resistance_ratio: float = 1.
    geometry_variation: float = 0.
    initial_radius_fraction: float = .35
    clot_width_mm: float = 2.5
    contact_distance_mm: float = .12
    lysis_mass_per_s: float = .36
    lysis_saturation: float = 4.
    particle_count: int = 8
    particle_radius_mm: float = .02
    spatial_fraction: float = .1
    max_substeps_per_control: int = 100000
    lubrication_floor: float = 1.
    robot_initialization: str = 'upstream_fixed'
    contact_model: str = 'point_target'
    progress_reward_scale: float = 0.
    reward_discount: float = .99
    particle_contact_penalty_per_s: float = 0.
    particle_initialization: str = 'length_uniform'
    inlet_flow_multiplier_min: float = 1.
    inlet_flow_multiplier_max: float = 1.
    clot_initialization: str = 'historical_sites'
    obstacle_observation: str = 'nearest'
    target_observation: str = 'nearest_euclidean'
    progress_potential: str = 'nearest'
    command_speed: str = 'bounded'
    particle_collision_event_penalty: float = 0.
    particle_near_penalty_per_s: float = 0.
    particle_safety_margin_mm: float = .15
    particle_prediction_horizon_s: float = 1.
    assumption_provenance: str = 'unmeasured engineering sensitivity, not calibrated physiology'

    def __post_init__(self):
        nonnegative = {'distal_resistance_ratio', 'geometry_variation', 'lysis_mass_per_s',
                       'initial_radius_fraction', 'progress_reward_scale', 'particle_contact_penalty_per_s',
                       'particle_collision_event_penalty', 'particle_near_penalty_per_s'}
        integers = {'num_robots': 1, 'particle_count': 0, 'max_substeps_per_control': 1}
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name == 'robot_initialization':
                if value not in ('upstream_fixed', 'distributed_branches'):
                    raise ValueError('Invalid robot_initialization')
            elif name == 'contact_model':
                if value not in ('point_target', 'stenosis_surface', 'localized_point'):
                    raise ValueError('Invalid contact_model')
            elif name == 'particle_initialization':
                if value not in ('length_uniform', 'mixed_branch_density'):
                    raise ValueError('Invalid particle_initialization')
            elif name == 'clot_initialization':
                if value not in ('historical_sites','random_branches'):
                    raise ValueError('Invalid clot_initialization')
            elif name == 'obstacle_observation':
                if value not in ('nearest','predictive_four','trajectory_four','trajectory_four_masked',
                                 'bounded_trajectory_four','bounded_linear_four','anchored_linear_four'):
                    raise ValueError('Invalid obstacle_observation')
            elif name == 'target_observation':
                if value not in ('nearest_euclidean', 'routed_assigned', 'routed_assigned_masked'):
                    raise ValueError('Invalid target_observation')
            elif name == 'progress_potential':
                if value not in ('nearest', 'mass_weighted'):
                    raise ValueError('Invalid progress_potential')
            elif name == 'command_speed':
                if value not in ('bounded', 'unit'):
                    raise ValueError('Invalid command_speed')
            elif name == 'assumption_provenance':
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('Explicit assumption provenance required')
            elif name in integers:
                if not isinstance(value, int) or isinstance(value, bool) or value < integers[name]:
                    raise ValueError(f'Invalid {name}')
            else:
                object.__setattr__(self, name, _finite_scalar(value, name, positive=name not in nonnegative))
        if not 0 <= self.initial_radius_fraction <= 1 or self.spatial_fraction > .25 or self.lubrication_floor > 1:
            raise ValueError('Invalid radius fraction or integration parameters')
        if self.reward_discount > 1:
            raise ValueError('Invalid reward_discount')
        if self.inlet_flow_multiplier_min > self.inlet_flow_multiplier_max:
            raise ValueError('Invalid inlet flow multiplier range')

    @classmethod
    def from_json(cls, path=CONFIG_PATH):
        data = json.loads(Path(path).read_text())
        return cls(**{f.name: data.get(f.name, f.default) if f.name in (
                      'robot_initialization', 'contact_model', 'progress_reward_scale', 'reward_discount',
                      'particle_contact_penalty_per_s', 'particle_initialization',
                      'inlet_flow_multiplier_min', 'inlet_flow_multiplier_max', 'clot_initialization',
                      'obstacle_observation', 'target_observation', 'progress_potential', 'command_speed',
                      'particle_collision_event_penalty',
                      'particle_near_penalty_per_s', 'particle_safety_margin_mm', 'particle_prediction_horizon_s')
                      else data[f.name] for f in fields(cls)})


class MCAPhysicalEnv(gym.Env):
    """Physical transport + synthetic contact kinetics; no hardware claim.

    Lost robots are masked permanently for the episode. Termination means all
    clots cleared OR all robots lost. The time horizon is a truncation. Particle
    contact can incur a configured avoidance cost. Particles are advected
    spheres with overlap diagnostics/cost, not a calibrated rigid RBC model.
    """

    metadata = {'render_modes': []}
    observation_schema = 'mca_physical_36_v1'

    def __init__(self, config=None, *, tree=None, clot_stations=None):
        super().__init__()
        self.config = config or DynamicsConfig.from_json()
        if self.config.contact_model == 'stenosis_surface':
            self.observation_schema = 'mca_surface_36_v2'
        elif self.config.contact_model == 'localized_point':
            self.observation_schema = 'mca_point_36_v3'
        if self.config.obstacle_observation != 'nearest':
            if self.config.contact_model != 'localized_point':
                raise ValueError('Predictive obstacle observations require localized point targets')
            self.observation_schema = 'mca_point_obstacles_76_v4'
        self.obs_dim = 76 if self.config.obstacle_observation != 'nearest' else 36
        if self.config.obstacle_observation not in ('nearest','predictive_four'):
            self.observation_schema = 'mca_point_trajectories_172_v5'
            self.obs_dim = 172
        if self.config.obstacle_observation in ('bounded_trajectory_four','bounded_linear_four'):
            self.observation_schema = 'mca_point_bounded_172_v6'
        elif self.config.obstacle_observation == 'anchored_linear_four':
            self.observation_schema = 'mca_point_anchored_172_v7'
        self.num_robots = self.config.num_robots
        self._fixed_tree = tree
        self._fixed_clot_stations = None if clot_stations is None else np.asarray(clot_stations, np.int32)
        self.num_clots = 4 if clot_stations is None else len(clot_stations)
        # Routed/assigned target features are appended; the whole prefix above
        # stays byte-identical so a v4/v5/v6/v7 parent expands into this schema.
        self.target_block = 0
        if self.config.target_observation != 'nearest_euclidean':
            if self.config.contact_model != 'localized_point':
                raise ValueError('Routed target observations require localized point targets')
            self.target_block = TARGET_SLOT_DIMS*self.num_clots + TARGET_SUMMARY_DIMS
            self.observation_schema = f'mca_point_routed_{self.obs_dim+self.target_block}_v8'
            self.obs_dim += self.target_block
        self.action_space = spaces.Box(-1., 1., (self.num_robots, 3), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'nodes': spaces.Box(-np.inf, np.inf, (self.num_robots, self.obs_dim), dtype=np.float32),
            'adjacency': spaces.Box(0., 1., (self.num_robots, self.num_robots), dtype=np.float32),
            'clot_state': spaces.Box(-np.inf, np.inf, (self.num_clots, 4), dtype=np.float32),
            'agent_mask': spaces.MultiBinary(self.num_robots),
        })
        self._done = True
        self._reset_called = False

    def _radii(self, masses):
        fraction = masses / self.initial_mass
        block = (self._occlusion_bump * fraction).max(axis=1) if len(masses) else 0.
        return self.flow_model.healthy_radius_mm * (1 - (1 - self.config.initial_radius_fraction) * block)

    def _sync_public_state(self):
        n = self.num_robots
        self.robot_positions = (self.positions_mm[:n] / self.units.mm_per_unit).astype(np.float32)
        ends = self.transport.ends[self.edges[:n]]
        points = self.transport.points[ends]
        distance = np.linalg.norm(points - self.positions_mm[:n, None], axis=-1)
        self.robot_stations = ends[np.arange(n), np.argmin(distance, axis=1)]
        self.robot_velocities = (self.velocity_mm_s * self.config.control_dt_s /
                                 self.units.mm_per_unit).astype(np.float32)
        self.agent_mask = self.active[:n].copy()
        self.clot_masses = self.masses.copy()

    def _sample_particles(self):
        weights = self.transport.length / self.transport.length.sum()
        self.particle_layout = 'length_uniform'
        if self.config.particle_initialization == 'mixed_branch_density':
            self.particle_layout = str(self.np_random.choice(
                ['length_uniform', 'branch_balanced', 'branch_clustered']))
            branches, inverse = np.unique(self.tree.branch_ids[self.transport.ends[:, 0]], return_inverse=True)
            branch_length = np.bincount(inverse, weights=self.transport.length)
            if self.particle_layout == 'branch_balanced':
                branch_weights = np.full(len(branches), 1 / len(branches))
            elif self.particle_layout == 'branch_clustered':
                branch_weights = .8*self.np_random.dirichlet(np.full(len(branches), .35)) + .2*branch_length/branch_length.sum()
            else:
                branch_weights = branch_length/branch_length.sum()
            weights = branch_weights[inverse]*self.transport.length/branch_length[inverse]
        points = []
        for _ in range(self.config.particle_count):
            for attempt in range(10000):
                e = int(self.np_random.choice(len(self.transport.ends),
                        p=weights))
                t = self.np_random.uniform()
                lumen = float(self.solution['radius_mm'][self.transport.ends[e]] @ np.array([1-t, t]))
                room = lumen - self.config.particle_radius_mm
                if room <= 0:
                    continue
                direction = self.transport.direction[e]
                helper = np.eye(3)[np.argmin(np.abs(direction))]
                normal = np.cross(direction, helper)
                normal /= np.linalg.norm(normal)
                binormal = np.cross(direction, normal)
                theta = self.np_random.uniform(0, 2*np.pi)
                offset = room * np.sqrt(self.np_random.uniform()) * .9
                point = self.transport.a[e] + t * self.transport.ab[e] + offset * (
                    np.cos(theta)*normal + np.sin(theta)*binormal)
                if self.config.particle_initialization == 'mixed_branch_density':
                    nearest = self.transport.nearest_edges(point[None])
                    _, actual_lumen, distance, _ = self.transport.coordinates(point[None], nearest, self.solution)
                    if nearest[0] != e or distance[0]+self.config.particle_radius_mm > actual_lumen[0]-1e-7:
                        continue
                    if points and np.linalg.norm(np.asarray(points)-point, axis=1).min() < 2*self.config.particle_radius_mm:
                        continue
                    if (self.config.clot_initialization=='random_branches' and
                            np.linalg.norm(self._reset_robot_positions_mm-point,axis=1).min() <=
                            self.config.robot_radius_mm+self.config.particle_radius_mm+1e-7):
                        continue
                points.append(point)
                break
            else:
                raise ValueError('Cannot place finite-size passive tracers in this lumen')
        return np.asarray(points, np.float64).reshape(-1, 3)

    def _sample_episode_flow(self):
        """Randomize driving pressure once per episode; all bodies share this flow."""
        lo, hi = self.config.inlet_flow_multiplier_min, self.config.inlet_flow_multiplier_max
        self.episode_flow_multiplier = float(self.np_random.uniform(lo, hi)) if lo != hi else lo
        self.flow_model.driving_pressure = self._base_driving_pressure*self.episode_flow_multiplier
        self.flow_model.inlet_flow_ml_min = self.config.inlet_flow_ml_min*self.episode_flow_multiplier
        # Compiled hydraulics contain a scalar pressure; do not reuse last episode's value.
        self._args_transport = None

    def initialization_record(self):
        """Small enough to log every reset, including the complete obstacle layout."""
        particle_branches = self.tree.branch_ids[self.transport.ends[self.edges[self.num_robots:], 0]]
        return dict(robot_initialization=self.config.robot_initialization,
                    clot_initialization=self.config.clot_initialization,
                    clot_stations=self.clot_stations.tolist(),clot_positions_mm=self.clot_positions_mm.tolist(),
                    clot_branch_ids=self.tree.branch_ids[self.clot_stations].tolist(),
                    robot_positions_mm=self.positions_mm[:self.num_robots].tolist(),
                    robot_branch_ids=self.tree.branch_ids[self.robot_stations].tolist(),
                    particle_initialization=self.config.particle_initialization,
                    particle_layout=self.particle_layout,
                    particle_positions_mm=self.positions_mm[self.num_robots:].tolist(),
                    particle_branch_ids=particle_branches.tolist(),
                    particle_branch_counts=np.bincount(particle_branches, minlength=len(self.tree.branches)).tolist(),
                    flow_multiplier=self.episode_flow_multiplier,
                    healthy_reference_flow_ml_min=self.flow_model.inlet_flow_ml_min)

    def _reset_clots(self):
        if self._fixed_clot_stations is not None:
            stations = self._fixed_clot_stations.copy()
        elif self.config.clot_initialization == 'random_branches':
            # Point targets stay in navigable branch interiors, independent of body starts.
            candidates = {}
            scale = self.units.mm_per_unit
            for bid,branch in enumerate(self.tree.branches):
                ids = np.arange(branch.start+1,branch.stop)
                clearance = (self.tree.arclength[ids]-self.tree.arclength[branch.start])*scale
                tail = (self.tree.arclength[branch.stop]-self.tree.arclength[ids])*scale
                ids = ids[(clearance > 1.) & (tail > 1.) &
                          (self.flow_model.healthy_radius_mm[ids]*self.config.initial_radius_fraction >
                           self.config.robot_radius_mm+1e-7)]
                if len(ids):candidates[bid] = ids
            if len(candidates)<self.num_clots:
                raise ValueError('Not enough navigable branches for distinct random clot targets')
            stations = None
            for _ in range(1000):
                bids = self.np_random.choice(list(candidates),size=self.num_clots,replace=False)
                selected = np.array([self.np_random.choice(candidates[int(bid)]) for bid in bids],np.int32)
                points = self.transport.points[selected]
                separation = np.linalg.norm(points[:,None]-points[None,:],axis=-1)
                separation[np.diag_indices(self.num_clots)] = np.inf
                if separation.min()>max(2*self.config.contact_distance_mm,1.):
                    stations=selected;break
            if stations is None:raise ValueError('Cannot sample separated clot targets')
        else:
            stations = []
            for site in self.tree.territory.clot_sites:
                branch = self.tree.branches[self.tree.segment_index[site.segment]]
                stations.append(branch.start+int(round(site.position*(branch.size-1))))
            stations = np.asarray(stations,np.int32)
        self.clot_stations = np.asarray(stations,np.int32)
        if (self.clot_stations.shape != (self.num_clots,) or
                np.any((self.clot_stations < 0) | (self.clot_stations >= self.tree.n_stations))):
            raise ValueError('Invalid clot stations')
        self.clot_positions_mm = self.transport.points[self.clot_stations]
        self.initial_mass = np.ones(self.num_clots,np.float64)
        self.masses = self.initial_mass.copy()
        arc = (self.tree.arclength[:,None]-self.tree.arclength[self.clot_stations])*self.units.mm_per_unit
        same = self.tree.branch_ids[:,None] == self.tree.branch_ids[self.clot_stations]
        self._occlusion_bump = np.exp(-.5*(arc/self.config.clot_width_mm)**2)*same
        routing = [self.tree.route_to(int(station)) for station in self.clot_stations]
        self.routes = [distance*self.units.mm_per_unit for distance, _ in routing]
        self._route_next_hop = [hop for _, hop in routing]

    def _reset_targets(self):
        """Clear the sticky allocation; a new episode never inherits targets."""
        self._assignment = None
        self._assignment_alive = None

    def _reset_avoidance(self):
        self._particle_overlaps = np.zeros((self.num_robots,self.config.particle_count),bool)
        self.episode_particle_contact_s = 0.
        self.episode_particle_collision_events = 0

    def _avoidance_outcome(self,contact_s,near_s,overlaps,events):
        self._particle_overlaps = overlaps
        self.episode_particle_contact_s += float(contact_s.sum())
        self.episode_particle_collision_events += int(events.sum())
        return (self.config.particle_contact_penalty_per_s*contact_s +
                self.config.particle_collision_event_penalty*events + self.config.particle_near_penalty_per_s*near_s)

    def _initial_robot_positions(self, options):
        if 'robot_positions_mm' in options:
            return np.asarray(options['robot_positions_mm'], np.float64)
        if self.config.robot_initialization == 'distributed_branches':
            return self._sample_robots()
        # Historical EXP22B/23 initialization, retained for reproducible replay.
        trunk = self.tree.branches[0]
        ids = np.linspace(trunk.start, trunk.start + max(1, (trunk.size-1)//5), self.num_robots)
        positions = np.array([self.transport.points[int(np.floor(i))] * (1-i%1) +
            self.transport.points[min(int(np.floor(i))+1, trunk.stop)]*(i%1) for i in ids])
        root_edge = int(np.flatnonzero(self.transport.edge_groups[:, 0] == self.transport.root_group)[0])
        positions[0] += self.transport.direction[root_edge] * (2*self.config.robot_radius_mm)
        return positions

    def _sample_robots(self):
        """Seeded, branch-stratified finite-radius starts, independent of targets.

        Branches are sampled without replacement per cycle; within each branch
        edges are length-weighted. Reject ambiguous projections, overlap, and
        centres within two robot radii of a branch endpoint. No flow, rewards,
        target locations, or physical body sizes are changed.
        """
        t = self.transport
        body = self.config.robot_radius_mm
        edge_branch = self.tree.branch_ids[t.ends[:, 0]]
        same_branch = edge_branch == self.tree.branch_ids[t.ends[:, 1]]
        candidates = {}
        for bid in np.unique(edge_branch):
            ids = np.flatnonzero(same_branch & (edge_branch == bid) &
                (self.solution['radius_mm'][t.ends].min(axis=1) > body + 1e-7))
            if ids.size:
                candidates[int(bid)] = ids
        if not candidates:
            raise ValueError('No navigable branch for robot initialization')
        positions = []
        while len(positions) < self.num_robots:
            placed = 0
            for bid in self.np_random.permutation(list(candidates)):
                ids = candidates[int(bid)]
                branch_points = t.points[np.flatnonzero(self.tree.branch_ids == bid)]
                weights = t.length[ids] / t.length[ids].sum()
                for _ in range(1000):
                    edge = int(self.np_random.choice(ids, p=weights))
                    fraction = self.np_random.uniform(.05, .95)
                    radius = float(self.solution['radius_mm'][t.ends[edge]] @ np.array([1-fraction, fraction]))
                    axis = t.direction[edge]
                    normal = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
                    normal /= np.linalg.norm(normal)
                    binormal = np.cross(axis, normal)
                    angle = self.np_random.uniform(0, 2*np.pi)
                    offset = .65 * (radius-body) * np.sqrt(self.np_random.uniform())
                    point = t.a[edge] + fraction*t.ab[edge] + offset*(
                        np.cos(angle)*normal + np.sin(angle)*binormal)
                    nearest = t.nearest_edges(point[None])
                    _, lumen, distance, _ = t.coordinates(point[None], nearest, self.solution)
                    if edge_branch[nearest[0]] != bid or distance[0]+body > lumen[0]-1e-7:
                        continue
                    if np.linalg.norm(branch_points[[0, -1]]-point, axis=1).min() < 2*body:
                        continue
                    if positions and np.linalg.norm(np.asarray(positions)-point, axis=1).min() < 2*body+1e-7:
                        continue
                    positions.append(point)
                    placed += 1
                    break
                if len(positions) == self.num_robots:
                    break
            if not placed:
                raise ValueError('Cannot place non-overlapping robots in navigable branches')
        return np.asarray(positions, np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        unknown = set(options) - {'robot_positions_mm', 'particle_positions_mm'}
        if unknown:
            raise ValueError(f'Unknown reset options: {sorted(unknown)}')
        self.tree = self._fixed_tree or build_territory('mca_m1_lvo', rng=self.np_random,
                        variation=self.config.geometry_variation, min_radius=0)
        scale = getattr(self.tree, 'physical_mm_per_unit', None)
        if scale is None:
            raise ValueError('Tree must explicitly declare physical_mm_per_unit')
        self.units = PhysicalUnits(scale, self.config.control_dt_s, self.config.robot_speed_mm_s,
                                   self.config.assumption_provenance)
        self.max_speed = self.units.robot_action_scale
        self.flow_model = PressureDrivenTreeFlow(self.tree, scale, self.config.inlet_flow_ml_min,
                                                 self.config.distal_resistance_ratio)
        self._base_driving_pressure = self.flow_model.driving_pressure
        self._sample_episode_flow()
        self.transport = PhysicalTubeTransport(self.flow_model, spatial_fraction=self.config.spatial_fraction,
                        max_substeps=self.config.max_substeps_per_control,
                        lubrication_floor=self.config.lubrication_floor)
        self._reset_clots()
        self.solution = self.flow_model.solve(self._radii(self.masses))
        robot_pos = self._initial_robot_positions(options)
        self._reset_robot_positions_mm = robot_pos
        self.particle_layout = 'explicit_positions'
        particle_pos = (np.asarray(options['particle_positions_mm'], np.float64)
                        if 'particle_positions_mm' in options else self._sample_particles())
        if robot_pos.shape != (self.num_robots, 3) or particle_pos.shape != (self.config.particle_count, 3):
            raise ValueError('Incorrect reset position shapes')
        self.positions_mm = np.concatenate((robot_pos, particle_pos))
        self.edges = self.transport.nearest_edges(self.positions_mm)
        self.body_radius = np.concatenate((np.full(self.num_robots, self.config.robot_radius_mm),
                                          np.full(self.config.particle_count, self.config.particle_radius_mm)))
        if not np.isfinite(self.positions_mm).all():
            raise ValueError('Nonfinite initial positions')
        _, radius, distance, _ = self.transport.coordinates(self.positions_mm, self.edges, self.solution)
        if np.any(distance + self.body_radius > radius + 1e-7):
            raise ValueError('Initial body does not fit the obstructed lumen')
        self.active = np.ones(len(self.positions_mm), bool)
        self.path_mm = np.zeros(len(self.positions_mm))
        self.exit_time_s = np.full(len(self.positions_mm), np.nan)
        self.exit_node = np.full(len(self.positions_mm), -1, np.int32)
        self.velocity_mm_s = np.zeros((self.num_robots, 3))
        self.elapsed_s, self.steps = 0., 0
        self._done, self._reset_called = False, True
        self._reset_avoidance()
        self._reset_targets()
        self._sync_public_state()
        return self._observation(), self._info()

    def _target_distances(self):
        if not self.num_clots:
            return np.empty((self.num_robots, 0))
        ends = self.transport.ends[self.edges[:self.num_robots]]
        _, _, _, t = self.transport.coordinates(self.positions_mm[:self.num_robots],
                                                self.edges[:self.num_robots], self.solution)
        length = self.transport.length[self.edges[:self.num_robots]]
        distances = np.stack([np.minimum(route[ends[:, 0]]+t*length,
                                        route[ends[:, 1]]+(1-t)*length) for route in self.routes], axis=1)
        distances[:, self.masses <= 0] = np.inf
        return distances

    def _solve_assignment(self, live):
        """Balanced geodesic robot->target allocation; observation only.

        Target columns are replicated so every remaining target gets a robot
        before any target gets a second one, which is what a nearest-target
        argmin cannot express. This assigns and never commands: no action, no
        waypoint and no velocity is derived from it anywhere.
        """
        from scipy.optimize import linear_sum_assignment
        assignment = np.full(self.num_robots, -1, np.int64)
        rows = np.flatnonzero(self.active[:self.num_robots])
        if not len(live) or not len(rows):
            return assignment
        cost = self._target_distances()[np.ix_(rows, live)]
        cost = np.where(np.isfinite(cost), cost, 1e9)
        # Stage 1 covers every remaining target exactly once: a rectangular
        # assignment uses each column at most once, so with robots >= targets no
        # target is left for nobody. Column replication alone does NOT guarantee
        # this - Hungarian will happily double up on two cheap targets instead.
        covered, columns = linear_sum_assignment(cost)
        assignment[rows[covered]] = live[columns]
        # Stage 2 sends the surplus robots to their own nearest remaining target.
        surplus = np.setdiff1d(np.arange(len(rows)), covered, assume_unique=False)
        if len(surplus):
            assignment[rows[surplus]] = live[np.argmin(cost[surplus], axis=1)]
        return assignment

    def _assigned_targets(self):
        """Sticky allocation: recomputed only when the remaining target set changes.

        EXP_0013 measured per-step reallocation as harmful (-9.8pp); holding the
        assignment between clearances keeps the observed target consistent.
        """
        live = np.flatnonzero(self.masses > 0)
        signature = live.tobytes()
        if self._assignment is None or self._assignment_alive != signature:
            self._assignment = self._solve_assignment(live)
            self._assignment_alive = signature
        return self._assignment

    def _route_directions(self, axis):
        """Unit world-frame direction along the centreline route to each target.

        Uses the same known-map `routes`/next-hop tables that `_target_distances`
        already reads, so this adds no information the geodesic distance feature
        did not already expose - only the bearing that went with it. Cleared
        targets stay zero.
        """
        n, t = self.num_robots, self.transport
        pos, edge = self.positions_mm[:n], self.edges[:n]
        ends = t.ends[edge]
        _, _, _, fraction = t.coordinates(pos, edge, self.solution)
        length = t.length[edge]
        distance = self._target_distances()
        directions = np.zeros((n, self.num_clots, 3))
        for j in range(self.num_clots):
            if self.masses[j] <= 0:
                continue
            route, station = self.routes[j], int(self.clot_stations[j])
            node = np.where(route[ends[:, 0]]+fraction*length <=
                            route[ends[:, 1]]+(1-fraction)*length, ends[:, 0], ends[:, 1])
            goal = t.points[node].copy()
            step = (np.linalg.norm(goal-axis, axis=1) < ROUTE_LOOKAHEAD_MM) & (node != station)
            if np.any(step):
                goal[step] = t.points[self._route_next_hop[j][node[step]]]
            goal = np.where((distance[:, j] < ROUTE_DIRECT_MM)[:, None], self.clot_positions_mm[j], goal)
            delta = goal-pos
            directions[:, j] = delta/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        return directions

    def _advance_particle_prediction(self, positions, edges, body, active, duration):
        return self.transport.advance(positions, edges, body, np.zeros_like(positions),
                                      active, self.solution, duration)

    def _observation(self):
        n = self.num_robots
        nodes = np.zeros((n, self.obs_dim), np.float32)
        pos, edge = self.positions_mm[:n], self.edges[:n]
        axis, radius, radial, _ = self.transport.coordinates(pos, edge, self.solution)
        flow = self.transport.velocity_mm_s(pos, edge, self.solution)
        frame = np.stack((self.tree.tangents[self.robot_stations], self.tree.normals[self.robot_stations],
                          self.tree.binormals[self.robot_stations]), axis=1)
        to_local = lambda value: np.einsum('nij,nj->ni', frame, value)
        nodes[:, :3] = self.robot_positions*2-1
        speed_scale = self.config.robot_speed_mm_s if self.config.contact_model != 'point_target' else 1000.
        nodes[:, 3:6] = to_local(self.velocity_mm_s)/speed_scale
        nodes[:, 6:9] = self.tree.tangents[self.robot_stations]
        nodes[:, 9:12] = to_local((pos-axis)/np.maximum(radial[:, None], 1e-12))
        nodes[:, 12] = (radius-radial-self.config.robot_radius_mm)/self.config.robot_radius_mm
        nodes[:, 13] = radius/1.5
        nodes[:, 14] = self.agent_mask
        distance = self._target_distances()
        if self.num_clots and np.any(self.masses > 0):
            target = np.argmin(distance, axis=1)
            nodes[:, 15] = distance[np.arange(n), target]/self.units.mm_per_unit
            nodes[:, 16:19] = to_local(self.clot_positions_mm[target]-pos)/self.units.mm_per_unit
            if self.config.contact_model == 'stenosis_surface':
                # Observe a surface location; this never modifies the action.
                tangent = self.tree.tangents[self.clot_stations[target]]
                radial_to_target = pos-self.clot_positions_mm[target]
                radial_to_target -= (radial_to_target*tangent).sum(axis=1)[:, None]*tangent
                norm = np.linalg.norm(radial_to_target, axis=1, keepdims=True)
                normal = np.where(norm > 1e-9, radial_to_target/np.maximum(norm, 1e-9),
                                  self.tree.normals[self.clot_stations[target]])
                target_radius = self.solution['radius_mm'][self.clot_stations[target]]
                goal = self.clot_positions_mm[target] + normal*(target_radius-self.config.robot_radius_mm-
                                                               .5*self.config.contact_distance_mm)[:, None]
                delta = goal-pos
                nodes[:, 16:19] = to_local(delta)/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            elif self.config.contact_model == 'localized_point':
                delta = self.clot_positions_mm[target]-pos
                nodes[:, 16:19] = to_local(delta)/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            nodes[:, 19] = self.masses[target]/self.initial_mass[target]
        nodes[:, 20] = max(0., 1-self.elapsed_s/self.config.episode_duration_s)
        nodes[:, 21:24] = to_local(flow)/speed_scale  # no clipping away high-flow information
        particle_ids = np.flatnonzero(self.active[n:]) + n
        if len(particle_ids):
            delta = self.positions_mm[particle_ids][None] - pos[:, None]
            d = np.linalg.norm(delta, axis=-1)
            nearest = np.argmin(d, axis=1)
            selected = particle_ids[nearest]
            particle_scale = 1.5 if self.config.contact_model == 'localized_point' else self.units.mm_per_unit
            nodes[:, 24:27] = to_local(self.positions_mm[selected]-pos)/particle_scale
            pv = self.transport.velocity_mm_s(self.positions_mm[selected], self.edges[selected], self.solution)
            nodes[:, 27:30] = to_local(pv)/speed_scale
            nodes[:, 30] = (d[np.arange(n), nearest]-self.config.robot_radius_mm-self.config.particle_radius_mm)/1.5
            nodes[:, 31] = 1
        nodes[:, 32] = self.config.robot_radius_mm/np.maximum(radius, 1e-12)
        nodes[:, 33] = radius/np.maximum(
            self.flow_model.healthy_radius_mm[self.robot_stations], 1e-12)
        nodes[:, 34] = np.log1p(np.linalg.norm(flow, axis=-1)/self.config.robot_speed_mm_s)
        nodes[:, 35] = self.agent_mask.mean()
        if self.config.obstacle_observation != 'nearest' and len(particle_ids):
            velocities = self.transport.velocity_mm_s(self.positions_mm[particle_ids],self.edges[particle_ids],self.solution)
            relative_velocity = velocities[None]-self.velocity_mm_s[:,None]
            relative_position = self.positions_mm[particle_ids][None]-pos[:,None]
            speed2 = np.sum(relative_velocity**2,axis=-1)
            horizon = self.config.particle_prediction_horizon_s
            closest_time = np.clip(-np.sum(relative_position*relative_velocity,axis=-1)/np.maximum(speed2,1e-12),0,horizon)
            closest_clearance = np.linalg.norm(relative_position+closest_time[:,:,None]*relative_velocity,axis=-1)-(
                self.config.robot_radius_mm+self.config.particle_radius_mm)
            # Rank by predicted clearance, not only current distance; approaching bodies come first.
            selected = np.argsort(closest_clearance,axis=1,kind='stable')[:,:4]
            for i in range(n):
                for k,j in enumerate(selected[i]):
                    start=36+10*k
                    nodes[i,start:start+3] = frame[i]@relative_position[i,j]/1.5
                    nodes[i,start+3:start+6] = frame[i]@relative_velocity[i,j]/self.config.robot_speed_mm_s
                    nodes[i,start+6] = (d[i,j]-self.config.robot_radius_mm-self.config.particle_radius_mm)/self.config.particle_safety_margin_mm
                    nodes[i,start+7] = closest_time[i,j]/horizon
                    nodes[i,start+8] = closest_clearance[i,j]/self.config.particle_safety_margin_mm
                    nodes[i,start+9] = 1.
        mode = self.config.obstacle_observation
        if mode in ('trajectory_four','bounded_trajectory_four','bounded_linear_four','anchored_linear_four') and len(particle_ids):
            if mode in ('bounded_linear_four','anchored_linear_four'):
                predictions, valid, exits = linear_particle_predictions(self.positions_mm[particle_ids], velocities)
            else:
                predictions, valid, exits = forecast_particles(self, particle_ids)
            features = trajectory_features(pos, self.velocity_mm_s, frame,
                self.positions_mm[particle_ids], velocities, predictions, valid, exits,
                self.config.robot_radius_mm, self.config.particle_radius_mm,
                self.config.robot_speed_mm_s, self.config.particle_safety_margin_mm,
                reference_velocities=np.zeros_like(self.velocity_mm_s) if mode=='anchored_linear_four' else None)
            nodes[:, 76:76+96] = features if mode=='trajectory_four' else bound_trajectory_features(features)
        if self.target_block:
            self._write_target_block(nodes, pos, axis, to_local)
        nodes[~self.agent_mask] = 0
        adjacency = np.outer(self.agent_mask, self.agent_mask).astype(np.float32)
        state = np.column_stack((self.clot_positions_mm/self.units.mm_per_unit*2-1,
                                 self.masses/self.initial_mass)).astype(np.float32)
        return dict(nodes=nodes, adjacency=adjacency, clot_state=state,
                    agent_mask=self.agent_mask.astype(np.int8))

    def _write_target_block(self, nodes, pos, axis, to_local):
        """Every remaining target plus this robot's allocation, appended in place.

        Per slot: route bearing in the robot frame (3), geodesic distance (1),
        remaining mass fraction (1), mine (1), teammates sharing it (1), alive
        (1). Summary: remaining targets, my geodesic distance, whether my target
        is alive, remaining mass fraction of the whole task.
        """
        n = self.num_robots
        start = self.obs_dim-self.target_block
        if not self.num_clots or self.config.target_observation == 'routed_assigned_masked':
            return
        distance = self._target_distances()
        fraction = self.masses/self.initial_mass
        assignment = self._assigned_targets()
        directions = self._route_directions(axis)
        alive = self.masses > 0
        shared = np.bincount(assignment[assignment >= 0], minlength=self.num_clots)
        span = max(self.units.mm_per_unit, 1e-12)
        for j in range(self.num_clots):
            slot = start+TARGET_SLOT_DIMS*j
            if not alive[j]:
                continue
            nodes[:, slot:slot+3] = to_local(directions[:, j])
            nodes[:, slot+3] = np.minimum(distance[:, j]/span, 1e3)
            nodes[:, slot+4] = fraction[j]
            nodes[:, slot+5] = assignment == j
            nodes[:, slot+6] = shared[j]/n
            nodes[:, slot+7] = 1.
        summary = start+TARGET_SLOT_DIMS*self.num_clots
        mine = assignment.clip(0)
        valid = (assignment >= 0) & alive[mine]
        nodes[:, summary] = alive.sum()/self.num_clots
        nodes[:, summary+1] = np.where(valid, np.minimum(distance[np.arange(n), mine]/span, 1e3), 0.)
        nodes[:, summary+2] = valid
        nodes[:, summary+3] = self.masses.sum()/self.initial_mass.sum()

    def _info(self):
        success = bool(self.num_clots and np.all(self.masses <= 0))
        protocol = {'stenosis_surface': 'EXP0026_in_vitro_surface',
                    'localized_point': 'EXP0027_in_vitro_points'}.get(self.config.contact_model, 'EXP0022B_engineering')
        return dict(protocol=protocol, observation_schema=self.observation_schema,
                    success=success, elapsed_s=self.elapsed_s, remaining_mass=float(self.masses.sum()),
                    collision_free_success=success and self.episode_particle_contact_s<=1e-12 and bool(self.active[:self.num_robots].all()),
                    episode_particle_contact_s=self.episode_particle_contact_s,
                    episode_particle_collision_events=self.episode_particle_collision_events,
                    inlet_flow_ml_min=self.solution['inlet_flow_ml_min'],
                    conservation_residual_mm3_s=self.solution['maximum_conservation_residual_mm3_s'],
                    active_robots=int(self.active[:self.num_robots].sum()),
                    lost_robots=int((~self.active[:self.num_robots]).sum()),
                    active_particles=int(self.active[self.num_robots:].sum()),
                    robot_path_mm=self.path_mm[:self.num_robots].copy(),
                    robot_exit_time_s=self.exit_time_s[:self.num_robots].copy(),
                    robot_exit_node=self.exit_node[:self.num_robots].copy(),
                    formal_training_ready=False, clinical_validity='unmeasured_engineering_assumptions')

    def _reward_potential(self):
        """Dense navigation feedback; no controller, demonstrations or success change.

        F = gamma*Phi(next)-Phi(now), with zero terminal potential. Time limits
        retain potential because training bootstraps them. Units are mm, and
        the policy uses the same discount. Legacy tasks have scale zero.
        """
        if not self.config.progress_reward_scale or not np.any(self.masses > 0):
            return np.zeros(self.num_robots)
        if self.config.progress_potential == 'mass_weighted':
            return self._mass_weighted_potential()
        distance = self._target_distances().min(axis=1)
        _, radius, radial, _ = self.transport.coordinates(
            self.positions_mm[:self.num_robots], self.edges[:self.num_robots], self.solution)
        if self.config.contact_model == 'localized_point':
            # Approach the discrete target on the centreline, not the tube wall.
            remaining = np.maximum(np.hypot(distance, radial)-self.config.contact_distance_mm, 0.)
            return -self.config.progress_reward_scale*remaining*self.active[:self.num_robots]
        gap = np.maximum(radius-radial-self.config.robot_radius_mm-.5*self.config.contact_distance_mm, 0)
        return -self.config.progress_reward_scale*np.hypot(distance, gap)*self.active[:self.num_robots]

    def _mass_weighted_potential(self):
        """Mass-weighted distance over ALL remaining targets, so Phi is monotone.

        The nearest-target form drops to the next target the instant one is
        cleared, which makes finishing a target a measured -6.17 mean penalty
        for the robot that finishes it. Weighting every target by its remaining
        mass fraction removes a cleared target's term instead, so clearing can
        only raise Phi. This stays a pure function of state, so the potential
        shaping remains policy-invariant, and its magnitude is no larger than
        the nearest-target form, so the gamma<1 survival drift does not grow.
        """
        n = self.num_robots
        distance = self._target_distances()
        _, _, radial, _ = self.transport.coordinates(
            self.positions_mm[:n], self.edges[:n], self.solution)
        alive = self.masses > 0
        weight = np.where(alive, self.masses/self.initial_mass.sum(), 0.)
        remaining = np.maximum(np.hypot(np.where(alive, distance, 0.), radial[:, None])
                               - self.config.contact_distance_mm, 0.)
        total = (remaining*weight).sum(axis=1)
        return -self.config.progress_reward_scale*total*self.active[:n]

    def _surface_contacts(self, positions, edges, solution, masses):
        """Contact with the lumen-facing annular stenosis surface, not its axis.

        The wall is already the maximum of mass-weighted Gaussian stenoses.
        Assign at most ONE wall-defining clot to each robot, within its 3-sigma
        treatment extent. This shares the actual interpolated lumen geometry;
        it adds neither adhesion nor automatic motion. contact_distance_mm is
        now the reaction gap from robot surface to clot surface.
        """
        _, radius, radial, fraction = self.transport.coordinates(positions, edges, solution)
        ends = self.transport.ends[edges]
        contribution = ((1-fraction[:, None])*self._occlusion_bump[ends[:, 0]] +
                        fraction[:, None]*self._occlusion_bump[ends[:, 1]])
        contribution *= masses/self.initial_mass
        selected = np.argmax(contribution, axis=1)
        rows = np.arange(len(positions))
        contact = np.zeros((len(positions), self.num_clots), bool)
        # Support uses the unweighted bump so late residual mass can clear.
        bump = ((1-fraction)*self._occlusion_bump[ends[:, 0], selected] +
                fraction*self._occlusion_bump[ends[:, 1], selected])
        contact[rows, selected] = ((radius-radial-self.config.robot_radius_mm <= self.config.contact_distance_mm) &
                                   (bump >= np.exp(-4.5)) & (masses[selected] > 0))
        return contact

    def step(self, action):
        if self._done or not self._reset_called:
            raise RuntimeError('Call reset before stepping a new or completed episode')
        action = np.asarray(action, np.float64)
        if action.shape != (self.num_robots, 3) or not np.isfinite(action).all():
            raise ValueError('Expected finite world-frame [num_robots,3] actions')
        action = bound_command(action, self.config.command_speed)
        commands = np.zeros_like(self.positions_mm)
        commands[:self.num_robots] = action*self.config.robot_speed_mm_s
        duration = min(self.config.control_dt_s, self.config.episode_duration_s-self.elapsed_s)
        potential_before = self._reward_potential()
        # Transactional: callbacks change only local copies. A numerical budget
        # error cannot leave mass/time partially advanced in the live env.
        masses = self.masses.copy()
        solution = self.solution
        agent_removed = np.zeros(self.num_robots)
        pair_contact_s = 0.
        particle_contact_s = np.zeros(self.num_robots)
        particle_near_s = np.zeros(self.num_robots)
        particle_events = np.zeros(self.num_robots,np.int64)
        particle_overlaps = self._particle_overlaps.copy()
        contact_s = np.zeros(self.num_robots)
        n = self.num_robots

        def callback(before, after, edges, previous_active, body_dt, dt):
            nonlocal solution, pair_contact_s
            # Midpoint quadrature for contact exposure, in seconds. A robot
            # that leaves within a substep gets only its time before exit.
            midpoint = (before+after)/2
            delta = midpoint[:n, None]-midpoint[None, :n]
            overlap = np.linalg.norm(delta, axis=-1) < 2*self.config.robot_radius_mm
            alive_pairs = np.outer(previous_active[:n], previous_active[:n])
            pair_contact_s += float((np.triu(overlap & alive_pairs, 1)*
                                     np.minimum(body_dt[:n, None], body_dt[None, :n])).sum())
            if self.config.particle_count:
                pd = np.linalg.norm(midpoint[:n, None]-midpoint[None, n:], axis=-1)
                po = (pd < self.config.robot_radius_mm+self.config.particle_radius_mm)
                po &= previous_active[:n, None] & previous_active[None, n:]
                particle_contact_s[:] += (po*np.minimum(body_dt[:n, None], body_dt[None, n:])).sum(axis=1)
                particle_events[:] += (po & ~particle_overlaps).sum(axis=1)
                particle_overlaps[:] = po
                gap = pd-self.config.robot_radius_mm-self.config.particle_radius_mm
                near = np.maximum(1-np.maximum(gap,0)/self.config.particle_safety_margin_mm,0)**2
                near *= previous_active[:n,None] & previous_active[None,n:]
                particle_near_s[:] += (near*np.minimum(body_dt[:n,None],body_dt[None,n:])).sum(axis=1)
            if not self.num_clots or not np.any(masses > 0) or self.config.lysis_mass_per_s == 0:
                return None
            euclidean = np.linalg.norm(midpoint[:n, None]-self.clot_positions_mm, axis=-1)
            end = self.transport.ends[edges[:n]]
            _, _, _, t = self.transport.coordinates(midpoint[:n], edges[:n], solution)
            length = self.transport.length[edges[:n]]
            geo = np.stack([np.minimum(route[end[:, 0]]+t*length,
                                      route[end[:, 1]]+(1-t)*length) for route in self.routes], axis=1)
            if self.config.contact_model == 'stenosis_surface':
                contact = self._surface_contacts(midpoint[:n], edges[:n], solution, masses)
                contact &= previous_active[:n, None]
            else:
                contact = ((euclidean <= self.config.contact_distance_mm) &
                           (geo <= self.config.contact_distance_mm) & previous_active[:n, None] & (masses > 0))
            exposure = contact * body_dt[:n, None]
            contact_s[:] += exposure.sum(axis=1)
            total = exposure.sum(axis=0)
            weighted = exposure*np.minimum(1., self.config.lysis_saturation*dt/np.maximum(total, 1e-30))
            amount = np.minimum(masses, self.config.lysis_mass_per_s*weighted.sum(axis=0))
            if np.any(amount > 0):
                agent_removed[:] += (weighted/np.maximum(weighted.sum(axis=0), 1e-30)*amount).sum(axis=1)
                masses[:] = np.maximum(masses-amount, 0)
                solution = self.flow_model.solve(self._radii(masses))
                return solution
            return None

        result = self.transport.advance(self.positions_mm, self.edges, self.body_radius,
                        commands, self.active, self.solution, duration, after_substep=callback)
        before = self.positions_mm[:n].copy()
        self.positions_mm, self.edges, self.active = result.positions_mm, result.edge, result.active
        self.masses, self.solution = masses, solution
        self.path_mm += result.path_mm
        new_exits = result.exit_node >= 0
        self.exit_node[new_exits] = result.exit_node[new_exits]
        self.exit_time_s[new_exits] = self.elapsed_s + result.exit_time_s[new_exits]
        self.elapsed_s += duration
        self.steps += 1
        self.velocity_mm_s = (self.positions_mm[:n]-before)/duration
        self.velocity_mm_s[~self.active[:n]] = 0
        self._sync_public_state()
        particle_penalty = self._avoidance_outcome(particle_contact_s,particle_near_s,particle_overlaps,particle_events)
        info = self._info()
        terminated = info['success'] or info['active_robots'] == 0
        truncated = not terminated and self.elapsed_s >= self.config.episode_duration_s - 1e-12
        self._done = terminated or truncated
        agent_reward = 10*agent_removed - result.wall_contact_s[:n] - .01*duration
        agent_reward -= particle_penalty
        shaping = self.config.reward_discount*self._reward_potential()-potential_before
        agent_reward += shaping
        already_lost = (~self.active[:n]) & (self.exit_time_s[:n] < self.elapsed_s-duration-1e-12)
        agent_reward[already_lost] = 0
        team_reward = 30. if info['success'] else 0.
        if self.config.particle_collision_event_penalty>0 and not info['collision_free_success']:team_reward=0.
        info.update(agent_rewards=agent_reward.astype(np.float32), team_reward=team_reward,
                    step_duration_s=duration, substeps=result.substeps,
                    contact_s=contact_s, removed_mass=float(agent_removed.sum()), shaping_rewards=shaping,
                    wall_contact_s=result.wall_contact_s[:n].copy(),
                    blocked_s=result.blocked_s[:n].copy(), robot_pair_contact_s=pair_contact_s,
                    particle_contact_s=particle_contact_s, particle_penalty=particle_penalty,
                    particle_collision_events=particle_events,particle_near_s=particle_near_s,
                    termination_reason=('all_clots_cleared' if info['success'] else
                        'all_robots_exited' if terminated else 'time_limit' if truncated else None))
        return self._observation(), float(agent_reward.mean()+team_reward), bool(terminated), bool(truncated), info
