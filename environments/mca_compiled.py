"""Compiled EXP22B integration, preserving mm/s, substep limits and events.

Numba fastmath is disabled. The reference MCAPhysicalEnv stays intact; this
backend is accepted only after paired trajectory/reward/exit checks.
"""
import numpy as np
import copy
import gymnasium as gym
from numba import njit
from environments.mca_physical_env import MCAPhysicalEnv, bound_command
from environments.mca_physical_dynamics import TransportResult


@njit(cache=True, inline='always')
def normdiff(p,q):
    x=p[0]-q[0];y=p[1]-q[1];z=p[2]-q[2]
    return np.sqrt((x*x+y*y)+z*z)


@njit(cache=True, inline='always')
def norm3(p):
    return np.sqrt((p[0]*p[0]+p[1]*p[1])+p[2]*p[2])


@njit(cache=True, inline='always')
def coord(p,e,geom,radii):
    points,ends,a,ab,length,direction,groups,candidates,valid,boundaries=geom
    t=min(1.,max(0.,np.sum((p-a[e])*ab[e])/(length[e]**2)))
    axis=a[e]+t*ab[e]
    radius=radii[ends[e,0]]*(1-t)+radii[ends[e,1]]*t
    return axis,radius,np.sqrt(np.sum((p-axis)**2)),t


@njit(cache=True, inline='always')
def velocity(p,e,body,command,geom,radii,flux,lubrication):
    _,r,radial,_=coord(p,e,geom,radii)
    f=lubrication+(1-lubrication)*min(1.,max(0.,(r-radial-body)/(4*body)))
    speed=0.
    if r>0: speed=(2*(flux[geom[1][e,1]]/(np.pi*r**2)))*max(1-(radial/r)**2,0.)
    return speed*geom[5][e]+command*f


@njit(cache=True, inline='always')
def project(p,previous,e,body,geom,radii):
    points,ends,a,ab,length,direction,groups,candidates,valid,boundaries=geom
    t=np.sum((p-a[e])*direction[e])/length[e]
    motion=np.sum((p-previous)*direction[e]);tol=1e-10/length[e]
    crossed=(t<=tol and motion<0) or (t>=1-tol and motion>0)
    new=e;ep=0 if motion<0 else 1;group=groups[e,ep]
    if crossed:
        best=-np.inf
        for j in range(candidates.shape[1]):
            other=candidates[e,j]
            if not valid[e,j] or other==e: continue
            if groups[other,0]!=group and groups[other,1]!=group: continue
            away=1. if groups[other,0]==group else -1.
            score=np.sum((direction[other]*away)*(p-previous))
            if score>best: best=score;new=other
    axis,r,_,_=coord(p,new,geom,radii)
    blocked_gate=False
    if crossed and new!=e:
        nep=1 if groups[new,1]==group else 0
        if min(radii[ends[e,ep]],radii[ends[new,nep]])-body<0:
            blocked_gate=True;new=e;axis,r,_,_=coord(p,e,geom,radii)
    delta=p-axis;radial=np.sqrt(np.sum(delta**2));limit=r-body
    blocked=limit<0 or blocked_gate
    wall=radial>max(limit,0.)
    fixed=axis+delta*min(1.,max(limit,0.)/max(radial,1e-30))
    if blocked: fixed=previous.copy();new=e
    return fixed,new,wall and not blocked,blocked


@njit(cache=True, inline='always')
def project_union(p,previous,e,body,geom,radii):
    """Union-of-tubes lumen; identical rule to PhysicalTubeTransport._project_union."""
    points,ends,a,ab,length,direction,groups,candidates,valid,boundaries=geom
    best_in=-1;best_ratio=np.inf;best_out=-1;best_viol=np.inf
    for j in range(candidates.shape[1]):
        if not valid[e,j]: continue
        c=candidates[e,j]
        axis,r,radial,_=coord(p,c,geom,radii)
        limit=r-body
        if limit<0: continue
        if radial<=limit:
            ratio=radial/max(r,1e-12)
            if ratio<best_ratio: best_ratio=ratio;best_in=c
        else:
            if radial-limit<best_viol: best_viol=radial-limit;best_out=c
    if best_in>=0:
        return p.copy(),best_in,False,False
    if best_out>=0:
        axis,r,radial,_=coord(p,best_out,geom,radii)
        limit=r-body
        fixed=axis+(p-axis)*(limit/max(radial,1e-30))
        return fixed,best_out,True,False
    return previous.copy(),e,False,True


@njit(cache=True, inline='always')
def project_any(p,previous,e,body,geom,radii,union):
    if union: return project_union(p,previous,e,body,geom,radii)
    return project(p,previous,e,body,geom,radii)


@njit(cache=True, inline='always')
def boundary(previous,p,e,body,geom,radii):
    points,ends,a,ab,length,direction,groups,candidates,valid,boundaries=geom
    node=-1;fraction=1.;point=p.copy()
    for ep in range(2):
        if not boundaries[e,ep]: continue
        sign=-1. if ep==0 else 1.
        b=points[ends[e,ep]];d=direction[e]*sign
        before=np.sum((previous-b)*d);after=np.sum((p-b)*d)
        if not (before<=1e-9 and after>0): continue
        alpha=min(1.,max(0.,-before/(after-before))) if after>before else 1.
        intersection=previous+alpha*(p-previous)
        aperture=radii[ends[e,ep]]-body
        if aperture>=0 and np.sqrt(np.sum((intersection-b)**2))<=aperture+1e-9:
            node=ends[e,ep];fraction=alpha;point=intersection
    return node,fraction,point


@njit(cache=True)
def solve_flow(radii,hyd):
    parent,order,length,child,counts,terminal,pressure,root=hyd
    edge=np.zeros(len(radii));eq=terminal.copy();flux=np.zeros(len(radii))
    for node in order[1:]:
        a=radii[parent[node]];b=radii[node]
        edge[node]=0. if length[node]==0 else np.inf if min(a,b)==0 else length[node]*.5*(a**-4+b**-4)
    for node in order[::-1]:
        if counts[node]:
            conductance=0.
            for k in range(counts[node]):
                c=child[node,k];path=edge[c]+eq[c]
                if path==0: raise ValueError('Zero resistance from internal node to outlet')
                conductance+=1/path
            eq[node]=1/conductance if conductance>0 else np.inf
    flux[root]=pressure/eq[root]
    for node in order:
        total=0.
        for k in range(counts[node]):
            c=child[node,k];total+=1/(edge[c]+eq[c])
        if total>0:
            for k in range(counts[node]):
                c=child[node,k];flux[c]=flux[node]*(1/(edge[c]+eq[c]))/total
    return flux,eq[root]


@njit(cache=True, nogil=True)
def integrate(pos,edge,active,body,command,duration,geom,hyd,radii,flux,masses,initial_mass,
              healthy,bump,radius_fraction,routes,clot_positions,nrobots,contact_distance,
              lysis_rate,saturation,spatial_fraction,lubrication,max_substeps,trace_dt,surface_contact,
              particle_overlaps,safety_margin,union=False):
    points,ends,a,ab,length,direction,groups,candidates,valid,boundaries=geom
    n=len(pos);path=np.zeros(n);walls=np.zeros(n);blocks=np.zeros(n)
    exit_time=np.full(n,np.nan);exit_node=np.full(n,-1,np.int32)
    removed=np.zeros(nrobots);contact=np.zeros(nrobots);particle=np.zeros(nrobots);pair=0.
    near=np.zeros(nrobots);events=np.zeros(nrobots,np.int64)
    elapsed=0.;count=0
    trace_size=(max_substeps+1 if trace_dt<0 else int(np.ceil(duration/trace_dt))+2) if trace_dt!=0 else 1
    trace_pos=np.empty((trace_size,n,3));trace_active=np.empty((trace_size,n),np.bool_)
    trace_mass=np.empty((trace_size,len(masses)));trace_time=np.zeros(trace_size)
    trace_count=0;next_trace=trace_dt
    if trace_dt!=0:
        trace_pos[0]=pos;trace_active[0]=active;trace_mass[0]=masses;trace_count=1
    for i in range(n):
        _,r,d,_=coord(pos[i],edge[i],geom,radii)
        if active[i] and d+body[i]>r+1e-7: raise ValueError('Active body starts outside the accessible lumen')
    while elapsed<duration-1e-14 and np.any(active):
        if count>=max_substeps: raise RuntimeError('Physical integration budget exceeded; flow was NOT capped')
        dt=duration-elapsed;v0=np.zeros_like(pos)
        for i in range(n):
            if not active[i]: continue
            e=edge[i];max_speed=0.;local_min=np.inf;edge_min=np.inf
            for k in range(candidates.shape[1]):
                if not valid[e,k]: continue
                c=candidates[e,k];minr=max(min(radii[ends[c,0]],radii[ends[c,1]]),body[i])
                max_speed=max(max_speed,2*flux[ends[c,1]]/(np.pi*minr**2))
                local_min=min(local_min,minr);edge_min=min(edge_min,length[c])
            max_speed+=norm3(command[i])
            if max_speed>0: dt=min(dt,min(spatial_fraction*local_min,.25*edge_min)/max_speed)
            v0[i]=velocity(pos[i],e,body[i],command[i],geom,radii,flux,lubrication)
            axial=np.sum((pos[i]-a[e])*direction[e]);speed=np.sum(v0[i]*direction[e])
            distance=length[e]-axial if speed>=0 else axial
            if distance>1e-10 and abs(speed)>0: dt=min(dt,distance/abs(speed))
        if dt<=0 or not np.isfinite(dt): raise FloatingPointError('Invalid physical integration step')
        before=pos.copy();previous_active=active.copy();body_dt=np.zeros(n)
        for i in range(n):
            if not active[i]: continue
            e=edge[i];current=pos[i].copy()
            midpoint,me,_,_=project_any(current+.5*dt*v0[i],current,e,body[i],geom,radii,union)
            vmid=velocity(midpoint,me,body[i],command[i],geom,radii,flux,lubrication)
            euler=current+dt*v0[i];axial=np.sum((euler-a[e])*direction[e])/length[e]
            proposal=current+dt*(vmid if 0<=axial<=1 and me==e else v0[i])
            fixed,ne,wall,blocked=project_any(proposal,current,e,body[i],geom,radii,union)
            node,alpha,bpoint=boundary(current,fixed,e,body[i],geom,radii)
            if node>=0:
                fixed=bpoint;wall=False;blocked=False;active[i]=False
                exit_node[i]=node;exit_time[i]=elapsed+dt*alpha
            pos[i]=fixed;edge[i]=ne;path[i]+=normdiff(fixed,current)
            walls[i]+=(dt if wall else 0.);blocks[i]+=(dt if blocked else 0.)
            body_dt[i]=dt*(alpha if node>=0 else 1.)
        mid=(before+pos)/2
        for i in range(nrobots):
            if not previous_active[i]: continue
            for j in range(i+1,nrobots):
                if previous_active[j] and normdiff(mid[i],mid[j])<2*body[i]:
                    pair+=min(body_dt[i],body_dt[j])
            for j in range(nrobots,n):
                overlap=previous_active[j] and normdiff(mid[i],mid[j])<body[i]+body[j]
                if overlap:
                    particle[i]+=min(body_dt[i],body_dt[j])
                    if not particle_overlaps[i,j-nrobots]:events[i]+=1
                particle_overlaps[i,j-nrobots]=overlap
                if previous_active[j]:
                    gap=normdiff(mid[i],mid[j])-body[i]-body[j]
                    risk=max(1-max(gap,0.)/safety_margin,0.)**2
                    near[i]+=risk*min(body_dt[i],body_dt[j])
        changed=False
        if lysis_rate>0 and np.any(masses>0):
            exposure=np.zeros((nrobots,len(masses)))
            for i in range(nrobots):
                if not previous_active[i]: continue
                e=edge[i];_,r,radial,t=coord(mid[i],e,geom,radii)
                if surface_contact:
                    selected=-1;best=0.;selected_bump=0.
                    for j in range(len(masses)):
                        b=(1-t)*bump[ends[e,0],j]+t*bump[ends[e,1],j]
                        contribution=b*masses[j]/initial_mass[j]
                        if contribution>best:
                            best=contribution;selected=j;selected_bump=b
                    if selected>=0 and selected_bump>=np.exp(-4.5) and r-radial-body[i]<=contact_distance:
                        exposure[i,selected]=body_dt[i];contact[i]+=body_dt[i]
                    continue
                for j in range(len(masses)):
                    if masses[j]<=0: continue
                    geo=min(routes[j,ends[e,0]]+t*length[e],routes[j,ends[e,1]]+(1-t)*length[e])
                    if geo<=contact_distance and normdiff(mid[i],clot_positions[j])<=contact_distance:
                        exposure[i,j]=body_dt[i];contact[i]+=body_dt[i]
            for j in range(len(masses)):
                total=exposure[:,j].sum();factor=min(1.,saturation*dt/max(total,1e-30))
                weighted=exposure[:,j]*factor;wsum=weighted.sum();amount=min(masses[j],lysis_rate*wsum)
                if amount>0:
                    changed=True;masses[j]=max(masses[j]-amount,0.)
                    for i in range(nrobots): removed[i]+=weighted[i]/max(wsum,1e-30)*amount
            if changed:
                for s in range(len(radii)):
                    block=0.
                    for j in range(len(masses)): block=max(block,bump[s,j]*(masses[j]/initial_mass[j]))
                    radii[s]=healthy[s]*(1-(1-radius_fraction)*block)
                flux,_=solve_flow(radii,hyd)
        elapsed+=dt;count+=1
        if trace_dt!=0 and (trace_dt<0 or elapsed>=next_trace or elapsed>=duration-1e-14 or not np.any(active)):
            trace_pos[trace_count]=pos;trace_active[trace_count]=active
            trace_mass[trace_count]=masses;trace_time[trace_count]=elapsed;trace_count+=1
            next_trace=(np.floor(elapsed/trace_dt)+1)*trace_dt if trace_dt>0 else 0.
    return pos,edge,active,path,walls,blocks,exit_time,exit_node,count,masses,radii,flux,removed,contact,particle,pair,trace_pos[:trace_count],trace_active[:trace_count],trace_mass[:trace_count],trace_time[:trace_count],near,events,particle_overlaps


class CompiledMCAPhysicalEnv(MCAPhysicalEnv):
    """Same task/observations with a compiled inner loop; Python remains reference."""
    def reset(self, *, seed=None, options=None):
        # Cache only deterministic nominal anatomy with no stenosis RNG draws.
        # Robot and particle starts are resampled from each episode's RNG.
        if not getattr(self, '_nominal_cache', False):
            result=super().reset(seed=seed,options=options)
            self._nominal_cache=(self._fixed_tree is None and self.config.geometry_variation==0
                                 and not self.tree.territory.stenoses)
            if self._nominal_cache:
                self._reset_solution=copy.deepcopy(self.solution)
            return result
        gym.Env.reset(self,seed=seed)
        options=options or {}
        unknown=set(options)-{'robot_positions_mm','particle_positions_mm'}
        if unknown: raise ValueError(f'Unknown reset options: {sorted(unknown)}')
        self.masses=self.initial_mass.copy()
        self._sample_episode_flow()
        if self.config.clot_initialization != 'historical_sites' and self._fixed_clot_stations is None:
            self._reset_clots()
        self.solution=(copy.deepcopy(self._reset_solution) if
                       self.config.inlet_flow_multiplier_min == self.config.inlet_flow_multiplier_max and
                       (self.config.clot_initialization == 'historical_sites' or self._fixed_clot_stations is not None) else
                       self.flow_model.solve(self._radii(self.masses)))
        robot_pos=self._initial_robot_positions(options)
        self._reset_robot_positions_mm=robot_pos
        self.particle_layout='explicit_positions'
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


    def _compiled_arguments(self):
        if getattr(self,'_args_transport',None) is self.transport:
            return self._args_cache
        t=self.transport;f=self.flow_model
        boundaries=np.column_stack((t.edge_groups[:,0]==t.root_group,
                                    np.isin(t.edge_groups[:,1],list(t.terminal_groups))))
        geom=(t.points,t.ends,t.a,t.ab,t.length,t.direction,t.edge_groups,t.candidates,t.candidate_valid,boundaries)
        counts=np.array([len(c) for c in f.children],np.int64)
        child=np.zeros((len(counts),max(1,int(counts.max()))),np.int64)
        for i,c in enumerate(f.children): child[i,:len(c)]=c
        hyd=(f.parent,np.array(f.order,np.int64),f.length,child,counts,f.terminal_resistance,f.driving_pressure,f.root)
        self._args_transport=self.transport
        self._args_cache=(geom,hyd)
        return self._args_cache

    def _advance_particle_prediction(self, positions, edges, body, active, duration):
        # Particle-only transport with frozen current radii/flux. No env.step,
        # no future actions or masses, and no mutation of actual simulator state.
        geom, hyd = self._compiled_arguments();c = self.config
        values = integrate(positions.copy(), edges.copy(), active.copy(), body,
            np.zeros_like(positions), duration, geom, hyd,
            self.solution['radius_mm'].copy(), self.solution['station_inflow_mm3_s'].copy(),
            self.masses.copy(), self.initial_mass, self.flow_model.healthy_radius_mm,
            self._occlusion_bump, c.initial_radius_fraction,
            np.asarray(self.routes).reshape(self.num_clots, -1) if self.num_clots else np.empty((0, len(self.transport.points))),
            self.clot_positions_mm, 0, c.contact_distance_mm, 0., c.lysis_saturation,
            c.spatial_fraction, c.lubrication_floor, c.max_substeps_per_control, 0., False,
            np.zeros((0, len(positions)), dtype=np.bool_), c.particle_safety_margin_mm, c.junction_model == 'union')
        return TransportResult(*values[:9], float(duration))

    def step(self, action):
        if self._done or not self._reset_called:
            raise RuntimeError('Call reset before stepping a new or completed episode')
        action=np.asarray(action,np.float64)
        if action.shape!=(self.num_robots,3) or not np.isfinite(action).all():
            raise ValueError('Expected finite world-frame [num_robots,3] actions')
        action=bound_command(self._apply_action_prior(action),self.config.command_speed)
        commands=np.zeros_like(self.positions_mm);commands[:self.num_robots]=action*self.config.robot_speed_mm_s
        duration=min(self.config.control_dt_s,self.config.episode_duration_s-self.elapsed_s)
        potential_before=self._reward_potential()
        assignment_before=self._shaping_assignment()
        geom,hyd=self._compiled_arguments();c=self.config;n=self.num_robots
        values=integrate(self.positions_mm.copy(),self.edges.copy(),self.active.copy(),self.body_radius,commands,
                         duration,geom,hyd,self.solution['radius_mm'].copy(),self.solution['station_inflow_mm3_s'].copy(),
                         self.masses.copy(),self.initial_mass,self.flow_model.healthy_radius_mm,self._occlusion_bump,
                         c.initial_radius_fraction,np.asarray(self.routes).reshape(self.num_clots,-1) if self.num_clots else np.empty((0,len(self.transport.points))),
                         self.clot_positions_mm,n,c.contact_distance_mm,c.lysis_mass_per_s,c.lysis_saturation,
                         c.spatial_fraction,c.lubrication_floor,c.max_substeps_per_control,float(getattr(self,"trace_dt_s",0.)),
                         c.contact_model == 'stenosis_surface',self._particle_overlaps.copy(),c.particle_safety_margin_mm,
                         c.junction_model == 'union')
        pos,edges,active,path,walls,blocks,exit_time,exit_node,count,masses,radii,flux,agent_removed,contact_s,particle_contact_s,pair_contact_s,trace_pos,trace_active,trace_mass,trace_time,particle_near_s,particle_events,particle_overlaps=values
        self.last_trace = dict(positions_mm=trace_pos, active=trace_active, masses=trace_mass, time_s=trace_time+self.elapsed_s)
        result=TransportResult(pos,edges,active,path,walls,blocks,exit_time,exit_node,count,duration)
        # Recompute the public flow dictionary once per control step with the
        # original solver. Inner-loop flow uses identical tree resistance laws.
        solution=self.flow_model.solve(radii) if np.any(agent_removed>0) else self.solution
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
        particle_penalty=self._avoidance_outcome(particle_contact_s,particle_near_s,particle_overlaps,particle_events)
        info = self._info()
        terminated = info['success'] or info['active_robots'] == 0
        truncated = not terminated and self.elapsed_s >= self.config.episode_duration_s - 1e-12
        self._done = terminated or truncated
        agent_reward = 10*agent_removed - result.wall_contact_s[:n] - .01*duration
        agent_reward -= particle_penalty
        shaping = self._shaping(potential_before, assignment_before)
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
