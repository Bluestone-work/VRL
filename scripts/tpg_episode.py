"""EXP0053: graph reservations and learned local controls over measured sensors."""
from collections import deque, Counter
from itertools import permutations
import hashlib
import json
import numpy as np

from scripts.aligned_option_episode import AlignedOptionEpisode, aligned_hashes, ROOT
from scripts.tracked_learning_episode import TrackedLearningEpisode
from marl.measured_options import joint_measured_projection,target_packet
from marl.measured_tpg import (scene_from_processor, measured_route, measured_assignment,
    polyline_prefix, polyline_distance, RollingTPG, tangent_frame)
from marl.tpg_learning import NODE_DIM,PAIR_DIM,LOW_DIM,HISTORY,N_LOW,CANDIDATE_DIM,causal_history

PROTOCOL=ROOT/'configs/experiments/EXP_0053_MEASURED_TPG.json'


def tpg_hashes():
    hashes=aligned_hashes()
    for name in ('marl/measured_tpg.py','marl/tpg_learning.py','scripts/tpg_episode.py',
                 'scripts/run_tpg_learning.py','scripts/run_tpg_study.py',
                 'configs/experiments/EXP_0053_MEASURED_TPG.json'):
        path=ROOT/name
        if not path.exists(): raise FileNotFoundError(path)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


class TPGEpisode(AlignedOptionEpisode):
    def __init__(self,*args,**kwargs):
        self.booted=False
        super().__init__(*args,**kwargs)
        self.protocol=json.loads(PROTOCOL.read_text())
        n=self.cfg.clusters
        self.tpg=RollingTPG(n,self.protocol['tpg'])
        self.history=deque(maxlen=HISTORY)
        self.graph_target_ids=np.full(n,-1,int)
        self.previous_active=np.zeros(n,bool)
        self.progress_positions=np.zeros((n,3)); self.progress_times=np.zeros(n)
        self.last_event_s=-np.inf; self.last_frame=-np.inf
        self.route_refreshes=self.target_reallocations=self.actual_target_switches=0
        self.unknown_route_agent_steps=self.ambiguous_route_agent_steps=0
        self.wait_agent_s=self.blocked_path_agent_s=0.
        self.low_counts=np.zeros(N_LOW,int)
        self.event_counts=Counter();self.decision_times=[]
        self.last_history_step=-1
        self.memory_prepared_step=-1
        self.booted=True
        self.refresh_graph()

    def prepare(self):
        if not self.booted:
            return AlignedOptionEpisode.prepare(self)
        self.packet=self.sensor.observe()
        self.refresh_graph()

    def refresh_graph(self):
        n=self.cfg.clusters
        # Simulator clock is the sensor acquisition/control clock, not a navigation label.
        self.scene=scene_from_processor(self.sensor.processor,self.packet,float(self.env.elapsed_s))
        scene=self.scene; now=scene.time_s
        reasons=[]
        changed_active=not np.array_equal(scene.active,self.previous_active)
        needs_target=any(scene.active[i] and (self.graph_target_ids[i]<0 or
            not scene.target_alive[self.graph_target_ids[i]]) for i in range(n))
        if needs_target:
            old=self.graph_target_ids.copy()
            self.graph_target_ids=measured_assignment(scene,old,self.protocol['route'])
            self.actual_target_switches+=int(np.sum((old>=0)&(old!=self.graph_target_ids)&scene.active))
            self.target_reallocations+=1;reasons.append('observed_target_event')
        self.routes=[]; self.prefixes=[]
        for i in range(n):
            target=scene.targets[self.graph_target_ids[i]] if self.graph_target_ids[i]>=0 else scene.positions[i]
            route=measured_route(scene.crops[i],scene.positions[i],target,self.protocol['route'])
            self.routes.append(route)
            self.prefixes.append(polyline_prefix(route.points,self.protocol['route']['lookahead_mm']))
        self.route_refreshes+=1
        processor=self.sensor.processor
        new_frame=processor.last_frame_time>self.last_frame
        fresh=np.array([new_frame and scene.active[i] and
            processor.tracks.get(('robot',i),(-np.inf,))[0]>=processor.last_frame_time-1e-9 and
            processor.crop_cache.get(i,(-np.inf,))[0]>=processor.last_frame_time-1e-9 for i in range(n)],bool)
        new_conflict=self.tpg.inspect(self.prefixes,scene.active,fresh)
        self.last_frame=self.sensor.processor.last_frame_time
        if new_conflict:reasons.append('new_geometric_conflict')
        if changed_active:reasons.append('track_state_change')
        stall=False
        for i in range(n):
            if not scene.active[i]:continue
            progressed=np.linalg.norm(scene.positions[i]-self.progress_positions[i])>=self.protocol['events']['progress_mm']
            near=self.routes[i].remaining_mm is not None and self.routes[i].remaining_mm<.3
            if progressed or near or not self.previous_active[i]:
                self.progress_positions[i]=scene.positions[i];self.progress_times[i]=now
            elif now-self.progress_times[i]>=self.protocol['events']['stall_s']:
                stall=True;self.progress_times[i]=now
        if stall and now-self.last_event_s>=self.protocol['events']['cooldown_s']:
            reasons.append('measured_no_progress')
        if self.tpg.calls==0 and scene.active.any():reasons.append('initial_schedule')
        self.needs_decision=bool(reasons and scene.active.any())
        self.event_reasons=tuple(sorted(set(reasons)))
        self.previous_active=scene.active.copy()
        self.build_features()
        self.build_low()

    def build_features(self):
        scene=self.scene;n=self.cfg.clusters
        nodes=np.zeros((n,NODE_DIM),np.float32)
        nodes[:,:11]=np.clip(self.packet.navigation[:,:11],-10,10)
        for i,route in enumerate(self.routes):
            nodes[i,11:14]=route.tangent
            nodes[i,14]=0. if route.remaining_mm is None else route.remaining_mm/10.
            nodes[i,15]=float(route.remaining_mm is not None)
            nodes[i,16:24]=[route.visible_length_mm/10.,route.lower_bound_mm/10.,route.radius_mm,
                route.curvature,float(route.ambiguous),scene.ages[i],float(self.tpg.edges[:,i].any()),float(scene.active[i])]
        pairs=np.zeros((n,n,PAIR_DIM),np.float32)
        for i in range(n):
            for j in range(n):
                if i==j or not (scene.active[i] and scene.active[j]):continue
                pairs[i,j,:3]=(scene.positions[j]-scene.positions[i])/6.
                pairs[i,j,3]=min(float(self.tpg.distances[i,j]),12.)/6.
                pairs[i,j,4]=float(self.tpg.edges[i,j])-float(self.tpg.edges[j,i])
                pairs[i,j,5]=1.
        self.nodes=np.clip(nodes,-10,10);self.pairs=np.clip(pairs,-10,10)

    def conventional_priority(self):
        costs=np.where(self.nodes[:,15]>.5,self.nodes[:,14],self.nodes[:,17]+.2)
        pending=set(range(self.cfg.clusters));admitted=[]
        while pending:
            ready=[i for i in pending if not any(self.tpg.edges[j,i] for j in pending)]
            node=min(ready,key=lambda i:(costs[i],i));admitted.append(node);pending.remove(node)
        order=tuple(admitted)
        return list(permutations(range(self.cfg.clusters))).index(order)

    def choose_priority(self,action):
        if not self.needs_decision:raise ValueError('No event requires a priority decision')
        order=list(permutations(range(self.cfg.clusters)))[int(action)]
        self.tpg.schedule(order)
        self.decision_times.append(self.scene.time_s)
        self.event_counts.update(self.event_reasons)
        self.last_event_s=self.scene.time_s
        self.needs_decision=False
        self.build_features();self.build_low()

    def build_low(self):
        n=self.cfg.clusters;packet=self.packet;scene=self.scene
        # Preserve the existing strong measured-memory nominal controller.
        # Cropped graph localization is not yet reliable enough to replace it.
        # Prepare this stateful library exactly once per physical control step.
        if self.memory_prepared_step!=self.steps:
            self.memory_packet=target_packet(packet,self.graph_target_ids)
            self.memory_features,self.memory_candidates,self.memory_valid,self.memory_details=self.library.prepare(self.memory_packet)
            self.memory_choices=self.library.conventional_choice(self.memory_packet,self.memory_valid,self.memory_details,memory=True)
            self.memory_prepared_step=self.steps
        ready=self.tpg.ready(scene.active)
        commands=np.zeros((n,N_LOW,3),float);valid=np.zeros((n,N_LOW),bool)
        candidate_features=np.zeros((n,N_LOW,CANDIDATE_DIM),np.float32)
        for i,route in enumerate(self.routes):
            drift=packet.navigation[i,3:6]-packet.navigation[i,:3]
            hold=-drift
            steer=polyline_prefix(route.points,self.protocol['route']['steer_mm'])[-1]-scene.positions[i]
            norm=np.linalg.norm(steer);t=steer/max(norm,1e-9)
            magnitude=min(1.,norm/.15)
            radial=packet.navigation[i,6:9]
            nominal=self.memory_candidates[i,self.memory_choices[i]].copy()
            route_command=magnitude*t-.35*radial-drift
            frame=tangent_frame(t)
            commands[i]=[nominal,route_command,.5*nominal-.5*drift,nominal+.2*frame[1],nominal-.2*frame[1],
                nominal+.2*frame[2],nominal-.2*frame[2],hold,-.5*t-.35*radial-drift]
            valid[i]=scene.active[i]
            if not ready[i] or route.ambiguous or norm<1e-8:
                valid[i,:7]=False
            # Yielding retreats must follow an observed branch away from a predecessor.
            incoming=np.flatnonzero(self.tpg.edges[:,i]&scene.active)
            if len(incoming):
                away=np.sum(scene.positions[i]-scene.positions[incoming],axis=0)
                escape_goal=scene.positions[i]+100*away/max(np.linalg.norm(away),1e-9)
                escape=measured_route(scene.crops[i],scene.positions[i],escape_goal,self.protocol['route'])
                e=polyline_prefix(escape.points,self.protocol['route']['steer_mm'])[-1]-scene.positions[i]
                e=e/max(np.linalg.norm(e),1e-9)
                commands[i,8]=.5*e-.35*radial-drift
                valid[i,8]=bool(not escape.ambiguous and e@away>1e-5)
            elif not ready[i]: valid[i,8]=False
            if route.ambiguous:valid[i,8]=False
            if not scene.active[i]:
                commands[i]=0.;valid[i,7]=True
            commands[i]/=np.maximum(np.linalg.norm(commands[i],axis=1,keepdims=True),1.)
            preferred=0 if valid[i,0] else 7
            # Supply the classical controller with the same retreat candidate.
            obstructs=any(polyline_distance(self.prefixes[j],scene.positions[i][None]) <
                self.protocol['tpg']['spacing_mm']+self.protocol['tpg']['margin_mm'] for j in incoming)
            if not ready[i] and obstructs and valid[i,8]:preferred=8
            candidate_features[i,:,:3]=commands[i]
            candidate_features[i,:,3]=commands[i]@t
            candidate_features[i,:,4]=commands[i]@radial
            candidate_features[i,:,5]=np.linalg.norm(commands[i],axis=1)
            candidate_features[i,:,6]=packet.navigation[i,9]
            candidate_features[i,:,7]=float(ready[i])
            candidate_features[i,preferred,8]=1.
        state=np.concatenate((packet.navigation,self.nodes[:,11:]),axis=1).astype(np.float32)
        assert state.shape==(n,LOW_DIM)
        if self.last_history_step==self.steps:self.history[-1]=state
        else:self.history.append(state);self.last_history_step=self.steps
        self.low_history=causal_history(self.history)
        self.low_commands=commands;self.low_valid=valid;self.low_candidates=candidate_features
        self.ready=ready

    def conventional_low(self):
        return self.low_candidates[:,:,8].argmax(axis=1)

    def step_control(self,actions):
        n=self.cfg.clusters;actions=np.asarray(actions,int)
        if self.needs_decision:raise ValueError('Resolve pending scheduling event first')
        if actions.shape!=(n,) or not self.low_valid[np.arange(n),actions].all():raise ValueError('Invalid local control')
        self.low_counts+=np.bincount(actions[self.scene.active],minlength=N_LOW)
        self.unknown_route_agent_steps+=sum(self.scene.active[i] and r.remaining_mm is None for i,r in enumerate(self.routes))
        self.ambiguous_route_agent_steps+=sum(self.scene.active[i] and r.ambiguous for i,r in enumerate(self.routes))
        observed_before=self.scene.active.copy()
        old_stamps=np.array([self.sensor.processor.tracks.get(('robot',i),(-np.inf,))[0] for i in range(n)])
        cmd=self.low_commands[np.arange(n),actions]
        cmd,checks=joint_measured_projection(cmd,self.packet,self.protocol['supervisor'],self.library.speed)
        self.supervisor_changes+=int(checks['changed']);self.supervisor_infeasible+=int(checks['residual']>1e-5)
        self.max_supervisor_residual=max(self.max_supervisor_residual,checks['residual'])
        self.missing_track_steps+=checks['missing_tracks']
        # Reuse frozen physical stepping and independent substep outcome scoring.
        self.features,self.candidates,self.valid,self.details=(self.memory_features,self.memory_candidates.copy(),
            self.memory_valid.copy(),self.memory_details)
        self.candidates[np.arange(n),self.memory_choices]=cmd
        self.valid[np.arange(n),self.memory_choices]=True
        self.details['projection_residual'][np.arange(n),self.memory_choices]=checks['residual']
        removal_before=self.previous_removal;spacing_before=self.spacing.pair_violation_s
        physical_before=self.env.active[:n].copy()  # reward instrumentation only
        waiting=int(np.sum(observed_before&~self.ready))
        _,done,_,_=TrackedLearningEpisode.step(self,self.memory_choices)
        dt=float(self.info['step_duration_s']);horizon=self.env.config.episode_duration_s
        self.wait_agent_s+=waiting*dt
        rc=self.protocol['reward']
        reward=rc['removed_fraction']*(self.previous_removal-removal_before)
        reward+=rc['removal_auc_fraction']*.5*(self.previous_removal+removal_before)*dt/horizon
        reward+=rc['wall_fraction']*float(np.sum(self.info['wall_contact_s']))/(n*horizon)
        reward+=rc['spacing_pair_fraction']*(self.spacing.pair_violation_s-spacing_before)/(max(n*(n-1)/2,1)*horizon)
        reward+=rc['particle_fraction']*float(np.sum(self.info['particle_contact_s']))/(n*horizon)
        reward+=rc['lost_fraction']*float(np.sum(physical_before&~self.env.active[:n]))/n
        reward+=rc['step_cost_per_horizon']*dt/horizon
        # The auxiliary label is a filtered measurement, never true velocity.
        fresh=np.array([self.sensor.processor.tracks.get(('robot',i),(-np.inf,))[0]>old_stamps[i] for i in range(n)],bool)
        aux_valid=observed_before&self.scene.active&fresh&(not done)
        aux_velocity=np.clip(self.scene.velocities/self.library.speed,-5,5).astype(np.float32)
        return float(reward),done,dict(commands=cmd.astype(np.float32),
            measured_velocity=aux_velocity,measurement_valid=aux_valid)

    def result(self,policy):
        result=super().result(policy)
        for key in ('option_steps','option_counts','memory_option_fraction','mean_option_control_steps'):
            result.pop(key,None)
        gaps=np.diff(self.decision_times)
        result.update(experiment='EXP_0053_MEASURED_TPG',source_hashes=tpg_hashes(),
            scheduler=policy,observation_contract=self.protocol['observation_contract'],
            scheduler_calls=self.tpg.calls,macro_steps=self.tpg.calls,target_reallocations=self.target_reallocations,
            actual_target_switches=self.actual_target_switches,priority_switches=self.tpg.priority_switches,
            reservation_releases=self.tpg.releases,route_refreshes=self.route_refreshes,
            scheduler_interval_mean_s=float(np.mean(gaps)) if len(gaps) else None,
            scheduler_interval_min_s=float(np.min(gaps)) if len(gaps) else None,
            event_counts=dict(self.event_counts),wait_agent_s=self.wait_agent_s,
            unknown_route_agent_steps=int(self.unknown_route_agent_steps),
            ambiguous_route_agent_steps=int(self.ambiguous_route_agent_steps),
            low_control_counts=self.low_counts.tolist(),physical_control_steps=self.steps,
            nominal_low_controller='frozen measured memory; graph-following is a separate selectable candidate',
            local_route_length_is_global=False,magnetic_spacing_calibrated=False,
            graph_global_truth_input=False,low_auxiliary_label='next filtered measured velocity',
            tpg_global_completeness=False,safety_certificate=False)
        return result
