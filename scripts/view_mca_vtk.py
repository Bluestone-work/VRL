"""EXP23 checkpoint replay using the established watch_gui.py VTK renderer.

Same compiled substep trajectories as the PyBullet option; rendering only.
Mouse orbit/zoom; space pause; R full anatomy; F treatment region; O orbit.
"""
import argparse
import copy
import json
import secrets
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from scripts.view_mca_pybullet import collect
from scripts.train_mca_physical import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--headless',action='store_true');ap.add_argument('--once',action='store_true')
    ap.add_argument('--fixed-seed',action='store_true',help='Repeat the same reset instead of showing different episode starts')
    ap.add_argument('--seed',type=int,help='Reproducible first layout; omitted uses a new seed per viewer session')
    ap.add_argument('--slowdown',type=float,default=.1)
    ap.add_argument('--playback-speed',type=float,help='Simulated seconds per wall second; rendering only')
    args=ap.parse_args();torch.set_num_threads(1)
    if args.seed is None:args.seed=900000000 if args.fixed_seed else 2000000000+secrets.randbelow(1000000000)
    speed=args.playback_speed if args.playback_speed is not None else 1/args.slowdown
    if not np.isfinite(speed) or speed<=0:ap.error('Playback speed must be finite and positive')
    from environments.pv_render import PVRenderer,PVStyle,View,centerline_tube
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);checkpoint=Path(args.checkpoint);checkpoint_anchor=checkpoint
    pointer=checkpoint_anchor.parent.parent/'active_run.json'
    if pointer.exists():
        checkpoint=Path(json.loads(pointer.read_text())['path'])/checkpoint_anchor.parent.name/checkpoint_anchor.name
    env,radii,frames,steps,info=collect(checkpoint,args.seed)
    surface=env.config.contact_model=='stenosis_surface'
    localized=env.config.contact_model=='localized_point'
    particle_display_scale=6. if localized else 3.
    clot_display_scale=3. if localized else 1.
    clot_radius=env.config.contact_distance_mm*clot_display_scale if localized else .4
    scale=env.units.mm_per_unit;tree=copy.copy(env.tree)
    tree.radii=(env.flow_model.healthy_radius_mm if surface else radii)/scale
    style=PVStyle(width=1500,height=940,hud_width=0,hud_height=0,wall_opacity=.38,
                  wall_sides=48,robot_radius=env.config.robot_radius_mm*4/scale,
                  background='#080b14',camera_margin=1.10,views=(View('MCA',None,28),),
                  clot_color='#ff7043' if localized else '#c8821e')
    renderer=PVRenderer(style,off_screen=args.headless)
    renderer.build(tree,env.num_robots,200,clot_draw_radius=clot_radius/scale)
    plot=renderer.plotter;pv=renderer.pv
    wall_actor=next((actor for actor in plot.renderer.actors.values() if hasattr(actor,'GetProperty') and
                     abs(actor.GetProperty().GetOpacity()-style.wall_opacity)<1e-6),None)
    plot.title=('MCA | Random clots, robots and obstacles' if env.config.clot_initialization=='random_branches' else
                'MCA | Distributed points | 32 moving particles') if localized else ('MCA | In-vitro pure RL replay' if surface else 'MCA | Pure RL policy replay | Distributed starts')
    plot.set_background('#080b14',top='#19263c')
    plot.add_axes(line_width=2,labels_off=False)
    plot.add_text(('MCA  /  ALL OBJECTS RANDOMIZED' if env.config.clot_initialization=='random_branches' else
                   'MCA  /  DISTRIBUTED POINT TARGETS') if localized else
                  ('MCA  /  IN-VITRO PURE RL' if surface else 'MCA  /  PURE RL'),
                  position=(18,885),font_size=22,color='#eef5ff',name='title')
    plot.add_text('Drag: orbit   Space: pause   1 / 5 / 0: 1x / 5x / 10x   S: true body sizes   T: trails   F: focus   R: full',
                  position='lower_left',font_size=11,color='#a9bfd5',name='controls')
    legend=('Blue: robots (4x display size)   Gold: stenosis inner surface   White: passive tracers\n'
            'In-vitro simulation | Original-scale MCA | Healthy reference flow: 0.01 mL/min') if surface else (
            'Blue: robots (4x display size)   Gold: clot-site markers   White: passive tracers\n'
            'Wall shows initial obstructed lumen | Engineering task | Geometry & flow are not changed by rendering')
    plot.add_text(legend,
                  position=(18,40),font_size=10,color='#a9bfd5',name='legend')
    if localized:
        plot.add_text(f'Default display sizes: robots 4x | clot markers 3x | {env.config.particle_count} particles 6x. S: actual body sizes\n'
                      'In-vitro low-flow simulation | Point contact: 0.12 mm | Red bodies: actual overlap, not display overlap',
                      position=(18,40),font_size=10,color='#a9bfd5',name='legend')
    # A restrained key light and rim light make the swept wall read as volume.
    center=tree.points.mean(0)
    plot.add_light(pv.Light(position=center+np.array([.5,-.7,.9]),focal_point=center,color='#e7efff',intensity=.65))
    plot.add_light(pv.Light(position=center+np.array([-.5,.6,.35]),focal_point=center,color='#ffb8ac',intensity=.45))
    plot.enable_anti_aliasing('ssaa' if args.headless else 'fxaa')
    for actor in plot.renderer.actors.values():
        if hasattr(actor,'GetProperty') and abs(actor.GetProperty().GetOpacity()-.38)<1e-6:
            prop=actor.GetProperty();prop.SetAmbient(.18);prop.SetDiffuse(.85);prop.SetSpecular(.65);prop.SetSpecularPower(35)
    particle_actors=[]
    surface_masses=None
    def update_surfaces(masses):
        nonlocal surface_masses
        if not surface or (surface_masses is not None and np.max(abs(surface_masses-masses))<1e-4):return
        current_radius=env._radii(masses)/scale
        for j in range(env.num_clots):
            name=f'stenosis_surface_{j}'
            plot.remove_actor(name,render=False)
            if masses[j]<=0:continue
            ids=np.flatnonzero(env._occlusion_bump[:,j]>=np.exp(-4.5))
            if len(ids)<2:continue
            line=pv.PolyData(tree.points[ids]);line.lines=np.r_[len(ids),np.arange(len(ids))]
            line['radius']=current_radius[ids]
            mesh=line.tube(scalars='radius',absolute=True,radius=float(current_radius[ids].mean()),n_sides=32,capping=False)
            plot.add_mesh(mesh,name=name,color='#e6ad48',opacity=.7,smooth_shading=True,
                          specular=.5,ambient=.4,render=False)
        surface_masses=masses.copy()
    for _ in range(env.config.particle_count):
        mesh=pv.Sphere(radius=env.config.particle_radius_mm*particle_display_scale/scale,theta_resolution=12,phi_resolution=12)
        particle_actors.append(plot.add_mesh(mesh,color='#e3edf5',ambient=.65,specular=.5,render=False))
    trail_mesh=pv.PolyData(np.zeros((2,3)));trail_mesh.lines=np.array([2,0,1])
    trail_actor=plot.add_mesh(trail_mesh,color='#7acfe8',line_width=2,opacity=.8,lighting=False,render=False)
    trail_actor.SetVisibility(False)
    status_actor=plot.add_text('',position=(18,710),font_size=11,color='#c2d4e8',name='status',render=False)
    state={'paused':False,'orbit':False,'trails':True,'real_size':False,'speed':speed,'clock_sim':0.,'clock_wall':time.monotonic()}
    home=copy.deepcopy(plot.camera_position)
    def play_time():
        return state['clock_sim']+(0. if state['paused'] else (time.monotonic()-state['clock_wall'])*state['speed'])
    def pause():
        state['clock_sim']=play_time();state['clock_wall']=time.monotonic();state['paused']=not state['paused']
    def set_speed(value):
        state['clock_sim']=play_time();state['clock_wall']=time.monotonic();state['speed']=value
    def trails():state['trails']=not state['trails']
    def sizes():state['real_size']=not state['real_size']
    def events():
        if not args.headless and plot.iren is not None:plot.iren.process_events()
    def snapshot(name):
        temp=out/(name+'.tmp.png');plot.screenshot(str(temp));temp.replace(out/(name+'.png'))
    def orbit(): state['orbit']=not state['orbit']
    def full(): plot.camera_position=home
    def focus():
        focus_pts=np.vstack((env.clot_positions_mm/scale,frames[0][1][:env.num_robots]/scale))
        target=focus_pts.mean(0);span=max(.05,float(np.linalg.norm(np.ptp(focus_pts,axis=0))))
        plot.camera_position=[target+np.array([.7,-1.,.9])*span,target,(0,0,1)]
    if not args.headless:
        for key,fn in [('space',pause),('o',orbit),('r',full),('f',focus),('t',trails),('s',sizes),
                       ('1',lambda:set_speed(1.)),('5',lambda:set_speed(5.)),('0',lambda:set_speed(10.))]: plot.add_key_event(key,fn)
        renderer.show_interactive()
    last_mtime=checkpoint.stat().st_mtime_ns;replay_index=0
    try:
        while True:
            if not args.headless and plot._closed: break
            pointer=checkpoint_anchor.parent.parent/'active_run.json'
            if pointer.exists():
                checkpoint=Path(json.loads(pointer.read_text())['path'])/checkpoint_anchor.parent.name/checkpoint_anchor.name
            mtime=checkpoint.stat().st_mtime_ns
            episode_seed=args.seed if args.fixed_seed else args.seed+replay_index
            if mtime!=last_mtime or (replay_index and not args.fixed_seed):
                env,radii,frames,steps,info=collect(checkpoint,episode_seed);last_mtime=mtime
            if env.config.clot_initialization=='random_branches' and wall_actor is not None:
                mesh=centerline_tube(tree.points,radii/scale,tree.branch_ids,style.wall_sides)
                wall_actor.mapper.dataset=mesh
            for i,pos in enumerate(env.clot_positions_mm/scale):
                plot.add_point_labels(np.array([pos]),[f'C{i+1}'],name=f'clot_label_{i}',point_size=0,font_size=12,
                                      text_color='#e8c88a',shape_opacity=0,always_visible=True)
            starts=frames[0][1][:env.num_robots]
            start_edges=env.transport.nearest_edges(starts)
            start_branches=env.tree.branch_ids[env.transport.ends[start_edges,0]]
            for i,bid in enumerate(start_branches):
                plot.add_point_labels(starts[i:i+1]/scale,[f'R{i+1} start / B{bid}'],
                                      name=f'start_label_{i}',font_size=13,text_color='#75d6ff',point_size=5,
                                      shape_opacity=.5,always_visible=True)
            status_path=checkpoint.parent/'status.json'
            worker_phase=json.loads(status_path.read_text()).get('phase','unknown') if status_path.exists() else 'unknown'
            atomic_json(out/'replay.json',dict(renderer='existing PVRenderer from watch_gui.py',checkpoint=str(checkpoint),
                       checkpoint_steps=steps,episode_seed=episode_seed,frames=len(frames),playback_slowdown=1/state['speed'],
                       playback_speed=state['speed'],wall_clock_playback=True,particle_trail_sim_seconds=30.,
                       robot_initialization=env.config.robot_initialization,initial_positions_mm=starts.tolist(),
                       initial_branch_ids=start_branches.tolist(),training_phase=worker_phase,
                       initial_particle_positions_mm=frames[0][1][env.num_robots:].tolist(),
                       particle_layout=env.particle_layout,flow_multiplier=env.episode_flow_multiplier,
                       clot_initialization=env.config.clot_initialization,clot_positions_mm=env.clot_positions_mm.tolist(),
                       healthy_reference_flow_ml_min=env.flow_model.inlet_flow_ml_min,
                       robot_display_scale=4,particle_display_scale=particle_display_scale,clot_surface_model=surface,
                       contact_model=env.config.contact_model,contact_distance_mm=env.config.contact_distance_mm,
                       clot_display_scale=clot_display_scale,particle_count=env.config.particle_count,
                       physical_duration_s=frames[-1][0],replay_index=replay_index,updated_at=time.time()))
            frame_times=np.asarray([f[0] for f in frames])
            first_particles=frames[0][1][env.num_robots:].copy()
            state['clock_sim']=0.;state['clock_wall']=time.monotonic()
            last_drawn=-1;last_snapshot=-np.inf;last_render=time.monotonic();rendered_frames=0
            while True:
                if not args.headless and plot._closed: return
                events()
                if state['paused']:
                    time.sleep(.03);continue
                if last_drawn<0:k=0
                elif args.headless:k=min(len(frames)-1,int(np.searchsorted(frame_times,min(10.,frame_times[-1]))))
                else:k=min(len(frames)-1,max(0,int(np.searchsorted(frame_times,play_time(),side='right')-1)))
                if k<=last_drawn:
                    time.sleep(.005);continue
                t,positions,active,masses=frames[k]
                positions=positions/scale
                # Colour tracks contact only when actual geometric proximity
                # is satisfied; exited robots are faded and remain at exits.
                distance=np.linalg.norm(positions[:env.num_robots,None]-env.clot_positions_mm[None]/scale,axis=-1)*scale
                if surface:
                    physical_pos=positions[:env.num_robots]*scale
                    edges=env.transport.nearest_edges(physical_pos)
                    solution=dict(env.solution,radius_mm=env._radii(masses))
                    contacts=env._surface_contacts(physical_pos,edges,solution,masses).any(axis=1)
                else:
                    contacts=((distance<=env.config.contact_distance_mm)&(masses>0)).any(axis=1)
                    if localized:
                        physical_pos=positions[:env.num_robots]*scale
                        edges=env.transport.nearest_edges(physical_pos)
                        _,_,_,fraction=env.transport.coordinates(physical_pos,edges,env.solution)
                        ends=env.transport.ends[edges];length=env.transport.length[edges]
                        geo=np.stack([np.minimum(route[ends[:,0]]+fraction*length,
                            route[ends[:,1]]+(1-fraction)*length) for route in env.routes],axis=1)
                        contacts=((distance<=env.config.contact_distance_mm)&
                                  (geo<=env.config.contact_distance_mm)&(masses>0)).any(axis=1)
                states=['contact' if active[i] and contacts[i] else 'transit' for i in range(env.num_robots)]
                physical=positions*scale
                pair_distance=np.linalg.norm(physical[:env.num_robots,None]-physical[None,env.num_robots:],axis=-1)
                overlaps=(pair_distance<env.config.robot_radius_mm+env.config.particle_radius_mm)
                overlaps &= active[:env.num_robots,None] & active[None,env.num_robots:]
                renderer.update(positions[:env.num_robots],states,env.clot_positions_mm/scale,masses,clot_radius/scale)
                if surface:
                    update_surfaces(masses)
                    for actor in renderer._clot_actors:actor.SetVisibility(False)
                for i,actor in enumerate(renderer._robot_actors):
                    actor.GetProperty().SetOpacity(1. if active[i] else .18)
                    actor.SetScale(.25 if state['real_size'] else 1.)
                    color=(1.,.1,.15) if overlaps[i].any() else pv.Color(
                        style.robot_contact_color if states[i]=='contact' else style.robot_color).float_rgb
                    actor.GetProperty().SetColor(*color)
                for j,actor in enumerate(particle_actors):
                    actor.SetPosition(*positions[env.num_robots+j]);actor.SetVisibility(bool(active[env.num_robots+j]))
                    actor.SetScale(1/particle_display_scale if state['real_size'] else 1.)
                    actor.GetProperty().SetColor(*((1.,.1,.15) if overlaps[:,j].any() else (.89,.93,.96)))
                start=int(np.searchsorted(frame_times,max(0.,t-30.)))
                ids=np.unique(np.linspace(start,k,min(20,k-start+1),dtype=int))
                if state['trails'] and len(ids)>1 and env.config.particle_count:
                    points=np.stack([frames[i][1][env.num_robots:] for i in ids]).transpose(1,0,2)/scale
                    trail_mesh.points=points.reshape(-1,3)
                    trail_mesh.lines=np.concatenate([np.r_[len(ids),np.arange(j*len(ids),(j+1)*len(ids))]
                                                     for j in range(env.config.particle_count)])
                    trail_actor.SetVisibility(True)
                else:trail_actor.SetVisibility(False)
                displacement=np.linalg.norm(positions[env.num_robots:]*scale-first_particles,axis=1)
                median_displacement=float(np.median(displacement)) if len(displacement) else 0.
                status_actor.SetInput(f'Checkpoint {steps:,} steps  |  t = {t:.2f} s  |  robots {active[:env.num_robots].sum()}/{env.num_robots}\n'
                    f'Removed {(4-masses.sum())/4*100:.3f}%  |  Playback {state["speed"]:g}x (visual only)\n'
                    f'Particles {active[env.num_robots:].sum()}/{env.config.particle_count} | median displacement {median_displacement:.3f} mm\n'
                    f'Layout: {env.particle_layout} | reference flow {env.flow_model.inlet_flow_ml_min*1000:.1f} uL/min\n'
                    f'Actual overlaps: {int(overlaps.sum())} (red) | body display: {"true size" if state["real_size"] else "enlarged; S toggles true size"}\n'
                    f'Cyan trails: last 30 simulated seconds | reset {episode_seed} | {worker_phase}')
                if state['orbit'] and not args.headless:plot.camera.azimuth+=2*(time.monotonic()-last_render)
                plot.render();last_render=time.monotonic();rendered_frames+=1
                if k==0:
                    snapshot('initial_scene')
                    state['clock_sim']=0.;state['clock_wall']=time.monotonic()
                if time.monotonic()-last_snapshot>=1. or args.headless or k==len(frames)-1:
                    if status_path.exists():worker_phase=json.loads(status_path.read_text()).get('phase','unknown')
                    snapshot('scene');last_snapshot=time.monotonic()
                    atomic_json(out/'playhead.json',dict(checkpoint=str(checkpoint),checkpoint_steps=steps,
                        episode_seed=episode_seed,replay_index=replay_index,frame_index=k,rendered_frames=rendered_frames,
                        displayed_sim_time_s=float(t),playback_speed=state['speed'],paused=state['paused'],
                        actual_robot_particle_overlaps=int(overlaps.sum()),true_body_display=state['real_size'],
                        particle_positions_mm=(positions[env.num_robots:]*scale).tolist(),
                        particle_displacement_mm=displacement.tolist(),median_particle_displacement_mm=median_displacement,
                        active_particles=int(active[env.num_robots:].sum()),training_phase=worker_phase,updated_at=time.time()))
                last_drawn=k
                if k==len(frames)-1 or (args.headless and k>0):break
            if args.once or args.headless: break
            replay_index+=1
    finally: renderer.close()


if __name__=='__main__':main()
