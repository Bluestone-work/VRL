"""Actual EXP0060 simulator replay GUI, with screenshots from identical VTK scene.

Input is a recorded trajectory, never synthetic/AI-generated artwork. Offscreen
mode renders the very same slider/size controls as the interactive GUI. The map
and ground-truth scene are for visualization ONLY, not policy observations.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--headless',action='store_true'); a=ap.parse_args()
    if a.headless:
        os.environ.pop('DISPLAY',None)
        os.environ.setdefault('PYVISTA_OFF_SCREEN','true')
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
    import pyvista as pv
    from marl.vascular_option_rl import OPTIONS
    data=json.loads(Path(a.trace).read_text()); frames=data['trace']; geo=data['geometry']; m=data['metrics']
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    P=np.array(geo['points']); ends=np.array(geo['ends']); radii=np.array(geo['healthy_radius'])
    plot=pv.Plotter(off_screen=a.headless,window_size=(1600,1000),title='Vascular DRL | simulator replay')
    plot.set_background('#0c1524',top='#21354a')
    plot.enable_depth_peeling(number_of_peels=100,occlusion_ratio=0.)
    plot.ren_win.SetMultiSamples(0)
    # Join degree-two segments for a smooth display tube; keep the measured
    # centreline samples and healthy radii at their original physical scale.
    adj={i:[] for i in range(len(P))}
    for u,v in ends:adj[u].append(v); adj[v].append(u)
    used=set(); paths=[]
    for start in sorted(adj,key=lambda i:len(adj[i])==2):
        for nxt in adj[start]:
            if tuple(sorted((start,nxt))) in used:continue
            path=[start,nxt]; used.add(tuple(sorted((start,nxt))))
            while len(adj[path[-1]])==2:
                candidates=[j for j in adj[path[-1]] if tuple(sorted((path[-1],j))) not in used]
                if not candidates:break
                nxt=candidates[0]; used.add(tuple(sorted((path[-1],nxt)))); path.append(nxt)
            paths.append(path)
    meshes=[]
    for path in paths:
        line=pv.PolyData(P[path]); line.lines=np.r_[len(path),np.arange(len(path))]
        line['radius_mm']=radii[path]
        meshes.append(line.tube(scalars='radius_mm',absolute=True,n_sides=32,capping=False))
    # Preserve tube normals directly; VTK 9.6's automatic smooth-surface filter
    # fails on the combined unstructured strip mesh in this runtime.
    wall=pv.MultiBlock(meshes).combine()
    plot.add_mesh(wall,color='#9ebed0',opacity=.35,smooth_shading=False,specular=.3,ambient=.35,name='vessel_wall')
    lines=pv.PolyData(P); lines.lines=np.c_[np.full(len(ends),2),ends].ravel()
    plot.add_mesh(lines,color='#627f9a',opacity=.6,line_width=1)
    plot.add_axes(); plot.add_text('VASCULAR MEMORY OPTION RL',position=(26,943),font_size=24,color='#edf6ff')
    plot.add_text('Actual simulator replay  |  union-of-tubes physics  |  predeployed clusters',
                  position=(28,910),font_size=12,color='#bed1df')
    plot.add_text(f"{m['anatomy']}   N={m['clusters']}   scene seed={m['seed']}\n"
                  f"Method: {m['method']}   checkpoint: {(m.get('checkpoint_sha256') or 'fixed control')[:16]}",
                  position=(28,853),font_size=11,color='#d4e3ee')
    plot.add_text('Blue / green / gold: clusters   Red: clot location markers   White: passive particles\n'
                  'Default display: clusters 4x, particles 4x. Clot markers are not reconstructed clot surfaces.\n'
                  'Truth shown for inspection only. Actor uses measured local observations. Distances in mm.',
                  position=(28,112),font_size=10,color='#b7cbd9')
    plot.add_text('Drag to orbit  |  Scroll to zoom  |  Time slider: replay  |  Checkbox: true body size',
                  position=(28,20),font_size=11,color='#e4edf5')
    state={'index':0,'true_size':False}; colors=['#35c4ff','#56e5ac','#ffcc59']
    def draw(index):
        k=min(max(int(round(index)),0),len(frames)-1); state['index']=k; frame=frames[k]
        q=np.array(frame['positions']); part=np.array(frame['particles']); masses=np.array(frame['masses'])
        size=1. if state['true_size'] else 4.
        for i,pos in enumerate(q):
            plot.add_mesh(pv.Sphere(radius=.08*size,center=pos),color=colors[i],name=f'robot{i}',render=False)
            trail=np.array([f['positions'][i] for f in frames[:k+1]])
            if len(trail)>1:
                plot.add_mesh(pv.lines_from_points(trail),color=colors[i],line_width=3,opacity=.9,name=f'trail{i}',render=False)
        if len(part):
            cloud=pv.PolyData(part)
            spheres=cloud.glyph(scale=False,orient=False,geom=pv.Sphere(radius=.02*size,theta_resolution=10,phi_resolution=10))
            plot.add_mesh(spheres,color='#eff6ff',name='particles',render=False)
        for j,pos in enumerate(geo['clot_positions']):
            frac=float(masses[j]/geo['initial_mass'][j])
            plot.add_mesh(pv.Sphere(radius=.25,center=pos),color='#ec5365' if frac>0 else '#485e70',
                          opacity=.8 if frac>0 else .25,name=f'clot{j}',render=False)
        d=np.linalg.norm(q[:,None]-q[None],axis=-1)
        spacing=float(d[np.triu_indices(len(q),1)].min()) if len(q)>1 else None
        labels='\n'.join(f"C{i}: {OPTIONS[o]}{' / TPG HOLD' if frame['holds'][i] else ''}" for i,o in enumerate(frame['options']))
        removal=100*(1-masses.sum()/np.sum(geo['initial_mass']))
        text=f"Time {frame['time']:.1f} s   |   Removed {removal:.1f}%\n"
        text+=f"Measured-control options:\n{labels}\n"
        text+=f"True nearest spacing: {spacing:.2f} mm" if spacing is not None else 'Single-cluster sequential baseline'
        plot.add_text(text,position=(28,645),font_size=12,color='#eff5fa',name='status',render=False)
        plot.render()
    def sizes(value):state['true_size']=bool(value); draw(state['index'])
    plot.add_checkbox_button_widget(sizes,value=False,position=(1460,120),size=25,color_on='#56e5ac')
    plot.add_text('True size',position=(1430,157),font_size=10,color='white')
    slider=plot.add_slider_widget(draw,[0,len(frames)-1],value=0,title='',pointa=(.15,.06),pointb=(.85,.06),
                                 style='modern',slider_width=.018,tube_width=.005,color='#b7cbd9')
    plot.camera_position='iso'; plot.reset_camera(); plot.camera.zoom(1.55)
    times=np.array([f['time'] for f in frames]); provenance=[]
    if a.headless:
        # Initialize the VTK render window; render() alone before show() omits
        # translucent scene actors on this offscreen driver.
        plot.show(auto_close=False,interactive=False)
        for label,t in [('start',0.),('10s',10.),('30s',30.),('last',float(times[-1]))]:
            k=int(np.argmin(abs(times-t))); draw(k); plot.render()
            slider.GetRepresentation().SetValue(k)
            slider.GetRepresentation().BuildRepresentation()
            plot.render()
            path=out/f'gui_{label}.png'; plot.screenshot(str(path))
            provenance.append(dict(file=path.name,frame=k,time_s=float(times[k]),scene_hash=m['scenario_hash'],
                                   checkpoint_sha256=m.get('checkpoint_sha256'),source_trace=str(Path(a.trace).resolve())))
        (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)); plot.close()
    else:plot.show()


if __name__=='__main__':main()
