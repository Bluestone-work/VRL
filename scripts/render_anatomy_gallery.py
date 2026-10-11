"""Gallery of the 14 vessel anatomies (pre-operative geometry, clots, cluster starts) with the PyBullet EGL renderer."""
import json, sys
from pathlib import Path
import numpy as np


def main(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
    from scripts.benchmark_lysis import LysisEpisode, dev_seeds
    from scripts.render_lysis_paper import Scene
    order = json.load(open('configs/evaluation_splits.json'))['anatomy_order']
    held = set(json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['held_out'])
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.4))
    for ax, an in zip(axes.ravel(), order):
        ep = LysisEpisode(3, an, dev_seeds(an, 1)[0], sensing='noise', flow_inlet_mm_s=.05)
        geo = dict(points=ep.env.transport.points.astype(float).tolist(), ends=np.asarray(ep.env.transport.ends).tolist(),
                   radius=np.asarray(ep.env.flow_model.healthy_radius_mm, float).tolist(),
                   clots=ep.env.clot_positions_mm.astype(float).tolist(), clot_mass0=ep.env.initial_mass.astype(float).tolist(),
                   body=float(ep.env.config.robot_radius_mm))
        sc = Scene(geo, 640, 480, vessel_alpha=.35); sc.clusters(3)
        P = np.asarray(geo['points']); sc.camera(P, None, -30.)
        img = sc.render(ep.env.positions_mm[:3].astype(float), ep.env.initial_mass.astype(float))
        ax.imshow(img); ax.set_axis_off()
        R = np.asarray(geo['radius'])
        ax.set_title(f"{an}{' (留出)' if an in held else ''}\n半径 {R.min():.2f}–{R.max():.1f} mm, 管段 {len(geo['ends'])}", fontsize=8)
        sc.close(); ep.close()
    for ax in axes.ravel()[len(order):]:
        ax.set_axis_off()
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(Path(out)/f'fig_anatomies.{e}', dpi=250)


if __name__ == '__main__':
    main(sys.argv[1])
