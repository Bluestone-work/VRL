# Image-based perception chain: paired probe (2026-10-05)

Question: does replacing the Gaussian-noise position model by an image -> detection -> tracking chain
(marl/image_sensing.py) change the deployable method's outcomes? Dev seeds k=0..4, 14 anatomies,
N=1 (pursuit_dep) and N=3 (pursuit_tpg_dep), union physics, identical scenes, horizon 300 s.

How the references obtained images (checked against the saved full texts):
- Medany 2025: simulation = Pygame, rendered 64x64 RGB synthetic images; real images only from their own
  physical set-up (inverted microscope + DSLR, 6-18 fps), processed by SAM segmentation, morphology,
  adaptive threshold, cluster detection and CSRT tracking; sim pretraining then on-hardware fine-tuning.
- An 2026 (Turbo): simulation policy input = state with N(0, 0.025) observation noise + multi-level domain
  randomisation; deployment uses YOLOv5 trained on 110 own camera images (augmented to 2200), image-to-world
  calibration. No public dataset in either paper.
Our chain follows the same pattern: synthetic rendering (PSF, read/shot noise, motion blur, latency), a
pixel-level detector, biplane 3-D fusion and tracking; controllers never read simulator positions.

Chain versions (each fixed from an observed failure, rows kept):
- v1 nearest-blob association (rows_v1_nearest_assoc.jsonl): track swaps when two clusters overlap -> N=3 -7.1 pp.
- v2 Hungarian on paired 3-D points (rows_v2_hungarian3d.jsonl): biplane ghost when two clusters share x and
  one more coordinate; a cluster was steered from a wrong z and washed out.
- v3 (rows.jsonl): joint hypothesis assignment over all tracks, shared (merged) blobs allowed with a penalty,
  coordinates measured only by a merged blob coast on pre-merge velocity, gate widened after a merge.

v3 results, image minus noise (paired bootstrap 95 % CI):

| N | metric | noise | image | diff [CI] |
|---|---|---:|---:|---|
| 1 | Safe Success | 78.6 | 84.3 | +5.7 [-2.9, +14.3] |
| 1 | particle events | 0.17 | 0.09 | -0.09 [-0.24, +0.07] |
| 3 | Safe Success | 80.0 | 77.1 | -2.9 [-10.0, +4.3] |
| 3 | particle events | 0.56 | 0.94 | +0.39 [+0.10, +0.72] |
| 3 | removal AUC | 0.127 | 0.229 | +0.10 [+0.08, +0.13] |

Perception error (active clusters): mean 0.13-0.14 mm, p95 0.15-0.16 mm (dominated by the one-frame
latency at ~1 mm/s), max 0.89 mm. No track divergence in v3.
Open item: at N=3 particle contacts rise significantly: particles are sub-pixel (radius 20 um, 1 px) and a
particle merged into a cluster's blob is invisible exactly when it is closest. This is the next target.
