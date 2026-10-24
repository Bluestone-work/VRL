# Navigation 3D gallery — 2026-10-18

The gallery renders **42 predeclared, independent illustration episodes**, one
per registered anatomy (14) and flow mode (weak, strong, pulsatile). These are
actual frozen-policy simulator rollouts, not synthetic success animations.
All outcomes, including failures and timeouts, are retained.

## Scope and provenance

- Model: `research/runs/NAVIGATION_V1/IPPO_s41/final_model.zip`, final 1M physical
  environment-step IPPO, deterministic K18 semantic actions. SHA256 is saved in
  the gallery manifest and each scene JSON.
- Observed training coverage was only `mca_m1_lvo` because the training manifest
  iterator was sequential. Do not describe this checkpoint as trained on all
  nine registered training anatomies.
- N=3, 16 dynamic obstacles, `route_encounter` placement, historical clot sites,
  zero latency, original 300 s horizon and navigation success criteria.
- Nominal inlet speeds: weak 0.025 mm/s; strong 0.1 mm/s; pulsatile 0.05 mm/s
  with the dataset's `Variation(strength=0.75, pulsatile=True)`. These are
  simulator units. The pulsatile variation setting can also affect patient
  response parameters, so this is not a pure waveform-only causal ablation.
- Requested seed for anatomy index i: `3908000000 + 1000*i`, reused across flow
  modes. These are dedicated visualization seeds, outside the registered
  train/validation/holdout namespaces. Accepted seeds and physical scene hashes
  are recorded separately; cross-flow scenes are not asserted to have identical
  accepted geometry.
- Viewing these anatomies is explicitly a development illustration. The
  registered holdout scenario rows are not loaded or evaluated, and these
  illustration scores must not be relabelled as sealed test performance.
- Actor inputs, global routing, safety layers, physics, and success criteria
  remain unchanged. True positions are used only after execution for rendering.
- This is navigation, with lysis disabled by the existing navigation task.
  Red objects mark target sites and do not shrink as if clearance occurred.
- The existing navigation Strict field means navigation success plus cumulative
  wall contact below 1 robot.s. It is not a zero-obstacle-collision criterion.

## Outputs

Default directory: `/home/wj/桌面/VRL_导航3D_GIF_14血管_20261018`.

- `index.html`: three-column animated gallery (weak / strong / pulsatile).
- `overview_14x3.jpg`: final-frame contact sheet, 14 rows and three columns.
- `manifest.json`: all predeclared jobs and model checksum.
- `results.json`, `completion.json`: completed scene outcomes and failure list.
- Per scene: `.gif`, final `.png`, `.json` with metrics and reset provenance,
  `.npz` with every 0.1 s physical state, executed world-frame command, selected
  action, target-reached state, and cumulative contacts/collisions.
- Rendering uses the existing 3D vessel mesh and perspective camera. Markers
  are enlarged, tracks are projected with the same camera, and playback is
  accelerated. Screen-space marker overlap alone is not a physical collision.

## Reproduce

```bash
cd /home/wj/桌面/vascular_marl_local.tar.
OPENBLAS_CORETYPE=GENERIC OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LYSIS_DISABLE_EGL=1 /home/wj/miniconda3/envs/v/bin/python \
  scripts/export_navigation_gallery.py --workers 3
```

`--limit 1 --workers 1` runs the first end-to-end check. Completed GIFs are
skipped on restart; existing raw traces are reused for interrupted renders.
`--job INDEX` regenerates one scene's GIF from its trace without another rollout.
Use a new `--out` directory for a different checkpoint.

`--render-only` rebuilds the full gallery from cached traces, without any new
simulation. The final camera fits the complete lumen envelope, including wall
radii, and keeps inactive/lost cluster positions visible in the reference view.

Validation includes comparison of the first five actions and physical states
with the unmodified `IPPOAgentVecEnv.step` path on the first scene. They match.
Trace checks cover finite 3D coordinates, three clusters and sixteen obstacles,
0.1 s timestamps, and cumulative wall/collision totals matching episode metrics.
