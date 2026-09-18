#!/usr/bin/env python3
"""Export the twenty parameterised vascular scenes as printable STL atlases."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "vascular_print_atlas_20260906"
SCENARIOS = [
    "straight", "bifurcation", "anastomosis", "stenotic", "multilevel",
    "mca_stroke", "pulmonary_saddle", "coronary_lm_bifurcation",
    "coronary_rca", "iliac_may_thurner", "popliteal_calf_dvt", "mca_m1_lvo",
    "ica_siphon", "ica_terminus_t", "carotid_bifurcation", "basilar_vertebral",
    "cerebral_venous_sinus", "sma_embolism", "femoropopliteal_pad",
    "renal_artery",
]
TARGET_MIN_LUMEN_DIAMETER = 2.4
DEFAULT_WALL_MM = 1.0
TARGET_LONGEST_MM = 150.0
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

sys.path.insert(0, str(ROOT))
from environments.vessel_geometry import build_vessel_tree


def _ensure_dirs() -> None:
    for name in ("meshes", "renders", "drawings", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def _write_stl(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(polydata)
    if not writer.Write():
        raise RuntimeError(f"failed to write {path}")


def _write_obj(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def _polydata_stats(polydata: vtk.vtkPolyData) -> dict:
    points = vtk_to_numpy(polydata.GetPoints().GetData()) if polydata.GetPoints() else np.empty((0, 3))
    edge_counts: dict[tuple[int, int], int] = {}
    for cell_idx in range(polydata.GetNumberOfPolys()):
        cell = polydata.GetCell(cell_idx)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        for j in range(len(ids)):
            edge = tuple(sorted((ids[j], ids[(j + 1) % len(ids)])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold = sum(1 for count in edge_counts.values() if count > 2)
    bounds = polydata.GetBounds()
    return {
        "vertices": int(polydata.GetNumberOfPoints()),
        "triangles": int(polydata.GetNumberOfPolys()),
        "bounds_mm": [float(x) for x in bounds] if bounds else [],
        "size_mm": [float(bounds[1] - bounds[0]), float(bounds[3] - bounds[2]), float(bounds[5] - bounds[4])] if bounds else [],
        "boundary_edges": int(boundary),
        "nonmanifold_edges": int(nonmanifold),
        "watertight_edge_test": bool(boundary == 0 and nonmanifold == 0),
    }


def _resample_axis(points: np.ndarray, radii: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-9:
        return points[:1], radii[:1]
    count = max(2, int(math.ceil(cumulative[-1] / max(spacing, 1e-6))) + 1)
    samples = np.linspace(0.0, cumulative[-1], count)
    sampled_points = np.column_stack([np.interp(samples, cumulative, points[:, axis]) for axis in range(3)])
    sampled_radii = np.interp(samples, cumulative, radii)
    return sampled_points.astype(np.float32), sampled_radii.astype(np.float32)


def _tree_to_print_coordinates(tree) -> tuple[np.ndarray, np.ndarray, float, str]:
    points = np.asarray(tree.points, dtype=np.float64)
    radii = np.asarray(tree.radii, dtype=np.float64)
    if hasattr(tree, "mm_per_unit") and getattr(tree, "mm_per_unit", 0.0):
        base_mm_per_unit = float(tree.mm_per_unit)
        coordinate_basis = "anatomical nominal millimetres"
    else:
        extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
        base_mm_per_unit = TARGET_LONGEST_MM / max(extent, 1e-9)
        coordinate_basis = "synthetic unit-cube mapped to laboratory millimetres"
    points_mm = points * base_mm_per_unit
    radii_mm = radii * base_mm_per_unit
    ext = np.max(points_mm.max(axis=0) - points_mm.min(axis=0))
    length_scale = TARGET_LONGEST_MM / max(ext, 1e-9)
    diameter_scale = TARGET_MIN_LUMEN_DIAMETER / max(2.0 * float(radii_mm.min()), 1e-9)
    print_scale = max(1.0, length_scale, diameter_scale)
    points_mm *= print_scale
    radii_mm *= print_scale
    return points_mm.astype(np.float32), radii_mm.astype(np.float32), float(print_scale), coordinate_basis


def _voxel_mesh(tree, points_mm: np.ndarray, radii_mm: np.ndarray, hollow: bool = False) -> tuple[vtk.vtkPolyData, dict]:
    outer_radii = radii_mm + DEFAULT_WALL_MM
    branch_samples = []
    all_sample_points = []
    all_sample_radii = []
    all_sample_inner_radii = []
    for branch in tree.branches:
        points = points_mm[branch.start: branch.stop + 1]
        lumen_radii = radii_mm[branch.start: branch.stop + 1]
        outer_branch_radii = outer_radii[branch.start: branch.stop + 1]
        sampled_points, sampled_radii = _resample_axis(points, outer_branch_radii, spacing=0.35)
        branch_samples.append((sampled_points, sampled_radii))
        all_sample_points.append(sampled_points)
        all_sample_radii.append(sampled_radii)
        all_sample_inner_radii.append(np.interp(
            np.linspace(0.0, 1.0, len(sampled_points)),
            np.linspace(0.0, 1.0, len(lumen_radii)), lumen_radii,
        ).astype(np.float32))
    axis_points = np.vstack(all_sample_points)
    axis_radii = np.concatenate(all_sample_radii)
    axis_inner_radii = np.concatenate(all_sample_inner_radii)
    voxel = float(np.clip(np.max(axis_points.max(axis=0) - axis_points.min(axis=0)) / 220.0, 0.30, 0.80))
    margin = float(axis_radii.max() + 2.0 * voxel)
    lo = axis_points.min(axis=0) - margin
    hi = axis_points.max(axis=0) + margin
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    dims = np.maximum(dims, 8)
    hi = lo + voxel * (dims - 1)
    tree_points = cKDTree(axis_points)
    volume = np.zeros(tuple(int(x) for x in dims), dtype=np.uint8)
    xs = lo[0] + voxel * np.arange(dims[0])
    ys = lo[1] + voxel * np.arange(dims[1])
    for z_start in range(0, int(dims[2]), 8):
        z_stop = min(int(dims[2]), z_start + 8)
        zs = lo[2] + voxel * np.arange(z_start, z_stop)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        distances, indices = tree_points.query(grid, workers=1)
        outer = axis_radii[indices] + 0.58 * voxel
        occupied = distances <= outer
        if hollow:
            inner = np.maximum(axis_inner_radii[indices] - 0.58 * voxel, 0.0)
            occupied &= distances >= inner
        volume[:, :, z_start:z_stop] = occupied.reshape((int(dims[0]), int(dims[1]), z_stop - z_start)).astype(np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.SetOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    image.SetSpacing(voxel, voxel, voxel)
    scalars = numpy_to_vtk(volume.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(scalars)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.5)
    contour.Update()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(10)
    smooth.SetPassBand(0.12)
    smooth.BoundarySmoothingOn()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(smooth.GetOutputPort())
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(tri.GetOutput())
    return result, {
        "voxel_mm": voxel,
        "grid": [int(x) for x in dims],
        "wall_thickness_mm": DEFAULT_WALL_MM,
        "model_type": "closed_hollow_wall_candidate" if hollow else "closed_solid_external_phantom",
    }


def _solid_mesh(tree, points_mm: np.ndarray, radii_mm: np.ndarray) -> tuple[vtk.vtkPolyData, dict]:
    return _voxel_mesh(tree, points_mm, radii_mm, hollow=False)


def _hollow_mesh(tree, points_mm: np.ndarray, radii_mm: np.ndarray) -> tuple[vtk.vtkPolyData, dict]:
    return _voxel_mesh(tree, points_mm, radii_mm, hollow=True)


def _clot_markers(tree, points_mm: np.ndarray) -> tuple[np.ndarray, list[str]]:
    markers = []
    labels = []
    territory = getattr(tree, "territory", None)
    segment_index = getattr(tree, "segment_index", {})
    if territory is not None:
        for site in territory.clot_sites:
            branch_id = segment_index.get(site.segment)
            if branch_id is None:
                continue
            branch = tree.branches[branch_id]
            station = int(round(branch.start + np.clip(site.position, 0.0, 1.0) * (branch.stop - branch.start)))
            markers.append(points_mm[station])
            labels.append(f"{site.segment} @ {site.position:.2f}: {site.mechanism}")
    else:
        terminal = [branch for branch in tree.branches if not any(other.parent == branch.branch_id for other in tree.branches)]
        for branch in terminal[:3]:
            station = int(round(branch.start + 0.72 * (branch.stop - branch.start)))
            markers.append(points_mm[station])
            labels.append(f"legacy nominal marker, branch {branch.branch_id}")
    return np.asarray(markers, dtype=np.float32).reshape((-1, 3)), labels


def _render_scene(scene: dict, path: Path) -> None:
    points = scene["points_mm"]
    tree = scene["tree"]
    markers = scene["clot_markers"]
    fig = plt.figure(figsize=(11.7, 8.3))
    grid = fig.add_gridspec(2, 2, top=0.88, left=0.05, right=0.98, bottom=0.08, hspace=0.24, wspace=0.14)
    views = [("3D print phantom", (24, 34)), ("Top projection", (0, 90)), ("Side projection", (90, 0)), ("Front projection", (0, 0))]
    colours = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(tree.branches), 1)))
    for axis, (title, view) in zip(fig.axes if fig.axes else [], views):
        pass
    for idx, (title, view) in enumerate(views):
        axis = fig.add_subplot(grid[idx // 2, idx % 2], projection="3d")
        for branch, colour in zip(tree.branches, colours):
            seg = points[branch.start: branch.stop + 1]
            width = max(1.0, float(np.mean(scene["radii_mm"][branch.start: branch.stop + 1])) * 0.20)
            axis.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colour, linewidth=width, alpha=0.9)
        if len(markers):
            axis.scatter(markers[:, 0], markers[:, 1], markers[:, 2], s=34, c="#d62728", marker="x", depthshade=False)
        axis.view_init(elev=view[1], azim=view[0])
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("X (mm)", fontsize=7)
        axis.set_ylabel("Y (mm)", fontsize=7)
        axis.set_zlabel("Z (mm)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.set_box_aspect(np.ptp(points, axis=0) + 1e-6)
    fig.suptitle(f"{scene['name']}  |  printable solid vascular phantom", fontsize=16, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scene_page(pdf: PdfPages, scene: dict) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.05, 0.94, f"{scene['name']}  /  {scene['description']}", fontsize=18, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD))
    fig.text(0.05, 0.905, "参数化合成血管体外打印模型（solid positive phantom）", fontsize=11, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
    image = plt.imread(scene["render_path"])
    axis = fig.add_axes([0.03, 0.29, 0.60, 0.57])
    axis.imshow(image)
    axis.axis("off")
    axis2 = fig.add_axes([0.68, 0.28, 0.29, 0.60])
    axis2.axis("off")
    rows = [
        ("类别 / category", scene["category"]),
        ("打印 STL", scene["stl_path"].name),
        ("OBJ 检查文件", scene["obj_path"].name),
        ("薄壁空腔候选 STL", scene["flow_stl_path"].name),
        ("打印倍率（相对生成几何）", f"{scene['print_scale']:.3f}x"),
        ("模型包围盒", " × ".join(f"{v:.1f} mm" for v in scene["mesh_stats"]["size_mm"])),
        ("中心线范围", " × ".join(f"{v:.1f} mm" for v in scene["centerline_size_mm"])),
        ("最小/最大管腔直径", f"{scene['lumen_diameter_min_mm']:.2f} / {scene['lumen_diameter_max_mm']:.2f} mm"),
        ("分支数量", str(len(scene["tree"].branches))),
        ("血栓标记数量", str(len(scene["clot_markers"]))),
        ("网格质量", "封闭边测试通过" if scene["mesh_stats"]["watertight_edge_test"] else "需要修复"),
    ]
    y = 0.98
    for label, value in rows:
        axis2.text(0.0, y, label, fontsize=9, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD), va="top")
        axis2.text(0.0, y - 0.032, str(value), fontsize=8.5, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT), va="top", wrap=True)
        y -= 0.092
    note = (
        "打印建议：STL 单位按毫米解释；实体正模默认壁厚参数为 "
        f"{DEFAULT_WALL_MM:.1f} mm。薄壁空腔候选 STL 需要在切片器或后处理中确认/打通端口，不应未经泄漏测试直接接泵。\n"
        "血栓红叉仅表示训练/解剖数据中的名义位置，不是额外实体。该文件不是患者级影像重建、医疗器械生产文件或临床使用模型。"
    )
    fig.text(0.05, 0.16, note, fontsize=9, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT), va="top", wrap=True)
    fig.text(0.05, 0.055, f"源几何：{scene['coordinate_basis']}；生成日期：2026-09-06；建议先打印 1 个小比例验证件，再批量打印。", fontsize=8, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
    pdf.savefig(fig)
    plt.close(fig)


def _write_readme(scenes: list[dict]) -> None:
    lines = [
        "# Vascular Print Atlas（20 场景）",
        "",
        "本目录由 `scripts/export_vascular_print_atlas.py` 生成，包含 20 个参数化合成血管树的 STL/OBJ、尺寸清单、场景图和总 PDF。",
        "",
        "## 打印口径",
        "- STL 坐标单位按毫米解释；每个场景采用各自的均匀打印倍率，避免改变分支角度和管径比例。",
        f"- 默认最小管腔直径目标为 {TARGET_MIN_LUMEN_DIAMETER:.1f} mm，外形按 {DEFAULT_WALL_MM:.1f} mm 的实体外扩生成。",
        "- `*_solid_phantom.stl` 是封闭实心外形，可用于教学、路径规划、机器人可视化和几何验证。",
        "- `*_hollow_wall_candidate.stl` 是薄壁空腔候选，可用于后续柔性材料/液路实验；端口需要后处理确认，分叉和泄漏仍需实物验证。",
        "",
        "## 文件对应",
        "每个场景目录内有 `*_solid_phantom.stl`、`*_hollow_wall_candidate.stl`、OBJ 检查文件和 `metadata.json`。",
        "",
        "## 科研边界",
        "这些场景是参数化合成血管，不等价于患者 CTA/MRA 的真实重建，不构成医疗器械文件，也没有经过临床材料、生物相容性或尺寸精度认证。",
        "",
        "## 20 个场景",
    ]
    lines.extend(f"- `{scene['name']}`：{scene['description']}" for scene in scenes)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_pdf(scenes: list[dict]) -> None:
    pdf_path = OUT / "PRINT_ATLAS.pdf"
    with PdfPages(pdf_path) as pdf:
        cover = plt.figure(figsize=(11.7, 8.3))
        cover.text(0.08, 0.78, "20 场景血管体外打印图谱", fontsize=28, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD))
        cover.text(0.08, 0.70, "Vascular Print Atlas | 20 parameterised vascular phantoms", fontsize=16, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
        cover.text(0.08, 0.58, "用途：3D 打印、机器人路径实验、几何验证、教学与可视化", fontsize=15, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
        cover.text(0.08, 0.49, "输出：20 个封闭 STL + OBJ 检查文件 + 尺寸/网格质量记录 + 逐场景图纸", fontsize=13, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
        cover.text(0.08, 0.32, "重要限制：本图谱使用参数化合成几何；默认是实心外形，不是患者级模型，也不是临床器械生产文件。", fontsize=12, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT), wrap=True)
        cover.text(0.08, 0.12, "生成日期：2026-09-06  |  项目：vascular_marl_local", fontsize=11, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT))
        pdf.savefig(cover)
        plt.close(cover)
        intro = plt.figure(figsize=(11.7, 8.3))
        intro.text(0.05, 0.94, "打印与验证说明", fontsize=22, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD))
        text = (
            "1. 直接打印：使用每个场景目录内的 *_solid_phantom.stl，切片器单位选择 mm。\n"
            "2. 方向：优先让最长主干水平放置；分叉朝上可减少支撑，但应依据实际打印机测试。\n"
            "3. 材料：刚性树脂适合几何验证；TPU/硅胶更接近柔性触感，但需要单独验证最小壁厚和回弹。\n"
            "4. 网格：本批次以体素并集+等值面提取生成，报告中给出体素尺寸、三角面数和封闭边测试。\n"
            "5. 空腔边界：当前 STL 是实心外形 phantom；不要把它直接接入液路。若做流体实验，需二次生成贯通腔体并做压力/泄漏测试。\n"
            "6. 血栓：红色叉号是名义训练/解剖位置，用于对照，不代表 STL 中已经植入了可溶解血栓实体。\n\n"
            "建议验收顺序：先在切片器打开 straight、bifurcation、pulmonary_saddle 三个文件，检查尺寸和分叉；再批量导入其余 17 个。"
        )
        intro.text(0.07, 0.80, text, fontsize=13, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT), va="top", linespacing=1.7)
        pdf.savefig(intro)
        plt.close(intro)
        for scene in scenes:
            _scene_page(pdf, scene)
        qa = plt.figure(figsize=(11.7, 8.3))
        qa.text(0.05, 0.94, "全量网格质量与文件清单", fontsize=22, fontproperties=matplotlib.font_manager.FontProperties(fname=FONT_BOLD))
        headers = ["Scenario", "STL MB", "Triangles", "Size mm", "Boundary edges", "Status"]
        rows = []
        for scene in scenes:
            stats = scene["mesh_stats"]
            rows.append([
                scene["name"], f"{scene['stl_path'].stat().st_size / 1e6:.2f}", str(stats["triangles"]),
                "×".join(f"{x:.0f}" for x in stats["size_mm"]), str(stats["boundary_edges"]),
                "PASS" if stats["watertight_edge_test"] else "REVIEW",
            ])
        table = qa.add_axes([0.04, 0.10, 0.92, 0.78])
        table.axis("off")
        rendered = table.table(cellText=rows, colLabels=headers, loc="upper left", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8)
        rendered.scale(1, 1.55)
        pdf.savefig(qa)
        plt.close(qa)


def main() -> None:
    _ensure_dirs()
    scenes = []
    for index, name in enumerate(SCENARIOS):
        rng = np.random.default_rng(20260906 + index)
        tree = build_vessel_tree(name, rng)
        points_mm, radii_mm, print_scale, coordinate_basis = _tree_to_print_coordinates(tree)
        mesh, mesh_extra = _solid_mesh(tree, points_mm, radii_mm)
        hollow_mesh, hollow_extra = _hollow_mesh(tree, points_mm, radii_mm)
        scene_dir = OUT / "meshes" / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        stl_path = scene_dir / f"{name}_solid_phantom.stl"
        obj_path = scene_dir / f"{name}_solid_phantom.obj"
        flow_stl_path = scene_dir / f"{name}_hollow_wall_candidate.stl"
        _write_stl(mesh, stl_path)
        _write_obj(mesh, obj_path)
        _write_stl(hollow_mesh, flow_stl_path)
        markers, marker_labels = _clot_markers(tree, points_mm)
        stats = _polydata_stats(mesh)
        hollow_stats = _polydata_stats(hollow_mesh)
        description = getattr(getattr(tree, "territory", None), "description", f"{name} parameterised vascular geometry")
        category = "anatomical territory" if hasattr(tree, "territory") else ("generated topology" if name in {"multilevel", "mca_stroke"} else "legacy topology")
        render_path = OUT / "renders" / f"{name}.png"
        centerline_size = np.ptp(points_mm, axis=0)
        scene = {
            "name": name,
            "description": description,
            "category": category,
            "tree": tree,
            "points_mm": points_mm,
            "radii_mm": radii_mm,
            "print_scale": print_scale,
            "coordinate_basis": coordinate_basis,
            "clot_markers": markers,
            "clot_labels": marker_labels,
            "stl_path": stl_path,
            "obj_path": obj_path,
            "flow_stl_path": flow_stl_path,
            "mesh_stats": {**stats, **mesh_extra},
            "flow_mesh_stats": {**hollow_stats, **hollow_extra},
            "centerline_size_mm": [float(x) for x in centerline_size],
            "lumen_diameter_min_mm": float(2.0 * radii_mm.min()),
            "lumen_diameter_max_mm": float(2.0 * radii_mm.max()),
            "render_path": render_path,
        }
        _render_scene(scene, render_path)
        metadata = {key: value for key, value in scene.items() if key not in {"tree", "points_mm", "radii_mm", "clot_markers", "render_path", "stl_path", "obj_path", "flow_stl_path"}}
        metadata["stl_file"] = stl_path.name
        metadata["obj_file"] = obj_path.name
        metadata["flow_wall_stl_file"] = flow_stl_path.name
        metadata["clot_markers_mm"] = markers.tolist()
        metadata["clot_marker_labels"] = marker_labels
        (scene_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        scenes.append(scene)
        print(f"[{index + 1:02d}/{len(SCENARIOS)}] {name}: solid={stats['triangles']} triangles, solid_watertight={stats['watertight_edge_test']}, flow_boundary_edges={hollow_stats['boundary_edges']}", flush=True)
    manifest = []
    for scene in scenes:
        stl = scene["stl_path"]
        manifest.append({
            "scenario": scene["name"],
            "stl": str(stl.relative_to(OUT)),
            "obj": str(scene["obj_path"].relative_to(OUT)),
            "flow_wall_stl": str(scene["flow_stl_path"].relative_to(OUT)),
            "render": str(scene["render_path"].relative_to(OUT)),
            "sha256_stl": hashlib.sha256(stl.read_bytes()).hexdigest(),
            "mesh": scene["mesh_stats"],
            "flow_mesh": scene["flow_mesh_stats"],
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "scenes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "category", "print_scale", "min_lumen_diameter_mm", "max_lumen_diameter_mm", "branches", "triangles", "boundary_edges", "watertight", "flow_boundary_edges", "stl", "flow_wall_stl"])
        writer.writeheader()
        for scene in scenes:
            writer.writerow({
                "scenario": scene["name"], "category": scene["category"], "print_scale": f"{scene['print_scale']:.6f}",
                "min_lumen_diameter_mm": f"{scene['lumen_diameter_min_mm']:.4f}", "max_lumen_diameter_mm": f"{scene['lumen_diameter_max_mm']:.4f}",
                "branches": len(scene["tree"].branches), "triangles": scene["mesh_stats"]["triangles"], "boundary_edges": scene["mesh_stats"]["boundary_edges"],
                "watertight": scene["mesh_stats"]["watertight_edge_test"], "flow_boundary_edges": scene["flow_mesh_stats"]["boundary_edges"],
                "stl": str(scene["stl_path"].relative_to(OUT)), "flow_wall_stl": str(scene["flow_stl_path"].relative_to(OUT)),
            })
    _write_readme(scenes)
    _make_pdf(scenes)
    summary = {"scenes": len(scenes), "watertight_pass": sum(s["mesh_stats"]["watertight_edge_test"] for s in scenes), "pdf": str((OUT / "PRINT_ATLAS.pdf").relative_to(ROOT))}
    (OUT / "logs" / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
