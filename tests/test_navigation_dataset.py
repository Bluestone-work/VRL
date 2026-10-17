import json
from pathlib import Path

from scripts.build_navigation_dataset import OUT_DIR, load_config, make_rows, validate_rows
from scripts.navigation_dataset import iter_rows


def test_navigation_manifest_is_frozen_and_disjoint():
    cfg, splits = load_config()
    rows = make_rows(cfg, splits)
    report = validate_rows(rows, cfg, splits)
    metadata = json.loads((OUT_DIR / "metadata.json").read_text())
    assert report["rows"] == metadata["row_count"] == 60480
    assert metadata["manifest_sha256"]
    assert len(list(iter_rows(split="holdout_test", clusters=3))) == 7200


def test_navigation_manifest_has_all_factor_levels():
    rows = list(iter_rows(split="train", clusters=3))
    assert {r["flow_mode"] for r in rows} == {"weak", "strong", "pulsatile"}
    assert {r["clot_pattern"] for r in rows} == {"historical_sites", "random_branches", "mixed_pattern"}
    assert {r["obstacle_density"] for r in rows} == {"low", "medium", "high", "stress"}
    assert {r["obstacle_layout"] for r in rows} == {"uniform", "route_encounter"}
    assert {r["clusters"] for r in rows} == {3}
