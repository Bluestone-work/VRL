"""Build a readable index for the slow GIF/MP4 matrix media."""
from __future__ import annotations

from pathlib import Path


ROOT = Path("experiments/full_matrix_20260906")


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    media_root = ROOT / "media_slow_v2" if (ROOT / "media_slow_v2").exists() else ROOT / "media_slow"
    methods = [line.split("|", 1)[0] for line in read_lines(ROOT / "methods.tsv")]
    scenarios = read_lines(ROOT / "scenarios.txt")
    total_gif = len(list(media_root.glob("*/*/*.gif")))
    total_mp4 = len(list(media_root.glob("*/*/*.mp4")))

    lines = [
        "# 慢速媒体索引",
        "",
        f"> 媒体根目录：`{media_root.name}/`；每个仿真步保留，固定镜头，5 FPS，最多 300 帧，约 60 秒/回合。",
        f"> 当前完成度：GIF `{total_gif}`，MP4 `{total_mp4}`；完整矩阵目标为 GIF `220`、MP4 `220`。",
        "> GIF 适合快速浏览，MP4 适合逐帧暂停；两者 HUD 信息一致。",
        "",
    ]
    for method in methods:
        lines.extend([f"## {method}", ""])
        for scenario in scenarios:
            gif = media_root / method / scenario / f"{method}.gif"
            mp4 = media_root / method / scenario / f"{method}.mp4"
            gif_ref = f"[`GIF`]({gif.relative_to(ROOT).as_posix()})" if gif.exists() else "GIF 待生成"
            mp4_ref = f"[`MP4`]({mp4.relative_to(ROOT).as_posix()})" if mp4.exists() else "MP4 待生成"
            lines.append(f"- `{scenario}` — {gif_ref} / {mp4_ref}")
        lines.append("")

    (ROOT / "MEDIA_SLOW_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ROOT / 'MEDIA_SLOW_INDEX.md'}: gif={total_gif} mp4={total_mp4}")


if __name__ == "__main__":
    main()
