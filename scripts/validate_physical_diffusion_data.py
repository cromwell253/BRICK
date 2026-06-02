"""
Validate graph-diffusion semi-synthetic Intel Lab data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "physical_diffusion_data"


def load_offsets(data_dir: Path) -> list[tuple[int, Path, np.lib.npyio.NpzFile]]:
    items = []
    paths = sorted(data_dir.glob("physical_diffusion_offset_*.npz"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for path in paths:
        offset = int(path.stem.rsplit("_", 1)[1])
        items.append((offset, path, np.load(path, allow_pickle=True)))
    if not items:
        raise FileNotFoundError(f"No physical_diffusion_offset_*.npz files in {data_dir}")
    return items


def wall_group_indices(walls: np.ndarray, source_indices: np.ndarray, n_sensors: int) -> dict[str, np.ndarray]:
    min_walls = np.min(walls[:, source_indices], axis=1)
    source_mask = np.zeros(n_sensors, dtype=bool)
    source_mask[source_indices] = True
    return {
        "source_room_1_2": np.where(source_mask)[0],
        "adjacent_wall1": np.where((~source_mask) & (min_walls <= 1))[0],
        "far_wall2plus": np.where((~source_mask) & (min_walls >= 2))[0],
    }


def mean_curve(delta: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.full(delta.shape[0], np.nan, dtype=np.float32)
    return delta[:, indices].mean(axis=1)


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def write_room_mean_plots(items, out_dir: Path) -> None:
    fig, axes = plt.subplots(len(items), 1, figsize=(9, 2.2 * len(items)), dpi=180, sharex=True)
    if len(items) == 1:
        axes = [axes]
    for ax, (offset, _path, data) in zip(axes, items):
        delta = data["delta_temperature"].astype(np.float64)
        source = data["source_room_indices"].astype(int)
        groups = wall_group_indices(data["walls"].astype(np.float64), source, delta.shape[1])
        x = np.arange(delta.shape[0])
        for name, color in [
            ("source_room_1_2", "#0072B2"),
            ("adjacent_wall1", "#009E73"),
            ("far_wall2plus", "#D55E00"),
        ]:
            ax.plot(x, mean_curve(delta, groups[name]), label=name, linewidth=1.8, color=color)
        ax.set_title(f"offset={offset} degC")
        ax.set_ylabel("mean delta (degC)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time step")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "physical_diffusion_room_mean_curves.png", bbox_inches="tight")
    fig.savefig(out_dir / "physical_diffusion_room_mean_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def write_schedule_plot(items, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=180)
    for offset, _path, data in items:
        delta = data["delta_temperature"].astype(np.float64)
        source = data["source_room_indices"].astype(int)
        ax.plot(np.arange(delta.shape[0]), mean_curve(delta, source), linewidth=1.7, label=f"{offset} degC")
    ax.set_xlabel("time step")
    ax.set_ylabel("Room 1/2 mean delta (degC)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "physical_diffusion_offset_schedule.png", bbox_inches="tight")
    fig.savefig(out_dir / "physical_diffusion_offset_schedule.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.data_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_offsets(args.data_dir)

    write_room_mean_plots(items, out_dir)
    write_schedule_plot(items, out_dir)

    wall_rows = []
    smooth_rows = []
    report = [
        "# Physical Diffusion Manifest Check",
        "",
        f"Data directory: `{args.data_dir}`",
        "",
        "| Offset | Late source mean | Relative error | Non-source/source | Max step | Step-ramp max step | Pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    all_pass = True

    for offset, path, data in items:
        delta = data["delta_temperature"].astype(np.float64)
        source = data["source_room_indices"].astype(int)
        walls = data["walls"].astype(np.float64)
        late_start = int(delta.shape[0] * 0.8)
        late = delta[late_start:]
        source_mean = float(late[:, source].mean()) if len(source) else math.nan
        other = data["other_room_indices"].astype(int)
        other_mean = float(late[:, other].mean()) if len(other) else 0.0
        rel_error = 0.0 if offset == 0 else abs(source_mean - offset) / offset
        non_source_ratio = 0.0 if source_mean == 0 else other_mean / source_mean

        min_walls = np.min(walls[:, source], axis=1)
        for wall_count in sorted(set(int(v) for v in min_walls)):
            idx = np.where(min_walls == wall_count)[0]
            wall_rows.append(
                {
                    "offset": offset,
                    "wall_count_to_source_min": wall_count,
                    "n_sensors": len(idx),
                    "late_mean_delta": finite_mean(late[:, idx]),
                    "late_max_delta": float(np.max(late[:, idx])) if len(idx) else math.nan,
                }
            )

        first_diff = np.diff(delta, axis=0)
        second_diff = np.diff(delta, n=2, axis=0)
        max_step = float(np.max(np.abs(first_diff))) if first_diff.size else 0.0
        mean_step = float(np.mean(np.abs(first_diff))) if first_diff.size else 0.0
        mean_second = float(np.mean(np.abs(second_diff))) if second_diff.size else 0.0
        step_ramp_max = float(offset / 48.0) if offset else 0.0
        smooth_rows.append(
            {
                "offset": offset,
                "max_abs_first_difference": max_step,
                "mean_abs_first_difference": mean_step,
                "mean_abs_second_difference": mean_second,
                "step_ramp_max_first_difference": step_ramp_max,
            }
        )

        by_wall = [r for r in wall_rows if r["offset"] == offset]
        wall_means = [float(r["late_mean_delta"]) for r in by_wall]
        wall_ok = True
        if offset > 0 and len(wall_means) >= 2:
            wall_ok = all(wall_means[i] >= wall_means[i + 1] - 0.05 * offset for i in range(len(wall_means) - 1))
        offset_ok = (offset == 0 and abs(source_mean) < 1e-5 and np.max(np.abs(delta)) < 1e-5) or rel_error <= 0.10
        nonsource_ok = offset == 0 or non_source_ratio < 0.85
        smooth_ok = offset == 0 or max_step < step_ramp_max
        passed = offset_ok and nonsource_ok and smooth_ok and wall_ok
        all_pass = all_pass and passed
        report.append(
            f"| {offset} | {source_mean:.4f} | {rel_error:.3f} | {non_source_ratio:.3f} | "
            f"{max_step:.4f} | {step_ramp_max:.4f} | {'yes' if passed else 'no'} |"
        )
        print(f"{path.name}: pass={passed} source_mean={source_mean:.4f} max_step={max_step:.4f}")

    wall_csv = out_dir / "wall_attenuation_summary.csv"
    with wall_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wall_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wall_rows)

    smooth_csv = out_dir / "temporal_smoothness_summary.csv"
    with smooth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(smooth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(smooth_rows)

    manifest_path = args.data_dir / "physical_diffusion_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report.extend(["", "## Parameters", "", "```json", json.dumps(manifest.get("params", {}), indent=2), "```"])
    report.extend(
        [
            "",
            "## Outputs",
            "",
            "- `physical_diffusion_room_mean_curves.png/pdf`",
            "- `physical_diffusion_offset_schedule.png/pdf`",
            "- `wall_attenuation_summary.csv`",
            "- `temporal_smoothness_summary.csv`",
            "",
            f"Overall diagnostic pass: {'yes' if all_pass else 'no'}",
        ]
    )
    md_path = out_dir / "physical_diffusion_manifest_check.md"
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(md_path)
    print(f"OVERALL_PASS={all_pass}")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
