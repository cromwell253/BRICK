"""
Generate physics-constrained graph-diffusion semi-synthetic Intel Lab data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = WORKSPACE_ROOT / "model" / "KITS-main" / "datasets" / "intel_lab"
OUT_DIR = SCRIPT_DIR / "physical_diffusion_data"

ROOM_GROUPS = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    1: [8],
    2: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
SOURCE_ROOMS = [1, 2]


def fill_temperature(df: pd.DataFrame) -> pd.DataFrame:
    hourly_mean = df.groupby(df.index.hour).transform("mean")
    df = df.fillna(hourly_mean)
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.fillna(df.mean())
    return df.fillna(0)


def build_weight_matrix(
    dist: np.ndarray,
    walls: np.ndarray,
    room: np.ndarray,
    sigma_d: float,
    lambda_wall: float,
    room_bonus: float,
    k_nearest: int,
) -> np.ndarray:
    n = dist.shape[0]
    dist_term = np.exp(-dist / sigma_d)
    wall_term = np.exp(-lambda_wall * walls)
    bonus = np.where(room > 0, room_bonus, 1.0)
    weight = dist_term * wall_term * bonus
    np.fill_diagonal(weight, 0.0)

    if k_nearest > 0:
        keep = np.zeros_like(weight, dtype=bool)
        for i in range(n):
            order = np.argsort(dist[i])
            neighbors = [j for j in order if j != i][:k_nearest]
            keep[i, neighbors] = True
        keep = keep | keep.T
        weight = np.where(keep, weight, 0.0)

    row_sum = weight.sum(axis=1)
    for i, value in enumerate(row_sum):
        if value <= 0:
            j = int(np.argsort(dist[i])[1])
            weight[i, j] = weight[j, i] = np.exp(-dist[i, j] / sigma_d)
    return weight.astype(np.float32)


def smooth_source_profile(length: int, start_fraction: float, tau_steps: float) -> np.ndarray:
    start_t = int(length * start_fraction)
    profile = np.zeros(length, dtype=np.float32)
    if start_t < length:
        t = np.arange(length - start_t, dtype=np.float32)
        profile[start_t:] = 1.0 - np.exp(-t / tau_steps)
    return profile


def simulate_unit_delta(
    length: int,
    weight: np.ndarray,
    source_indices: np.ndarray,
    eta: float,
    beta: float,
    start_fraction: float,
    source_tau_steps: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_sum = weight.sum(axis=1, keepdims=True)
    p = np.divide(weight, row_sum, out=np.zeros_like(weight), where=row_sum > 0)
    source_profile = smooth_source_profile(length, start_fraction, source_tau_steps)
    source_vec = np.zeros(weight.shape[0], dtype=np.float32)
    source_vec[source_indices] = 1.0

    delta = np.zeros((length, weight.shape[0]), dtype=np.float32)
    for t in range(length - 1):
        neighbor = p @ delta[t]
        delta[t + 1] = (
            delta[t]
            + eta * (neighbor - delta[t])
            - beta * delta[t]
            + source_profile[t] * source_vec
        )
        delta[t + 1] = np.maximum(delta[t + 1], 0.0)
    return delta, source_profile


def calibrate_delta(delta_unit: np.ndarray, source_indices: np.ndarray, offset: float) -> tuple[np.ndarray, float, float]:
    if offset == 0:
        return np.zeros_like(delta_unit), 0.0, 0.0
    late_start = int(delta_unit.shape[0] * 0.8)
    unit_late_mean = float(delta_unit[late_start:, source_indices].mean())
    if unit_late_mean <= 0:
        raise ValueError("Unit diffusion produced zero late source response.")
    scale = float(offset / unit_late_mean)
    delta = delta_unit * scale
    clip_max = 1.25 * offset
    clipped = float(np.mean(delta > clip_max))
    delta = np.clip(delta, 0.0, clip_max).astype(np.float32)
    return delta, scale, clipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offsets", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.006)
    parser.add_argument("--sigma-d", type=float, default=8.0)
    parser.add_argument("--lambda-wall", type=float, default=1.7)
    parser.add_argument("--room-bonus", type=float, default=1.2)
    parser.add_argument("--k-nearest", type=int, default=10)
    parser.add_argument("--start-fraction", type=float, default=0.2)
    parser.add_argument("--source-tau-steps", type=float, default=96.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = DATA_DIR / "intel_lab.h5"
    df_raw = pd.DataFrame(pd.read_hdf(h5_path, "temperature"))
    natural_mask = (~df_raw.isna()).astype("uint8").values
    base = fill_temperature(df_raw).values.astype(np.float32)

    dist = np.load(DATA_DIR / "dist.npy").astype(np.float32)
    walls = np.load(DATA_DIR / "walls.npy").astype(np.float32)
    room = np.load(DATA_DIR / "room.npy").astype(np.float32)
    coords = np.load(DATA_DIR / "coords.npy").astype(np.float32)

    n_sensors = base.shape[1]
    source_indices = np.array(
        [i for rid in SOURCE_ROOMS for i in ROOM_GROUPS[rid] if i < n_sensors],
        dtype=np.int64,
    )
    source_set = set(int(i) for i in source_indices)
    other_indices = np.array([i for i in range(n_sensors) if i not in source_set], dtype=np.int64)

    weight = build_weight_matrix(
        dist=dist,
        walls=walls,
        room=room,
        sigma_d=args.sigma_d,
        lambda_wall=args.lambda_wall,
        room_bonus=args.room_bonus,
        k_nearest=args.k_nearest,
    )
    delta_unit, source_profile = simulate_unit_delta(
        length=base.shape[0],
        weight=weight,
        source_indices=source_indices,
        eta=args.eta,
        beta=args.beta,
        start_fraction=args.start_fraction,
        source_tau_steps=args.source_tau_steps,
    )

    params = {
        "eta": args.eta,
        "beta": args.beta,
        "sigma_d": args.sigma_d,
        "lambda_wall": args.lambda_wall,
        "room_bonus": args.room_bonus,
        "k_nearest": args.k_nearest,
        "start_fraction": args.start_fraction,
        "source_tau_steps": args.source_tau_steps,
        "source_rooms": SOURCE_ROOMS,
        "equation": "delta[t+1] = delta[t] + eta * (P @ delta[t] - delta[t]) - beta * delta[t] + source[t]",
    }
    manifest = {
        "source_h5": str(h5_path),
        "shape": list(base.shape),
        "offsets": [],
        "params": params,
        "calibration_rule": "Scale unit-source diffusion so late-period Room 1/2 mean delta matches target offset.",
        "wall_count_source": "datasets/intel_lab/walls.npy",
    }

    for offset in args.offsets:
        delta, scale, clipped_fraction = calibrate_delta(delta_unit, source_indices, float(offset))
        physical = (base + delta).astype(np.float32)
        out_name = f"physical_diffusion_offset_{int(offset)}.npz"
        out_path = args.out_dir / out_name
        late_start = int(base.shape[0] * 0.8)
        late_source_mean = float(delta[late_start:, source_indices].mean())
        np.savez_compressed(
            out_path,
            temperature=physical,
            original_temperature=base,
            delta_temperature=delta,
            natural_mask=natural_mask,
            offset=np.array(float(offset), dtype=np.float32),
            source_room_indices=source_indices,
            other_room_indices=other_indices,
            diffusion_weight_matrix=weight,
            diffusion_params=json.dumps(params),
            source_profile=source_profile,
            coords=coords,
            walls=walls,
            room=room,
        )
        manifest["offsets"].append(
            {
                "offset": float(offset),
                "file": out_name,
                "calibration_scale": scale,
                "late_source_mean_delta": late_source_mean,
                "late_source_error": late_source_mean - float(offset),
                "clipped_fraction": clipped_fraction,
            }
        )
        print(f"saved {out_path} late_source_mean_delta={late_source_mean:.4f}")

    manifest_path = args.out_dir / "physical_diffusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
