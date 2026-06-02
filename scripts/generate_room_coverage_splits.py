"""
Generate room-coverage sparse sensor splits for Intel Lab experiments.

The main constraint is that every separated room has at least one observed
sensor. The isolated single-sensor room (mote_id=11, zero-based index 8) is
therefore always observed in the main protocol.
"""

from pathlib import Path
import csv

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "model" / "KITS-main" / "datasets" / "intel_lab"
OUT_DIR = Path(__file__).resolve().parent / "room_coverage_splits"

ROOM_GROUPS = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    1: [8],
    2: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


def make_split(n_sensors, miss_rate, seed):
    rng = np.random.RandomState(seed)
    target_known = int(np.ceil(n_sensors * (1.0 - miss_rate)))
    target_known = max(target_known, len(ROOM_GROUPS))

    known = set()
    for room_id, indices in ROOM_GROUPS.items():
        indices = [i for i in indices if i < n_sensors]
        if room_id == 1 and 8 in indices:
            chosen = 8
        else:
            chosen = int(rng.choice(indices))
        known.add(chosen)

    remaining = [i for i in range(n_sensors) if i not in known]
    need = max(0, target_known - len(known))
    if need > 0:
        known.update(int(x) for x in rng.choice(remaining, size=need, replace=False))

    known_mask = np.zeros(n_sensors, dtype=bool)
    known_mask[sorted(known)] = True
    hidden_mask = ~known_mask
    return known_mask, hidden_mask


def validate_split(known_mask, mote_ids):
    problems = []
    if not known_mask[8]:
        problems.append("mote_id=11/index=8 is not observed")
    for room_id, indices in ROOM_GROUPS.items():
        indices = [i for i in indices if i < len(known_mask)]
        if not known_mask[indices].any():
            problems.append(f"room {room_id} has no observed sensor")
    if problems:
        raise RuntimeError("; ".join(problems))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mote_ids = np.load(DATA_DIR / "mote_ids.npy")
    coords = np.load(DATA_DIR / "coords.npy")
    n_sensors = len(mote_ids)

    rows = []
    for miss_rate in [0.25, 0.5, 0.75]:
        for seed in [1, 2, 3]:
            known_mask, hidden_mask = make_split(n_sensors, miss_rate, seed)
            validate_split(known_mask, mote_ids)
            known_indices = np.where(known_mask)[0]
            hidden_indices = np.where(hidden_mask)[0]

            path = OUT_DIR / f"split_roomcov_mr{miss_rate:g}_seed{seed}.npz"
            np.savez_compressed(
                path,
                known_mask=known_mask,
                hidden_mask=hidden_mask,
                known_indices=known_indices,
                hidden_indices=hidden_indices,
                mote_ids=mote_ids,
                coords=coords,
                seed=seed,
                miss_rate=miss_rate,
                split_type="room_coverage",
                room_coverage_note=(
                    "Every room has at least one observed sensor; isolated "
                    "mote_id=11 is always observed."
                ),
            )

            row = {
                "miss_rate": miss_rate,
                "seed": seed,
                "known_count": int(known_mask.sum()),
                "hidden_count": int(hidden_mask.sum()),
                "known_mote_ids": " ".join(str(int(x)) for x in mote_ids[known_mask]),
                "hidden_mote_ids": " ".join(str(int(x)) for x in mote_ids[hidden_mask]),
                "mote11_observed": bool(known_mask[8]),
                "split_file": str(path),
            }
            for room_id, indices in ROOM_GROUPS.items():
                indices = [i for i in indices if i < n_sensors]
                row[f"room{room_id}_known_count"] = int(known_mask[indices].sum())
            rows.append(row)
            print(
                f"[roomcov] mr={miss_rate:g} seed={seed} "
                f"known={known_mask.sum()}/{n_sensors}, "
                f"hidden={hidden_mask.sum()}, mote11_observed={known_mask[8]}"
            )

    summary = OUT_DIR / "split_roomcov_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
