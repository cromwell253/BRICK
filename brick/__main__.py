"""Training, prediction and scoring commands for the published partitions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from . import build
from .data import BuildingData
from .inference import predict
from .training import seed_all, sample_plan, optimizer, objective, clip_gradients

SCENES = ['Hospital_2', 'Office_1', 'Residence_2', 'Office_2', 'Residence_1', 'Residence_3']


def targets(data, times, queries):
    path = data.root / 'data' / data.scene / 'temperature.csv'
    if times.min() < 731 or times.max() >= data.val_end or not np.isin(queries, data.val_queries).all():
        raise ValueError('Scoring is restricted to the development query endpoints')
    requested = {}
    for index, (time, query) in enumerate(zip(times, queries)):
        requested.setdefault(int(time), []).append((index, str(query)))
    result = np.empty(len(times), dtype=np.float64)
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            time = int(row['frame'])
            if time > int(times.max()):
                break
            for index, query in requested.get(time, []):
                result[index] = float(row[query])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict', 'score', 'report'])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--scene', choices=SCENES, default='Hospital_2')
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--seed', type=int, choices=[505, 606, 707, 1103, 1201, 1301], default=505)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/run'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'report':
        print((args.root / 'results/comparison_summary.json').read_text(encoding='utf-8'))
        return
    data = BuildingData(args.root, args.scene, args.fold, args.device)
    times = np.tile(data.val_times, len(data.val_queries))
    queries = np.repeat(data.val_queries, len(data.val_times))
    if args.command == 'score':
        sealed = np.load(args.output / 'predictions.npz')
        if not np.array_equal(sealed['times'], times) or not np.array_equal(sealed['queries'], queries):
            raise ValueError('Prediction partition differs from the registered partition')
        errors = sealed['prediction'] - targets(data, times, queries)
        print(json.dumps(dict(MAE_C=float(np.abs(errors).mean()), RMSE_C=float(np.sqrt(np.square(errors).mean()))), indent=2))
        return
    seed_all(args.seed)
    model = build(data)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'predict':
        checkpoint = args.checkpoint or args.root / f'checkpoints/{args.scene}/fold_{args.fold}/seed_{args.seed}.pt'
        model.load_state_dict(torch.load(checkpoint, map_location=data.device, weights_only=True), strict=True)
        prediction = predict(model, data, times, queries)
        np.savez_compressed(args.output / 'predictions.npz', prediction=prediction, times=times, queries=queries)
        print('Saved predictions')
        return
    opt = optimizer(model)
    plan_path = args.root / f'configs/plans/{args.scene}/fold_{args.fold}/seed_{args.seed}.npz'
    if plan_path.exists():
        with np.load(plan_path) as archive:
            plan_times = archive['times']
            plan_queries = archive['queries']
    else:
        plan_times, plan_queries = sample_plan(data, SCENES.index(data.scene) % 3, args.seed)
    best = float('inf')
    history = []
    for epoch in range(48):
        model.train()
        for start in range(0, 128, 16):
            loss = objective(model, data, plan_times[epoch, start:start + 16], plan_queries[epoch, start:start + 16])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradients(model)
            opt.step()
        if (epoch + 1) % 2 == 0:
            prediction = predict(model, data, times, queries)
            mae = float(np.abs(prediction - targets(data, times, queries)).mean())
            if mae < best:
                best = mae
                torch.save(model.state_dict(), args.output / 'best.pt')
            history.append(dict(epoch=epoch + 1, updates=(epoch + 1) * 8, development_MAE_C=mae))
            (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(f'Epoch {epoch + 1}: development MAE {mae:.6f}')


if __name__ == '__main__':
    main()
