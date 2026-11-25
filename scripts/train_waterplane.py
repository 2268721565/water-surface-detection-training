"""Compare direct and water-plane-aligned underwater assistance to an above-water detector."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_paired_fusion import (
    CONF, EDGE, SEED, VIEWS, Dual, Pairs, ap_per_class, batch_targets, collate,
    correct_predictions, decode, stratified_batches, write_csv, write_json,
)


OFFSETS = (-32, 0, 32)


def plane_camera(camera, water_z):
    """Map (world x, world y, 1) on the interface to homogeneous image pixels."""
    pose = np.asarray(camera["camera_to_world_row_major"], dtype=float)
    eye = np.asarray(camera["eye_m"], dtype=float)
    np.testing.assert_allclose(pose[3, :3], eye, atol=1e-7)
    # USD camera basis vectors are rows. Optical coordinates are right, down, forward.
    rotation = np.diag([1, -1, -1]) @ pose[:3, :3]
    fx, fy, cx, cy = camera["intrinsics_px"]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    plane = np.column_stack((rotation[:, :2], rotation @ (np.array([0, 0, water_z]) - eye)))
    return intrinsics @ plane


def calibration(root, split, output):
    cases = {}
    for name in ("train", "val"):
        rows = (split / f"{name}.jsonl").read_text().splitlines()
        for line in rows:
            row = json.loads(line)
            label = json.loads((root / row["labels"]).read_text())
            key = json.dumps({key: label[key] for key in ("camera", "waterline_z_m")}, sort_keys=True)
            cases[key] = cases.get(key, 0) + 1
    if len(cases) != 1:
        raise ValueError("This run requires fixed cameras and water level over train and validation")
    info = json.loads(next(iter(cases)))
    water_z = info["waterline_z_m"]
    cameras = [info["camera"][view] for view in VIEWS]
    above, below = (plane_camera(camera, water_z) for camera in cameras)
    inverse = np.linalg.inv(above)
    homography = below @ inverse
    points = np.array([[-0.7, 0.1, 1], [0, 1.3, 1], [0.65, 2, 1]]).T
    projected = []
    for camera in cameras:
        pose = np.asarray(camera["camera_to_world_row_major"])
        xyz = np.vstack((points[:2], np.full(points.shape[1], water_z)))
        optical = (pose[:3, :3] @ (xyz - pose[3, :3, None])) * np.array([[1], [-1], [-1]])
        assert np.all(optical[2] > 0)
        fx, fy, cx, cy = camera["intrinsics_px"]
        projected.append(optical[:2] / optical[2] * np.array([[fx], [fy]]) + np.array([[cx], [cy]]))
    transfer = homography @ np.vstack((projected[0], np.ones(points.shape[1])))
    error = float(np.max(np.abs(transfer[:2] / transfer[2] - projected[1])))
    if error > 1e-6:
        raise ValueError(f"Plane projection check failed: {error}")
    report = {**info, "checked_labels": sum(cases.values()), "above_to_underwater": homography.tolist(),
              "inverse_depth_row": inverse[2].tolist(), "projection_error_pixels": error,
              "water_plane_points_xy_m": points[:2].T.tolist(),
              "above_pixels": projected[0].T.tolist(), "underwater_pixels": projected[1].T.tolist()}
    write_json(output / "geometry.json", report)
    return homography, inverse[2]


def sample_grids(shape, homography, inverse_depth_row, use_plane, device):
    height, width = shape
    y, x = np.meshgrid((np.arange(height) + 0.5) * EDGE/height,
                       (np.arange(width) + 0.5) * EDGE/width, indexing="ij")
    # Undo training letterbox before projection. The original frame is 1280 x 960.
    coords = np.stack((x / .75, (y - 120) / .75, np.ones_like(x)), axis=-1)
    mapped = coords @ homography.T
    inverse_depth = coords @ inverse_depth_row
    valid = (inverse_depth > 1e-8) & (mapped[..., 2] * inverse_depth > 1e-8)
    valid &= (coords[..., 1] >= 0) & (coords[..., 1] < 960)
    safe_depth = np.where(np.abs(mapped[..., 2]) > 1e-8, mapped[..., 2], 1)
    source = mapped[..., :2] / safe_depth[..., None] if use_plane else coords[..., :2]
    grids = []
    for offset in OFFSETS:
        u, v = source[..., 0], source[..., 1] + offset
        usable = valid & (u >= 0) & (u < 1280) & (v >= 0) & (v < 960)
        norm = np.stack((2*u*.75/EDGE - 1, 2*(v*.75+120)/EDGE - 1), axis=-1)
        norm[~usable] = 2
        grids.append(torch.as_tensor(norm, dtype=torch.float32, device=device)[None])
    return grids


class AboveFusion(Dual):
    def __init__(self, checkpoints, device, transform, use_plane):
        super().__init__(checkpoints, device)
        widths = [layer.partner_weight.shape[0] for layer in self.fusion[0]]
        self.fusion = nn.ModuleList(nn.Conv2d(width*len(OFFSETS), width, 1, bias=False)
                                    for width in widths).to(device)
        for layer in self.fusion:
            nn.init.zeros_(layer.weight)
        self.losses = self.losses[:1]
        self.device = device
        self.transform = transform
        self.use_plane = use_plane
        self.grids = {}

    def forward_head(self, maps, index, mode, training=False):
        if index != 0:
            raise ValueError("Only above-water boxes are produced")
        head = self.detectors[0].model[-1]
        head.training = training
        if mode == "single":
            return head(maps[0])
        output = []
        for own, other, layer in zip(*maps, self.fusion):
            shape = own.shape[-2:]
            if shape not in self.grids:
                self.grids[shape] = sample_grids(shape, *self.transform, self.use_plane, self.device)
            if mode == "shuffled":
                other = other.roll(1, 0)
            samples = [F.grid_sample(other, grid.expand(len(other), -1, -1, -1),
                                     mode="bilinear", padding_mode="zeros", align_corners=False)
                       for grid in self.grids[shape]]
            output.append(own + layer(torch.cat(samples, dim=1)))
        return head(output)


@torch.inference_mode()
def validate(model, loader, device, modes, output=None):
    stats = {mode: [] for mode in modes}
    groups = defaultdict(list)
    for images, rows in loader:
        maps = model.features([image.to(device, non_blocking=True).float().div_(255) for image in images])
        predictions = {mode: decode(model, maps, 0, mode) for mode in modes}
        for i, row in enumerate(rows):
            events = {"key": row["key"], "targets": len(row["truth"][0])}
            for mode in modes:
                boxes = predictions[mode][i]
                hit, ids, classes = correct_predictions(boxes, row["truth"][0])
                stats[mode].append((hit, boxes[:, 4], boxes[:, 5], classes))
                kept = ids[boxes[:, 4] >= CONF]
                events[mode] = [int(item) for item in kept if item >= 0]
                events[f"{mode}_fp"] = int((kept < 0).sum())
            groups["all", "all"].append(events)
            groups["above", row["above_factor"]].append(events)
            groups["pair", f"{row['above_factor']}/{row['underwater_factor']}"].append(events)
    metrics = {}
    for mode, rows in stats.items():
        ap = ap_per_class(*(np.concatenate(values) for values in zip(*rows)), plot=False)[5]
        metrics[mode] = {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean())}
    if output is not None:
        summary = []
        for (group, condition), rows in sorted(groups.items()):
            count = sum(row["targets"] for row in rows)
            entry = {"grouping": group, "condition": condition, "images": len(rows), "targets": count}
            for mode in modes:
                tp = sum(len(row[mode]) for row in rows)
                fp = sum(row[f"{mode}_fp"] for row in rows)
                entry.update({f"{mode}_tp": tp, f"{mode}_fp": fp, f"{mode}_recall": tp/count})
            entry["recovered"] = sum(len(set(row["fusion"]) - set(row["single"])) for row in rows)
            entry["lost"] = sum(len(set(row["single"]) - set(row["fusion"])) for row in rows)
            entry["recovered_only_when_paired"] = sum(
                len(set(row["fusion"]) - set(row["single"]) - set(row["shuffled"])) for row in rows
            )
            summary.append(entry)
        write_csv(output / "validation_by_condition.csv", summary)
        (output / "validation_events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in groups["all", "all"]), encoding="utf-8"
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "plane"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    cv2.setNumThreads(0)
    device = torch.device(f"cuda:{args.device}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    transform = calibration(data, args.split, output)
    print("GEOMETRY_OK", flush=True)
    checkpoints = [args.runs / f"{view}_single_v1/fit/weights/best.pt" for view in VIEWS]
    write_json(output / "config.json", {
        "mode": args.mode, "epochs": args.epochs, "batch": args.batch, "seed": SEED,
        "source_offsets_px": OFFSETS, "selection": "above-water validation mAP50-95",
        "split": str(args.split), "projection_band": "forward rays below above-water horizon",
        "checkpoints": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
    })
    for subset in ("train", "val"):
        shutil.copyfile(args.split / f"{subset}.jsonl", output / f"{subset}.jsonl")
    train = Pairs(data, args.split / "train.jsonl", 1400)
    val = Pairs(data, args.split / "val.jsonl", 300)
    common = {"num_workers": 5, "collate_fn": collate, "pin_memory": True, "persistent_workers": True}
    training = DataLoader(train, batch_size=args.batch, shuffle=True, **common)
    validating = DataLoader(val, batch_sampler=stratified_batches(val, args.batch), **common)
    model = AboveFusion(checkpoints, device, transform, args.mode == "plane")
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    best, best_epoch = -1.0, 0
    records = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for step, (images, rows) in enumerate(training):
            maps = model.features([image.to(device, non_blocking=True).float().div_(255) for image in images])
            if epoch == 1 and step == 0:
                with torch.no_grad():
                    torch.testing.assert_close(model.forward_head(maps, 0, "fusion")[0],
                                               model.forward_head(maps, 0, "single")[0],
                                               atol=1e-3, rtol=1e-4)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_head(maps, 0, "fusion", training=True)
            loss = model.losses[0](prediction, batch_targets(rows, 0, device))[0].sum() / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.fusion.parameters(), 10)
            optimizer.step()
            losses.append(loss.item())
            if args.smoke and step == 2:
                sample = DataLoader(val, batch_sampler=stratified_batches(val, args.batch)[:2], collate_fn=collate)
                print("SMOKE_OK", validate(model, sample, device, ("single", "fusion", "shuffled"), output))
                return
        scheduler.step()
        scores = validate(model, validating, device, ("fusion",))["fusion"]
        records.append({"epoch": epoch, "loss": float(np.mean(losses)), **scores})
        write_csv(output / "training.csv", records)
        if scores["map50_95"] > best:
            best, best_epoch = scores["map50_95"], epoch
            torch.save(model.fusion.state_dict(), output / "fusion.pt.partial")
            (output / "fusion.pt.partial").replace(output / "fusion.pt")
        print("EPOCH", json.dumps(records[-1]), flush=True)
        if epoch >= 6 and epoch - best_epoch >= 4:
            break
    model.fusion.load_state_dict(torch.load(output / "fusion.pt", map_location=device, weights_only=True))
    write_json(output / "validation_map.json", {
        "best_epoch": best_epoch,
        **validate(model, validating, device, ("single", "fusion", "shuffled"), output),
    })
    print("WATERPLANE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
