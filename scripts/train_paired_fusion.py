"""Fit feature concatenation on fixed YOLO11n detectors and score paired views."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO, __version__
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.nms import non_max_suppression


VIEWS = ("above", "underwater")
CLASSES = ("bottle", "can")
SEED = 42
EDGE = 960
CONF = 0.25
IOUS = np.linspace(0.50, 0.95, 10)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode {path}")
    if image.shape[:2] != (960, 1280):
        raise ValueError(f"Unexpected image shape: {path}: {image.shape}")
    resized = cv2.resize(image, (EDGE, 720), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, 120, 120, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)))


class Pairs(Dataset):
    def __init__(self, root, path, expected):
        self.rows = []
        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["key"] in keys or row["split"] != path.stem:
                raise ValueError(f"Duplicate or misplaced sample: {row['key']}")
            keys.add(row["key"])
            label = json.loads((root / row["labels"]).read_text(encoding="utf-8"))
            row["paths"] = [root / row[f"{view}_image"] for view in VIEWS]
            row["truth"] = [
                [{"id": obj["instance_id"], "category": CLASSES.index(obj["category"]),
                  "box": obj["boxes_xyxy"][view]}
                 for obj in label["instances"] if obj["boxes_xyxy"][view] is not None]
                for view in VIEWS
            ]
            self.rows.append(row)
        if len(self.rows) != expected:
            raise ValueError(f"Expected {expected} samples in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return [read_image(path) for path in row["paths"]], row


def collate(samples):
    images, rows = zip(*samples)
    return [torch.stack(items) for items in zip(*images)], rows


def stratified_batches(dataset, size):
    """Same-condition batches, with at least two images for partner shuffling."""
    strata = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        strata[row["group"], row["above_factor"], row["underwater_factor"]].append(index)
    batches = []
    for indices in strata.values():
        for start in range(0, len(indices), size):
            batch = indices[start:start+size]
            if len(batch) == 1:
                # Move an item from the preceding batch to keep both batches usable.
                if start == 0:
                    raise ValueError("Shuffling requires two independent pairs per condition")
                batch.insert(0, batches[-1].pop())
            batches.append(batch)
    return batches


def layers_before_head(model, image):
    cache = []
    for layer in model.model[:-1]:
        if layer.f != -1:
            image = cache[layer.f] if isinstance(layer.f, int) else [
                image if index == -1 else cache[index] for index in layer.f
            ]
        image = layer(image)
        cache.append(image if layer.i in model.save else None)
    return [cache[index] for index in model.model[-1].f]


class ConcatProjection(nn.Module):
    """Concatenate views, preserve the target block, and learn the partner block."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("own_block", torch.zeros(width, width, 1, 1))
        self.partner_weight = nn.Parameter(torch.zeros(width, width, 1, 1))

    def forward(self, own, other):
        kernel = torch.cat((self.own_block, self.partner_weight), dim=1)
        return own + F.conv2d(torch.cat((own, other), dim=1), kernel)


class Dual(nn.Module):
    def __init__(self, checkpoints, device):
        super().__init__()
        self.detectors = nn.ModuleList(
            YOLO(str(path)).model.float().to(device).eval().requires_grad_(False)
            for path in checkpoints
        )
        widths = [module[0].conv.in_channels for module in self.detectors[0].model[-1].cv2]
        self.fusion = nn.ModuleList(
            nn.ModuleList(ConcatProjection(width) for width in widths) for _ in VIEWS
        ).to(device)
        self.losses = []
        for detector in self.detectors:
            detector.args = get_cfg()
            self.losses.append(v8DetectionLoss(detector))

    @torch.no_grad()
    def features(self, images):
        return [layers_before_head(model, image) for model, image in zip(self.detectors, images)]

    def forward_head(self, maps, index, mode, training=False):
        head = self.detectors[index].model[-1]
        head.training = training
        features = maps[index]
        if mode == "dual":
            features = [layer(a, b) for layer, a, b in zip(self.fusion[index], features, maps[1-index])]
        elif mode == "shuffled":
            features = [layer(a, b.roll(1, 0)) for layer, a, b in zip(self.fusion[index], features, maps[1-index])]
        return head(features)


def batch_targets(rows, index, device):
    targets = []
    for batch_id, row in enumerate(rows):
        for obj in row["truth"][index]:
            x1, y1, x2, y2 = obj["box"]
            targets.append([batch_id, obj["category"], (x1+x2)/2560, ((y1+y2)*0.375+120)/EDGE,
                            (x2-x1)/1280, (y2-y1)/1280])
    tensor = torch.tensor(targets, dtype=torch.float32, device=device).reshape(-1, 6)
    return {"batch_idx": tensor[:, 0], "cls": tensor[:, 1:2], "bboxes": tensor[:, 2:]}


def correct_predictions(predictions, truth):
    target_boxes = np.array([obj["box"] for obj in truth], dtype=float).reshape(-1, 4)
    target_classes = np.array([obj["category"] for obj in truth], dtype=int)
    hit = np.zeros((len(predictions), len(IOUS)), dtype=bool)
    assigned_ids = np.full(len(predictions), -1, dtype=int)
    if len(truth) and len(predictions):
        boxes = predictions[:, :4]
        overlap = np.maximum(0, np.minimum(boxes[:, None, 2:], target_boxes[:, 2:])
                             - np.maximum(boxes[:, None, :2], target_boxes[:, :2])).prod(axis=-1)
        union = ((boxes[:, 2:] - boxes[:, :2]).prod(axis=1)[:, None]
                 + (target_boxes[:, 2:] - target_boxes[:, :2]).prod(axis=1) - overlap)
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        iou[predictions[:, 5, None] != target_classes] = 0
        for column, threshold in enumerate(IOUS):
            candidates = iou.copy()
            for row in np.flatnonzero(iou.max(axis=1) >= threshold):
                match = int(candidates[row].argmax())
                if candidates[row, match] >= threshold:
                    hit[row, column] = True
                    candidates[:, match] = 0
                    if column == 0:
                        assigned_ids[row] = int(truth[match]["id"])
    return hit, assigned_ids, target_classes


def decode(model, maps, index, mode):
    raw = model.forward_head(maps, index, mode)
    batches = non_max_suppression(raw[0], conf_thres=0.001, iou_thres=0.7, nc=2, max_det=300)
    decoded = []
    for batch in batches:
        boxes = batch.detach().cpu().numpy().astype(float)
        boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] / 0.75).clip(0, 1280)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - 120) / 0.75).clip(0, 960)
        decoded.append(boxes)
    return decoded


@torch.inference_mode()
def evaluate(model, loader, device, modes, output=None):
    records = defaultdict(list)
    aps = {}
    for images, rows in loader:
        maps = model.features([image.to(device, non_blocking=True).float().div_(255) for image in images])
        for index, view in enumerate(VIEWS):
            for mode in modes:
                for row, predictions in zip(rows, decode(model, maps, index, mode), strict=True):
                    truth = row["truth"][index]
                    hit, ids, target_classes = correct_predictions(predictions, truth)
                    keep = predictions[:, 4] >= CONF
                    records[view, mode].append({
                        "key": row["key"], "group": row["group"], "above_factor": row["above_factor"],
                        "underwater_factor": row["underwater_factor"], "truth": truth,
                        "hit": hit, "target_classes": target_classes, "predictions": predictions,
                        "matched_ids": ids, "keep": keep,
                    })
    for (view, mode), rows in records.items():
        hit = np.concatenate([row["hit"] for row in rows])
        preds = np.concatenate([row["predictions"] for row in rows])
        truth = np.concatenate([row["target_classes"] for row in rows])
        ap = ap_per_class(hit, preds[:, 4], preds[:, 5], truth, plot=False)[5]
        aps[f"{view}/{mode}"] = {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean())}
    if output is not None:
        compare(records, output)
    return aps


def compare(records, output):
    comparisons = []
    for view in VIEWS:
        for single, dual, shuffled in zip(*(records[view, mode] for mode in ("single", "dual", "shuffled")), strict=True):
            if not single["key"] == dual["key"] == shuffled["key"]:
                raise ValueError("The same sample must be compared across modes")
            row = {key: single[key] for key in ("key", "group", "above_factor", "underwater_factor", "truth")}
            row["view"] = view
            modes = {}
            for mode, case in zip(("single", "dual", "shuffled"), (single, dual, shuffled)):
                ids = case["matched_ids"][case["keep"]]
                modes[mode] = set(ids[ids >= 0].tolist())
                row[mode] = [{"box": p[:4].tolist(), "score": float(p[4]), "category": int(p[5]),
                              "matched_id": int(instance)}
                             for p, instance in zip(case["predictions"][case["keep"]], ids)]
                row[f"{mode}_tp"] = len(modes[mode])
                row[f"{mode}_fp"] = int((ids < 0).sum())
            row["recovered_ids"] = sorted(modes["dual"] - modes["single"])
            row["lost_ids"] = sorted(modes["single"] - modes["dual"])
            comparisons.append(row)
    (output / "paired_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in comparisons), encoding="utf-8"
    )
    groups = defaultdict(list)
    for row in comparisons:
        view = row["view"]
        groups[view, "all", "all"].append(row)
        groups[view, "target_condition", row[f"{view}_factor"]].append(row)
        groups[view, "condition_pair", row["above_factor"] + "/" + row["underwater_factor"]].append(row)
    aggregated = []
    rng = np.random.default_rng(SEED)
    for (view, grouping, condition), rows in sorted(groups.items()):
        n = len(rows)
        targets = np.array([len(row["truth"]) for row in rows])
        indices = rng.integers(n, size=(2000, n))
        row = {"view": view, "grouping": grouping, "condition": condition,
               "images": n, "targets": int(targets.sum())}
        for mode in ("single", "dual", "shuffled"):
            tp, fp = (sum(item[f"{mode}_{field}"] for item in rows) for field in ("tp", "fp"))
            row.update({f"{mode}_tp": tp, f"{mode}_fp": fp,
                        f"{mode}_recall": tp / targets.sum() if targets.sum() else 0,
                        f"{mode}_precision": tp / (tp + fp) if tp + fp else 0})
        row["recovered"] = sum(len(item["recovered_ids"]) for item in rows)
        row["lost"] = sum(len(item["lost_ids"]) for item in rows)
        for baseline in ("single", "shuffled"):
            delta = np.array([item["dual_tp"] - item[f"{baseline}_tp"] for item in rows])
            denominator = targets[indices].sum(axis=1)
            draws = delta[indices].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
            lower, upper = np.quantile(draws, (0.025, 0.975)) if len(draws) else (0, 0)
            row[f"gain_vs_{baseline}"] = row["dual_recall"] - row[f"{baseline}_recall"]
            row[f"gain_vs_{baseline}_ci_low"] = float(lower)
            row[f"gain_vs_{baseline}_ci_high"] = float(upper)
        aggregated.append(row)
    write_csv(output / "condition_comparison.csv", aggregated)


def preview(dataset, output):
    rows = [json.loads(line) for line in (output / "paired_predictions.jsonl").read_text().splitlines()]
    by_key = {row["key"]: row for row in dataset.rows}
    destination = output / "examples"
    destination.mkdir()
    for view in VIEWS:
        options = [row for row in rows if row["view"] == view]
        for kind, key in (("recovered", "recovered_ids"), ("lost", "lost_ids")):
            row = max(options, key=lambda row: len(row[key]))
            if not row[key]:
                continue
            sample = by_key[row["key"]]
            index = VIEWS.index(view)
            panels = Image.new("RGB", (1280*3, 960))
            for column, mode in enumerate(("partner", "single", "dual")):
                with Image.open(sample["paths"][1-index if mode == "partner" else index]) as image:
                    canvas = image.convert("RGB")
                pen = ImageDraw.Draw(canvas)
                pen.text((16, 16), f"{mode} | {row['key']}", fill="white", stroke_width=2, stroke_fill="black")
                if mode != "partner":
                    for prediction in row[mode]:
                        color = ("#ffdf34", "#56f5ff")[prediction["category"]]
                        pen.rectangle(prediction["box"], outline=color, width=2)
                    for truth in row["truth"]:
                        if truth["id"] in row[key]:
                            pen.rectangle(truth["box"], outline="#ff28af", width=3)
                panels.paste(canvas, (1280*column, 0))
            panels.save(destination / f"{view}_{kind}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(f"cuda:{args.device}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [args.runs / f"{view}_single_v1/fit/weights/best.pt" for view in VIEWS]
    write_json(output / "config.json", {
        "split": str(args.split), "epochs": args.epochs, "batch": args.batch, "seed": SEED,
        "initial_checkpoints": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
        "input_size": EDGE, "learning_rate": 0.001, "optimizer": "AdamW", "lightweight_fusion_only": True,
        "confidence": CONF, "match_iou": float(IOUS[0]), "ultralytics": __version__,
        "selection": "mean validation mAP50-95 across both views",
        "shuffled": "rotate partner features inside same-condition batches at evaluation",
    })
    train = Pairs(args.data.resolve(), args.split / "train.jsonl", 1400)
    val = Pairs(args.data.resolve(), args.split / "val.jsonl", 300)
    common = {"collate_fn": collate, "num_workers": 5, "pin_memory": True, "persistent_workers": True}
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, **common)
    val_loader = DataLoader(val, batch_size=args.batch, **common)
    model = Dual(checkpoints, device)
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    log = []
    best = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for iteration, (images, rows) in enumerate(train_loader):
            maps = model.features([image.to(device, non_blocking=True).float().div_(255) for image in images])
            if epoch == 1 and iteration == 0:
                for index in range(2):
                    with torch.no_grad():
                        single = model.forward_head(maps, index, "single")[0]
                        paired = model.forward_head(maps, index, "dual")[0]
                    torch.testing.assert_close(paired, single, atol=1e-3, rtol=1e-4)
                print("IDENTITY_CHECK_OK", flush=True)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model.losses[index](model.forward_head(maps, index, "dual", training=True),
                                           batch_targets(rows, index, device))[0].sum()
                       for index in range(2)) / (2 * len(rows))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{iteration}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.fusion.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())
            if args.smoke and iteration == 2:
                smoke_loader = DataLoader(val, batch_sampler=stratified_batches(val, args.batch)[:2],
                                          collate_fn=collate)
                scores = evaluate(model, smoke_loader, device, ("single", "dual", "shuffled"), output)
                print(f"SMOKE_OK loss={loss.item():.5f} scores={json.dumps(scores)}", flush=True)
                return
        schedule.step()
        metric = evaluate(model, val_loader, device, ("dual",))
        score = sum(x["map50_95"] for x in metric.values()) / 2
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "mean_map50_95": score,
                    **{view + "_" + name: value for view in VIEWS
                       for name, value in metric[f"{view}/dual"].items()}})
        write_csv(output / "training.csv", log)
        if score > best:
            best, best_epoch = score, epoch
            temp = output / "fusion.pt.partial"
            torch.save(model.fusion.state_dict(), temp)
            temp.replace(output / "fusion.pt")
        print("EPOCH", json.dumps(log[-1]), flush=True)
        if epoch >= 8 and epoch - best_epoch >= 5:
            break
    model.fusion.load_state_dict(torch.load(output / "fusion.pt", map_location=device, weights_only=True))
    print(f"BEST_EPOCH {best_epoch}", flush=True)
    for split in ("train", "val", "test"):
        shutil.copyfile(args.split / f"{split}.jsonl", output / f"{split}.jsonl")
    write_json(output / "validation_map.json", evaluate(model, val_loader, device, ("single", "dual")))
    # Test is first opened after model selection and metric thresholds are fixed.
    test = Pairs(args.data.resolve(), args.split / "test.jsonl", 300)
    test_output = output / "test_results"
    test_output.mkdir()
    test_loader = DataLoader(test, batch_sampler=stratified_batches(test, args.batch), **common)
    write_json(test_output / "map.json", evaluate(model, test_loader, device,
                                                 ("single", "dual", "shuffled"), test_output))
    preview(test, test_output)
    print("FUSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
