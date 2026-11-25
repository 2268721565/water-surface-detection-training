"""Audit paired capture data and check underwater detection on 24 training images."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import struct

from PIL import Image, ImageDraw


GROUPS = {
    "01_ideal_600": 600,
    "02_above_hard_500": 500,
    "03_underwater_hard_600": 600,
    "04_dual_hard_300": 300,
}
VIEWS = ("above", "underwater")
CLASSES = ("bottle", "can")
SIZE = (1280, 960)
SEED = 42
CONFIDENCE = 0.25


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    candidates = defaultdict(list)
    report = {}
    layouts = defaultdict(list)
    for group, expected_count in GROUPS.items():
        folder = root / group
        ids = set()
        factors = Counter()
        visibility = Counter()
        categories = Counter()
        for line in (folder / "pairs.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            pair_id = row["pair_id"]
            key = f"{group}/{pair_id}"
            check(pair_id not in ids, f"Duplicate pair {key}")
            ids.add(pair_id)
            label = json.loads((folder / row["labels"]).read_text(encoding="utf-8"))
            check(label["pair_id"] == pair_id and label["images"] == row["images"], key)
            check(tuple(label["resolution_wh"]) == SIZE, key)
            for view in VIEWS:
                with (folder / row["images"][view]).open("rb") as image:
                    header = image.read(24)
                check(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", key)
                check(struct.unpack(">II", header[16:24]) == SIZE, key)
            factors["/".join(label.get(f"{view}_factor", "ideal") for view in VIEWS)] += 1
            instance_ids = set()
            layout = []
            targets = []
            for instance in label["instances"]:
                category = instance["category"]
                instance_id = instance["instance_id"]
                check(category in CLASSES and instance_id not in instance_ids, key)
                instance_ids.add(instance_id)
                categories[category] += 1
                seen = []
                for view in VIEWS:
                    box = instance["boxes_xyxy"][view]
                    check(instance["visibility"][view] == (box is not None), key)
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box
                    check(0 <= x1 < x2 <= SIZE[0] and 0 <= y1 < y2 <= SIZE[1], key)
                    seen.append(view)
                    if view == "underwater":
                        targets.append((CLASSES.index(category), box))
                check(bool(seen), key)
                visibility["+".join(seen)] += 1
                if "position_m" in instance and "yaw_deg" in instance:
                    layout.append((category, *instance["position_m"], instance["yaw_deg"]))
            if layout and len(layout) == len(instance_ids):
                layouts[tuple(sorted(layout))].append(key)
            factor = label.get("underwater_factor")
            subset = "ideal" if group == "01_ideal_600" else (
                "mild_turbidity" if group == "03_underwater_hard_600" and factor == "mild_turbidity" else None
            )
            clear_classes = {c for c, (x1, y1, x2, y2) in targets if min(x2-x1, y2-y1) >= 15}
            if subset and len(clear_classes) == len(CLASSES):
                candidates[subset].append({
                    "key": key, "subset": subset, "targets": targets,
                    "above": str(folder / row["images"]["above"]),
                    "underwater": str(folder / row["images"]["underwater"]),
                    "label": str(folder / row["labels"]),
                })
        check(len(ids) == expected_count, f"{group}: expected {expected_count}, found {len(ids)}")
        report[group] = {"pairs": len(ids), "factors": factors, "visibility": visibility,
                         "categories": categories}
    duplicates = [keys for keys in layouts.values() if len(keys) > 1]
    return candidates, {"groups": report, "exact_layout_metadata_pairs": sum(map(len, layouts.values())),
                        "exact_layout_duplicates": duplicates}


def stage(candidates, output):
    rng = random.Random(SEED)
    selected = []
    for subset in ("ideal", "mild_turbidity"):
        pool = sorted(candidates[subset], key=lambda row: row["key"])
        check(len(pool) >= 12, f"{subset}: fewer than 12 eligible samples")
        selected.extend(rng.sample(pool, 12))
    images = output / "dataset" / "images"
    labels = output / "dataset" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    width, height = SIZE
    for row in selected:
        source = Path(row["underwater"])
        with Image.open(source) as image:
            image.load()
            check(image.size == SIZE and image.mode == "RGB", str(source))
        stem = row["key"].replace("/", "_")
        (images / f"{stem}.png").symlink_to(source)
        lines = []
        for category, (x1, y1, x2, y2) in row["targets"]:
            values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
            lines.append(f"{category} " + " ".join(f"{value:.9f}" for value in values))
        label = labels / f"{stem}.txt"
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        loaded = []
        for line in label.read_text().splitlines():
            category, cx, cy, w, h = map(float, line.split())
            box = [round((cx-w/2)*width), round((cy-h/2)*height),
                   round((cx+w/2)*width), round((cy+h/2)*height)]
            loaded.append((int(category), box))
        check(loaded == row["targets"], f"Box conversion changed {row['key']}")
    write_json(output / "selection.json", selected)
    yaml = output / "smoke.yaml"
    yaml.write_text(f"path: {json.dumps(str(images.parent))}\n"
                    "train: images\nval: images\nnames: [bottle, can]\n", encoding="utf-8")
    return selected, yaml


def draw(source, boxes, destination, caption):
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    pen = ImageDraw.Draw(canvas)
    pen.rectangle((0, 0, 520, 25), fill="black")
    pen.text((8, 6), caption, fill="white")
    for category, box, score in boxes:
        x1, y1, x2, y2 = box
        color = ("#ffdf34", "#56f5ff")[category]
        pen.rectangle((x1, y1, x2-1, y2-1), outline=color, width=3)
        name = CLASSES[category] if score is None else f"{CLASSES[category]} {score:.2f}"
        top = max(0, y1-17)
        pen.rectangle((x1, top, x1+pen.textlength(name)+6, top+16), fill="black")
        pen.text((x1+3, top+2), name, fill=color)
    canvas.save(destination, quality=94)


def iou(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2)-max(x1, a1))*max(0, min(y2, b2)-max(y1, b1))
    return intersection / ((x2-x1)*(y2-y1) + (a2-a1)*(b2-b1) - intersection)


def compare(selected, model, stage_name, output, device):
    totals = Counter(tp=0, fp=0, fn=0)
    results = model.predict([row["underwater"] for row in selected], imgsz=960,
                            conf=CONFIDENCE, device=device, stream=True, verbose=False)
    for index, (row, result) in enumerate(zip(selected, results, strict=True)):
        boxes = [(int(category), box, score) for category, box, score in zip(
            result.boxes.cls.int().tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
        )]
        unmatched = set(range(len(row["targets"])))
        for category, box, _ in sorted(boxes, key=lambda item: -item[2]):
            overlap, target = max(
                ((iou(box, row["targets"][i][1]), i) for i in unmatched
                 if row["targets"][i][0] == category),
                default=(0, None),
            )
            if overlap >= 0.5:
                unmatched.remove(target)
                totals["tp"] += 1
            else:
                totals["fp"] += 1
        totals["fn"] += len(unmatched)
        if index in (0, 12):
            draw(row["underwater"], boxes, output / f"{row['subset']}_{stage_name}.jpg",
                 f"{row['key']} | {stage_name} | confidence >= {CONFIDENCE}")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int, default=240)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    candidates, findings = audit(args.data.resolve())
    write_json(output / "audit.json", findings)
    print("AUDIT_COMPLETE", flush=True)
    selected, yaml = stage(candidates, output)
    for index in (0, 12):
        row = selected[index]
        draw(row["underwater"], [(c, b, None) for c, b in row["targets"]],
             output / f"{row['subset']}_labels.jpg", f"{row['key']} | ground truth")
    example = selected[12]
    for view in VIEWS:
        with Image.open(example[view]) as image:
            image.save(output / f"example_{view}.png")
    write_json(output / "example_label.json", json.loads(Path(example["label"]).read_text()))
    print("STAGING_COMPLETE", flush=True)

    import torch
    from ultralytics import YOLO

    device = 0
    initial = output / "initial.pt"
    model = YOLO(str(args.weights.resolve()) if args.weights else "yolo11n.yaml")

    def capture_initial(trainer):
        torch.save({"model": deepcopy(trainer.model).cpu().half()}, initial)

    model.add_callback("on_train_start", capture_initial)
    model.train(data=str(yaml), project=str(output), name="fit", epochs=args.epochs, imgsz=960,
                batch=8, device=device, workers=2, seed=SEED, patience=0, amp=False,
                pretrained=args.weights is not None,
                optimizer="AdamW", lr0=0.002, warmup_epochs=0.0, cos_lr=True, val=False,
                mosaic=0, translate=0, scale=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0,
                plots=False)
    final = output / "fit" / "weights" / "last.pt"
    metrics = {"samples": len(selected), "epochs": args.epochs, "confidence": CONFIDENCE, "iou": 0.5,
               "initialization": str(args.weights) if args.weights else "random",
               "evaluation_data": "same 24 images used for training"}
    for name, checkpoint in (("before", initial), ("after", final)):
        metrics[name] = compare(selected, YOLO(str(checkpoint)), name, output, device)
    write_json(output / "metrics.json", metrics)
    print("SMOKE_COMPLETE", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
