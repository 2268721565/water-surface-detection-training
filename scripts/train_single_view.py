"""Train and validate a single-view detector from a fixed paired split."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil


CLASSES = ("bottle", "can")
SCORE_THRESHOLD = 0.25
MATCH_IOU = 0.50


def prepare(data, split, run, view):
    dataset = run / "dataset"
    index = {}
    yaml_lines = [f"path: {json.dumps(str(dataset))}", "names: [bottle, can]"]
    seen = set()
    for subset, count in (("train", 1400), ("val", 300)):
        source = split / f"{subset}.jsonl"
        shutil.copyfile(source, run / source.name)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        if len(rows) != count:
            raise ValueError(f"{subset}: expected {count} rows, found {len(rows)}")
        images = dataset / "images" / subset
        labels = dataset / "labels" / subset
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for row in rows:
            if row["key"] in seen or row["split"] != subset:
                raise ValueError(f"Duplicate or misplaced sample: {row['key']}")
            seen.add(row["key"])
            stem = row["key"].replace("/", "_")
            document = json.loads((data / row["labels"]).read_text(encoding="utf-8"))
            width, height = document["resolution_wh"]
            truth = []
            lines = []
            for instance in document["instances"]:
                box = instance["boxes_xyxy"][view]
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Invalid {view} box at {row['key']}: {box}")
                category = CLASSES.index(instance["category"])
                yolo_box = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
                lines.append(f"{category} " + " ".join(f"{value:.9f}" for value in yolo_box))
                truth.append((category, box))
            image = data / row[f"{view}_image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            (images / f"{stem}.png").symlink_to(image)
            (labels / f"{stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            if subset == "val":
                index[str(image)] = (row, truth)
        yaml_lines.append(f"{subset}: images/{subset}")
    config = run / "data.yaml"
    config.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return config, index


def validate(model, index, run, device):
    import numpy as np
    from PIL import Image, ImageDraw

    counts = defaultdict(Counter)
    examples = set()
    predictions = []
    previews = run / "validation_examples"
    previews.mkdir(exist_ok=True)
    paths = list(index)
    for start in range(0, len(paths), 8):
        batch = paths[start:start+8]
        results = model.predict(batch, imgsz=960, conf=SCORE_THRESHOLD, device=device, verbose=False)
        for path, result in zip(batch, results, strict=True):
            row, truth = index[path]
            key = (row["group"], row["above_factor"], row["underwater_factor"])
            totals = counts[key]
            totals["images"] += 1
            raw = result.boxes.data.cpu().numpy()
            targets = np.array([box for _, box in truth], dtype=float).reshape(-1, 4)
            categories = np.array([category for category, _ in truth])
            areas = (targets[:, 2]-targets[:, 0]) * (targets[:, 3]-targets[:, 1])
            available = np.ones(len(truth), dtype=bool)
            matched = []
            for x1, y1, x2, y2, score, category in sorted(raw, key=lambda box: -box[4]):
                span = (np.minimum(targets[:, 2:], (x2, y2))
                        - np.maximum(targets[:, :2], (x1, y1))).clip(0)
                overlap = span.prod(axis=1)
                union = areas + (x2-x1)*(y2-y1) - overlap
                iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
                iou[~available | (categories != int(category))] = 0
                item = int(iou.argmax()) if len(iou) else None
                accepted = item is not None and iou[item] >= MATCH_IOU
                if accepted:
                    available[item] = False
                totals["tp" if accepted else "fp"] += 1
                matched.append({"category": CLASSES[int(category)], "box": list(map(float, (x1, y1, x2, y2))),
                                "confidence": float(score), "matched": bool(accepted)})
            totals["fn"] += int(available.sum())
            predictions.append({"key": row["key"], "predictions": matched})
            if key not in examples:
                examples.add(key)
                with Image.open(path) as image:
                    source = image.convert("RGB")
                annotated = source.copy()
                pen = ImageDraw.Draw(annotated)
                for category, (x1, y1, x2, y2) in truth:
                    color = ("#ffdf34", "#56f5ff")[category]
                    pen.rectangle((x1, y1, x2-1, y2-1), outline=color, width=3)
                pair = Image.new("RGB", (source.width * 2, source.height))
                pair.paste(annotated, (0, 0))
                pair.paste(Image.fromarray(result.plot()[:, :, ::-1]), (source.width, 0))
                pair.save(previews / (row["key"].replace("/", "_") + ".jpg"), quality=92)
    fields = ("group", "above_factor", "underwater_factor", "images", "tp", "fp", "fn",
              "precision", "recall")
    with (run / "validation_by_condition.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        aggregate = Counter()
        for totals in counts.values():
            aggregate.update(totals)
        for key, totals in [*sorted(counts.items()), (("all", "", ""), aggregate)]:
            tp, fp, fn = (totals[field] for field in ("tp", "fp", "fn"))
            writer.writerow((*key, totals["images"], tp, fp, fn,
                             tp/(tp+fp) if tp+fp else "", tp/(tp+fn) if tp+fn else ""))
    (run / "validation_predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--view", choices=("above", "underwater"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    from ultralytics import YOLO, __version__

    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    config, validation = prepare(args.data.resolve(), args.split.resolve(), run, args.view)
    source = str(args.weights.resolve()) if args.weights else "yolo11n.yaml"
    provenance = {
        "view": args.view, "split": str(args.split), "ultralytics": __version__,
        "initialization": source,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest() if args.weights else None,
        "confidence_threshold_for_counts": SCORE_THRESHOLD, "match_iou_for_counts": MATCH_IOU,
    }
    (run / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("DATA_READY train=1400 val=300", flush=True)

    model = YOLO(source)
    fit = model.train(data=str(config), project=str(run), name="fit", epochs=args.epochs,
                      imgsz=960, batch=args.batch, device=args.device, workers=6, seed=42,
                      patience=12, amp=False, pretrained=args.weights is not None,
                      mosaic=0, translate=0, scale=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, plots=False)
    best = run / "fit" / "weights" / "best.pt"
    detector = YOLO(str(best))
    print("BEST_CHECKPOINT_WRITTEN", best, flush=True)
    validate(detector, validation, run, args.device)
    (run / "validation_map.json").write_text(json.dumps({
        "checkpoint": "fit/weights/best.pt", "images": len(validation),
        "metrics": {key: float(value) for key, value in fit.results_dict.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print("FORMAL_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
