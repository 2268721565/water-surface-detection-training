"""Build the fixed paired split for the four formal pool datasets."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import random


GROUPS = {
    "01_ideal_600": 600,
    "02_above_hard_500": 500,
    "03_underwater_hard_600": 600,
    "04_dual_hard_300": 300,
}
SPLITS = ("train", "val", "test")
SEED = 42
HELD_OUT_COUNT = 300


def read_samples(root):
    samples = []
    layouts = {}
    keys = set()
    index_hashes = {}
    for group, expected in GROUPS.items():
        index = root / group / "pairs.jsonl"
        content = index.read_bytes()
        index_hashes[group] = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        if len(rows) != expected:
            raise ValueError(f"{group}: expected {expected} pairs, found {len(rows)}")
        for row in rows:
            pair_id = row["pair_id"]
            key = f"{group}/{pair_id}"
            if key in keys:
                raise ValueError(f"Duplicate pair ID: {key}")
            keys.add(key)

            labels = PurePosixPath(group, row["labels"]).as_posix()
            label = json.loads((root / labels).read_text(encoding="utf-8"))
            if label["pair_id"] != pair_id or label["images"] != row["images"]:
                raise ValueError(f"Index and label disagree: {key}")
            layout = tuple(sorted(
                (obj["category"], *obj["position_m"], obj["yaw_deg"])
                for obj in label["instances"]
            ))
            if layout in layouts:
                raise ValueError(f"Repeated layout: {layouts[layout]} and {key}; assign together")
            layouts[layout] = key

            images = {view: PurePosixPath(group, row["images"][view]).as_posix()
                      for view in ("above", "underwater")}
            for path in images.values():
                if not (root / path).is_file():
                    raise FileNotFoundError(root / path)
            samples.append({
                "key": key,
                "group": group,
                "pair_id": pair_id,
                "condition": label["condition"],
                "above_factor": label.get("above_factor", "ideal"),
                "underwater_factor": label.get("underwater_factor", "ideal"),
                "above_image": images["above"],
                "underwater_image": images["underwater"],
                "labels": labels,
            })
    return samples, index_hashes


def stratify(samples, forced_train):
    strata = defaultdict(list)
    for sample in samples:
        key = (sample["group"], sample["above_factor"], sample["underwater_factor"])
        strata[key].append(sample)
    # Largest remainder keeps exactly 300 pairs in each held-out split.
    total = len(samples)
    quotas = {key: divmod(len(rows) * HELD_OUT_COUNT, total)
              for key, rows in strata.items()}
    counts = {key: quotient for key, (quotient, _) in quotas.items()}
    remainder = HELD_OUT_COUNT - sum(counts.values())
    for key in sorted(quotas, key=lambda key: (-quotas[key][1], key))[:remainder]:
        counts[key] += 1

    rng = random.Random(SEED)
    partitions = {split: [] for split in SPLITS}
    distribution = {}
    for key, rows in sorted(strata.items()):
        reserved = [row for row in rows if row["key"] in forced_train]
        candidates = sorted((row for row in rows if row["key"] not in forced_train),
                            key=lambda row: row["key"])
        rng.shuffle(candidates)
        count = counts[key]
        if len(candidates) < count * 2:
            raise ValueError(f"Too few held-out candidates for {key}")
        subsets = {
            "train": reserved + candidates[2*count:],
            "val": candidates[:count],
            "test": candidates[count:2*count],
        }
        distribution[key] = Counter({split: len(rows) for split, rows in subsets.items()})
        for split, rows in subsets.items():
            partitions[split].extend(dict(row, split=split, used_in_smoke=row["key"] in forced_train)
                                     for row in rows)
    for rows in partitions.values():
        rows.sort(key=lambda row: row["key"])
    return partitions, distribution


def write_outputs(output, partitions, distribution, index_hashes):
    output.mkdir(parents=True)
    combined = [row for rows in partitions.values() for row in rows]
    with (output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    for split, rows in partitions.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (output / f"{split}.jsonl").write_text(content, encoding="utf-8", newline="\n")

    lines = [
        "# 水上水下配对检测数据划分 v1", "",
        "每行是一对图。图片和标签路径相对于服务器的 `pool simulation/training_data/`。",
        "划分按数据组、水上条件和水下条件分层。每层验证与测试对数相同，其余归入训练。",
        f"随机种子为 `{SEED}`。试训使用的 24 对全部归入训练，在总表中标为 `used_in_smoke=True`。",
        "检查了位置和偏航角记录完全相同的布局，未发现重复。", "",
        "| 数据组 | 水上条件 | 水下条件 | 训练 | 验证 | 测试 | 合计 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, counts in distribution.items():
        lines.append("| " + " | ".join(map(str, (*key, *(counts[s] for s in SPLITS),
                                                     sum(counts.values())))) + " |")
    totals = [len(partitions[split]) for split in SPLITS]
    lines.append(f"| **总计** | | | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | "
                 f"**{sum(totals)}** |")
    lines.extend(["", "## 原始索引校验值", ""])
    lines.extend(f"- `{group}/pairs.jsonl` 的 SHA-256 为 `{digest}`。"
                 for group, digest in index_hashes.items())
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--smoke-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    samples, index_hashes = read_samples(args.data)
    selection = json.loads(args.smoke_selection.read_text(encoding="utf-8"))
    forced_train = {row["key"] for row in selection}
    keys = {row["key"] for row in samples}
    if len(selection) != 24 or len(forced_train) != 24 or not forced_train <= keys:
        raise ValueError("Expected 24 distinct formal samples from the smoke run")
    partitions, distribution = stratify(samples, forced_train)
    write_outputs(args.output, partitions, distribution, index_hashes)


if __name__ == "__main__":
    main()
