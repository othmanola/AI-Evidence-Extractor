"""Evaluate AI Evidence Extractor detections against a hand-labeled CSV.

CSV columns: video,timestamp,label,x1,y1,x2,y2
Use one row per annotated object/frame. Predictions may use the same schema.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    video: str
    timestamp: float
    label: str
    coords: tuple[float, float, float, float]


def load_boxes(path: Path) -> list[Box]:
    boxes = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for line, row in enumerate(reader, start=2):
            try:
                # Accept the normalized annotation schema and the app's current
                # detections CSV (Video, Timestamp, Object, BBox).
                lower = {str(k).casefold(): k for k in row}
                video = row[lower.get("video", "video")].strip()
                label_key = lower.get("label", lower.get("object", "label"))
                label = row[label_key].strip().split()[-1].casefold()
                time_key = lower.get("timestamp", "timestamp")
                time_text = row[time_key].strip()
                if ":" in time_text:
                    parts = [float(part) for part in time_text.split(":")]
                    timestamp = sum(part * 60 ** i for i, part in enumerate(reversed(parts)))
                else:
                    timestamp = float(time_text)
                if "bbox" in lower:
                    coords = tuple(float(v) for v in ast.literal_eval(row[lower["bbox"]]))
                else:
                    coords = tuple(float(row[lower[k]]) for k in ("x1", "y1", "x2", "y2"))
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    raise ValueError("box has non-positive area")
                boxes.append(Box(video, timestamp, label, coords))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid annotation ({exc})") from exc
    return boxes


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def evaluate(ground_truth: list[Box], predictions: list[Box],
             iou_threshold: float = 0.5, time_tolerance: float = 0.5) -> dict:
    """Greedily match same-class boxes within a time tolerance, one-to-one."""
    candidates = []
    for pi, pred in enumerate(predictions):
        for gi, truth in enumerate(ground_truth):
            if pred.video != truth.video or pred.label != truth.label:
                continue
            dt = abs(pred.timestamp - truth.timestamp)
            overlap = iou(pred.coords, truth.coords)
            if dt <= time_tolerance and overlap >= iou_threshold:
                candidates.append((-overlap, dt, pi, gi))
    candidates.sort()
    used_p, used_g = set(), set()
    per_class = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    for _, _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
    classes = {b.label for b in ground_truth} | {b.label for b in predictions}
    for label in classes:
        gt_count = sum(b.label == label for b in ground_truth)
        pred_count = sum(b.label == label for b in predictions)
        tp = sum(predictions[pi].label == label for pi in used_p)
        per_class[label].update(tp=tp, fp=pred_count - tp, fn=gt_count - tp)

    def scores(counts: Counter) -> dict:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"TP": tp, "FP": fp, "FN": fn,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "F1": round(f1, 4)}

    total = Counter()
    for counts in per_class.values():
        total.update(counts)
    return {"settings": {"IoU_threshold": iou_threshold,
                         "time_tolerance_seconds": time_tolerance},
            "overall": scores(total),
            "per_class": {label: scores(counts)
                          for label, counts in sorted(per_class.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path, help="Labeled CSV")
    parser.add_argument("predictions", type=Path, help="Prediction CSV")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--time-tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not 0 <= args.iou <= 1 or args.time_tolerance < 0:
        parser.error("--iou must be in [0,1] and --time-tolerance must be nonnegative")
    report = evaluate(load_boxes(args.ground_truth), load_boxes(args.predictions),
                      args.iou, args.time_tolerance)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
