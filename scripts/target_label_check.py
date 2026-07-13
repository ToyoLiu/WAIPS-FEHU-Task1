#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Check Task-1b target-label changes between two runs on common dev guids."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "ntcir19_fehu-master" / "ntcir19_fehu-master"
GOLD = PROJECT / "dataset" / "gold_labels" / "dev" / "dev_human_values.json"
RUNS = PROJECT / "output" / "task1" / "runs"
L1_VALUES = PROJECT / "dataset" / "hv_categories" / "human_value_level1_values.json"
OUT = ROOT / "label_level_analysis"

DEFAULT_TARGETS = {
    "22",  # Have good health
    "26",  # Have a safe country
    "27",  # Have a stable society
    "19",  # Have social recognition
    "20",  # Have a good reputation
    "13",  # Be capable
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reverse_mapping(name_to_id: dict) -> dict:
    return {str(v): str(k) for k, v in name_to_id.items()}


def resolve_run(run_tag: str) -> Path:
    candidate = RUNS / run_tag / "pred_task1b.json"
    if candidate.exists():
        return candidate
    path = Path(run_tag)
    if path.exists():
        return path
    raise FileNotFoundError(f"Cannot resolve run: {run_tag}")


def parse_task1b(path: Path) -> dict[tuple[str, str], set[str]]:
    data = load_json(path)
    out = defaultdict(set)
    for article in data:
        guid = str(article["guid"])
        for hv in article.get("article_human_values", []):
            actor = str(hv["actor"])
            l1 = str(hv["l1_value"])
            direction = str(hv["direction"])
            out[(guid, actor)].add(f"{direction}:{l1}")
    return dict(out)


def guids_in_pred(path: Path) -> set[str]:
    return {str(item["guid"]) for item in load_json(path)}


def safe_div(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def metrics(label: str, gold: dict, pred: dict, guids: set[str]) -> dict:
    tp = fp = fn = 0
    instances = {iid for iid in set(gold) | set(pred) if iid[0] in guids}
    for iid in instances:
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        tp += int(label in g and label in p)
        fp += int(label not in g and label in p)
        fn += int(label in g and label not in p)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_count": tp + fn,
        "pred_count": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }


def label_name(label: str, names: dict[str, str]) -> str:
    direction, l1 = label.split(":", 1)
    d = "aligned" if direction == "1" else "contradictory"
    return f"{d}: {names.get(l1, l1)}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--targets", nargs="*", default=sorted(DEFAULT_TARGETS))
    parser.add_argument("--out", default=str(OUT / "target_label_check.csv"))
    args = parser.parse_args()

    left_path = resolve_run(args.left)
    right_path = resolve_run(args.right)
    common_guids = guids_in_pred(left_path) & guids_in_pred(right_path)

    names = reverse_mapping(load_json(L1_VALUES))
    gold = parse_task1b(GOLD)
    left = parse_task1b(left_path)
    right = parse_task1b(right_path)

    rows = []
    for l1 in args.targets:
        for direction in ("0", "1"):
            label = f"{direction}:{l1}"
            lm = metrics(label, gold, left, common_guids)
            rm = metrics(label, gold, right, common_guids)
            rows.append({
                "label": label,
                "label_name": label_name(label, names),
                "gold_count": lm["gold_count"],
                "left_tp": lm["tp"],
                "left_fp": lm["fp"],
                "left_fn": lm["fn"],
                "left_precision": round(lm["precision"], 6),
                "left_recall": round(lm["recall"], 6),
                "left_f1": round(lm["f1"], 6),
                "right_tp": rm["tp"],
                "right_fp": rm["fp"],
                "right_fn": rm["fn"],
                "right_precision": round(rm["precision"], 6),
                "right_recall": round(rm["recall"], 6),
                "right_f1": round(rm["f1"], 6),
                "diff_tp": rm["tp"] - lm["tp"],
                "diff_fp": rm["fp"] - lm["fp"],
                "diff_fn": rm["fn"] - lm["fn"],
                "diff_recall": round(rm["recall"] - lm["recall"], 6),
                "diff_f1": round(rm["f1"] - lm["f1"], 6),
            })

    write_csv(Path(args.out), rows)
    print(f"Scope: {len(common_guids)} common guids")
    print(f"Wrote: {Path(args.out).resolve()}")
    for row in rows:
        if row["gold_count"] == 0 and row["right_fp"] == row["left_fp"]:
            continue
        print(
            f"{row['label_name']:<42} gold={row['gold_count']:<3} "
            f"F1 {row['left_f1']:.3f}->{row['right_f1']:.3f} "
            f"R {row['left_recall']:.3f}->{row['right_recall']:.3f} "
            f"TP {row['left_tp']}->{row['right_tp']} "
            f"FP {row['left_fp']}->{row['right_fp']}"
        )


if __name__ == "__main__":
    main()
