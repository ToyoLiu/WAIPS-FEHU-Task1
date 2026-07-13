#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare error sets between two FEHU Task-1 prediction runs.

This answers questions like:
  - Which FP labels did the right run remove compared with the left run?
  - Which FN labels did the right run newly introduce?
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def find_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        os.path.join(script_dir, "ntcir19_fehu-master", "ntcir19_fehu-master"),
        os.path.join(script_dir, "ntcir19_fehu-master"),
    ]
    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "dataset"))
            and os.path.exists(os.path.join(candidate, "evaluation.py"))
        ):
            return candidate
    raise FileNotFoundError("Could not find FEHU project root.")


BASE_DIR = find_base_dir()
RUNS_DIR = os.path.join(BASE_DIR, "output", "task1", "runs")
DEV_LABELS = os.path.join(BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")
L1_VALUES = os.path.join(BASE_DIR, "dataset", "hv_categories", "human_value_level1_values.json")
L2_VALUES = os.path.join(BASE_DIR, "dataset", "hv_categories", "human_value_level2_values.json")
OUT_DIR = os.path.join(BASE_DIR, "error_analysis", "task1", "comparisons")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reverse_mapping(name_to_id):
    return {str(v): str(k) for k, v in name_to_id.items()}


def run_file(run_tag, filename):
    path = os.path.join(RUNS_DIR, run_tag, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def parse_task1a(path):
    data = load_json(path)
    out = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for hv in item.get("article_human_values", []):
            out[(guid, str(hv["actor"]))].add(str(hv["l2_value"]))
    return dict(out)


def parse_task1b(path):
    data = load_json(path)
    out = defaultdict(set)
    for item in data:
        guid = str(item["guid"])
        for hv in item.get("article_human_values", []):
            label = f'{str(hv["direction"])}:{str(hv["l1_value"])}'
            out[(guid, str(hv["actor"]))].add(label)
    return dict(out)


def direction_label_name(label, l1_id_to_name):
    direction, l1 = label.split(":", 1)
    dname = "aligned" if direction == "1" else "contradictory"
    return f"{dname}:{l1_id_to_name.get(l1, l1)}"


def error_sets(gold, pred):
    fp = set()
    fn = set()
    for iid in set(gold) | set(pred):
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        for label in p - g:
            fp.add((iid, label))
        for label in g - p:
            fn.add((iid, label))
    return fp, fn


def summarize(changes, label_name):
    counter = Counter(label for _, label in changes)
    rows = []
    for label, count in counter.most_common():
        rows.append({
            "label": label,
            "label_name": label_name(label),
            "count": count,
        })
    return rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["label", "label_name", "count"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detail_rows(changes, label_name, limit=500):
    rows = []
    for (guid, actor), label in sorted(changes)[:limit]:
        rows.append({
            "guid": guid,
            "actor": actor,
            "label": label,
            "label_name": label_name(label),
        })
    return rows


def compare_task(task_name, gold, left, right, label_name, out_dir):
    left_fp, left_fn = error_sets(gold, left)
    right_fp, right_fn = error_sets(gold, right)

    removed_fp = left_fp - right_fp
    new_fp = right_fp - left_fp
    removed_fn = left_fn - right_fn
    new_fn = right_fn - left_fn

    print(f"\n{task_name}")
    print(f"  Left FP/FN:  {len(left_fp)} / {len(left_fn)}")
    print(f"  Right FP/FN: {len(right_fp)} / {len(right_fn)}")
    print(f"  Removed FP:  {len(removed_fp)}")
    print(f"  New FP:      {len(new_fp)}")
    print(f"  Removed FN:  {len(removed_fn)}")
    print(f"  New FN:      {len(new_fn)}")

    for name, changes in [
        ("removed_fp", removed_fp),
        ("new_fp", new_fp),
        ("removed_fn", removed_fn),
        ("new_fn", new_fn),
    ]:
        write_csv(os.path.join(out_dir, f"{task_name}_{name}_summary.csv"), summarize(changes, label_name))
        write_csv(
            os.path.join(out_dir, f"{task_name}_{name}_examples.csv"),
            detail_rows(changes, label_name),
            ["guid", "actor", "label", "label_name"],
        )


def main():
    parser = argparse.ArgumentParser(description="Compare FEHU Task-1 error sets.")
    parser.add_argument("--left", required=True, help="Baseline/left run tag.")
    parser.add_argument("--right", required=True, help="Comparison/right run tag.")
    parser.add_argument("--out_tag", default=None)
    args = parser.parse_args()

    l1_id_to_name = reverse_mapping(load_json(L1_VALUES))
    l2_id_to_name = reverse_mapping(load_json(L2_VALUES))

    gold_1a = parse_task1a(DEV_LABELS)
    gold_1b = parse_task1b(DEV_LABELS)
    left_1a = parse_task1a(run_file(args.left, "pred_task1a.json"))
    left_1b = parse_task1b(run_file(args.left, "pred_task1b.json"))
    right_1a = parse_task1a(run_file(args.right, "pred_task1a.json"))
    right_1b = parse_task1b(run_file(args.right, "pred_task1b.json"))

    out_tag = args.out_tag or f"{args.left}_vs_{args.right}"
    out_dir = os.path.join(OUT_DIR, out_tag)

    compare_task("task1a", gold_1a, left_1a, right_1a, lambda x: l2_id_to_name.get(x, x), out_dir)
    compare_task("task1b", gold_1b, left_1b, right_1b, lambda x: direction_label_name(x, l1_id_to_name), out_dir)
    print(f"\nWrote comparison CSV files to: {out_dir}")


if __name__ == "__main__":
    main()
