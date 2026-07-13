#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a label-aware FEHU Task-1 ensemble.

Default strategy:
  - Task-1a: use targeted_more_shot_balanced directly.
  - Task-1b: start from majority_vote_full, then add targeted_more_shot_balanced
    predictions only for selected labels where it showed complementary recall.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


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

DEFAULT_BASE = "dev_deepseek_ensemble_majority_vote_full"
DEFAULT_TARGET = "dev_deepseek_deepseek-v4-flash_targeted_more_shot_balanced_temp0p1_max8000"
DEFAULT_OUT = "dev_deepseek_ensemble_label_aware_targeted_balanced"

# Direction uses evaluation.py convention: 1 = aligned, 0 = contradictory.
DEFAULT_ADD_LABELS = {
    "0:0",   # contradictory: Be creative
    "0:22",  # contradictory: Have good health
    "0:27",  # contradictory: Have a stable society
    "0:44",  # contradictory: Have equality
    "0:48",  # contradictory: Have harmony with nature
    "1:13",  # aligned: Be capable
    "1:22",  # aligned: Have good health
    "1:27",  # aligned: Have a stable society
    "1:35",  # aligned: Be humble
    "1:46",  # aligned: Have a world at peace
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_dir(run_tag):
    path = os.path.join(RUNS_DIR, run_tag)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path


def parse_task1a(path):
    data = load_json(path)
    out = defaultdict(set)
    guids = []
    for item in data:
        guid = str(item["guid"])
        guids.append(guid)
        for hv in item.get("article_human_values", []):
            out[guid].add((str(hv["actor"]), str(hv["l2_value"])))
    return guids, dict(out)


def parse_task1b(path):
    data = load_json(path)
    out = defaultdict(set)
    guids = []
    for item in data:
        guid = str(item["guid"])
        guids.append(guid)
        for hv in item.get("article_human_values", []):
            out[guid].add((str(hv["actor"]), str(hv["l1_value"]), str(hv["direction"])))
    return guids, dict(out)


def format_task1a(guids, labels_by_guid):
    rows = []
    for guid in guids:
        labels = sorted(labels_by_guid.get(guid, set()), key=lambda x: (x[0], int(x[1])))
        rows.append({
            "guid": guid,
            "article_human_values": [
                {"actor": actor, "l2_value": l2}
                for actor, l2 in labels
            ],
        })
    return rows


def format_task1b(guids, labels_by_guid):
    rows = []
    for guid in guids:
        labels = sorted(labels_by_guid.get(guid, set()), key=lambda x: (x[0], int(x[1]), x[2]))
        rows.append({
            "guid": guid,
            "article_human_values": [
                {"actor": actor, "l1_value": l1, "direction": direction}
                for actor, l1, direction in labels
            ],
        })
    return rows


def parse_label_list(values):
    labels = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                labels.add(item)
    return labels


def build_label_aware(base_tag, target_tag, add_labels, task1a_source):
    base = run_dir(base_tag)
    target = run_dir(target_tag)

    base_1a_guids, base_1a = parse_task1a(os.path.join(base, "pred_task1a.json"))
    base_1b_guids, base_1b = parse_task1b(os.path.join(base, "pred_task1b.json"))
    target_1a_guids, target_1a = parse_task1a(os.path.join(target, "pred_task1a.json"))
    target_1b_guids, target_1b = parse_task1b(os.path.join(target, "pred_task1b.json"))

    guids = sorted(set(base_1b_guids) & set(target_1b_guids))

    if task1a_source == "base":
        task1a = {guid: set(base_1a.get(guid, set())) for guid in guids}
    elif task1a_source == "target":
        task1a = {guid: set(target_1a.get(guid, set())) for guid in guids}
    else:
        raise ValueError(f"Unknown task1a_source: {task1a_source}")

    task1b = {guid: set(base_1b.get(guid, set())) for guid in guids}
    added = 0
    for guid in guids:
        for actor, l1, direction in target_1b.get(guid, set()):
            if f"{direction}:{l1}" in add_labels:
                before = len(task1b[guid])
                task1b[guid].add((actor, l1, direction))
                added += len(task1b[guid]) - before

    return guids, task1a, task1b, added


def main():
    parser = argparse.ArgumentParser(description="Create a label-aware FEHU Task-1 ensemble.")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Stable base run tag.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Run tag used to add selected labels.")
    parser.add_argument("--out_tag", default=DEFAULT_OUT)
    parser.add_argument(
        "--task1a_source",
        choices=["base", "target"],
        default="target",
        help="Which run to use for Task-1a output.",
    )
    parser.add_argument(
        "--add_label",
        action="append",
        default=[],
        help="Extra direction:l1 labels to add, e.g. 1:22 or comma-separated 1:22,0:27.",
    )
    parser.add_argument(
        "--only_labels",
        action="append",
        default=[],
        help="Override defaults with this direction:l1 label list.",
    )
    args = parser.parse_args()

    add_labels = parse_label_list(args.only_labels) if args.only_labels else set(DEFAULT_ADD_LABELS)
    add_labels |= parse_label_list(args.add_label)

    guids, task1a_labels, task1b_labels, added = build_label_aware(
        args.base,
        args.target,
        add_labels,
        args.task1a_source,
    )

    out_dir = os.path.join(RUNS_DIR, args.out_tag)
    task1a = format_task1a(guids, task1a_labels)
    task1b = format_task1b(guids, task1b_labels)
    write_json(os.path.join(out_dir, "pred_task1a.json"), task1a)
    write_json(os.path.join(out_dir, "pred_task1b.json"), task1b)

    total_1a = sum(len(item["article_human_values"]) for item in task1a)
    total_1b = sum(len(item["article_human_values"]) for item in task1b)
    print(f"Wrote label-aware ensemble run: {out_dir}")
    print(f"Base: {args.base}")
    print(f"Target: {args.target}")
    print(f"Task-1a source: {args.task1a_source}")
    print(f"Added labels: {', '.join(sorted(add_labels))}")
    print(f"New Task-1b predictions added from target: {added}")
    print(f"Articles: {len(guids)}")
    print(f"Task-1a predictions: {total_1a} (avg {total_1a / len(guids):.1f}/article)")
    print(f"Task-1b predictions: {total_1b} (avg {total_1b / len(guids):.1f}/article)")


if __name__ == "__main__":
    main()
