#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create FEHU Task-1 ensemble prediction files from existing runs.

Supported methods:
  majority_vote  - keep labels predicted by at least 2 of 3 runs
  task_specific  - Task-1a from compact, Task-1b from majority vote
"""

from __future__ import annotations

import argparse
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


DEFAULT_RUNS = {
    "baseline": "dev_deepseek_deepseek-v4-flash_baseline_temp0p1_max8000",
    "compact": "dev_deepseek_deepseek-v4-flash_compact_temp0p1_max8000",
    "compact_strict": "dev_deepseek_deepseek-v4-flash_compact_strict_temp0p1_max8000",
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


def majority_vote(label_sets, threshold=2):
    out = {}
    all_guids = sorted(set().union(*[set(s.keys()) for s in label_sets]))
    for guid in all_guids:
        counter = Counter()
        for labels in label_sets:
            counter.update(labels.get(guid, set()))
        out[guid] = {label for label, count in counter.items() if count >= threshold}
    return out


def load_runs(run_tags):
    task1a_sets = []
    task1b_sets = []
    common_guids = None
    for tag in run_tags:
        path = run_dir(tag)
        guids_1a, labels_1a = parse_task1a(os.path.join(path, "pred_task1a.json"))
        guids_1b, labels_1b = parse_task1b(os.path.join(path, "pred_task1b.json"))
        guids = set(guids_1a) & set(guids_1b)
        common_guids = guids if common_guids is None else common_guids & guids
        task1a_sets.append(labels_1a)
        task1b_sets.append(labels_1b)
    return sorted(common_guids), task1a_sets, task1b_sets


def build_ensemble(method, run_tags):
    guids, task1a_sets, task1b_sets = load_runs(run_tags)
    if method == "majority_vote":
        task1a = majority_vote(task1a_sets)
        task1b = majority_vote(task1b_sets)
    elif method == "task_specific":
        # Task-1a uses compact directly; Task-1b uses majority vote.
        compact_index = 1
        task1a = task1a_sets[compact_index]
        task1b = majority_vote(task1b_sets)
    else:
        raise ValueError(f"Unknown method: {method}")
    return guids, task1a, task1b


def main():
    parser = argparse.ArgumentParser(description="Create FEHU Task-1 ensemble runs.")
    parser.add_argument("--method", choices=["majority_vote", "task_specific"], required=True)
    parser.add_argument("--baseline", default=DEFAULT_RUNS["baseline"])
    parser.add_argument("--compact", default=DEFAULT_RUNS["compact"])
    parser.add_argument("--compact_strict", default=DEFAULT_RUNS["compact_strict"])
    parser.add_argument("--out_tag", default=None)
    args = parser.parse_args()

    run_tags = [args.baseline, args.compact, args.compact_strict]
    out_tag = args.out_tag or f"dev_deepseek_ensemble_{args.method}"
    out_dir = os.path.join(RUNS_DIR, out_tag)

    guids, task1a_labels, task1b_labels = build_ensemble(args.method, run_tags)
    task1a = format_task1a(guids, task1a_labels)
    task1b = format_task1b(guids, task1b_labels)

    write_json(os.path.join(out_dir, "pred_task1a.json"), task1a)
    write_json(os.path.join(out_dir, "pred_task1b.json"), task1b)

    total_1a = sum(len(item["article_human_values"]) for item in task1a)
    total_1b = sum(len(item["article_human_values"]) for item in task1b)
    print(f"Wrote ensemble run: {out_dir}")
    print(f"Method: {args.method}")
    print(f"Articles: {len(guids)}")
    print(f"Task-1a predictions: {total_1a} (avg {total_1a / len(guids):.1f}/article)")
    print(f"Task-1b predictions: {total_1b} (avg {total_1b / len(guids):.1f}/article)")


if __name__ == "__main__":
    main()
