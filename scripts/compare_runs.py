#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two FEHU Task-1 prediction runs on dev.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


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
DEV_LABELS = os.path.join(BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")
RUNS_DIR = os.path.join(BASE_DIR, "output", "task1", "runs")


def resolve_run_path(value, filename):
    if os.path.isdir(value):
        return os.path.join(value, filename)
    candidate = os.path.join(RUNS_DIR, value, filename)
    if os.path.exists(candidate):
        return candidate
    return value


def load_pred_guids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["guid"]) for item in data}


def filter_by_guid(labels, guids):
    if guids is None:
        return labels
    return {iid: values for iid, values in labels.items() if iid[0] in guids}


def evaluate_run(run, guids=None):
    sys.path.insert(0, BASE_DIR)
    from evaluation import (
        parse_task1a,
        parse_task1b,
        evaluate,
        l2_universe,
        l1_dir_universe,
        direction_reverse_rate_gold_excl,
    )

    pred_1a_path = resolve_run_path(run, "pred_task1a.json")
    pred_1b_path = resolve_run_path(run, "pred_task1b.json")

    gold_1a = parse_task1a(DEV_LABELS)
    pred_1a = parse_task1a(pred_1a_path)
    gold_1b = parse_task1b(DEV_LABELS)
    pred_1b = parse_task1b(pred_1b_path)

    gold_1a = filter_by_guid(gold_1a, guids)
    pred_1a = filter_by_guid(pred_1a, guids)
    gold_1b = filter_by_guid(gold_1b, guids)
    pred_1b = filter_by_guid(pred_1b, guids)

    task1a = evaluate(gold_1a, pred_1a, l2_universe())
    task1b = evaluate(gold_1b, pred_1b, l1_dir_universe())
    drr, rev_cnt, denom_cnt, _ = direction_reverse_rate_gold_excl(gold_1b, pred_1b)
    task1b["direction_reverse_rate"] = drr
    task1b["direction_reverse_count"] = rev_cnt
    task1b["direction_reverse_denom"] = denom_cnt
    return {"task1a": task1a, "task1b": task1b}


def metric_row(name, left, right):
    diff = right - left
    return f"{name:<28} {left:>8.4f} {right:>8.4f} {diff:>+8.4f}"


def main():
    parser = argparse.ArgumentParser(description="Compare two FEHU Task-1 runs.")
    parser.add_argument("--left", required=True, help="Left run directory, run tag, or pred_task1a path prefix.")
    parser.add_argument("--right", required=True, help="Right run directory, run tag, or pred_task1a path prefix.")
    parser.add_argument("--all_dev", action="store_true",
                        help="Evaluate on all dev articles instead of the common predicted guids.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    args = parser.parse_args()

    guids = None
    if not args.all_dev:
        left_1a = resolve_run_path(args.left, "pred_task1a.json")
        right_1a = resolve_run_path(args.right, "pred_task1a.json")
        guids = load_pred_guids(left_1a) & load_pred_guids(right_1a)

    left = evaluate_run(args.left, guids)
    right = evaluate_run(args.right, guids)

    if args.json:
        print(json.dumps({"left": left, "right": right}, indent=2, ensure_ascii=False))
        return

    print(f"{'Metric':<28} {'Left':>8} {'Right':>8} {'Diff':>8}")
    if guids is not None:
        print(f"Scope: common predicted dev articles ({len(guids)} guids)")
    else:
        print("Scope: all dev articles")
    print("-" * 55)
    for task in ("task1a", "task1b"):
        print(task)
        for metric in ("micro_f1", "macro_f1", "micro_precision", "micro_recall"):
            print(metric_row(metric, left[task][metric], right[task][metric]))
        if task == "task1b":
            print(metric_row("direction_reverse_rate", left[task]["direction_reverse_rate"], right[task]["direction_reverse_rate"]))


if __name__ == "__main__":
    main()
