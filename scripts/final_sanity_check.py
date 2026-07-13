#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sanity-check FEHU Task-1 prediction files before submission."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
DEV_EVENTS = os.path.join(BASE_DIR, "dataset", "dev", "dev_event_base.json")
DEV_GOLD = os.path.join(BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")
L1_TO_L2 = os.path.join(BASE_DIR, "dataset", "hv_categories", "level_1_to_level_2.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run(value):
    run_dir = os.path.join(RUNS_DIR, value)
    if os.path.isdir(run_dir):
        return run_dir
    if os.path.isdir(value):
        return value
    raise FileNotFoundError(f"Run directory not found: {value}")


def load_event_actors(path):
    data = load_json(path)
    guid_to_actors = {}
    for item in data:
        actors = set()
        for actor_map in item.get("actors", []):
            actors.update(str(actor_id) for actor_id in actor_map.values())
        guid_to_actors[str(item["guid"])] = actors
    return guid_to_actors


def normalize_l1_to_l2(raw):
    mapping = {}
    for key, value in raw.items():
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Unexpected level_1_to_level_2 value for {key}: {value}")
            value = value[0]
        mapping[str(key)] = str(value)
    return mapping


def check_prediction_file(path, task, guid_to_actors, l1_to_l2):
    data = load_json(path)
    issues = []

    if not isinstance(data, list):
        issues.append(f"{task}: top-level JSON is not a list")
        return issues, {}

    guid_counts = Counter()
    pred_counts = 0
    duplicate_counts = 0
    actor_label_sets = defaultdict(set)
    l2_from_task1b = defaultdict(set)

    allowed_l2 = {str(i) for i in range(20)}
    allowed_l1 = {str(i) for i in range(54)}
    allowed_dir = {"0", "1"}

    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            issues.append(f"{task}: row {row_idx} is not an object")
            continue

        guid = str(row.get("guid", ""))
        guid_counts[guid] += 1
        if guid not in guid_to_actors:
            issues.append(f"{task}: unknown guid {guid!r}")
            valid_actors = set()
        else:
            valid_actors = guid_to_actors[guid]

        values = row.get("article_human_values")
        if not isinstance(values, list):
            issues.append(f"{task}: guid {guid} article_human_values is not a list")
            continue

        for hv_idx, hv in enumerate(values):
            pred_counts += 1
            if not isinstance(hv, dict):
                issues.append(f"{task}: guid {guid} item {hv_idx} is not an object")
                continue

            actor = str(hv.get("actor", ""))
            if actor not in valid_actors:
                issues.append(f"{task}: guid {guid} invalid actor {actor!r}")

            if task == "task1a":
                if set(hv.keys()) != {"actor", "l2_value"}:
                    issues.append(f"{task}: guid {guid} item {hv_idx} unexpected keys {sorted(hv.keys())}")
                l2 = str(hv.get("l2_value", ""))
                if l2 not in allowed_l2:
                    issues.append(f"{task}: guid {guid} actor {actor} invalid l2_value {l2!r}")
                key = (guid, actor, l2)
            else:
                if set(hv.keys()) != {"actor", "l1_value", "direction"}:
                    issues.append(f"{task}: guid {guid} item {hv_idx} unexpected keys {sorted(hv.keys())}")
                l1 = str(hv.get("l1_value", ""))
                direction = str(hv.get("direction", ""))
                if l1 not in allowed_l1:
                    issues.append(f"{task}: guid {guid} actor {actor} invalid l1_value {l1!r}")
                if direction not in allowed_dir:
                    issues.append(f"{task}: guid {guid} actor {actor} invalid direction {direction!r}")
                key = (guid, actor, l1, direction)
                if l1 in l1_to_l2:
                    l2_from_task1b[(guid, actor)].add(l1_to_l2[l1])

            if key in actor_label_sets[guid]:
                duplicate_counts += 1
                issues.append(f"{task}: duplicate prediction {key}")
            actor_label_sets[guid].add(key)

    missing_guids = sorted(set(guid_to_actors) - set(guid_counts))
    extra_guid_dupes = sorted(guid for guid, count in guid_counts.items() if count > 1)
    if missing_guids:
        issues.append(f"{task}: missing {len(missing_guids)} dev guids")
    if extra_guid_dupes:
        issues.append(f"{task}: duplicated guid rows {extra_guid_dupes[:10]}")

    summary = {
        "rows": len(data),
        "predictions": pred_counts,
        "duplicate_predictions": duplicate_counts,
        "missing_guids": len(missing_guids),
        "duplicated_guid_rows": len(extra_guid_dupes),
        "l2_from_task1b": l2_from_task1b,
    }
    return issues, summary


def main():
    parser = argparse.ArgumentParser(description="Sanity-check a FEHU Task-1 run.")
    parser.add_argument("--run", required=True, help="Run tag or run directory.")
    parser.add_argument(
        "--event_file",
        default=DEV_EVENTS,
        help="Event base JSON used to validate guid/actor IDs. Defaults to dev_event_base.json.",
    )
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    pred_1a = os.path.join(run_dir, "pred_task1a.json")
    pred_1b = os.path.join(run_dir, "pred_task1b.json")

    missing_files = [p for p in (pred_1a, pred_1b) if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"Missing prediction file(s): {missing_files}")

    guid_to_actors = load_event_actors(args.event_file)
    l1_to_l2 = normalize_l1_to_l2(load_json(L1_TO_L2))

    issues_a, summary_a = check_prediction_file(pred_1a, "task1a", guid_to_actors, l1_to_l2)
    issues_b, summary_b = check_prediction_file(pred_1b, "task1b", guid_to_actors, l1_to_l2)

    task1a_l2 = defaultdict(set)
    for row in load_json(pred_1a):
        guid = str(row["guid"])
        for hv in row.get("article_human_values", []):
            task1a_l2[(guid, str(hv["actor"]))].add(str(hv["l2_value"]))

    mapped_l2_missing = 0
    for iid, l2_values in summary_b["l2_from_task1b"].items():
        missing = l2_values - task1a_l2.get(iid, set())
        mapped_l2_missing += len(missing)

    print(f"Run: {args.run}")
    print(f"Path: {run_dir}")
    print("")
    print("Task-1a:")
    print(f"  rows: {summary_a['rows']}")
    print(f"  predictions: {summary_a['predictions']}")
    print(f"  duplicate predictions: {summary_a['duplicate_predictions']}")
    print(f"  missing dev guids: {summary_a['missing_guids']}")
    print("")
    print("Task-1b:")
    print(f"  rows: {summary_b['rows']}")
    print(f"  predictions: {summary_b['predictions']}")
    print(f"  duplicate predictions: {summary_b['duplicate_predictions']}")
    print(f"  missing dev guids: {summary_b['missing_guids']}")
    print("")
    print("Cross-check:")
    print(f"  Task-1b L1 values whose mapped L2 is absent from Task-1a: {mapped_l2_missing}")
    print("")

    issues = issues_a + issues_b
    if issues:
        print(f"FAIL: {len(issues)} issue(s) found")
        for issue in issues[:50]:
            print(f"  - {issue}")
        if len(issues) > 50:
            print(f"  ... {len(issues) - 50} more")
        raise SystemExit(1)

    print("PASS: no schema/id/duplicate/guid issues found")

    if mapped_l2_missing:
        print("NOTE: Task-1a and Task-1b are not perfectly L1->L2 consistent.")
        print("      This is not necessarily a submission-format error, but it is worth knowing.")


if __name__ == "__main__":
    main()
