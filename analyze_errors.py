#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FEHU Task-1 error analysis for dev predictions.

Outputs CSV files under:
  ntcir19_fehu-master/ntcir19_fehu-master/error_analysis/task1/
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
DEV_ARTICLES = os.path.join(BASE_DIR, "dataset", "dev", "dev_event_base.json")
DEV_LABELS = os.path.join(BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")
L1_VALUES = os.path.join(BASE_DIR, "dataset", "hv_categories", "human_value_level1_values.json")
L2_VALUES = os.path.join(BASE_DIR, "dataset", "hv_categories", "human_value_level2_values.json")
PRED_1A = os.path.join(BASE_DIR, "output", "task1", "pred_task1a.json")
PRED_1B = os.path.join(BASE_DIR, "output", "task1", "pred_task1b.json")
OUT_DIR = os.path.join(BASE_DIR, "error_analysis", "task1")
RUNS_DIR = os.path.join(BASE_DIR, "output", "task1", "runs")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_file(run_tag, filename):
    if not run_tag:
        return None
    run_dir = os.path.join(RUNS_DIR, run_tag)
    return os.path.join(run_dir, filename)


def safe_div(n, d):
    return 0.0 if d == 0 else n / d


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def reverse_mapping(name_to_id):
    return {str(v): str(k) for k, v in name_to_id.items()}


def build_article_context():
    articles = load_json(DEV_ARTICLES)
    context = {}
    actor_names = {}
    for article in articles:
        guid = str(article["guid"])
        context[guid] = {
            "title": article.get("title", ""),
            "content": article.get("content", ""),
        }
        for actor_dict in article.get("actors", []):
            for name, aid in actor_dict.items():
                actor_names[(guid, str(aid))] = str(name)
    return context, actor_names


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
            direction = str(hv["direction"])
            l1 = str(hv["l1_value"])
            out[(guid, str(hv["actor"]))].add(f"{direction}:{l1}")
    return dict(out)


def per_label_rows(gold, pred, label_ids, label_name):
    rows = []
    for label in label_ids:
        tp = fp = fn = 0
        instances = set(gold) | set(pred)
        for iid in instances:
            g = gold.get(iid, set())
            p = pred.get(iid, set())
            tp += int(label in g and label in p)
            fp += int(label not in g and label in p)
            fn += int(label in g and label not in p)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        rows.append({
            "label_id": label,
            "label_name": label_name(label),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": tp + fn,
            "pred_count": tp + fp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1(precision, recall), 6),
        })
    return rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_error_examples(gold, pred, task, label_name, actor_names, article_context, limit_per_type=200):
    rows = []
    counters = Counter()
    for iid in sorted(set(gold) | set(pred)):
        guid, actor = iid
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        for label in sorted(p - g):
            counters[("FP", label)] += 1
            if counters[("FP", label)] <= limit_per_type:
                rows.append(example_row("FP", task, guid, actor, label, label_name, actor_names, article_context))
        for label in sorted(g - p):
            counters[("FN", label)] += 1
            if counters[("FN", label)] <= limit_per_type:
                rows.append(example_row("FN", task, guid, actor, label, label_name, actor_names, article_context))
    return rows


def example_row(error_type, task, guid, actor, label, label_name, actor_names, article_context):
    return {
        "error_type": error_type,
        "task": task,
        "guid": guid,
        "title": article_context.get(guid, {}).get("title", ""),
        "actor": actor,
        "actor_name": actor_names.get((guid, actor), ""),
        "label": label,
        "label_name": label_name(label),
    }


def summarize_label_errors(per_label):
    fp_rows = sorted(per_label, key=lambda r: (-r["fp"], r["label_id"]))
    fn_rows = sorted(per_label, key=lambda r: (-r["fn"], r["label_id"]))
    low_precision = sorted(
        [r for r in per_label if r["pred_count"] >= 5],
        key=lambda r: (r["precision"], -r["pred_count"], r["label_id"]),
    )
    low_recall = sorted(
        [r for r in per_label if r["gold_count"] >= 5],
        key=lambda r: (r["recall"], -r["gold_count"], r["label_id"]),
    )
    return fp_rows, fn_rows, low_precision, low_recall


def split_dir_label(label):
    direction, l1 = label.split(":", 1)
    return direction, l1


def direction_label_name(label, l1_id_to_name):
    direction, l1 = split_dir_label(label)
    dname = "aligned" if direction == "1" else "contradictory"
    return f"{dname}:{l1_id_to_name.get(l1, l1)}"


def collect_reversals(gold_1b, pred_1b, l1_id_to_name, actor_names, article_context):
    rows = []
    for iid in sorted(set(gold_1b) | set(pred_1b)):
        guid, actor = iid
        g = gold_1b.get(iid, set())
        p = pred_1b.get(iid, set())
        g_by_l1 = defaultdict(set)
        p_by_l1 = defaultdict(set)
        for label in g:
            d, l1 = split_dir_label(label)
            g_by_l1[l1].add(d)
        for label in p:
            d, l1 = split_dir_label(label)
            p_by_l1[l1].add(d)

        for l1, g_dirs in sorted(g_by_l1.items(), key=lambda x: int(x[0])):
            if len(g_dirs) != 1:
                continue
            gold_dir = next(iter(g_dirs))
            pred_dirs = p_by_l1.get(l1, set())
            opposite = "0" if gold_dir == "1" else "1"
            if opposite in pred_dirs:
                rows.append({
                    "guid": guid,
                    "title": article_context.get(guid, {}).get("title", ""),
                    "actor": actor,
                    "actor_name": actor_names.get((guid, actor), ""),
                    "l1_value": l1,
                    "l1_name": l1_id_to_name.get(l1, l1),
                    "gold_direction": "aligned" if gold_dir == "1" else "contradictory",
                    "pred_direction": "aligned" if opposite == "1" else "contradictory",
                })
    return rows


def instance_summary(gold_1a, pred_1a, gold_1b, pred_1b, actor_names, article_context):
    rows = []
    for iid in sorted(set(gold_1a) | set(pred_1a) | set(gold_1b) | set(pred_1b)):
        guid, actor = iid
        g1a = gold_1a.get(iid, set())
        p1a = pred_1a.get(iid, set())
        g1b = gold_1b.get(iid, set())
        p1b = pred_1b.get(iid, set())
        rows.append({
            "guid": guid,
            "title": article_context.get(guid, {}).get("title", ""),
            "actor": actor,
            "actor_name": actor_names.get((guid, actor), ""),
            "task1a_tp": len(g1a & p1a),
            "task1a_fp": len(p1a - g1a),
            "task1a_fn": len(g1a - p1a),
            "task1b_tp": len(g1b & p1b),
            "task1b_fp": len(p1b - g1b),
            "task1b_fn": len(g1b - p1b),
            "task1a_gold": len(g1a),
            "task1a_pred": len(p1a),
            "task1b_gold": len(g1b),
            "task1b_pred": len(p1b),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze FEHU Task-1 dev errors.")
    parser.add_argument("--run_tag", default=None,
                        help="Prediction run tag under output/task1/runs/. Overrides pred paths.")
    parser.add_argument("--gold", default=DEV_LABELS)
    parser.add_argument("--pred_task1a", default=PRED_1A)
    parser.add_argument("--pred_task1b", default=PRED_1B)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.run_tag:
        args.pred_task1a = resolve_run_file(args.run_tag, "pred_task1a.json")
        args.pred_task1b = resolve_run_file(args.run_tag, "pred_task1b.json")
        args.out_dir = os.path.join(OUT_DIR, args.run_tag)

    l1_id_to_name = reverse_mapping(load_json(L1_VALUES))
    l2_id_to_name = reverse_mapping(load_json(L2_VALUES))
    article_context, actor_names = build_article_context()

    gold_1a = parse_task1a(args.gold)
    pred_1a = parse_task1a(args.pred_task1a)
    gold_1b = parse_task1b(args.gold)
    pred_1b = parse_task1b(args.pred_task1b)

    task1a_rows = per_label_rows(
        gold_1a,
        pred_1a,
        [str(i) for i in range(20)],
        lambda label: l2_id_to_name.get(label, label),
    )
    task1b_rows = per_label_rows(
        gold_1b,
        pred_1b,
        [f"{d}:{i}" for d in ("0", "1") for i in range(54)],
        lambda label: direction_label_name(label, l1_id_to_name),
    )

    write_csv(os.path.join(args.out_dir, "task1a_per_label.csv"), task1a_rows)
    write_csv(os.path.join(args.out_dir, "task1b_per_label.csv"), task1b_rows)

    for task_name, rows in (("task1a", task1a_rows), ("task1b", task1b_rows)):
        fp_rows, fn_rows, low_precision, low_recall = summarize_label_errors(rows)
        write_csv(os.path.join(args.out_dir, f"{task_name}_top_fp_labels.csv"), fp_rows[:30])
        write_csv(os.path.join(args.out_dir, f"{task_name}_top_fn_labels.csv"), fn_rows[:30])
        write_csv(os.path.join(args.out_dir, f"{task_name}_low_precision_labels.csv"), low_precision[:30])
        write_csv(os.path.join(args.out_dir, f"{task_name}_low_recall_labels.csv"), low_recall[:30])

    task1a_examples = collect_error_examples(
        gold_1a,
        pred_1a,
        "task1a",
        lambda label: l2_id_to_name.get(label, label),
        actor_names,
        article_context,
    )
    task1b_examples = collect_error_examples(
        gold_1b,
        pred_1b,
        "task1b",
        lambda label: direction_label_name(label, l1_id_to_name),
        actor_names,
        article_context,
    )
    write_csv(os.path.join(args.out_dir, "task1a_error_examples.csv"), task1a_examples)
    write_csv(os.path.join(args.out_dir, "task1b_error_examples.csv"), task1b_examples)

    reversals = collect_reversals(gold_1b, pred_1b, l1_id_to_name, actor_names, article_context)
    write_csv(os.path.join(args.out_dir, "task1b_direction_reversals.csv"), reversals)

    instances = instance_summary(gold_1a, pred_1a, gold_1b, pred_1b, actor_names, article_context)
    write_csv(os.path.join(args.out_dir, "instance_error_summary.csv"), instances)

    print(f"Wrote error analysis CSV files to: {args.out_dir}")
    print(f"Task-1a labels analyzed: {len(task1a_rows)}")
    print(f"Task-1b labels analyzed: {len(task1b_rows)}")
    print(f"Direction reversals: {len(reversals)}")


if __name__ == "__main__":
    main()
