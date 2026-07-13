#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare FEHU Task-1b label-level performance between two dev runs.

Default comparison:
  DeepSeek V4 Flash baseline full vs majority_vote_full

Outputs:
  label_level_analysis/task1b_label_comparison.csv
  label_level_analysis/task1b_frequency_group_summary.csv
  label_level_analysis/task1b_top_improved.png
  label_level_analysis/task1b_top_degraded.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "ntcir19_fehu-master" / "ntcir19_fehu-master"
GOLD = PROJECT / "dataset" / "gold_labels" / "dev" / "dev_human_values.json"
L1_VALUES = PROJECT / "dataset" / "hv_categories" / "human_value_level1_values.json"
RUNS = PROJECT / "output" / "task1" / "runs"
OUT = ROOT / "label_level_analysis"

DEFAULT_LEFT = "dev_deepseek_deepseek-v4-flash_baseline_full_temp0p1_max8000"
DEFAULT_RIGHT = "dev_deepseek_ensemble_majority_vote_full"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reverse_mapping(name_to_id: dict) -> dict:
    return {str(v): str(k) for k, v in name_to_id.items()}


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


def safe_div(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def f1_score(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def label_name(label: str, l1_names: dict[str, str]) -> str:
    direction, l1 = label.split(":", 1)
    direction_name = "aligned" if direction == "1" else "contradictory"
    return f"{direction_name}: {l1_names.get(l1, l1)}"


def frequency_group(gold_count: int) -> str:
    if gold_count >= 20:
        return "frequent (>=20)"
    if gold_count >= 5:
        return "medium (5-19)"
    if gold_count >= 1:
        return "infrequent (1-4)"
    return "no gold"


def metrics_for_label(label: str, gold: dict, pred: dict) -> dict:
    tp = fp = fn = 0
    for iid in set(gold) | set(pred):
        g = gold.get(iid, set())
        p = pred.get(iid, set())
        tp += int(label in g and label in p)
        fp += int(label not in g and label in p)
        fn += int(label in g and label not in p)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = f1_score(precision, recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_count": tp + fn,
        "pred_count": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["frequency_group"]].append(row)

    out = []
    for group in ["frequent (>=20)", "medium (5-19)", "infrequent (1-4)", "no gold"]:
        items = grouped.get(group, [])
        if not items:
            continue
        n = len(items)
        out.append({
            "frequency_group": group,
            "num_labels": n,
            "avg_gold_count": round(sum(r["gold_count"] for r in items) / n, 3),
            "left_avg_f1": round(sum(r["left_f1"] for r in items) / n, 6),
            "right_avg_f1": round(sum(r["right_f1"] for r in items) / n, 6),
            "diff_avg_f1": round(sum(r["diff_f1"] for r in items) / n, 6),
            "left_avg_recall": round(sum(r["left_recall"] for r in items) / n, 6),
            "right_avg_recall": round(sum(r["right_recall"] for r in items) / n, 6),
            "diff_avg_recall": round(sum(r["diff_recall"] for r in items) / n, 6),
            "left_avg_precision": round(sum(r["left_precision"] for r in items) / n, 6),
            "right_avg_precision": round(sum(r["right_precision"] for r in items) / n, 6),
            "diff_avg_precision": round(sum(r["diff_precision"] for r in items) / n, 6),
        })
    return out


def plot_bars(rows: list[dict], path: Path, title: str, value_key: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skip plot {path.name}: {exc}")
        return

    rows = list(reversed(rows))
    labels = [r["label_name"].replace("contradictory: ", "contra: ") for r in rows]
    values = [r[value_key] for r in rows]
    colors = ["#24885c" if v >= 0 else "#b84848" for v in values]

    plt.rcParams["font.family"] = "Times New Roman"
    fig_h = max(4.2, 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(9.2, fig_h), dpi=180)
    ax.barh(labels, values, color=colors, height=0.68)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel(value_key.replace("_", " "), fontsize=13)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(values):
        ha = "left" if v >= 0 else "right"
        offset = 0.004 if v >= 0 else -0.004
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Task-1b label-level metrics between two runs.")
    parser.add_argument("--left", default=DEFAULT_LEFT, help="Left/base run tag under output/task1/runs")
    parser.add_argument("--right", default=DEFAULT_RIGHT, help="Right/new run tag under output/task1/runs")
    parser.add_argument("--out_dir", default=str(OUT), help="Output directory")
    parser.add_argument("--topn", type=int, default=25, help="Number of labels in improved/degraded charts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    l1_names = reverse_mapping(load_json(L1_VALUES))
    gold = parse_task1b(GOLD)
    left = parse_task1b(RUNS / args.left / "pred_task1b.json")
    right = parse_task1b(RUNS / args.right / "pred_task1b.json")

    rows = []
    labels = [f"{d}:{i}" for d in ("0", "1") for i in range(54)]
    for label in labels:
        left_m = metrics_for_label(label, gold, left)
        right_m = metrics_for_label(label, gold, right)
        row = {
            "label": label,
            "label_name": label_name(label, l1_names),
            "frequency_group": frequency_group(left_m["gold_count"]),
            "gold_count": left_m["gold_count"],
            "left_pred_count": left_m["pred_count"],
            "right_pred_count": right_m["pred_count"],
            "left_tp": left_m["tp"],
            "left_fp": left_m["fp"],
            "left_fn": left_m["fn"],
            "right_tp": right_m["tp"],
            "right_fp": right_m["fp"],
            "right_fn": right_m["fn"],
            "left_precision": left_m["precision"],
            "left_recall": left_m["recall"],
            "left_f1": left_m["f1"],
            "right_precision": right_m["precision"],
            "right_recall": right_m["recall"],
            "right_f1": right_m["f1"],
            "diff_precision": right_m["precision"] - left_m["precision"],
            "diff_recall": right_m["recall"] - left_m["recall"],
            "diff_f1": right_m["f1"] - left_m["f1"],
            "diff_fp": right_m["fp"] - left_m["fp"],
            "diff_fn": right_m["fn"] - left_m["fn"],
        }
        rows.append(row)

    display_rows = []
    for row in rows:
        display_rows.append({
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in row.items()
        })

    fields = list(display_rows[0].keys())
    write_csv(out_dir / "task1b_label_comparison.csv", display_rows, fields)
    summary = group_summary(rows)
    write_csv(out_dir / "task1b_frequency_group_summary.csv", summary, list(summary[0].keys()))

    improved = sorted(
        [r for r in rows if r["gold_count"] > 0],
        key=lambda r: (r["diff_f1"], r["gold_count"]),
        reverse=True,
    )[: args.topn]
    degraded = sorted(
        [r for r in rows if r["gold_count"] > 0],
        key=lambda r: (r["diff_f1"], -r["gold_count"]),
    )[: args.topn]
    write_csv(
        out_dir / "task1b_top_improved.csv",
        [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in improved],
        fields,
    )
    write_csv(
        out_dir / "task1b_top_degraded.csv",
        [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in degraded],
        fields,
    )

    plot_bars(improved, out_dir / "task1b_top_improved.png", "Top Improved Labels: Majority Vote vs Baseline", "diff_f1")
    plot_bars(degraded, out_dir / "task1b_top_degraded.png", "Top Degraded Labels: Majority Vote vs Baseline", "diff_f1")

    print(f"Compared left={args.left}")
    print(f"      vs right={args.right}")
    print(f"Wrote outputs to: {out_dir.resolve()}")
    print("\nFrequency group summary:")
    for row in summary:
        print(
            f"- {row['frequency_group']}: labels={row['num_labels']}, "
            f"F1 {row['left_avg_f1']:.4f}->{row['right_avg_f1']:.4f} "
            f"({row['diff_avg_f1']:+.4f}), "
            f"Recall {row['left_avg_recall']:.4f}->{row['right_avg_recall']:.4f} "
            f"({row['diff_avg_recall']:+.4f})"
        )


if __name__ == "__main__":
    main()
