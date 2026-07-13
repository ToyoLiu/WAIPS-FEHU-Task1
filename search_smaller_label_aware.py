#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Search smaller label-aware ensembles over the current selected labels."""

from __future__ import annotations

import itertools
import os
import sys
from collections import defaultdict

import label_aware_ensemble as lae


sys.path.insert(0, lae.BASE_DIR)
from evaluation import evaluate, parse_task1b, l1_dir_universe, direction_reverse_rate_gold_excl


DEV_LABELS = os.path.join(lae.BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")


def to_eval_task1b(guids, labels_by_guid):
    out = defaultdict(set)
    for guid in guids:
        for actor, l1, direction in labels_by_guid.get(guid, set()):
            out[(guid, actor)].add(f"{direction}:{l1}")
    return dict(out)


def score_subset(labels):
    guids, _, task1b, added = lae.build_label_aware(
        lae.DEFAULT_BASE,
        lae.DEFAULT_TARGET,
        set(labels),
        "target",
    )
    gold = parse_task1b(DEV_LABELS)
    pred = to_eval_task1b(guids, task1b)
    metrics = evaluate(gold, pred, l1_dir_universe())
    drr, rev_count, denom_count, _ = direction_reverse_rate_gold_excl(gold, pred)
    metrics["direction_reverse_rate"] = drr
    metrics["direction_reverse_count"] = rev_count
    metrics["direction_reverse_denom"] = denom_count
    metrics["added"] = added
    return metrics


def main():
    labels = sorted(lae.DEFAULT_ADD_LABELS)
    rows = []
    for size in range(1, len(labels) + 1):
        for subset in itertools.combinations(labels, size):
            m = score_subset(subset)
            rows.append({
                "size": size,
                "labels": ",".join(subset),
                "micro_f1": m["micro_f1"],
                "macro_f1": m["macro_f1"],
                "precision": m["micro_precision"],
                "recall": m["micro_recall"],
                "drr": m["direction_reverse_rate"],
                "added": m["added"],
            })

    rows.sort(key=lambda r: (r["micro_f1"], r["macro_f1"], -r["drr"]), reverse=True)

    print("Top subsets by Task-1b micro F1")
    print("rank size micro_f1 macro_f1 precision recall drr added labels")
    for i, r in enumerate(rows[:20], 1):
        print(
            f"{i:>2} {r['size']:>4} {r['micro_f1']:.4f} {r['macro_f1']:.4f} "
            f"{r['precision']:.4f} {r['recall']:.4f} {r['drr']:.4f} "
            f"{r['added']:>5} {r['labels']}"
        )

    print("\nBest subsets with DRR <= 0.0230")
    filtered = [r for r in rows if r["drr"] <= 0.0230]
    for i, r in enumerate(filtered[:15], 1):
        print(
            f"{i:>2} {r['size']:>4} {r['micro_f1']:.4f} {r['macro_f1']:.4f} "
            f"{r['precision']:.4f} {r['recall']:.4f} {r['drr']:.4f} "
            f"{r['added']:>5} {r['labels']}"
        )


if __name__ == "__main__":
    main()
