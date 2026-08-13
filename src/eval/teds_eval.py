"""
Evaluates predicted CSV vs ground-truth CSV using a table-structure-aware
metric rather than raw string equality, since a single dropped row shifts
every subsequent line in naive comparison and would make the metric
uselessly harsh/uninformative.

Reports three complementary numbers per example:
  - cell_accuracy: fraction of ground-truth cells correctly recovered
    after row/column alignment (via difflib, treating rows as sequences)
  - row_recall: fraction of ground-truth rows found (order-independent,
    exact row match) -- catches wholesale dropped/hallucinated rows
  - header_exact_match: whether the header row matches exactly, since
    header errors cascade into every downstream use of the extracted data

Usage:
    python teds_eval.py --checkpoint <path> --val-manifest data/manifests/val.jsonl
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
from dataclasses import dataclass


@dataclass
class EvalResult:
    cell_accuracy: float
    row_recall: float
    header_exact_match: bool
    n_gt_rows: int
    n_pred_rows: int


def parse_csv_text(text: str) -> list[list[str]]:
    try:
        return [row for row in csv.reader(io.StringIO(text))]
    except csv.Error:
        return []  # malformed prediction -- scored as zero below, not a crash


def score_example(pred_text: str, gt_text: str) -> EvalResult:
    pred_rows = parse_csv_text(pred_text)
    gt_rows = parse_csv_text(gt_text)

    if not gt_rows:
        raise ValueError("Ground truth CSV parsed to zero rows -- check source data")

    header_exact_match = bool(pred_rows) and pred_rows[0] == gt_rows[0]

    # Row-level recall: exact string match per row, order-independent.
    # Using a multiset comparison so duplicate rows are handled correctly.
    from collections import Counter

    gt_row_strs = [",".join(r) for r in gt_rows]
    pred_row_strs = [",".join(r) for r in pred_rows]
    gt_counter = Counter(gt_row_strs)
    pred_counter = Counter(pred_row_strs)
    matched_rows = sum((gt_counter & pred_counter).values())
    row_recall = matched_rows / len(gt_row_strs)

    # Cell-level accuracy: align rows via difflib SequenceMatcher (treats
    # each row as an atomic token), then compare cells within matched rows.
    # Unmatched/inserted/deleted rows contribute zero correct cells for
    # their ground-truth cell count, which is the correct penalty.
    matcher = difflib.SequenceMatcher(a=gt_row_strs, b=pred_row_strs, autojunk=False)
    total_gt_cells = sum(len(r) for r in gt_rows)
    correct_cells = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for gi, pi in zip(range(i1, i2), range(j1, j2)):
                correct_cells += len(gt_rows[gi])  # exact row match => all cells correct
        elif tag == "replace":
            # Cell-by-cell comparison for approximately-aligned rows
            for gi, pi in zip(range(i1, i2), range(j1, j2)):
                gt_row = gt_rows[gi]
                pred_row = pred_rows[pi] if pi < len(pred_rows) else []
                for gc, pc in zip(gt_row, pred_row):
                    if gc == pc:
                        correct_cells += 1

    cell_accuracy = correct_cells / total_gt_cells if total_gt_cells else 0.0

    return EvalResult(
        cell_accuracy=cell_accuracy,
        row_recall=row_recall,
        header_exact_match=header_exact_match,
        n_gt_rows=len(gt_rows),
        n_pred_rows=len(pred_rows),
    )


def run_model_inference(checkpoint: str, image_path: str, prompt: str) -> str:
    """Placeholder for actual model inference call -- wire up to your
    trained checkpoint (or the AWQ-quantized export, to eval the exact
    artifact that ships to laptops) once training completes."""
    raise NotImplementedError(
        "Wire this up to transformers.pipeline or the AWQ inference API "
        "once you have a checkpoint to evaluate."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--val-manifest", type=str, required=True)
    args = ap.parse_args()

    results: list[EvalResult] = []
    with open(args.val_manifest) as f:
        for line in f:
            record = json.loads(line)
            pred_text = run_model_inference(
                args.checkpoint, record["image_path"], record["prompt"]
            )
            result = score_example(pred_text, record["target"])
            results.append(result)

    n = len(results)
    print(f"Evaluated {n} examples")
    print(f"Mean cell accuracy:     {sum(r.cell_accuracy for r in results) / n:.4f}")
    print(f"Mean row recall:        {sum(r.row_recall for r in results) / n:.4f}")
    print(f"Header exact match:     {sum(r.header_exact_match for r in results) / n:.4f}")


if __name__ == "__main__":
    main()
