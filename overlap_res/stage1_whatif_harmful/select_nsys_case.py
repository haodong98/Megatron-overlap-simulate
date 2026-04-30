#!/usr/bin/env python3
"""Select the most useful what-if case for an Nsight Systems sample."""

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-jsonl", required=True)
    parser.add_argument("--variants-json", required=True)
    parser.add_argument("--output-env", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    variants = {
        item["id"]: item
        for item in json.loads(Path(args.variants_json).read_text(encoding="utf-8"))["variants"]
    }
    rows = _read_rows(Path(args.atlas_jsonl))
    selected = _select(rows, variants)

    env_path = Path(args.output_env)
    env_path.write_text(_env_text(selected), encoding="utf-8")
    output_json = Path(args.output_json) if args.output_json else env_path.with_suffix(".json")
    output_json.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selected, indent=2, sort_keys=True))


def _read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _select(rows, variants):
    scored = []
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id not in variants:
            continue
        if abs(_num(row.get("target_offset_pct"))) > 1.0e-9:
            continue
        overlap = _num(row.get("actual_overlap_ms"))
        overlap_a = _num(row.get("pairwise_overlap_pct_a"))
        overlap_b = _num(row.get("pairwise_overlap_pct_b"))
        min_overlap_pct = min(overlap_a, overlap_b)
        benefit = _num(row.get("benefit_vs_serial_same_stream"))
        tax = _num(row.get("spin_corrected_tax_ms"))
        valid_overlap = overlap > 0.05 and min_overlap_pct >= 0.30
        harmful = row.get("classification") == "harmful_overlap" and benefit < 0 and valid_overlap
        score = (
            2 if harmful else 1 if valid_overlap else 0,
            -benefit if harmful else tax,
            overlap,
            min_overlap_pct,
        )
        scored.append((score, row))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        row = scored[0][1]
        variant = variants[row["pair_id"]]
        reason = (
            "harmful_overlap_candidate"
            if row.get("classification") == "harmful_overlap"
            else "highest_tax_valid_overlap"
        )
    else:
        variant = variants.get("anchor_ar_simple") or next(iter(variants.values()))
        row = {}
        reason = "fallback_anchor_no_aggregated_rows"

    return {
        "variant": variant["id"],
        "reason": reason,
        "nsys_case": variant["nsys_case"],
        "nccl_algo": variant["nccl_algo"],
        "nccl_proto": variant["nccl_proto"],
        "nccl_min_nchannels": variant.get("nccl_min_nchannels"),
        "classification": row.get("classification"),
        "benefit_vs_serial_same_stream": row.get("benefit_vs_serial_same_stream"),
        "spin_corrected_tax_ms": row.get("spin_corrected_tax_ms"),
        "actual_overlap_ms": row.get("actual_overlap_ms"),
        "pairwise_overlap_pct_a": row.get("pairwise_overlap_pct_a"),
        "pairwise_overlap_pct_b": row.get("pairwise_overlap_pct_b"),
    }


def _env_text(selection):
    values = {
        "WHATIF_SELECTED_VARIANT": selection["variant"],
        "WHATIF_SELECTED_REASON": selection["reason"],
        "WHATIF_SELECTED_NSYS_CASE": selection["nsys_case"],
        "WHATIF_SELECTED_NCCL_ALGO": selection["nccl_algo"],
        "WHATIF_SELECTED_NCCL_PROTO": selection["nccl_proto"],
        "WHATIF_SELECTED_NCCL_MIN_NCHANNELS": selection.get("nccl_min_nchannels") or "",
    }
    return "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in values.items())


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
