#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine B0/B1/B2/B3 experiment summaries into one comparison table.")
    parser.add_argument("--b0", required=True, help="Path to B0 batch_summary.json")
    parser.add_argument("--b1", required=True, help="Path to B1 batch_summary.json")
    parser.add_argument("--b2", required=True, help="Path to B2 repaint_summary.json")
    parser.add_argument("--b3", required=True, help="Path to B3 repaint_summary.json")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        ("B0 Original / No Control", _load_json(args.b0)),
        ("B1 Edit Only", _load_json(args.b1)),
        ("B2 Edit + Repaint, No Preserve", _load_json(args.b2)),
        ("B3 Full Method", _load_json(args.b3)),
    ]

    table_rows = []
    for name, payload in rows:
        table_rows.append({
            "Method": name,
            "N": payload.get("N"),
            "CSR": payload.get("CSR"),
            "MPA": payload.get("MPA"),
            "CR": payload.get("CR"),
            "RSR": payload.get("RSR"),
            "SPR": payload.get("SPR"),
        })

    with open(out_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(table_rows, f, ensure_ascii=False, indent=2)

    with open(out_dir / "comparison_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "N", "CSR", "MPA", "CR", "RSR", "SPR"])
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"[OK] wrote: {out_dir / 'comparison_metrics.csv'}")


if __name__ == "__main__":
    main()
