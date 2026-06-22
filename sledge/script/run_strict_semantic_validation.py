from __future__ import annotations

import argparse
import json
from pathlib import Path

from sledge.semantic_control.io import load_raw_scene, save_json
from sledge.semantic_control.semantic_validator import validate_scene_against_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict semantic validation on an edited sledge_raw.gz.")
    parser.add_argument("--edited_raw", type=str, required=True, help="Path to edited sledge_raw.gz.")
    parser.add_argument("--semantic_report", type=str, required=True, help="Path to semantic_report.json from editing.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output validation JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene, source_format = load_raw_scene(args.edited_raw)
    with open(args.semantic_report, "r", encoding="utf-8") as fp:
        report_payload = json.load(fp)

    validation = validate_scene_against_report(scene, report_payload)

    payload = {
        "edited_raw": args.edited_raw,
        "semantic_report": args.semantic_report,
        "source_format": source_format,
        "strict_validation": validation,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        save_json(args.output_json, payload)
        print(f"[OK] Saved strict validation to {args.output_json}")


if __name__ == "__main__":
    main()
