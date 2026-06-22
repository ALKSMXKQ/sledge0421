from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sledge.semantic_control.primitive_compiler import compile_spec_to_ops, ops_to_dicts
from sledge.semantic_control.spec_checker import check_spec
from sledge.semantic_control.spec_io import load_spec, load_specs_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check compositional HazardSemanticSpec JSON files and compile primitive ops."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec_json", type=str, help="Path to one hazard semantic spec JSON.")
    group.add_argument("--spec_dir", type=str, help="Directory containing hazard semantic spec JSON files.")

    parser.add_argument("--strict", action="store_true", help="Treat suspicious combinations as errors.")
    parser.add_argument("--print_ops", action="store_true", help="Print compiled primitive ops.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save a check report JSON.")
    return parser.parse_args()


def build_report_for_spec(path: Path, strict: bool, print_ops: bool) -> Dict[str, Any]:
    spec = load_spec(path, normalize=True)
    check_report = check_spec(spec, strict=strict)
    ops = compile_spec_to_ops(spec) if check_report.valid else []

    report: Dict[str, Any] = {
        "path": str(path),
        "spec_id": spec.spec_id,
        "canonical_type": spec.canonical_type,
        "semantic_signature": spec.semantic_signature,
        "check": check_report.to_dict(),
        "num_ops": len(ops),
    }
    if print_ops:
        report["ops"] = ops_to_dicts(ops)
    return report


def main() -> None:
    args = parse_args()

    reports: List[Dict[str, Any]] = []
    if args.spec_json:
        p = Path(args.spec_json)
        reports.append(build_report_for_spec(p, strict=args.strict, print_ops=args.print_ops))
    else:
        root = Path(args.spec_dir)
        for p in sorted(root.glob("*.json")):
            reports.append(build_report_for_spec(p, strict=args.strict, print_ops=args.print_ops))

    payload = {"reports": reports}

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"[OK] Saved report to {out}")


if __name__ == "__main__":
    main()
