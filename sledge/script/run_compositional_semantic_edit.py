from __future__ import annotations

import argparse
from pathlib import Path

from sledge.semantic_control.compositional_editor import CompositionalSemanticSceneEditor
from sledge.semantic_control.io import load_raw_scene, save_json, save_raw_scene
from sledge.semantic_control.spec_io import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply compositional primitive-based semantic editing to one sledge_raw.gz.")
    parser.add_argument("--input_raw", type=str, required=True, help="Input sledge_raw.gz path.")
    parser.add_argument("--spec_json", type=str, required=True, help="HazardSemanticSpec JSON path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--output_name", type=str, default="sledge_raw.gz", help="Output feature filename. Keep as sledge_raw.gz for cache compatibility.")
    parser.add_argument("--strict_check", action="store_true", help="Enable strict spec checking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_raw = Path(args.input_raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene, source_format = load_raw_scene(input_raw)
    spec = load_spec(args.spec_json, normalize=True)

    editor = CompositionalSemanticSceneEditor(strict_check=args.strict_check)
    edited_scene, edit_result, report = editor.edit(scene, spec)

    output_raw = output_dir / args.output_name
    output_report = output_dir / "semantic_report.json"

    save_raw_scene(output_raw, edited_scene, source_format=source_format)
    save_json(output_report, {"input_raw": str(input_raw), "spec_json": str(args.spec_json), "output_raw": str(output_raw), "source_format": source_format, "edit_result": edit_result.to_dict(), "report": report})

    validation = report.get("validation", {})
    print(f"[OK] saved edited scene: {output_raw}")
    print(f"[OK] saved report: {output_report}")
    print(f"[OK] source_format: {source_format}")
    print(f"[OK] semantic_signature: {report.get('semantic_signature')}")
    print(f"[OK] SSR: {validation.get('semantic_satisfaction_rate')}")
    print(f"[OK] overall_pass: {validation.get('overall_pass')}")


if __name__ == "__main__":
    main()
