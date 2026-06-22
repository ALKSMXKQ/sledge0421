from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run compositional semantic editing on sledge_raw.gz files.")
    parser.add_argument("--input_root", type=str, required=True, help="Root directory containing sledge_raw.gz files.")
    parser.add_argument("--spec_json", type=str, required=True, help="HazardSemanticSpec JSON path.")
    parser.add_argument("--output_root", type=str, required=True, help="Output root directory.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of input files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = sorted(Path(args.input_root).rglob("sledge_raw.gz"))[: args.limit]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(input_files)} input files.")
    for i, raw in enumerate(input_files):
        sample_id = f"{i:05d}_{raw.parent.name}"
        out_dir = output_root / sample_id
        cmd = ["python", "sledge/script/run_compositional_semantic_edit.py", "--input_raw", str(raw), "--spec_json", args.spec_json, "--output_dir", str(out_dir)]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"[OK] Finished batch editing. Output root: {output_root}")


if __name__ == "__main__":
    main()
