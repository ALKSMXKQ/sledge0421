#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 允许从 $SLEDGE_DEVKIT_ROOT 下运行
_THIS = Path(__file__).resolve()
for _p in _THIS.parents:
    if (_p / "sledge").is_dir():
        sys.path.append(str(_p))
        break

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.semantic_control import NaturalLanguagePromptParser, PromptAlignmentEvaluator
from sledge.script.run_half_denoise_from_tiered_cache import (
    basic_scene_compliance,
    summarize_multiscenario_semantics,
)


@dataclass(frozen=True)
class TargetSpec:
    scenario_type: str
    severity_level: str
    prompt: str


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original SLEDGE generated scenario_cache scenes under natural-hit mode. "
            "For each generated scene, score it against all benchmark target prompts "
            "(scenario_type x severity_level), then report weighted CSR / MPA / CR."
        )
    )
    parser.add_argument(
        "--scenario-cache-root",
        required=True,
        help="Generated scenario cache root, e.g. .../scenario_cache/log/us-ma-boston",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Benchmark scenario_manifest.csv used only to derive target prompts and target weights",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--alignment-threshold", type=float, default=0.70)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--method-name",
        type=str,
        default="G0",
        help="Method label used in the final master-table CSV row",
    )
    parser.add_argument(
        "--emit-master-table-row",
        action="store_true",
        help="Also emit a single-row CSV shaped for the final master comparison table",
    )
    return parser


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_gz_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _load_vector(path: Path) -> SledgeVector:
    obj = _load_gz_pickle(path)
    if isinstance(obj, SledgeVector):
        return obj
    if isinstance(obj, dict):
        return SledgeVector.deserialize(obj)
    raise TypeError(f"Unsupported object type in {path}: {type(obj)}")


def _collect_generated_paths(cache_root: Path) -> List[Path]:
    return sorted(cache_root.rglob("sledge_vector.gz"))


def _extract_target_specs_and_weights(
    manifest_rows: List[Dict[str, str]]
) -> Tuple[List[TargetSpec], Dict[Tuple[str, str], float], Dict[str, Dict[str, float]]]:
    """
    返回：
      1) 唯一 target 列表（scenario_type, severity_level, prompt）
      2) 全局权重：按 manifest 中该 target 出现频率 / 总数
      3) 场景类型内部权重：按某个 scenario_type 下 severity 的相对比例
    """
    prompt_by_key: Dict[Tuple[str, str], str] = {}
    global_counter: Counter = Counter()
    per_type_counter: Dict[str, Counter] = defaultdict(Counter)

    for row in manifest_rows:
        st = str(row["scenario_type"]).strip()
        sev = str(row["severity_level"]).strip()
        prompt = str(row["prompt"]).strip()
        key = (st, sev)

        global_counter[key] += 1
        per_type_counter[st][sev] += 1

        if key not in prompt_by_key:
            prompt_by_key[key] = prompt

    targets = [
        TargetSpec(scenario_type=st, severity_level=sev, prompt=prompt_by_key[(st, sev)])
        for (st, sev) in sorted(prompt_by_key.keys())
    ]

    total = float(sum(global_counter.values()))
    global_weights = {k: v / total for k, v in global_counter.items()}

    within_type_weights: Dict[str, Dict[str, float]] = {}
    for st, counter in per_type_counter.items():
        subtotal = float(sum(counter.values()))
        within_type_weights[st] = {sev: cnt / subtotal for sev, cnt in counter.items()}

    return targets, global_weights, within_type_weights


def _evaluate_scene_against_target(
    vector: SledgeVector,
    target: TargetSpec,
    threshold: float,
    prompt_parser: NaturalLanguagePromptParser,
    alignment_evaluator: PromptAlignmentEvaluator,
) -> Dict[str, Any]:
    prompt_spec = prompt_parser.parse(target.prompt)
    prompt_spec.scenario_type = target.scenario_type
    prompt_spec.severity_level = target.severity_level

    alignment = alignment_evaluator.evaluate(vector, prompt_spec)
    semantic = summarize_multiscenario_semantics(
        alignment=alignment,
        prompt_spec=prompt_spec,
        vector=vector,
        threshold=float(threshold),
    )
    return {
        "alignment_total": float(getattr(alignment, "total", 0.0)),
        "semantic_pass": bool(semantic["semantic_pass"]),
        "alignment_details": dict(getattr(alignment, "details", {}) or {}),
        "alignment_notes": list(getattr(alignment, "notes", []) or []),
        "semantic_summary": semantic,
    }


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_cache_root = Path(args.scenario_cache_root).resolve()
    manifest_path = Path(args.manifest).resolve()

    generated_paths = _collect_generated_paths(scenario_cache_root)
    if args.max_scenes is not None:
        generated_paths = generated_paths[: args.max_scenes]

    manifest_rows = _load_manifest_rows(manifest_path)
    targets, global_weights, within_type_weights = _extract_target_specs_and_weights(manifest_rows)

    prompt_parser = NaturalLanguagePromptParser()
    alignment_evaluator = PromptAlignmentEvaluator()

    # 针对每个 target: 收集所有生成场景对它的 alignment / semantic_pass
    per_target_alignment: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_target_pass: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    # G0 的 compliance 不依赖 prompt
    compliance_all: List[float] = []

    scene_level_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    results_jsonl = out_dir / "results.jsonl"
    with open(results_jsonl, "w", encoding="utf-8") as fp:
        for idx, scene_path in enumerate(generated_paths, start=1):
            try:
                vector = _load_vector(scene_path)

                compliance = basic_scene_compliance(vector)
                compliant = bool(compliance["compliant"])
                compliance_all.append(1.0 if compliant else 0.0)

                scene_record = {
                    "scene_index": idx,
                    "generated_scene_path": str(scene_path.resolve()),
                    "compliant": compliant,
                    "compliance_issues": list(compliance["issues"]),
                    "targets": {},
                }

                for target in targets:
                    key = (target.scenario_type, target.severity_level)
                    ev = _evaluate_scene_against_target(
                        vector=vector,
                        target=target,
                        threshold=float(args.alignment_threshold),
                        prompt_parser=prompt_parser,
                        alignment_evaluator=alignment_evaluator,
                    )

                    per_target_alignment[key].append(float(ev["alignment_total"]))
                    per_target_pass[key].append(1.0 if bool(ev["semantic_pass"]) else 0.0)

                    scene_record["targets"][f"{target.scenario_type}__{target.severity_level}"] = {
                        "prompt": target.prompt,
                        "alignment_total": float(ev["alignment_total"]),
                        "semantic_pass": bool(ev["semantic_pass"]),
                    }

                fp.write(json.dumps(scene_record, ensure_ascii=False) + "\n")
                scene_level_records.append(scene_record)

            except Exception as exc:
                errors.append({
                    "scene_index": idx,
                    "generated_scene_path": str(scene_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })

    # 1) 每个 target（9类）
    per_target_summary: Dict[str, Any] = {}
    for target in targets:
        key = (target.scenario_type, target.severity_level)
        per_target_summary[f"{target.scenario_type}__{target.severity_level}"] = {
            "scenario_type": target.scenario_type,
            "severity_level": target.severity_level,
            "prompt": target.prompt,
            "weight_global": float(global_weights[key]),
            "CSR": _mean(per_target_pass[key]),
            "MPA": _mean(per_target_alignment[key]),
        }

    # 2) 按 scenario_type 聚合（在该场景类型内部按 severity 分布加权）
    per_scenario_type_summary: Dict[str, Any] = {}
    scenario_types = sorted(set(t.scenario_type for t in targets))
    for st in scenario_types:
        sev_weights = within_type_weights.get(st, {})
        csr = 0.0
        mpa = 0.0
        for sev, w in sev_weights.items():
            key = (st, sev)
            csr += w * _mean(per_target_pass[key])
            mpa += w * _mean(per_target_alignment[key])

        per_scenario_type_summary[st] = {
            "CSR": float(csr),
            "MPA": float(mpa),
            "CR": _mean(compliance_all),  # G0 的 CR 与 prompt 无关，统一复制
            "severity_weights": sev_weights,
        }

    # 3) 全局加权（按 manifest 分布）
    weighted_csr = 0.0
    weighted_mpa = 0.0
    for target in targets:
        key = (target.scenario_type, target.severity_level)
        w = global_weights[key]
        weighted_csr += w * _mean(per_target_pass[key])
        weighted_mpa += w * _mean(per_target_alignment[key])

    summary = {
        "mode": "natural_hit_rate",
        "scenario_cache_root": str(scenario_cache_root),
        "manifest": str(manifest_path),
        "N_generated_scenes": len(scene_level_records),
        "num_errors": len(errors),
        "targets": per_target_summary,
        "by_scenario_type": per_scenario_type_summary,
        "weighted": {
            "CSR": float(weighted_csr),
            "MPA": float(weighted_mpa),
            "CR": _mean(compliance_all),
            "RSR": None,
            "SPR": None,
        },
    }

    with open(out_dir / "natural_hit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if errors:
        with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    # 每个 target 的详细 CSV
    with open(out_dir / "natural_hit_by_target.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario_type", "severity_level", "prompt", "weight_global", "CSR", "MPA"],
        )
        writer.writeheader()
        for v in per_target_summary.values():
            writer.writerow(v)

    # 每个 scenario_type 的加权结果 CSV
    with open(out_dir / "natural_hit_by_scenario_type.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario_type", "CSR", "MPA", "CR"],
        )
        writer.writeheader()
        for st, v in per_scenario_type_summary.items():
            writer.writerow({
                "scenario_type": st,
                "CSR": v["CSR"],
                "MPA": v["MPA"],
                "CR": v["CR"],
            })

    # 全局加权结果 CSV
    with open(out_dir / "natural_hit_weighted_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["N_generated_scenes", "CSR", "MPA", "CR", "RSR", "SPR"],
        )
        writer.writeheader()
        writer.writerow({
            "N_generated_scenes": len(scene_level_records),
            "CSR": summary["weighted"]["CSR"],
            "MPA": summary["weighted"]["MPA"],
            "CR": summary["weighted"]["CR"],
            "RSR": summary["weighted"]["RSR"],
            "SPR": summary["weighted"]["SPR"],
        })

    # 生成一行 master-table CSV，便于拼成你图里的大表
    if args.emit_master_table_row:
        ped_key = "sudden_pedestrian_crossing" if "sudden_pedestrian_crossing" in per_scenario_type_summary else "pedestrian_crossing"
        hb_key = "hard_brake"
        ci_key = "cut_in"

        ped = per_scenario_type_summary.get(ped_key, {"CSR": "", "MPA": "", "CR": ""})
        hb = per_scenario_type_summary.get(hb_key, {"CSR": "", "MPA": "", "CR": ""})
        ci = per_scenario_type_summary.get(ci_key, {"CSR": "", "MPA": "", "CR": ""})

        with open(out_dir / "master_table_row_G0.csv", "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "Method",
                "CSR_pedestrian", "CSR_hard_brake", "CSR_cut_in", "CSR_weighted",
                "MPA_pedestrian", "MPA_hard_brake", "MPA_cut_in", "MPA_weighted",
                "SPR_pedestrian", "SPR_hard_brake", "SPR_cut_in", "SPR_weighted",
                "CR_pedestrian", "CR_hard_brake", "CR_cut_in", "CR_weighted",
                "RSR_pedestrian", "RSR_hard_brake", "RSR_cut_in", "RSR_weighted",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "Method": args.method_name,

                "CSR_pedestrian": ped["CSR"],
                "CSR_hard_brake": hb["CSR"],
                "CSR_cut_in": ci["CSR"],
                "CSR_weighted": summary["weighted"]["CSR"],

                "MPA_pedestrian": ped["MPA"],
                "MPA_hard_brake": hb["MPA"],
                "MPA_cut_in": ci["MPA"],
                "MPA_weighted": summary["weighted"]["MPA"],

                # G0 没有 repaint，所以留空
                "SPR_pedestrian": "",
                "SPR_hard_brake": "",
                "SPR_cut_in": "",
                "SPR_weighted": "",

                # G0 的 CR 与 prompt 无关，为了适配总表格式，重复写入三列
                "CR_pedestrian": summary["weighted"]["CR"],
                "CR_hard_brake": summary["weighted"]["CR"],
                "CR_cut_in": summary["weighted"]["CR"],
                "CR_weighted": summary["weighted"]["CR"],

                # G0 没有 repaint，所以留空
                "RSR_pedestrian": "",
                "RSR_hard_brake": "",
                "RSR_cut_in": "",
                "RSR_weighted": "",
            })

    print(f"[OK] wrote: {out_dir / 'natural_hit_summary.json'}")
    print(
        f"[OK] Weighted CSR={summary['weighted']['CSR']:.6f}, "
        f"MPA={summary['weighted']['MPA']:.6f}, "
        f"CR={summary['weighted']['CR']:.6f}"
    )


if __name__ == "__main__":
    main()