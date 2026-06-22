from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sledge.semantic_control.hazard_spec import HazardSemanticSpec
from sledge.semantic_control.spec_presets import normalize_spec


class HazardSpecJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, HazardSemanticSpec):
            return obj.to_dict()
        return super().default(obj)


def load_spec(path_like: str | Path, normalize: bool = True) -> HazardSemanticSpec:
    path = Path(path_like)
    with open(path, "r", encoding="utf-8") as fp:
        data: Dict[str, Any] = json.load(fp)
    spec = HazardSemanticSpec.from_dict(data)
    if normalize:
        spec = normalize_spec(spec)
    return spec


def save_spec(spec: HazardSemanticSpec, path_like: str | Path) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(spec.to_dict(), fp, ensure_ascii=False, indent=2)
    return path


def load_specs_from_dir(spec_dir: str | Path, normalize: bool = True) -> List[HazardSemanticSpec]:
    root = Path(spec_dir)
    specs: List[HazardSemanticSpec] = []
    for p in sorted(root.glob("*.json")):
        specs.append(load_spec(p, normalize=normalize))
    return specs


def spec_to_json_dict(spec: HazardSemanticSpec) -> Dict[str, Any]:
    return spec.to_dict()
