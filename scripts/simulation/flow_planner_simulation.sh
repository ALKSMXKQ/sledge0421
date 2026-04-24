#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
export TENSORBOARD_LOG_PATH="/home16T/home8T_1/leitingting/nuplan/exp/tensorboard/flow_planner"
export RANK=0

export NUPLAN_DEVKIT_ROOT="/home16T/home8T_1/leitingting/nuplan-devkit"
export SLEDGE_ROOT="/home16T/home8T_1/leitingting/sledge_workspace/sledge"
export FLOW_PLANNER_ROOT="/home16T/home8T_1/leitingting/Flow-Planner"

export NUPLAN_DATA_ROOT="/home16T/home8T_1/leitingting/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home16T/home8T_1/leitingting/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home16T/home8T_1/leitingting/nuplan/exp"

# 让本地源码优先于 site-packages
export PYTHONPATH="${NUPLAN_DEVKIT_ROOT}:${SLEDGE_ROOT}:${FLOW_PLANNER_ROOT}:${PYTHONPATH:-}"

# 这里必须保留 SLEDGE 的 reactive 配置
# 因为你现在跑的是 SLEDGE gz cache，不是 nuPlan 原始 val14
CHALLENGE="sledge_reactive_agents"

BRANCH_NAME="flow_planner_release"

# 继续使用你之前已经验证过的推理配置
CONFIG_FILE="/home16T/home8T_1/leitingting/checkpoints/flow_planner/model_config_infer.yaml"
CKPT_FILE="/home16T/home8T_1/leitingting/checkpoints/flow_planner/model.pth"

# SLEDGE gz 场景缓存目录
SCENARIO_CACHE="/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario"

echo "=== Path check ==="
echo "NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT}"
echo "SLEDGE_ROOT=${SLEDGE_ROOT}"
echo "FLOW_PLANNER_ROOT=${FLOW_PLANNER_ROOT}"
echo "SCENARIO_CACHE=${SCENARIO_CACHE}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "CKPT_FILE=${CKPT_FILE}"
echo "PYTHONPATH=${PYTHONPATH}"

echo "=== File check ==="
[[ -f "${CONFIG_FILE}" ]] || { echo "[ERROR] CONFIG_FILE not found: ${CONFIG_FILE}"; exit 1; }
[[ -f "${CKPT_FILE}" ]] || { echo "[ERROR] CKPT_FILE not found: ${CKPT_FILE}"; exit 1; }
[[ -d "${SCENARIO_CACHE}" ]] || { echo "[ERROR] SCENARIO_CACHE not found: ${SCENARIO_CACHE}"; exit 1; }

echo "[OK] CONFIG_FILE exists"
echo "[OK] CKPT_FILE exists"
echo "[OK] SCENARIO_CACHE exists"

echo "=== Config sanity check ==="
CONFIG_FILE_ENV="${CONFIG_FILE}" python - <<'PY'
from omegaconf import OmegaConf
import os

cfg = OmegaConf.load(os.environ["CONFIG_FILE_ENV"])
print("[INFO] top-level keys:", list(cfg.keys())[:20])
assert "model" in cfg, "config missing key: model"
assert "core" in cfg, "config missing key: core"
print("[OK] config contains model/core")
PY

echo "=== Launch simulation (parallel / reactive) ==="

FILENAME=$(basename "${CKPT_FILE}")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

export SLEDGE_RANDOM_SAMPLE=1000
export SLEDGE_RANDOM_SEED=42

python "${SLEDGE_ROOT}/sledge/script/run_simulation.py" \
    +simulation="${CHALLENGE}" \
    planner=flow_planner \
    planner.flow_planner.config_path="${CONFIG_FILE}" \
    planner.flow_planner.ckpt_path="${CKPT_FILE}" \
    ++planner.flow_planner.enable_ema=false \
    ++planner.flow_planner.device=cuda \
    observation=sledge_agents_observation \
    scenario_builder=nuplan \
    cache.scenario_cache_path="${SCENARIO_CACHE}" \
    run_metric=true \
    experiment_uid="flow_planner/sledge_cache/${BRANCH_NAME}/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S")" \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=8 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.5 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://sledge.script.config.common,pkg://sledge.script.experiments,pkg://flow_planner.nuplan_simulation,pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.simulation,pkg://nuplan.planning.script.experiments]"