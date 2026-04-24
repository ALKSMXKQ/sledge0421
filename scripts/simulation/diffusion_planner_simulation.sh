#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
export SLEDGE_RANDOM_SAMPLE=1000
export SLEDGE_RANDOM_SEED=42

CHALLENGE="sledge_reactive_agents"

ARGS_FILE="/home16T/home8T_1/leitingting/Diffusion-Planner/checkpoints/args.json"
CKPT_FILE="/home16T/home8T_1/leitingting/Diffusion-Planner/checkpoints/model.pth"
SCENARIO_CACHE="/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario"

echo "=== CUDA check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible gpu 0:", torch.cuda.get_device_name(0))
PY

echo "=== File check ==="
test -f "$ARGS_FILE"
test -f "$CKPT_FILE"

python "$SLEDGE_DEVKIT_ROOT/sledge/script/run_simulation.py" \
  +simulation="$CHALLENGE" \
  planner=diffusion_planner \
  planner.diffusion_planner.config.args_file="$ARGS_FILE" \
  planner.diffusion_planner.ckpt_path="$CKPT_FILE" \
  planner.diffusion_planner.device=cuda \
  observation=sledge_agents_observation \
  scenario_builder=nuplan \
  cache.scenario_cache_path="$SCENARIO_CACHE" \
  worker=ray_distributed \
  distributed_mode='SINGLE_NODE' \
  worker.threads_per_node=8 \
  number_of_gpus_allocated_per_simulation=0.5 \
  enable_simulation_progress_bar=true \
  run_metric=true \
  hydra.searchpath="[pkg://sledge.script.config.common,pkg://sledge.script.experiments,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.simulation,pkg://nuplan.planning.script.experiments]"