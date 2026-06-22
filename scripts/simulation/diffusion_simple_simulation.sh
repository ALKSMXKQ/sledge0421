#!/bin/bash
export PYTHONPATH="/home16T/home8T_1/leitingting/Diffusion-Planner:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

CHALLENGE=sledge_reactive_agents
SCENARIO_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_simulation.py \
  +simulation=$CHALLENGE \
  planner=diffusion_planner \
  observation=sledge_agents_observation \
  scenario_builder=nuplan \
  cache.scenario_cache_path=$SCENARIO_CACHE_PATH \
  worker=sequential \
  number_of_cpus_allocated_per_simulation=1 \
  number_of_gpus_allocated_per_simulation=1