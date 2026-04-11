CHALLENGE=sledge_reactive_agents 
SCENARIO_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario1

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_closed_planner \
observation=sledge_agents_observation \
scenario_builder=nuplan \
cache.scenario_cache_path=$SCENARIO_CACHE_PATH