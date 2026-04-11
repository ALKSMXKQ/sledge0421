


后半段流程：
original raw + edited raw
→ sledge_raw_feature_processing
→ encode_raster
→ diff mask + ROI mask
→ half-denoise refinement
→ semantic/compliance 筛选
→ 导出 simulator-ready sledge_vector.gz

将原始场景修改为稀缺高危三场景：
python sledge/script/build_multiscenario_raw_cache.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --glob-pattern "**/sledge_raw.gz" \
  --crossing-ratio 0.20 \
  --cut-in-ratio 0.30 \
  --hard-brake-ratio 0.50 \
  --mild-ratio 0.50 \
  --moderate-ratio 0.35 \
  --aggressive-ratio 0.15

  --max-scenes 500 \


半扩散生成：
python $SLEDGE_DEVKIT_ROOT/sledge/script/run_half_denoise_from_tiered_cache.py \
  --original-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --edited-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_refine_output \
  --scenario-cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt \
  --diffusion-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint \
  --guidance-scale 4.0 \
  --low-noise-start-step-seq 10,12,14 \
  --repair-attempts 6 \
  --save-visuals \
  --save-latents


评估实验
评估B0:sledge
python $SLEDGE_DEVKIT_ROOT/sledge/script/evaluate/evaluate_generated_scenario_cache.py \
  --scenario-cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache/log/us-ma-boston \
  --manifest /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache/scenario_manifest.csv \
  --method-name G0 \
  --emit-master-table-row \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/eval_G0_original_sledge

评估B1:仅编辑不扩散
python $SLEDGE_DEVKIT_ROOT/sledge/script/evaluate/evaluate_manifest_baseline.py \
  --manifest /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache/scenario_manifest.csv \
  --which edited \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/eval_B1_edit_only_accept70 \
  --accepted-only \
  --manifest-min-alignment 0.7 \
  --max-scenes 50


评估B3:完整框架
python $SLEDGE_DEVKIT_ROOT/sledge/script/evaluate/evaluate_manifest_baseline.py \
  --manifest /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache/scenario_manifest.csv \
  --which edited \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/eval_B1_edit_only

筛选掉有问题的仿真场景
cd /home16T/home8T_1/leitingting/sledge_workspace/sledge


python sledge/script/filter_existing_scenarios_by_metrics.py \
  --metrics-parquet /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/simulation/sledge_reactive_agents/2026.04.08.22.07.56/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2026.04.08.22.07.56.parquet \
  --scenario-cache-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_multiscenario0 \
  --finite-only \
  --in-place

将 sledge_raw.gz 转换为 sledge_vector.gz
python /home16T/home8T_1/leitingting/sledge_workspace/sledge/scripts/convert_raw_cache_to_sim_vector_cache.py \
  --input /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache/2021.05.12.22.00.38_veh-35_00215_00995 \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_semantic_check \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml

python /mnt/data/convert_raw_cache_to_sim_vector_cache.py \
  --input /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_semantic_check \
  --config /path/to/your_config.yaml \
  --max-scenes 50 \
  --save-raster-npz

将原始数据和修改后的数据对应都转换为sledge_vector.gz
python /home16T/home8T_1/leitingting/sledge_workspace/sledge/sledge/script/build_paired_original_edited_vector_caches.py \
  --manifest /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache/scenario_manifest.csv \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/paired_compare_cache \
  --max-scenes 50 \
  --accepted-only \
  --copy-edited-metadata