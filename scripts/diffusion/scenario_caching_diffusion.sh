JOB_NAME=scenario_caching
AUTOENCODER_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
DIFFUSION_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint
DIFFUSION_MODEL=dit_b_model # [dit_s_model, dit_b_model, dit_l_model, dit_xl_model]
SEED=0

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_diffusion.py \
py_func=scenario_caching \
seed=$SEED \
job_name=$JOB_NAME \
+diffusion=training_dit_model \
diffusion_model=$DIFFUSION_MODEL \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
diffusion_checkpoint=$DIFFUSION_CHECKPOINT
