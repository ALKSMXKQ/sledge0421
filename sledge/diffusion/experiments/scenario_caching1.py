from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from accelerate.logging import get_logger
from accelerate import Accelerator
import gzip
import pickle

from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint

logger = get_logger(__name__, log_level="INFO")

# 1. 定义场景映射表
SCENARIO_TYPE_MAP = {
    "high_magnitude_speed": 0,
    "medium_magnitude_speed": 1,
    "traversing_intersection": 2,
    "traversing_traffic_light_intersection": 3,
    "unknown": 4
}
INDEX_TO_SCENARIO = {v: k for k, v in SCENARIO_TYPE_MAP.items()}


def run_scenario_caching(cfg: DictConfig) -> None:
    """
    Applies the diffusion model generate and cache scenarios.
    """
    accelerator = Accelerator()

    logger.info("Building pipeline from checkpoint...")
    pipeline = build_pipeline_from_checkpoint(cfg)
    pipeline.to("cuda")
    logger.info("Building pipeline from checkpoint...DONE!")

    logger.info("Scenario caching (Directed Generation)...")
    storing_mechanism = FeatureCachePickle()
    current_cache_size: int = 0

    num_classes = 5

    # 2. 【核心修改】定向生成逻辑
    # 从配置中获取想要生成的场景类型，如果没有指定，默认生成 "all"
    target_scenario = cfg.get("target_scenario", "all")

    if target_scenario != "all" and target_scenario in SCENARIO_TYPE_MAP:
        # 如果指定了具体场景，就只生成这一种
        target_idx = SCENARIO_TYPE_MAP[target_scenario]
        class_labels = [target_idx] * cfg.inference_batch_size
        logger.info(f"Targeted Generation Mode: {target_scenario} (Index: {target_idx})")
    else:
        # 如果没有指定，则均匀混合生成所有的5种场景
        class_labels = list(range(num_classes)) * (cfg.inference_batch_size // num_classes)
        # 补齐不足一个 batch 的余数
        class_labels += list(range(num_classes))[:cfg.inference_batch_size - len(class_labels)]
        logger.info("Mixed Generation Mode: Generating a mix of all 5 scenarios.")

    num_total_batches = (cfg.cache.scenario_cache_size // cfg.inference_batch_size) + 1

    for _ in tqdm(range(num_total_batches), desc=f"Generating {target_scenario} Scenarios..."):
        sledge_vector_list = pipeline(
            class_labels=class_labels,
            num_inference_timesteps=cfg.num_inference_timesteps,
            guidance_scale=cfg.guidance_scale,
            num_classes=num_classes,
        )

        for sledge_vector, label_index in zip(sledge_vector_list, class_labels):
            sledge_vector_numpy: SledgeVector = sledge_vector.torch_to_numpy()

            scenario_name = INDEX_TO_SCENARIO[label_index]

            # 创建带有场景分类名称的文件夹
            base_path = (
                    Path(cfg.cache.scenario_cache_path)
                    / "log"
                    / scenario_name
                    / str(current_cache_size)
            )
            base_path.mkdir(parents=True, exist_ok=True)

            # 保存生成的道路和轨迹向量
            storing_mechanism.store_computed_feature_to_folder(
                base_path / "sledge_vector",
                sledge_vector_numpy
            )

            # 保存对应的标签信息文件（模拟打标过程）
            label_data = {"id": label_index, "name": scenario_name}
            with gzip.open(base_path / "scenario_type.gz", "wb") as f:
                pickle.dump(label_data, f)

            current_cache_size += 1
            if current_cache_size >= cfg.cache.scenario_cache_size:
                break

    logger.info("Scenario caching...DONE!")
    return None
