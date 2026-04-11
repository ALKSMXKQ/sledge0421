import inspect
from typing import List, Optional, Union

import torch

from diffusers import DiffusionPipeline
from diffusers.models import DiTTransformer2DModel
from diffusers.schedulers import DDPMScheduler

from sledge.autoencoder.modeling.models.rvae.rvae_decoder import RVAEDecoder
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVector


class LDMPipeline(DiffusionPipeline):
    """Latent Diffusion Model pipeline for unconditional or img2img-style scene generation."""

    def __init__(self, decoder: RVAEDecoder, transformer: DiTTransformer2DModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(decoder=decoder, transformer=transformer, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        num_inference_timesteps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_classes: int = 4,
        init_latents: Optional[torch.Tensor] = None,
        start_timestep_index: int = 0,
        preserve_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Union[List[SledgeVector], tuple[List[SledgeVector], torch.Tensor]]:
        """
        Generate scenes from pure noise or from an encoded scene latent.

        Args:
            class_labels: map/city labels.
            num_inference_timesteps: number of reverse denoising steps.
            guidance_scale: classifier-free guidance scale.
            generator: optional generator.
            eta: scheduler eta where supported.
            num_classes: number of class labels used during training.
            init_latents: optional encoded scene latent. When provided, the pipeline
                performs img2img-style generation by first noising this latent and then
                denoising from ``start_timestep_index`` onward.
            start_timestep_index: reverse process starting index over scheduler.timesteps.
                ``0`` means standard generation from the noisiest step. ``30`` with
                50 steps means: add noise at scheduler.timesteps[30] and denoise from there.
            preserve_mask: optional latent-space mask shaped ``[B,1,H,W]`` or ``[B,C,H,W]``.
                Masked regions are repeatedly reset to the source latent at every step,
                which preserves user-edited entities during denoising.
            noise: optional noise tensor for deterministic experiments.
            return_latents: whether to also return the final latent tensor.
        """

        batch_size = len(class_labels)
        class_labels_tensor, class_labels_input = self._prepare_class_labels(class_labels, num_classes)

        self.scheduler.set_timesteps(num_inference_timesteps, device=self.device)
        timesteps = self.scheduler.timesteps
        start_timestep_index = int(max(0, min(start_timestep_index, len(timesteps) - 1)))

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {"eta": eta} if accepts_eta else {}

        latent_shape = (
            batch_size,
            self.transformer.config.in_channels,
            self.transformer.config.sample_size,
            self.transformer.config.sample_size,
        )

        if init_latents is None:
            latents = torch.randn(latent_shape, generator=generator, device=self.device)
            latents = latents * self.scheduler.init_noise_sigma
            source_latents = None
            source_noise = None
            denoise_timesteps = timesteps
        else:
            init_latents = init_latents.to(self.device)
            if init_latents.shape != latent_shape:
                raise ValueError(f"init_latents shape {tuple(init_latents.shape)} does not match expected {latent_shape}")
            source_latents = init_latents
            source_noise = noise if noise is not None else torch.randn(source_latents.shape, generator=generator, device=self.device, dtype=source_latents.dtype)
            source_noise = source_noise.to(self.device)
            start_timestep = timesteps[start_timestep_index]
            timestep_batch = torch.full((batch_size,), int(start_timestep.item()), device=self.device, dtype=torch.long)
            latents = self.scheduler.add_noise(source_latents, source_noise, timestep_batch)
            denoise_timesteps = timesteps[start_timestep_index:]

        preserve_mask = self._prepare_preserve_mask(preserve_mask, latents) if preserve_mask is not None else None

        for t in self.progress_bar(denoise_timesteps):
            if preserve_mask is not None and source_latents is not None and source_noise is not None:
                timestep_batch = torch.full((batch_size,), int(t.item()), device=self.device, dtype=torch.long)
                source_noised = self.scheduler.add_noise(source_latents, source_noise, timestep_batch)
                latents = preserve_mask * source_noised + (1.0 - preserve_mask) * latents

            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # --- 再次修正后的调试代码 ---
            if class_labels is not None:
                # 只要 class_labels 有 tolist 属性就转，没有就当 list 处理
                labels_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)

                max_val = max(labels_list)
                min_val = min(labels_list)

                # 获取模型配置的上限（DiT 常用 num_embeds 这个 key）
                limit = getattr(self.transformer.config, "num_embeds", "未知")

                print("\n" + "=" * 30)
                print(f"[DEBUG] 类别标签检查:")
                print(f"  - 标签内容: {labels_list}")
                print(f"  - 标签范围: [{min_val}, {max_val}]")
                print(f"  - 模型允许上限 (num_embeds): {limit}")
                print("=" * 30 + "\n")

                if limit != "未知" and max_val >= limit:
                    print(f"🚨 警告: 发现越界！最大值 {max_val} 必须小于 {limit}")
            # --- 调试代码结束 ---
            noise_prediction = self.transformer(
                hidden_states=latent_model_input,
                class_labels=class_labels_input,
                timestep=t.unsqueeze(0),
            ).sample

            cond_eps, uncond_eps = torch.split(noise_prediction, batch_size, dim=0)
            guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            latents = self.scheduler.step(guided_eps, t, latents, **extra_kwargs).prev_sample

        vector_output = self.decoder.decode(latents).unpack()
        if return_latents:
            return vector_output, latents
        return vector_output

    def _prepare_class_labels(self, class_labels: List[int], num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        class_labels_tensor = torch.tensor(class_labels, device=self.device).reshape(-1)
        class_null = torch.tensor([num_classes] * len(class_labels), device=self.device)
        class_labels_input = torch.cat([class_labels_tensor, class_null], 0)
        return class_labels_tensor, class_labels_input

    @staticmethod
    def _prepare_preserve_mask(mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        if mask.ndim != 4:
            raise ValueError(f"preserve_mask must be 4D, got shape {tuple(mask.shape)}")
        if mask.shape[1] == 1:
            mask = mask.repeat(1, latents.shape[1], 1, 1)
        if mask.shape != latents.shape:
            raise ValueError(f"preserve_mask shape {tuple(mask.shape)} must match latents {tuple(latents.shape)} or have 1 channel")
        return mask.clamp(0.0, 1.0)
