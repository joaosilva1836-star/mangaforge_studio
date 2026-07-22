"""
Implementação concreta do ImageGenerationPipeline usando a lib `diffusers`.

Esta é a ÚNICA camada que sabe que PyTorch/diffusers existem. Todo o
resto do sistema (services, domain) não importa torch nem diffusers
diretamente — isso é o que permite trocar para ComfyUI/ONNX Runtime
no futuro (módulo AI Optimizer) sem alterar a lógica de negócio.

NOTA: requer GPU NVIDIA com CUDA + `pip install diffusers transformers
accelerate torch pillow`. Em máquinas sem GPU compatível, use
MockImageGenerationPipeline (infra/mock_pipeline.py) para desenvolver
e testar a integração sem custo de VRAM.
"""
from __future__ import annotations

import os
import random
import uuid

from mangaforge_studio.interfaces.pipeline import (
    GenerationRequest,
    GenerationResult,
    ImageGenerationPipeline,
    PipelineFactory,
)


class DiffusersSDXLPipeline(ImageGenerationPipeline):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_paths: list[str] | None = None,
        device: str = "cuda",
        dtype: str = "fp16",
        output_dir: str = "./output/tmp",
    ):
        import torch
        from diffusers import StableDiffusionXLPipeline

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._device = device

        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        ).to(device)

        # Otimizações de VRAM (o AI Optimizer decide quais ativar conforme o hardware)
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers pode não estar disponível; segue sem ele

        for lora_path in lora_paths or []:
            self._pipe.load_lora_weights(lora_path)

        self._model_id = model_id

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import torch

        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=self._device).manual_seed(seed)

        result = self._pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        image = result.images[0]

        out_path = os.path.join(self._output_dir, f"{uuid.uuid4()}.png")
        image.save(out_path)

        return GenerationResult(
            image_path=out_path,
            seed_used=seed,
            model_used=self._model_id,
            metadata={"steps": request.steps, "guidance_scale": request.guidance_scale},
        )

    def upscale(self, image_path: str, scale: int = 2) -> GenerationResult:
        # Placeholder: plugar Real-ESRGAN ou similar aqui.
        raise NotImplementedError("Upscale ainda não implementado neste MVP")

    def inpaint(self, request: GenerationRequest) -> GenerationResult:
        # Placeholder: trocar para StableDiffusionXLInpaintPipeline aqui.
        raise NotImplementedError("Inpaint ainda não implementado neste MVP")


class DiffusersPipelineFactory(PipelineFactory):
    """Factory Pattern: escolhe SDXL ou FLUX conforme o base_model do
    StyleProfile. Se receber um `hardware_profile` (vindo do AI Optimizer),
    usa a precisão e o device escolhidos automaticamente por ele em vez
    de valores fixos — é o que conecta o Hardware Manager a este módulo
    sem o Studio precisar saber nada sobre GPUs."""

    def __init__(self, device: str = "cuda", dtype: str = "fp16", hardware_profile=None):
        if hardware_profile is not None:
            dtype = hardware_profile.precision.value
        self._device = device
        self._dtype = dtype
        self._hardware_profile = hardware_profile
        self._cache: dict[str, ImageGenerationPipeline] = {}

    def create(self, base_model: str, lora_paths: list[str]) -> ImageGenerationPipeline:
        cache_key = f"{base_model}:{','.join(sorted(lora_paths))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if base_model == "sdxl":
            pipeline = DiffusersSDXLPipeline(lora_paths=lora_paths, device=self._device, dtype=self._dtype)
            if self._hardware_profile is not None and self._hardware_profile.enable_cpu_offload:
                try:
                    pipeline._pipe.enable_model_cpu_offload()
                except Exception:
                    pass  # sem suporte no ambiente atual; segue rodando normalmente na GPU
        elif base_model == "flux":
            raise NotImplementedError(
                "Pipeline FLUX ainda não implementado neste MVP — adicionar "
                "FluxPipeline do diffusers aqui quando disponível"
            )
        else:
            raise ValueError(f"base_model desconhecido: {base_model}")

        self._cache[cache_key] = pipeline
        return pipeline
