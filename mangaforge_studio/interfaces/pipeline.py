"""
Abstração do pipeline de geração de imagem.

A camada de serviço fala apenas com `ImageGenerationPipeline`. A
implementação concreta (SDXL via diffusers, FLUX, ComfyUI remoto, etc.)
fica isolada em infra/. Isso é o que permite ao AI Optimizer (módulo
futuro) trocar de modelo/precisão sem que o Studio saiba de nada.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.0
    seed: int | None = None
    lora_paths: list[str] = field(default_factory=list)
    control_image_path: str | None = None  # para ControlNet (pose, linha, etc.)
    init_image_path: str | None = None  # para img2img / inpainting
    mask_image_path: str | None = None  # para inpainting/outpainting


@dataclass
class GenerationResult:
    image_path: str
    seed_used: int
    model_used: str
    metadata: dict = field(default_factory=dict)


class ImageGenerationPipeline(ABC):
    """Contrato que qualquer backend de geração deve implementar."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    @abstractmethod
    def upscale(self, image_path: str, scale: int = 2) -> GenerationResult: ...

    @abstractmethod
    def inpaint(self, request: GenerationRequest) -> GenerationResult: ...


class PipelineFactory(ABC):
    """Factory Pattern: decide qual pipeline instanciar (SDXL vs FLUX)
    conforme o StyleProfile e o hardware disponível."""

    @abstractmethod
    def create(self, base_model: str, lora_paths: list[str]) -> ImageGenerationPipeline: ...
