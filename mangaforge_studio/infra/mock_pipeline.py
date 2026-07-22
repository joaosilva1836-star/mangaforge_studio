"""
Pipeline "falso" que gera imagens sólidas com o prompt escrito em cima.
Serve para desenvolver e testar toda a orquestração (services, API,
storyboard, export) numa máquina sem GPU NVIDIA — sem gastar tempo/VRAM
com SDXL de verdade. Trocar por DiffusersSDXLPipeline em produção.
"""
from __future__ import annotations

import hashlib
import os
import random
import uuid

from mangaforge_studio.interfaces.pipeline import (
    GenerationRequest,
    GenerationResult,
    ImageGenerationPipeline,
    PipelineFactory,
)


class MockImageGenerationPipeline(ImageGenerationPipeline):
    def __init__(self, output_dir: str = "./output/tmp"):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

    def _color_from_prompt(self, prompt: str) -> tuple[int, int, int]:
        h = hashlib.md5(prompt.encode()).hexdigest()
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from PIL import Image, ImageDraw

        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        color = self._color_from_prompt(request.prompt)
        img = Image.new("RGB", (request.width, request.height), color)
        draw = ImageDraw.Draw(img)
        text = request.prompt[:60]
        draw.text((10, 10), text, fill=(255, 255, 255))
        draw.text((10, 30), f"seed={seed}", fill=(255, 255, 255))

        out_path = os.path.join(self._output_dir, f"{uuid.uuid4()}.png")
        img.save(out_path)

        return GenerationResult(image_path=out_path, seed_used=seed, model_used="mock")

    def upscale(self, image_path: str, scale: int = 2) -> GenerationResult:
        from PIL import Image

        img = Image.open(image_path)
        img = img.resize((img.width * scale, img.height * scale))
        out_path = os.path.join(self._output_dir, f"{uuid.uuid4()}_upscaled.png")
        img.save(out_path)
        return GenerationResult(image_path=out_path, seed_used=0, model_used="mock-upscale")

    def inpaint(self, request: GenerationRequest) -> GenerationResult:
        return self.generate(request)


class MockPipelineFactory(PipelineFactory):
    def create(self, base_model: str, lora_paths: list[str]) -> ImageGenerationPipeline:
        return MockImageGenerationPipeline()
