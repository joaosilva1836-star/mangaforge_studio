"""
Entidades puras do Hardware Manager + AI Optimizer.

Sem dependência de torch/nvidia-smi aqui — só descrevem os dados e os
perfis, exatamente como a spec original definiu (8/12/16/24/48GB).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class GPUInfo:
    name: str = "unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cuda_available: bool = False
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    compute_capability: str = "unknown"
    supports_bf16: bool = False  # Ampere+ (RTX 30xx/A40/A100 em diante)


@dataclass
class SystemInfo:
    cpu_count: int = 0
    ram_total_mb: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)


@dataclass
class HardwareProfile:
    """Um dos 5 perfis definidos na spec (8/12/16/24/48GB), já traduzido
    em parâmetros concretos que o AI Optimizer aplica na geração."""
    name: str
    vram_tier_gb: int
    precision: Precision
    max_resolution: int
    default_batch_size: int
    max_concurrent_loras: int
    enable_cpu_offload: bool
    enable_quantization: bool
    enable_multi_model_cache: bool  # "carregar vários modelos simultaneamente" (perfil 48GB)
    notes: str


# Perfis fixos conforme a especificação original do MangaForge AI Enterprise.
# O AI Optimizer escolhe um destes com base na VRAM detectada.
PROFILES: dict[str, HardwareProfile] = {
    "8gb": HardwareProfile(
        name="8gb", vram_tier_gb=8, precision=Precision.FP16,
        max_resolution=512, default_batch_size=1, max_concurrent_loras=1,
        enable_cpu_offload=True, enable_quantization=True, enable_multi_model_cache=False,
        notes="Modelos leves, batch pequeno, 512x512, CPU offload, quantização.",
    ),
    "12gb": HardwareProfile(
        name="12gb", vram_tier_gb=12, precision=Precision.FP16,
        max_resolution=1024, default_batch_size=1, max_concurrent_loras=2,
        enable_cpu_offload=False, enable_quantization=False, enable_multi_model_cache=False,
        notes="SDXL, LoRAs, 1024x1024, ControlNet.",
    ),
    "16gb": HardwareProfile(
        name="16gb", vram_tier_gb=16, precision=Precision.FP16,
        max_resolution=1024, default_batch_size=2, max_concurrent_loras=3,
        enable_cpu_offload=False, enable_quantization=True, enable_multi_model_cache=False,
        notes="FLUX quantizado, ControlNet, múltiplos LoRAs.",
    ),
    "24gb": HardwareProfile(
        name="24gb", vram_tier_gb=24, precision=Precision.BF16,
        max_resolution=1536, default_batch_size=4, max_concurrent_loras=4,
        enable_cpu_offload=False, enable_quantization=False, enable_multi_model_cache=False,
        notes="Treinamento de LoRA, SDXL, FLUX, batch alto.",
    ),
    "48gb": HardwareProfile(
        name="48gb", vram_tier_gb=48, precision=Precision.BF16,
        max_resolution=2048, default_batch_size=8, max_concurrent_loras=8,
        enable_cpu_offload=False, enable_quantization=False, enable_multi_model_cache=True,
        notes="Modo Enterprise: múltiplos modelos simultâneos, batch muito alto, 2048x2048+.",
    ),
}
