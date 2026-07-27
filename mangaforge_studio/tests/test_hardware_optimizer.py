"""
Testa a seleção automática de perfil do AI Optimizer a partir do que o
Hardware Manager detecta — sem depender de uma GPU real (SystemInfo é
construído à mão). Cobre os 5 tiers da spec (8/12/16/24/48 GB) e o
fallback sem GPU.
"""
from __future__ import annotations

import pytest

from mangaforge_studio.hardware.entities import GPUInfo, Precision, SystemInfo
from mangaforge_studio.hardware.optimizer import AIOptimizer


def _system(vram_mb: int, supports_bf16: bool = True) -> SystemInfo:
    return SystemInfo(
        cpu_count=8,
        ram_total_mb=32000,
        gpus=[GPUInfo(name="test-gpu", vram_total_mb=vram_mb, vram_free_mb=vram_mb, supports_bf16=supports_bf16)],
    )


@pytest.mark.parametrize(
    ("vram_mb", "expected_profile"),
    [
        (48 * 1024, "48gb"),
        (24 * 1024, "24gb"),
        (16 * 1024, "16gb"),
        (12 * 1024, "12gb"),
        (8 * 1024, "8gb"),
        (6 * 1024, "8gb"),  # abaixo do menor tier cai no perfil conservador
    ],
)
def test_profile_selection_by_vram(vram_mb: int, expected_profile: str) -> None:
    optimizer = AIOptimizer()
    profile = optimizer.recommend_profile(_system(vram_mb))
    assert profile.name == expected_profile


def test_no_gpu_falls_back_to_conservative_profile() -> None:
    optimizer = AIOptimizer()
    profile = optimizer.recommend_profile(SystemInfo(cpu_count=4, ram_total_mb=8000, gpus=[]))
    assert profile.name == "8gb"
    assert profile.enable_cpu_offload is True


def test_bf16_downgrades_to_fp16_when_gpu_lacks_support() -> None:
    optimizer = AIOptimizer()
    # 24 GB tier normalmente usa BF16, mas GPU sem suporte deve cair para FP16.
    profile = optimizer.recommend_profile(_system(24 * 1024, supports_bf16=False))
    assert profile.precision == Precision.FP16


def test_explain_mentions_gpu_and_profile() -> None:
    optimizer = AIOptimizer()
    system = _system(48 * 1024)
    profile = optimizer.recommend_profile(system)
    text = optimizer.explain(system, profile)
    assert "48gb" in text
    assert "test-gpu" in text
