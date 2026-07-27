"""
AI Optimizer: pega o que o Hardware Manager detectou e decide, sozinho,
qual perfil usar e quais parâmetros aplicar na geração — sem o usuário
precisar configurar nada manualmente.
"""
from __future__ import annotations

from mangaforge_studio.hardware.entities import PROFILES, HardwareProfile, Precision, SystemInfo


class AIOptimizer:
    # Limites inferiores de VRAM (em GB) pra cada perfil, do menor pro maior.
    _TIERS = [(48, "48gb"), (24, "24gb"), (16, "16gb"), (12, "12gb"), (8, "8gb")]

    def recommend_profile(self, system: SystemInfo) -> HardwareProfile:
        """Escolhe o melhor perfil com base na GPU com mais VRAM livre.
        Se não houver GPU NVIDIA disponível, cai no perfil mais conservador
        (8gb) com CPU offload forçado — ainda funciona, só mais lento."""
        if not system.gpus:
            return PROFILES["8gb"]

        best_gpu = max(system.gpus, key=lambda g: g.vram_total_mb)
        vram_gb = best_gpu.vram_total_mb / 1024

        for min_gb, profile_name in self._TIERS:
            if vram_gb >= min_gb * 0.9:  # 90% de margem pra overhead do driver/SO
                profile = PROFILES[profile_name]
                # Ajusta a precisão dinamicamente: só usa bf16 se a GPU realmente suportar
                if profile.precision == Precision.BF16 and not best_gpu.supports_bf16:
                    profile = self._downgrade_precision(profile)
                return profile

        return PROFILES["8gb"]

    def _downgrade_precision(self, profile: HardwareProfile) -> HardwareProfile:
        from dataclasses import replace

        return replace(profile, precision=Precision.FP16)

    def explain(self, system: SystemInfo, profile: HardwareProfile) -> str:
        if not system.gpus:
            return (
                "Nenhuma GPU NVIDIA detectada — rodando em modo degradado "
                f"(perfil '{profile.name}', CPU offload forçado). Gerações vão ser lentas."
            )
        gpu = system.gpus[0]
        return (
            f"GPU detectada: {gpu.name} ({gpu.vram_total_mb} MB VRAM). "
            f"Perfil escolhido: '{profile.name}' — {profile.notes} "
            f"Precisão: {profile.precision.value}."
        )
