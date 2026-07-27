"""
Hardware Manager: detecta GPU NVIDIA, VRAM, CUDA, RAM e CPU.

Usa `nvidia-smi` (sempre disponível em qualquer máquina com driver NVIDIA
instalado, com ou sem PyTorch) como fonte primária — assim funciona mesmo
antes de instalar torch/diffusers. Usa `torch.cuda` como fonte secundária
para checar compute capability / suporte a bf16 com mais precisão.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

from mangaforge_studio.hardware.entities import GPUInfo, SystemInfo


class HardwareDetector:
    def detect(self) -> SystemInfo:
        gpus = self._detect_gpus_nvidia_smi()
        if not gpus:
            gpus = self._detect_gpus_torch()

        return SystemInfo(
            cpu_count=os.cpu_count() or 0,
            ram_total_mb=self._detect_ram_mb(),
            gpus=gpus,
        )

    # --- GPU ---

    def _detect_gpus_nvidia_smi(self) -> list[GPUInfo]:
        if shutil.which("nvidia-smi") is None:
            return []

        query = (
            "name,memory.total,memory.free,driver_version,compute_cap"
        )
        try:
            output = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return []

        gpus = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            name, mem_total, mem_free, driver, compute_cap = parts[:5]
            try:
                major = int(compute_cap.split(".")[0])
            except (ValueError, IndexError):
                major = 0

            gpus.append(
                GPUInfo(
                    name=name,
                    vram_total_mb=int(float(mem_total)),
                    vram_free_mb=int(float(mem_free)),
                    cuda_available=True,
                    cuda_version=self._detect_cuda_version(),
                    driver_version=driver,
                    compute_capability=compute_cap,
                    supports_bf16=major >= 8,  # Ampere (RTX 30xx/A40/A100) e superior
                )
            )
        return gpus

    def _detect_gpus_torch(self) -> list[GPUInfo]:
        try:
            import torch
        except ImportError:
            return []

        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            gpus.append(
                GPUInfo(
                    name=props.name,
                    vram_total_mb=int(total_bytes / (1024 * 1024)),
                    vram_free_mb=int(free_bytes / (1024 * 1024)),
                    cuda_available=True,
                    cuda_version=torch.version.cuda or "unknown",
                    driver_version="unknown",
                    compute_capability=f"{props.major}.{props.minor}",
                    supports_bf16=props.major >= 8,
                )
            )
        return gpus

    def _detect_cuda_version(self) -> str:
        try:
            output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=10).decode()
            match = re.search(r"CUDA Version:\s*([\d.]+)", output)
            return match.group(1) if match else "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    # --- RAM ---

    def _detect_ram_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except (FileNotFoundError, ValueError, IndexError):
            pass
        return 0
