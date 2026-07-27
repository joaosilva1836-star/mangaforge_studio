"""
Implementações concretas do ImageProcessor.

- `PillowImageProcessor`: CPU, sempre disponível (só depende de Pillow).
- `CudaImageProcessor`: usa PyTorch/CUDA para redimensionar/filtrar na GPU
  quando há placa NVIDIA; cai para o caminho CPU em qualquer falha.

`create_image_processor()` escolhe automaticamente: GPU se disponível
(e não desabilitada), senão CPU — atendendo ao requisito da spec de
"usar CUDA se houver GPU NVIDIA, senão CPU automaticamente".
"""
from __future__ import annotations

from PIL import Image, ImageFilter

from mangaforge_studio.interfaces.image_processor import ImageProcessor


class PillowImageProcessor(ImageProcessor):
    @property
    def backend(self) -> str:
        return "cpu"

    def resize(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        return image.resize(size, Image.LANCZOS)

    def denoise(self, image: Image.Image) -> Image.Image:
        # Mediana de raio 1: remove pontos isolados de ruído preservando linhas.
        return image.filter(ImageFilter.MedianFilter(size=3))

    def sharpen_lines(self, image: Image.Image) -> Image.Image:
        # Unsharp mask suave: recupera nitidez perdida na reamostragem sem halos.
        return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=2))

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        return image.convert("L")


class CudaImageProcessor(ImageProcessor):
    """Redimensiona/filtra na GPU via PyTorch. Mantém uma instância CPU como
    fallback para qualquer operação que falhe ou não valha a pena na GPU."""

    def __init__(self) -> None:
        import torch  # importado só quando a GPU é escolhida

        self._torch = torch
        self._device = "cuda"
        self._cpu = PillowImageProcessor()

    @property
    def backend(self) -> str:
        return "cuda"

    def _to_tensor(self, image: Image.Image):  # noqa: ANN202 - tensor do torch
        import numpy as np

        arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        # HWC -> NCHW
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    def _to_image(self, tensor) -> Image.Image:  # noqa: ANN001
        import numpy as np

        arr = tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    def resize(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        if image.size == size:
            return image
        try:
            tensor = self._to_tensor(image)
            w, h = size
            resized = self._torch.nn.functional.interpolate(
                tensor, size=(h, w), mode="bicubic", align_corners=False, antialias=True
            )
            return self._to_image(resized)
        except Exception:  # noqa: BLE001 — qualquer erro de GPU cai pro CPU
            return self._cpu.resize(image, size)

    def denoise(self, image: Image.Image) -> Image.Image:
        # Denoise fino é barato e seguro na CPU (preserva linhas); mantém CPU.
        return self._cpu.denoise(image)

    def sharpen_lines(self, image: Image.Image) -> Image.Image:
        return self._cpu.sharpen_lines(image)

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        return self._cpu.to_grayscale(image)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def create_image_processor(use_gpu: bool | None = None) -> ImageProcessor:
    """Factory: `use_gpu=None` decide automaticamente (CUDA se houver, senão
    CPU); `True`/`False` força. Nunca levanta — se a GPU falhar ao inicializar,
    cai para CPU."""
    want_gpu = _cuda_available() if use_gpu is None else use_gpu
    if want_gpu:
        try:
            return CudaImageProcessor()
        except Exception:  # noqa: BLE001
            return PillowImageProcessor()
    return PillowImageProcessor()
