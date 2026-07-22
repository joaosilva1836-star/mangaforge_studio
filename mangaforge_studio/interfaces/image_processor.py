"""
Contrato do processador de imagem usado na exportação profissional.

A implementação concreta (Pillow/CPU ou CUDA/GPU) fica em infra/. O
ExportService fala apenas com esta abstração, então trocar CPU por GPU
(ou por outra lib) não altera a lógica de exportação — mesma ideia de
Clean Architecture já usada no pipeline de geração.

O tipo `Image` do Pillow é importado só sob TYPE_CHECKING para manter a
camada de interfaces sem dependência de runtime em PIL.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


class ImageProcessor(ABC):
    """Operações de pixel de alta qualidade para preparar a página antes de
    salvar. Todos os métodos recebem e devolvem imagens Pillow RGB/L."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """`"cuda"` ou `"cpu"` — usado para relatório/telemetria."""

    @abstractmethod
    def resize(self, image: Image, size: tuple[int, int]) -> Image:
        """Redimensiona preservando nitidez (reamostragem de alta qualidade)."""

    @abstractmethod
    def denoise(self, image: Image) -> Image:
        """Remove ruído leve sem destruir linhas finas/hachuras."""

    @abstractmethod
    def sharpen_lines(self, image: Image) -> Image:
        """Realça bordas (unsharp mask) para manter linhas nítidas após resize."""

    @abstractmethod
    def to_grayscale(self, image: Image) -> Image:
        """Converte para tons de cinza preservando contraste de hachuras."""
