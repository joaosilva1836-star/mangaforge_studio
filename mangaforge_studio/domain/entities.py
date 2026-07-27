"""
Entidades de domínio do MangaForge Studio.

Seguindo Clean Architecture: esta camada não depende de nada externo
(sem PyTorch, sem FastAPI, sem diffusers). São apenas dataclasses puras
que representam os conceitos do negócio.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


def new_id() -> str:
    return str(uuid.uuid4())


class AssetStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


class ExportFormat(str, Enum):
    PDF = "pdf"
    CBZ = "cbz"
    PNG = "png"
    WEBP = "webp"
    JPG = "jpg"
    SVG = "svg"


class PageSize(str, Enum):
    """Tamanhos de papel suportados na exportação. B5 é o padrão de mangá
    japonês."""
    A4 = "a4"
    A5 = "a5"
    B5 = "b5"
    LETTER = "letter"
    CUSTOM = "custom"


class Orientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


# Dimensões físicas em milímetros (largura, altura) no modo retrato.
PAGE_SIZE_MM: dict[PageSize, tuple[float, float]] = {
    PageSize.A4: (210.0, 297.0),
    PageSize.A5: (148.0, 210.0),
    PageSize.B5: (176.0, 250.0),
    PageSize.LETTER: (215.9, 279.4),
}

# DPIs padrão da spec: web / tela / impressão / alta qualidade / arquivo mestre.
STANDARD_DPIS: tuple[int, ...] = (72, 150, 300, 600, 1200)


@dataclass
class ExportSettings:
    """Todas as opções de exportação profissional num único objeto de valor.

    Puro (sem PIL/torch): descreve o *quê* exportar; o *como* fica nos
    services/infra. Dimensões físicas + DPI viram pixels no ExportService.
    """
    fmt: ExportFormat = ExportFormat.PNG
    dpi: int = 300
    page_size: PageSize = PageSize.A4
    orientation: Orientation = Orientation.PORTRAIT
    custom_size_mm: tuple[float, float] | None = None  # usado quando page_size == CUSTOM
    quality: int = 95            # 1-100, para JPG/WEBP com perda
    lossless: bool = False       # WEBP sem perda
    scale: float = 1.0           # multiplica a resolução final
    margins_mm: float = 0.0      # margem interna (branca) em cada lado
    bleed_mm: float = 0.0        # sangria adicionada além do tamanho do papel
    grayscale: bool = False      # converter para tons de cinza (preserva hachuras)
    smart: bool = True           # limpeza inteligente antes de exportar
    use_gpu: bool | None = None  # None = automático (CUDA se houver, senão CPU)

    def physical_size_mm(self) -> tuple[float, float]:
        """Tamanho do papel em mm já considerando orientação e tamanho custom
        (sem sangria)."""
        if self.page_size == PageSize.CUSTOM:
            if not self.custom_size_mm:
                raise ValueError("page_size=custom exige custom_size_mm=(largura_mm, altura_mm)")
            w, h = self.custom_size_mm
        else:
            w, h = PAGE_SIZE_MM[self.page_size]
        if self.orientation == Orientation.LANDSCAPE:
            w, h = h, w
        return (w, h)

    def target_pixels(self) -> tuple[int, int]:
        """Converte tamanho físico (mm) + sangria + DPI + escala em pixels."""
        w_mm, h_mm = self.physical_size_mm()
        w_mm += 2 * self.bleed_mm
        h_mm += 2 * self.bleed_mm
        px_per_mm = self.dpi / 25.4
        w_px = max(1, int(round(w_mm * px_per_mm * self.scale)))
        h_px = max(1, int(round(h_mm * px_per_mm * self.scale)))
        return (w_px, h_px)


@dataclass
class StyleProfile:
    """Perfil de estilo aprendido a partir de um dataset do usuário."""
    id: str = field(default_factory=new_id)
    name: str = "default"
    lora_paths: list[str] = field(default_factory=list)
    base_model: str = "sdxl"  # "sdxl" | "flux"
    trigger_words: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Character:
    id: str = field(default_factory=new_id)
    name: str = ""
    description: str = ""
    reference_image_paths: list[str] = field(default_factory=list)
    style_profile_id: str | None = None
    seed: int | None = None
    status: AssetStatus = AssetStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Scene:
    id: str = field(default_factory=new_id)
    description: str = ""
    character_ids: list[str] = field(default_factory=list)
    background_prompt: str = ""
    status: AssetStatus = AssetStatus.DRAFT


@dataclass
class Panel:
    """Um quadro dentro de uma página."""
    id: str = field(default_factory=new_id)
    scene_id: str | None = None
    prompt: str = ""
    layout_box: tuple[float, float, float, float] = (0, 0, 1, 1)  # x, y, w, h (normalizado 0-1)
    image_path: str | None = None
    status: AssetStatus = AssetStatus.DRAFT


@dataclass
class Page:
    id: str = field(default_factory=new_id)
    chapter_id: str | None = None
    page_number: int = 1
    panels: list[Panel] = field(default_factory=list)
    resolution: tuple[int, int] = (1024, 1536)
    status: AssetStatus = AssetStatus.DRAFT


@dataclass
class Chapter:
    id: str = field(default_factory=new_id)
    title: str = ""
    pages: list[Page] = field(default_factory=list)
    order: int = 1


@dataclass
class Project:
    id: str = field(default_factory=new_id)
    title: str = ""
    style_profile_id: str | None = None
    characters: list[str] = field(default_factory=list)  # character ids
    chapters: list[Chapter] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
