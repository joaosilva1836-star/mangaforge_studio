from __future__ import annotations

from pydantic import BaseModel, Field


class CreateCharacterRequest(BaseModel):
    name: str
    description: str
    style_profile_id: str | None = None
    seed: int | None = None


class CharacterResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    reference_image_paths: list[str]


class GenerateReferenceSheetRequest(BaseModel):
    poses: list[str] | None = None


class BuildChapterRequest(BaseModel):
    title: str
    beats: list[dict]
    order: int = 1
    panels_per_page: int = 4


class ExportChapterRequest(BaseModel):
    page_image_paths: list[str]
    format: str  # "pdf" | "cbz" | "png" | "webp" | "jpg" | "svg"


class ExportSettingsSchema(BaseModel):
    """Opções de exportação profissional (MangaForge Export)."""
    format: str = "png"                              # png|webp|jpg|pdf|cbz|svg
    dpi: int = Field(default=300, ge=1, le=2400)     # 72/150/300/600/1200
    page_size: str = "a4"                            # a4|a5|b5|letter|custom
    orientation: str = "portrait"                    # portrait|landscape
    custom_size_mm: tuple[float, float] | None = None
    quality: int = Field(default=95, ge=1, le=100)
    lossless: bool = False
    scale: float = Field(default=1.0, gt=0)
    margins_mm: float = Field(default=0.0, ge=0)
    bleed_mm: float = Field(default=0.0, ge=0)
    grayscale: bool = False
    smart: bool = True
    use_gpu: bool | None = None


class ExportPageRequest(BaseModel):
    source_image_path: str
    chapter_number: int = 1
    page_number: int = 1
    settings: ExportSettingsSchema = Field(default_factory=ExportSettingsSchema)


class ExportChapterProRequest(BaseModel):
    page_image_paths: list[str]
    chapter_number: int = 1
    settings: ExportSettingsSchema = Field(default_factory=ExportSettingsSchema)


class ExportMangaRequest(BaseModel):
    # chave = número do capítulo, valor = páginas renderizadas desse capítulo
    chapters: dict[int, list[str]]
    settings: ExportSettingsSchema = Field(default_factory=ExportSettingsSchema)


class ExportResultResponse(BaseModel):
    output_path: str
    file_url: str | None = None
    backend: str
    settings: ExportSettingsSchema


class ExportOptionsResponse(BaseModel):
    formats: list[str]
    dpis: list[int]
    page_sizes: list[str]
    orientations: list[str]
