from __future__ import annotations

from pydantic import BaseModel


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
    format: str  # "pdf" | "cbz" | "png" | "webp" | "svg"
