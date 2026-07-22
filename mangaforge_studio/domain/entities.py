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
    SVG = "svg"


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
