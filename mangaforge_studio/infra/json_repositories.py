"""
Implementação concreta dos repositórios usando arquivos JSON em disco.

Simples e 100% offline — adequado para o MVP. Pode ser trocado depois
por SQLite/Postgres implementando as mesmas interfaces em
interfaces/repositories.py, sem tocar nos services.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import TypeVar

from mangaforge_studio.domain.entities import Character, Page, Project, Scene, StyleProfile
from mangaforge_studio.interfaces.repositories import (
    CharacterRepository,
    PageRepository,
    ProjectRepository,
    SceneRepository,
    StyleProfileRepository,
)

T = TypeVar("T")


class _JsonRepository:
    """Base genérica: cada entidade vira um arquivo `<id>.json` dentro
    de `base_dir`."""

    entity_cls: type

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, entity_id: str) -> str:
        return os.path.join(self._base_dir, f"{entity_id}.json")

    def get(self, entity_id: str):
        path = self._path(entity_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self.entity_cls(**data)

    def save(self, entity):
        with open(self._path(entity.id), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(entity), f, ensure_ascii=False, indent=2, default=str)
        return entity

    def delete(self, entity_id: str) -> None:
        path = self._path(entity_id)
        if os.path.exists(path):
            os.remove(path)

    def list(self):
        entities = []
        for filename in os.listdir(self._base_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self._base_dir, filename), encoding="utf-8") as f:
                    entities.append(self.entity_cls(**json.load(f)))
        return entities


class JsonCharacterRepository(_JsonRepository, CharacterRepository):
    entity_cls = Character

    def __init__(self, base_dir: str = "./data/characters"):
        super().__init__(base_dir)


class JsonSceneRepository(_JsonRepository, SceneRepository):
    entity_cls = Scene

    def __init__(self, base_dir: str = "./data/scenes"):
        super().__init__(base_dir)


class JsonPageRepository(_JsonRepository, PageRepository):
    """Sobrescreve get/list pois Page contém uma lista aninhada de
    Panel (dataclass); json.load por si só devolveria dicts crus para
    cada panel, então precisamos reidratar manualmente."""

    entity_cls = Page

    def __init__(self, base_dir: str = "./data/pages"):
        super().__init__(base_dir)

    def _hydrate(self, data: dict) -> Page:
        from mangaforge_studio.domain.entities import Panel

        data = dict(data)
        data["panels"] = [Panel(**p) for p in data.get("panels", [])]
        data["resolution"] = tuple(data.get("resolution", (1024, 1536)))
        return Page(**data)

    def get(self, entity_id: str):
        path = self._path(entity_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return self._hydrate(json.load(f))

    def list(self):
        pages = []
        for filename in os.listdir(self._base_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self._base_dir, filename), encoding="utf-8") as f:
                    pages.append(self._hydrate(json.load(f)))
        return pages


class JsonProjectRepository(_JsonRepository, ProjectRepository):
    entity_cls = Project

    def __init__(self, base_dir: str = "./data/projects"):
        super().__init__(base_dir)


class JsonStyleProfileRepository(_JsonRepository, StyleProfileRepository):
    entity_cls = StyleProfile

    def __init__(self, base_dir: str = "./data/styles"):
        super().__init__(base_dir)
