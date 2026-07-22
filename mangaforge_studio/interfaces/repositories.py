"""
Contratos de repositório (Repository Pattern).

A camada de serviço depende apenas destas interfaces, nunca de uma
implementação concreta (ex.: arquivos em disco, SQLite, Postgres).
Isso permite trocar o backend de armazenamento sem tocar na lógica
de negócio — um dos pilares do Clean Architecture pedido na spec.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mangaforge_studio.domain.entities import (
    Character,
    Page,
    Project,
    Scene,
    StyleProfile,
)

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    @abstractmethod
    def get(self, entity_id: str) -> T | None: ...

    @abstractmethod
    def save(self, entity: T) -> T: ...

    @abstractmethod
    def delete(self, entity_id: str) -> None: ...

    @abstractmethod
    def list(self) -> list[T]: ...


class CharacterRepository(Repository[Character]):
    pass


class SceneRepository(Repository[Scene]):
    pass


class PageRepository(Repository[Page]):
    pass


class ProjectRepository(Repository[Project]):
    pass


class StyleProfileRepository(Repository[StyleProfile]):
    pass
