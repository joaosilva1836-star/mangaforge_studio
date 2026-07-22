"""
Testa o fluxo completo — personagem -> storyboard -> página -> export —
usando o MockPipelineFactory, então roda sem GPU e sem PyTorch/diffusers
instalados. Serve para validar a integração entre as camadas antes de
gastar VRAM de verdade.

Rodar com: pytest mangaforge_studio/tests/test_end_to_end.py -v
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from mangaforge_studio.domain.entities import ExportFormat, StyleProfile
from mangaforge_studio.infra.json_repositories import (
    JsonCharacterRepository,
    JsonPageRepository,
    JsonSceneRepository,
    JsonStyleProfileRepository,
)
from mangaforge_studio.infra.mock_pipeline import MockPipelineFactory
from mangaforge_studio.infra.page_compositor import PageCompositor
from mangaforge_studio.services.character_service import CharacterService
from mangaforge_studio.services.export_service import ExportService
from mangaforge_studio.services.page_service import PageService
from mangaforge_studio.services.storyboard_service import StoryboardService


@pytest.fixture
def tmp_data_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_full_pipeline(tmp_data_dir):
    character_repo = JsonCharacterRepository(base_dir=f"{tmp_data_dir}/characters")
    scene_repo = JsonSceneRepository(base_dir=f"{tmp_data_dir}/scenes")
    page_repo = JsonPageRepository(base_dir=f"{tmp_data_dir}/pages")
    style_repo = JsonStyleProfileRepository(base_dir=f"{tmp_data_dir}/styles")

    pipeline_factory = MockPipelineFactory()
    compositor = PageCompositor()

    character_service = CharacterService(character_repo, style_repo, pipeline_factory, output_dir=f"{tmp_data_dir}/char_out")
    storyboard_service = StoryboardService(scene_repo, page_repo)
    page_service = PageService(page_repo, style_repo, pipeline_factory, compositor, output_dir=f"{tmp_data_dir}/pages_out")
    export_service = ExportService(project_dir=f"{tmp_data_dir}/exports")

    # 1. Style profile
    style = StyleProfile(name="shonen-ink", base_model="sdxl")
    style_repo.save(style)

    # 2. Personagem consistente
    character = character_service.create_character(
        name="Kaito",
        description="young ronin swordsman, spiky black hair, scar over left eye",
        style_profile_id=style.id,
    )
    character = character_service.generate_reference_sheet(character.id, poses=["front view", "side view"])
    assert character.status.value == "ready"
    assert len(character.reference_image_paths) == 2

    # 3. Storyboard -> capítulo com páginas
    beats = [
        {
            "scene_description": "Kaito walks into a rainy village at dusk",
            "character_ids": [character.id],
            "panel_prompts": [
                "Kaito walking, rain, wide shot",
                "close up of Kaito's determined face",
            ],
        }
    ]
    chapter = storyboard_service.build_chapter(title="Capitulo 1", beats=beats, panels_per_page=2)
    assert len(chapter.pages) == 1

    # 4. Gerar a página (renderiza cada quadro + compõe)
    page = page_service.generate_page(chapter.pages[0].id, style_profile_id=style.id)
    assert page.status.value == "ready"
    assert all(p.image_path for p in page.panels)

    # 5. Exportar como CBZ
    composed_page_path = f"{tmp_data_dir}/pages_out/page_{page.page_number}.png"
    output = export_service.export(chapter, [composed_page_path], ExportFormat.CBZ)
    assert output.endswith(".cbz")
