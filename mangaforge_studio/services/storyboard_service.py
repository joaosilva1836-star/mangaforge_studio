from __future__ import annotations

from mangaforge_studio.domain.entities import Chapter, Page, Panel, Scene
from mangaforge_studio.interfaces.repositories import PageRepository, SceneRepository


class StoryboardService:
    """Transforma um roteiro em texto simples em uma estrutura de
    capítulos/páginas/quadros pronta para geração.

    A divisão de layout aqui é heurística e determinística (sem custo de
    GPU); a parte de IA generativa (texto -> roteiro estruturado) pode
    ser plugada depois via Ollama, conforme a spec pede compatibilidade
    com modelos de linguagem locais.
    """

    # Layouts de quadro pré-definidos por número de painéis na página
    _LAYOUTS: dict[int, list[tuple[float, float, float, float]]] = {
        1: [(0, 0, 1, 1)],
        2: [(0, 0, 1, 0.5), (0, 0.5, 1, 0.5)],
        3: [(0, 0, 1, 0.34), (0, 0.34, 0.5, 0.33), (0.5, 0.34, 0.5, 0.33)],
        4: [(0, 0, 0.5, 0.5), (0.5, 0, 0.5, 0.5), (0, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)],
    }

    def __init__(self, scene_repo: SceneRepository, page_repo: PageRepository):
        self._scenes = scene_repo
        self._pages = page_repo

    def build_chapter(
        self,
        title: str,
        beats: list[dict],
        order: int = 1,
        panels_per_page: int = 4,
    ) -> Chapter:
        """`beats` é uma lista de dicts simples, ex.:
        [{"scene_description": "...", "character_ids": [...], "panel_prompts": ["...", "..."]}]
        """
        chapter = Chapter(title=title, order=order)
        panels: list[Panel] = []

        for beat in beats:
            scene = Scene(
                description=beat.get("scene_description", ""),
                character_ids=beat.get("character_ids", []),
                background_prompt=beat.get("background_prompt", beat.get("scene_description", "")),
            )
            self._scenes.save(scene)

            for prompt in beat.get("panel_prompts", [beat.get("scene_description", "")]):
                panels.append(Panel(scene_id=scene.id, prompt=prompt))

        layout = self._LAYOUTS.get(panels_per_page, self._LAYOUTS[4])
        page_number = 1
        for i in range(0, len(panels), panels_per_page):
            page_panels = panels[i : i + panels_per_page]
            for panel, box in zip(page_panels, layout, strict=False):
                panel.layout_box = box
            page = Page(chapter_id=chapter.id, page_number=page_number, panels=page_panels)
            self._pages.save(page)
            chapter.pages.append(page)
            page_number += 1

        return chapter
