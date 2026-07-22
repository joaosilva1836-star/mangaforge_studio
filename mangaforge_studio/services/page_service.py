from __future__ import annotations

import os

from mangaforge_studio.domain.entities import AssetStatus, Page
from mangaforge_studio.domain.exceptions import GenerationError, NotFoundError
from mangaforge_studio.interfaces.pipeline import GenerationRequest, PipelineFactory
from mangaforge_studio.interfaces.repositories import PageRepository, StyleProfileRepository


class PageService:
    """Gera as imagens de cada quadro de uma página e compõe o resultado
    final. A composição real (colar os quadros na tela seguindo o
    layout_box) fica em infra/page_compositor.py para manter esta classe
    livre de dependências de imagem (PIL etc.)."""

    def __init__(
        self,
        page_repo: PageRepository,
        style_repo: StyleProfileRepository,
        pipeline_factory: PipelineFactory,
        compositor,  # infra.page_compositor.PageCompositor — injetado, sem import direto aqui
        output_dir: str = "./output/pages",
    ):
        self._pages = page_repo
        self._styles = style_repo
        self._pipeline_factory = pipeline_factory
        self._compositor = compositor
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_page(self, page_id: str, style_profile_id: str | None = None) -> Page:
        page = self._pages.get(page_id)
        if page is None:
            raise NotFoundError(f"Page {page_id} não encontrada")

        style = self._styles.get(style_profile_id) if style_profile_id else None
        lora_paths = style.lora_paths if style else []
        base_model = style.base_model if style else "sdxl"
        pipeline = self._pipeline_factory.create(base_model=base_model, lora_paths=lora_paths)

        page.status = AssetStatus.GENERATING
        self._pages.save(page)

        try:
            for panel in page.panels:
                request = GenerationRequest(
                    prompt=panel.prompt,
                    lora_paths=lora_paths,
                    width=int(page.resolution[0] * panel.layout_box[2]),
                    height=int(page.resolution[1] * panel.layout_box[3]),
                )
                result = pipeline.generate(request)
                panel.image_path = result.image_path
                panel.status = AssetStatus.READY

            composed_path = os.path.join(self._output_dir, f"page_{page.page_number}.png")
            self._compositor.compose(page, composed_path)
        except Exception as exc:  # noqa: BLE001
            page.status = AssetStatus.FAILED
            self._pages.save(page)
            raise GenerationError(f"Falha ao gerar página {page.page_number}: {exc}") from exc

        page.status = AssetStatus.READY
        return self._pages.save(page)
