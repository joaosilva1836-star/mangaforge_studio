from __future__ import annotations

from mangaforge_studio.domain.entities import Page
from mangaforge_studio.domain.exceptions import GenerationError


class PageCompositor:
    """Compõe os quadros já gerados (panel.image_path) em uma única
    imagem de página, respeitando o layout_box (x, y, w, h normalizados
    de 0 a 1) definido pelo StoryboardService."""

    def compose(self, page: Page, output_path: str) -> str:
        from PIL import Image

        canvas_w, canvas_h = page.resolution
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        for panel in page.panels:
            if not panel.image_path:
                raise GenerationError(f"Panel {panel.id} não possui imagem gerada")
            panel_img = Image.open(panel.image_path).convert("RGB")

            x, y, w, h = panel.layout_box
            box_w = int(w * canvas_w)
            box_h = int(h * canvas_h)
            panel_img = panel_img.resize((box_w, box_h))

            paste_x = int(x * canvas_w)
            paste_y = int(y * canvas_h)
            canvas.paste(panel_img, (paste_x, paste_y))

        canvas.save(output_path)
        return output_path
