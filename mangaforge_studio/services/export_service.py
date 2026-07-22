from __future__ import annotations

import os
import zipfile

from mangaforge_studio.domain.entities import Chapter, ExportFormat
from mangaforge_studio.domain.exceptions import ExportError


class ExportService:
    """Exporta um capítulo já renderizado para os formatos pedidos na spec.

    Requer que cada Page do capítulo já tenha sido processada pelo
    PageService (ou seja, que o compositor já tenha gerado o PNG final
    de cada página em disco).
    """

    def __init__(self, output_dir: str = "./output/exports"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(self, chapter: Chapter, page_image_paths: list[str], fmt: ExportFormat) -> str:
        if not page_image_paths:
            raise ExportError("Nenhuma página renderizada para exportar")

        safe_title = "".join(c for c in chapter.title if c.isalnum() or c in (" ", "_", "-")).strip() or "chapter"

        if fmt == ExportFormat.CBZ:
            return self._export_cbz(safe_title, page_image_paths)
        if fmt == ExportFormat.PDF:
            return self._export_pdf(safe_title, page_image_paths)
        if fmt in (ExportFormat.PNG, ExportFormat.WEBP):
            return self._export_images(safe_title, page_image_paths, fmt)
        if fmt == ExportFormat.SVG:
            raise ExportError(
                "Exportação SVG requer vetorização prévia (módulo de vetorização "
                "ainda não implementado neste MVP)"
            )
        raise ExportError(f"Formato não suportado: {fmt}")

    def _export_cbz(self, title: str, page_paths: list[str]) -> str:
        out_path = os.path.join(self._output_dir, f"{title}.cbz")
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, path in enumerate(page_paths, start=1):
                zf.write(path, arcname=f"page_{i:03d}{os.path.splitext(path)[1]}")
        return out_path

    def _export_pdf(self, title: str, page_paths: list[str]) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ExportError("Pillow é necessário para exportar PDF (pip install pillow)") from exc

        images = [Image.open(p).convert("RGB") for p in page_paths]
        out_path = os.path.join(self._output_dir, f"{title}.pdf")
        images[0].save(out_path, save_all=True, append_images=images[1:])
        return out_path

    def _export_images(self, title: str, page_paths: list[str], fmt: ExportFormat) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ExportError("Pillow é necessário para exportar imagens") from exc

        folder = os.path.join(self._output_dir, title)
        os.makedirs(folder, exist_ok=True)
        for i, path in enumerate(page_paths, start=1):
            img = Image.open(path)
            out_path = os.path.join(folder, f"page_{i:03d}.{fmt.value}")
            img.save(out_path)
        return folder
