"""
MangaForge Export — exportação profissional de páginas de mangá.

Preserva a qualidade máxima (linhas nítidas, hachuras, tons de cinza, sem
compressão visível) e suporta:

- formatos: PNG, WEBP, JPG, PDF, CBZ e SVG (raster embutido — vetorização
  real depende do módulo de vetorização, ainda não implementado);
- DPI: 72/150/300/600/1200 (1200 = arquivo mestre);
- tamanhos: A4, A5, B5, Letter e personalizado, com orientação;
- margens, sangria, escala, qualidade/compressão e tons de cinza;
- limpeza inteligente antes de exportar (denoise + realce de linhas);
- aceleração CUDA quando há GPU NVIDIA (senão CPU, automaticamente);
- nomeação automática (`Capitulo_001_Pagina_001`) e organização em
  `Projeto/Export/{PNG,PDF,WEBP,CBZ,JPG,SVG,MASTER}/`;
- página única, capítulo inteiro, mangá completo e exportação em lote.
"""
from __future__ import annotations

import base64
import os
import zipfile
from collections.abc import Callable, Iterable, Mapping

from mangaforge_studio.domain.entities import ExportFormat, ExportSettings
from mangaforge_studio.domain.exceptions import ExportError
from mangaforge_studio.infra.image_processor import create_image_processor
from mangaforge_studio.interfaces.image_processor import ImageProcessor

# Formatos que geram um único arquivo com várias páginas.
_MULTIPAGE_FORMATS = {ExportFormat.PDF, ExportFormat.CBZ}
_MASTER_DPI = 1200


class ExportService:
    """Exporta páginas já renderizadas (PNGs em disco) com qualidade de
    publicação. `processor_factory` é injetável para testes/DI."""

    def __init__(
        self,
        project_dir: str = "./output/projects/default",
        processor_factory: Callable[[bool | None], ImageProcessor] = create_image_processor,
    ) -> None:
        self._project_dir = project_dir
        self._processor_factory = processor_factory

    # ------------------------------------------------------------------ API

    def export_page(
        self,
        source_image_path: str,
        chapter_number: int,
        page_number: int,
        settings: ExportSettings | None = None,
    ) -> str:
        """Exporta uma única página. Para PDF/CBZ, gera um arquivo de 1 página."""
        settings = settings or ExportSettings()
        if settings.fmt in _MULTIPAGE_FORMATS:
            return self.export_chapter([source_image_path], chapter_number, settings)
        processor = self._make_processor(settings)
        return self._write_single_image(source_image_path, chapter_number, page_number, settings, processor)

    def export_chapter(
        self,
        page_image_paths: list[str],
        chapter_number: int,
        settings: ExportSettings | None = None,
    ) -> str:
        """Exporta um capítulo inteiro. PDF/CBZ viram um arquivo; formatos de
        imagem geram um arquivo por página numa pasta do capítulo. Retorna o
        caminho do arquivo/pasta gerado."""
        settings = settings or ExportSettings()
        if not page_image_paths:
            raise ExportError("Nenhuma página renderizada para exportar")

        processor = self._make_processor(settings)
        prepared = [self._prepare(p, settings, processor) for p in page_image_paths]

        if settings.fmt == ExportFormat.PDF:
            return self._write_pdf(prepared, chapter_number, settings)
        if settings.fmt == ExportFormat.CBZ:
            return self._write_cbz(prepared, chapter_number, settings)

        # Formatos de imagem/SVG: um arquivo por página, dentro de uma pasta.
        out_dir = os.path.join(self._format_dir(settings), self._chapter_stem(chapter_number))
        os.makedirs(out_dir, exist_ok=True)
        for i, image in enumerate(prepared, start=1):
            self._save_image(image, os.path.join(out_dir, self._page_stem(chapter_number, i)), settings)
        return out_dir

    def export_manga(
        self,
        chapters: Mapping[int, list[str]],
        settings: ExportSettings | None = None,
    ) -> list[str]:
        """Exporta o mangá completo (várias chapters) em lote. Retorna a lista
        de caminhos gerados, um por capítulo."""
        settings = settings or ExportSettings()
        outputs: list[str] = []
        for chapter_number in sorted(chapters):
            outputs.append(self.export_chapter(chapters[chapter_number], chapter_number, settings))
        return outputs

    def export_batch(
        self,
        jobs: Iterable[tuple[list[str], int, ExportSettings]],
    ) -> list[str]:
        """Exportação em lote com settings potencialmente diferentes por job.
        Cada job é (page_image_paths, chapter_number, settings)."""
        return [self.export_chapter(paths, ch, settings) for paths, ch, settings in jobs]

    def verify_quality(self, source_image_path: str, settings: ExportSettings) -> dict[str, object]:
        """Checagem de qualidade pré-exportação: compara a resolução da fonte
        com o alvo (DPI/tamanho) e sinaliza se haverá upscaling significativo."""
        from PIL import Image

        with Image.open(source_image_path) as img:
            src_w, src_h = img.size
        tgt_w, tgt_h = settings.target_pixels()
        upscale_factor = max(tgt_w / src_w, tgt_h / src_h)
        warnings: list[str] = []
        if upscale_factor > 2.0:
            warnings.append(
                f"Fonte {src_w}x{src_h} vai ser ampliada ~{upscale_factor:.1f}x para {tgt_w}x{tgt_h} "
                f"({settings.dpi} DPI); considere gerar em resolução maior para nitidez máxima."
            )
        return {
            "source_size": (src_w, src_h),
            "target_size": (tgt_w, tgt_h),
            "upscale_factor": round(upscale_factor, 2),
            "warnings": warnings,
        }

    # -------------------------------------------------- compat (assinatura antiga)

    def export(
        self,
        chapter,  # noqa: ANN001 — Chapter; mantido por compatibilidade
        page_image_paths: list[str],
        fmt: ExportFormat,
        settings: ExportSettings | None = None,
    ) -> str:
        """Assinatura retrocompatível com a versão anterior do Studio. Usa
        `chapter.order` como número do capítulo e o `fmt` informado."""
        settings = settings or ExportSettings()
        settings.fmt = fmt
        chapter_number = getattr(chapter, "order", 1) or 1
        return self.export_chapter(page_image_paths, chapter_number, settings)

    # ------------------------------------------------------------- internals

    def _make_processor(self, settings: ExportSettings) -> ImageProcessor:
        return self._processor_factory(settings.use_gpu)

    def _prepare(self, source_image_path: str, settings: ExportSettings, processor: ImageProcessor):  # noqa: ANN202
        """Abre a página, aplica limpeza inteligente e ajusta ao tamanho/DPI
        alvo (fit preservando proporção, margens brancas e sangria)."""
        from PIL import Image

        if not os.path.exists(source_image_path):
            raise ExportError(f"Arquivo de página não encontrado: {source_image_path}")

        image = Image.open(source_image_path).convert("RGB")

        if settings.smart:
            image = processor.denoise(image)
            image = processor.sharpen_lines(image)

        target_w, target_h = settings.target_pixels()
        margin_px = int(round(settings.margins_mm * settings.dpi / 25.4 * settings.scale))
        content_w = max(1, target_w - 2 * margin_px)
        content_h = max(1, target_h - 2 * margin_px)

        # Fit preservando proporção dentro da área de conteúdo.
        scale = min(content_w / image.width, content_h / image.height)
        fitted = processor.resize(image, (max(1, int(image.width * scale)), max(1, int(image.height * scale))))

        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset = ((target_w - fitted.width) // 2, (target_h - fitted.height) // 2)
        canvas.paste(fitted, offset)

        if settings.grayscale:
            canvas = processor.to_grayscale(canvas)
        return canvas

    # --- nomeação / organização ---

    @staticmethod
    def _page_stem(chapter_number: int, page_number: int) -> str:
        return f"Capitulo_{chapter_number:03d}_Pagina_{page_number:03d}"

    @staticmethod
    def _chapter_stem(chapter_number: int) -> str:
        return f"Capitulo_{chapter_number:03d}"

    def _format_dir(self, settings: ExportSettings) -> str:
        """`Projeto/Export/<SUBPASTA>/`. DPI de arquivo mestre (1200) vai para
        MASTER; senão, subpasta com o nome do formato."""
        subdir = "MASTER" if settings.dpi >= _MASTER_DPI else settings.fmt.value.upper()
        path = os.path.join(self._project_dir, "Export", subdir)
        os.makedirs(path, exist_ok=True)
        return path

    # --- writers por formato ---

    def _write_single_image(
        self, source: str, chapter_number: int, page_number: int, settings: ExportSettings, processor: ImageProcessor
    ) -> str:
        image = self._prepare(source, settings, processor)
        out_base = os.path.join(self._format_dir(settings), self._page_stem(chapter_number, page_number))
        return self._save_image(image, out_base, settings)

    def _save_image(self, image, out_base: str, settings: ExportSettings) -> str:  # noqa: ANN001
        """Salva `image` no formato de imagem pedido, com metadados de DPI e
        parâmetros de qualidade/compressão adequados. `out_base` é sem extensão."""
        dpi = (settings.dpi, settings.dpi)
        fmt = settings.fmt

        if fmt == ExportFormat.PNG:
            out = f"{out_base}.png"
            image.save(out, format="PNG", dpi=dpi, optimize=True)
        elif fmt == ExportFormat.JPG:
            out = f"{out_base}.jpg"
            image.convert("RGB").save(
                out, format="JPEG", dpi=dpi, quality=settings.quality, optimize=True, subsampling=0
            )
        elif fmt == ExportFormat.WEBP:
            out = f"{out_base}.webp"
            image.save(out, format="WEBP", lossless=settings.lossless, quality=settings.quality, method=6)
        elif fmt == ExportFormat.SVG:
            out = f"{out_base}.svg"
            self._write_svg(image, out, settings)
        else:
            raise ExportError(f"Formato de imagem não suportado aqui: {fmt}")
        return out

    def _write_svg(self, image, out_path: str, settings: ExportSettings) -> None:  # noqa: ANN001
        """SVG com o raster embutido (base64). Não é vetorização real — é o
        que dá para fazer 'quando possível' sem o módulo de vetorização."""
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG", dpi=(settings.dpi, settings.dpi))
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        w, h = image.size
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<image width="{w}" height="{h}" '
            f'href="data:image/png;base64,{encoded}"/></svg>'
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg)

    def _write_pdf(self, images: list, chapter_number: int, settings: ExportSettings) -> str:  # noqa: ANN001
        out = os.path.join(self._format_dir(settings), f"{self._chapter_stem(chapter_number)}.pdf")
        rgb = [im.convert("RGB") for im in images]
        rgb[0].save(
            out,
            format="PDF",
            save_all=True,
            append_images=rgb[1:],
            resolution=float(settings.dpi),
        )
        return out

    def _write_cbz(self, images: list, chapter_number: int, settings: ExportSettings) -> str:  # noqa: ANN001
        import io

        out = os.path.join(self._format_dir(settings), f"{self._chapter_stem(chapter_number)}.cbz")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, image in enumerate(images, start=1):
                buf = io.BytesIO()
                image.save(buf, format="PNG", dpi=(settings.dpi, settings.dpi), optimize=True)
                zf.writestr(f"{self._page_stem(chapter_number, i)}.png", buf.getvalue())
        return out
