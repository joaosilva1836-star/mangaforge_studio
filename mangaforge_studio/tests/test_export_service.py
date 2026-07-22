from __future__ import annotations

import os
import zipfile

import pytest
from PIL import Image

from mangaforge_studio.domain.entities import (
    ExportFormat,
    ExportSettings,
    Orientation,
    PageSize,
)
from mangaforge_studio.infra.image_processor import PillowImageProcessor, create_image_processor
from mangaforge_studio.services.export_service import ExportService


@pytest.fixture
def source_page(tmp_path) -> str:
    path = tmp_path / "page_1.png"
    Image.new("RGB", (600, 900), (200, 210, 220)).save(path)
    return str(path)


def _service(tmp_path) -> ExportService:
    # força CPU para determinismo (a máquina de CI não tem GPU de qualquer forma)
    return ExportService(project_dir=str(tmp_path / "proj"), processor_factory=lambda _: PillowImageProcessor())


# ------------------------------------------------------------- ExportSettings

def test_target_pixels_a4_300dpi() -> None:
    s = ExportSettings(page_size=PageSize.A4, dpi=300)
    w, h = s.target_pixels()
    assert (w, h) == (2480, 3508)  # 210mm x 297mm @ 300 DPI


def test_orientation_swaps_dimensions() -> None:
    portrait = ExportSettings(page_size=PageSize.A4, orientation=Orientation.PORTRAIT).physical_size_mm()
    landscape = ExportSettings(page_size=PageSize.A4, orientation=Orientation.LANDSCAPE).physical_size_mm()
    assert portrait == (210.0, 297.0)
    assert landscape == (297.0, 210.0)


def test_bleed_and_scale_grow_pixels() -> None:
    base = ExportSettings(page_size=PageSize.B5, dpi=300).target_pixels()
    bleed = ExportSettings(page_size=PageSize.B5, dpi=300, bleed_mm=3).target_pixels()
    scaled = ExportSettings(page_size=PageSize.B5, dpi=300, scale=2.0).target_pixels()
    assert bleed[0] > base[0] and bleed[1] > base[1]
    assert abs(scaled[0] - base[0] * 2) <= 1 and abs(scaled[1] - base[1] * 2) <= 1


def test_custom_size_requires_dimensions() -> None:
    with pytest.raises(ValueError):
        ExportSettings(page_size=PageSize.CUSTOM).physical_size_mm()
    ok = ExportSettings(page_size=PageSize.CUSTOM, custom_size_mm=(100.0, 200.0)).physical_size_mm()
    assert ok == (100.0, 200.0)


# ------------------------------------------------------------------ naming

def test_page_and_chapter_naming() -> None:
    assert ExportService._page_stem(1, 2) == "Capitulo_001_Pagina_002"
    assert ExportService._chapter_stem(12) == "Capitulo_012"


# --------------------------------------------------------------- formats

@pytest.mark.parametrize("fmt", [ExportFormat.PNG, ExportFormat.WEBP, ExportFormat.JPG, ExportFormat.SVG])
def test_export_chapter_image_formats(tmp_path, source_page, fmt) -> None:
    svc = _service(tmp_path)
    settings = ExportSettings(fmt=fmt, page_size=PageSize.B5, dpi=150)
    out_dir = svc.export_chapter([source_page, source_page], chapter_number=1, settings=settings)
    assert os.path.isdir(out_dir)
    assert fmt.value.upper() in out_dir
    files = sorted(os.listdir(out_dir))
    assert files == ["Capitulo_001_Pagina_001." + fmt.value, "Capitulo_001_Pagina_002." + fmt.value]


def test_export_chapter_pdf_single_file(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    out = svc.export_chapter([source_page, source_page], 1, ExportSettings(fmt=ExportFormat.PDF, dpi=150))
    assert out.endswith("Capitulo_001.pdf")
    assert os.path.getsize(out) > 0


def test_export_chapter_cbz_contains_named_pages(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    out = svc.export_chapter([source_page, source_page], 3, ExportSettings(fmt=ExportFormat.CBZ, dpi=150))
    assert out.endswith("Capitulo_003.cbz")
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["Capitulo_003_Pagina_001.png", "Capitulo_003_Pagina_002.png"]


def test_master_dpi_routes_to_master_folder(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    out = svc.export_page(source_page, 1, 1, ExportSettings(fmt=ExportFormat.PNG, dpi=1200, page_size=PageSize.A5))
    assert os.path.join("Export", "MASTER") in out


def test_svg_embeds_raster(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    out_dir = svc.export_chapter([source_page], 1, ExportSettings(fmt=ExportFormat.SVG, dpi=72))
    svg_file = os.path.join(out_dir, "Capitulo_001_Pagina_001.svg")
    text = open(svg_file, encoding="utf-8").read()
    assert text.startswith("<svg") and "data:image/png;base64," in text


def test_grayscale_output_is_single_channel(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    out_dir = svc.export_chapter(
        [source_page], 1, ExportSettings(fmt=ExportFormat.PNG, dpi=72, grayscale=True)
    )
    img = Image.open(os.path.join(out_dir, "Capitulo_001_Pagina_001.png"))
    assert img.mode == "L"


def test_exported_page_matches_target_pixels(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    settings = ExportSettings(fmt=ExportFormat.PNG, page_size=PageSize.A5, dpi=72)
    out = svc.export_page(source_page, 1, 1, settings)
    assert Image.open(out).size == settings.target_pixels()


# --------------------------------------------------------- manga / batch

def test_export_manga_batches_chapters(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    outputs = svc.export_manga({1: [source_page], 2: [source_page, source_page]}, ExportSettings(fmt=ExportFormat.PDF))
    assert len(outputs) == 2
    assert outputs[0].endswith("Capitulo_001.pdf")
    assert outputs[1].endswith("Capitulo_002.pdf")


def test_verify_quality_flags_big_upscale(tmp_path, source_page) -> None:
    svc = _service(tmp_path)
    report = svc.verify_quality(source_page, ExportSettings(page_size=PageSize.A4, dpi=600))
    assert report["upscale_factor"] > 2.0
    assert report["warnings"]


# ------------------------------------------------------- processor factory

def test_create_image_processor_defaults_to_cpu_without_gpu() -> None:
    # sem GPU nesta máquina -> backend cpu; forçar False sempre dá cpu
    assert create_image_processor(False).backend == "cpu"


def test_empty_chapter_raises(tmp_path) -> None:
    from mangaforge_studio.domain.exceptions import ExportError

    svc = _service(tmp_path)
    with pytest.raises(ExportError):
        svc.export_chapter([], 1, ExportSettings())
