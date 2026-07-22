"""
API REST do MangaForge Studio.

Rodar com:
    uvicorn mangaforge_studio.api.main:app --reload

Swagger disponível automaticamente em /docs (FastAPI gera isso sozinho).

Por padrão usa o MockPipelineFactory (sem GPU) para você testar a API
imediatamente. Para gerar com SDXL de verdade, troque
`USE_MOCK_PIPELINE = False` abaixo (requer GPU NVIDIA + diffusers
instalado).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mangaforge_studio.api.schemas import (
    BuildChapterRequest,
    CharacterResponse,
    CreateCharacterRequest,
    ExportChapterProRequest,
    ExportChapterRequest,
    ExportMangaRequest,
    ExportOptionsResponse,
    ExportPageRequest,
    ExportResultResponse,
    ExportSettingsSchema,
    GenerateReferenceSheetRequest,
)
from mangaforge_studio.domain.entities import (
    STANDARD_DPIS,
    Chapter,
    ExportFormat,
    ExportSettings,
    Orientation,
    PageSize,
)
from mangaforge_studio.domain.exceptions import MangaForgeError
from mangaforge_studio.hardware.detector import HardwareDetector
from mangaforge_studio.hardware.optimizer import AIOptimizer
from mangaforge_studio.infra.json_repositories import (
    JsonCharacterRepository,
    JsonPageRepository,
    JsonSceneRepository,
    JsonStyleProfileRepository,
)
from mangaforge_studio.infra.page_compositor import PageCompositor
from mangaforge_studio.services.character_service import CharacterService
from mangaforge_studio.services.export_service import ExportService
from mangaforge_studio.services.page_service import PageService
from mangaforge_studio.services.storyboard_service import StoryboardService


def _resolve_use_mock() -> bool:
    """Decide automaticamente entre o pipeline real (SDXL via diffusers) e o
    mock, sem exigir edição de código:

    - Se a variável de ambiente ``MANGAFORGE_MOCK`` estiver definida, ela manda
      (``1/true/yes/on`` força mock; ``0/false/no/off`` força o pipeline real).
    - Caso contrário, usa o pipeline real quando houver GPU CUDA + ``diffusers``
      disponíveis; senão cai no mock. Assim, num pod com GPU (ex.: RunPod) a
      geração real liga sozinha, e numa máquina sem GPU tudo continua rodando.
    """
    override = os.getenv("MANGAFORGE_MOCK")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    try:
        import torch

        if torch.cuda.is_available():
            import importlib.util

            if importlib.util.find_spec("diffusers") is not None:
                return False
    except Exception:  # noqa: BLE001 — qualquer falha aqui significa "sem GPU utilizável"
        pass
    return True


USE_MOCK_PIPELINE = _resolve_use_mock()

if USE_MOCK_PIPELINE:
    from mangaforge_studio.infra.mock_pipeline import MockPipelineFactory as _PipelineFactoryImpl
else:
    from mangaforge_studio.infra.diffusion_pipeline import DiffusersPipelineFactory as _PipelineFactoryImpl

app = FastAPI(
    title="MangaForge Studio API",
    description="Geração de personagens, storyboard e páginas de mangá — offline.",
    version="0.1.0",
)

# Libera acesso de qualquer origem (necessário pro frontend rodar fora da pod,
# ex: aberto localmente no seu navegador). Em produção, restrinja a origins
# específicas em vez de "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("output", exist_ok=True)
app.mount("/files", StaticFiles(directory="output"), name="files")

# Serve o dashboard na raiz — abrir a URL do pod (ex.: RunPod) já mostra a UI
# funcional, sem precisar abrir o index.html manualmente nem digitar a URL da API
# (o front detecta a própria origem). O arquivo fica dentro do pacote.
_FRONTEND_INDEX = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(_FRONTEND_INDEX)


def to_public_url(local_path: str | None) -> str | None:
    """Converte um caminho local (ex: ./output/tmp/x.png) numa URL servível
    pelo /files (ex: /files/tmp/x.png). Retorna o caminho original se não
    estiver dentro de ./output (ex: já for uma URL, ou vazio)."""
    if not local_path:
        return local_path
    normalized = local_path.replace("\\", "/")
    marker = "output/"
    if marker in normalized:
        return "/files/" + normalized.split(marker, 1)[1]
    return local_path

# --- Hardware Manager + AI Optimizer: roda uma vez na subida da API ---
hardware_detector = HardwareDetector()
ai_optimizer = AIOptimizer()
detected_system = hardware_detector.detect()
hardware_profile = ai_optimizer.recommend_profile(detected_system)
print("[Hardware Manager]", ai_optimizer.explain(detected_system, hardware_profile))

# --- Composição de dependências (Dependency Injection manual e explícita) ---
character_repo = JsonCharacterRepository()
scene_repo = JsonSceneRepository()
page_repo = JsonPageRepository()
style_repo = JsonStyleProfileRepository()

if USE_MOCK_PIPELINE:
    pipeline_factory = _PipelineFactoryImpl()
else:
    pipeline_factory = _PipelineFactoryImpl(hardware_profile=hardware_profile)

compositor = PageCompositor()

character_service = CharacterService(character_repo, style_repo, pipeline_factory)
storyboard_service = StoryboardService(scene_repo, page_repo)
page_service = PageService(page_repo, style_repo, pipeline_factory, compositor)
export_service = ExportService()


@app.exception_handler(MangaForgeError)
async def mangaforge_error_handler(request, exc: MangaForgeError):  # noqa: ANN001
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/characters", response_model=CharacterResponse, tags=["characters"])
def create_character(payload: CreateCharacterRequest):
    character = character_service.create_character(
        name=payload.name,
        description=payload.description,
        style_profile_id=payload.style_profile_id,
        seed=payload.seed,
    )
    return CharacterResponse(
        id=character.id,
        name=character.name,
        description=character.description,
        status=character.status.value,
        reference_image_paths=[to_public_url(p) for p in character.reference_image_paths],
    )


@app.post(
    "/characters/{character_id}/reference-sheet",
    response_model=CharacterResponse,
    tags=["characters"],
)
def generate_reference_sheet(character_id: str, payload: GenerateReferenceSheetRequest):
    character = character_service.generate_reference_sheet(character_id, poses=payload.poses)
    return CharacterResponse(
        id=character.id,
        name=character.name,
        description=character.description,
        status=character.status.value,
        reference_image_paths=[to_public_url(p) for p in character.reference_image_paths],
    )


@app.get("/characters/{character_id}", response_model=CharacterResponse, tags=["characters"])
def get_character(character_id: str):
    character = character_repo.get(character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character não encontrado")
    return CharacterResponse(
        id=character.id,
        name=character.name,
        description=character.description,
        status=character.status.value,
        reference_image_paths=[to_public_url(p) for p in character.reference_image_paths],
    )


@app.post("/chapters/build", tags=["storyboard"])
def build_chapter(payload: BuildChapterRequest):
    chapter: Chapter = storyboard_service.build_chapter(
        title=payload.title,
        beats=payload.beats,
        order=payload.order,
        panels_per_page=payload.panels_per_page,
    )
    return {
        "chapter_id": chapter.id,
        "title": chapter.title,
        "page_ids": [p.id for p in chapter.pages],
    }


@app.post("/pages/{page_id}/generate", tags=["pages"])
def generate_page(page_id: str, style_profile_id: str | None = None):
    page = page_service.generate_page(page_id, style_profile_id=style_profile_id)
    return {
        "page_id": page.id,
        "status": page.status.value,
        "panel_image_paths": [to_public_url(p.image_path) for p in page.panels],
        "composed_page_url": to_public_url(f"./output/pages/page_{page.page_number}.png"),
    }


def _build_export_settings(schema: ExportSettingsSchema) -> ExportSettings:
    """Converte o schema da API (strings) no objeto de domínio ExportSettings,
    validando formato/tamanho/orientação com mensagens de erro claras."""
    try:
        fmt = ExportFormat(schema.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Formato inválido: {schema.format}") from exc
    try:
        page_size = PageSize(schema.page_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Tamanho inválido: {schema.page_size}") from exc
    try:
        orientation = Orientation(schema.orientation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Orientação inválida: {schema.orientation}") from exc
    if page_size == PageSize.CUSTOM and not schema.custom_size_mm:
        raise HTTPException(status_code=400, detail="page_size=custom exige custom_size_mm")
    return ExportSettings(
        fmt=fmt,
        dpi=schema.dpi,
        page_size=page_size,
        orientation=orientation,
        custom_size_mm=schema.custom_size_mm,
        quality=schema.quality,
        lossless=schema.lossless,
        scale=schema.scale,
        margins_mm=schema.margins_mm,
        bleed_mm=schema.bleed_mm,
        grayscale=schema.grayscale,
        smart=schema.smart,
        use_gpu=schema.use_gpu,
    )


def _export_result(output_path: str, settings: ExportSettings, schema: ExportSettingsSchema) -> ExportResultResponse:
    from mangaforge_studio.infra.image_processor import create_image_processor

    return ExportResultResponse(
        output_path=output_path,
        file_url=to_public_url(output_path),
        backend=create_image_processor(settings.use_gpu).backend,
        settings=schema,
    )


@app.get("/export/options", response_model=ExportOptionsResponse, tags=["export"])
def export_options():
    """Formatos, DPIs, tamanhos e orientações suportados pelo MangaForge Export."""
    return ExportOptionsResponse(
        formats=[f.value for f in ExportFormat],
        dpis=list(STANDARD_DPIS),
        page_sizes=[s.value for s in PageSize],
        orientations=[o.value for o in Orientation],
    )


@app.post("/export/page", response_model=ExportResultResponse, tags=["export"])
def export_page_pro(payload: ExportPageRequest):
    """Exporta uma página única com qualidade profissional (DPI, tamanho,
    margens, sangria, tons de cinza, limpeza inteligente, GPU/CPU)."""
    settings = _build_export_settings(payload.settings)
    output_path = export_service.export_page(
        payload.source_image_path, payload.chapter_number, payload.page_number, settings
    )
    return _export_result(output_path, settings, payload.settings)


@app.post("/export/chapter", response_model=ExportResultResponse, tags=["export"])
def export_chapter_pro(payload: ExportChapterProRequest):
    """Exporta um capítulo inteiro. PDF/CBZ viram um arquivo; formatos de
    imagem/SVG geram um arquivo por página numa pasta do capítulo."""
    settings = _build_export_settings(payload.settings)
    output_path = export_service.export_chapter(
        payload.page_image_paths, payload.chapter_number, settings
    )
    return _export_result(output_path, settings, payload.settings)


@app.post("/export/manga", tags=["export"])
def export_manga_pro(payload: ExportMangaRequest):
    """Exporta o mangá completo (vários capítulos) em lote."""
    settings = _build_export_settings(payload.settings)
    outputs = export_service.export_manga(payload.chapters, settings)
    return {
        "outputs": [
            {"output_path": p, "file_url": to_public_url(p)} for p in outputs
        ],
        "settings": payload.settings,
    }


# Rota legada (compatibilidade). Registrada por último de propósito: como usa um
# path param (`{chapter_title}`), se viesse antes ela capturaria /export/page,
# /export/chapter, etc. A ordem de declaração é a ordem de resolução no FastAPI.
@app.post("/export/{chapter_title}", tags=["export"])
def export_chapter_legacy(chapter_title: str, payload: ExportChapterRequest):
    fake_chapter = Chapter(title=chapter_title)
    try:
        fmt = ExportFormat(payload.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Formato inválido: {payload.format}") from exc

    output_path = export_service.export(fake_chapter, payload.page_image_paths, fmt)
    return {"output_path": output_path, "file_url": to_public_url(output_path)}


@app.get("/hardware", tags=["system"])
def get_hardware():
    """Mostra o que o Hardware Manager detectou e qual perfil o AI
    Optimizer escolheu automaticamente — sem precisar configurar nada."""
    return {
        "cpu_count": detected_system.cpu_count,
        "ram_total_mb": detected_system.ram_total_mb,
        "gpus": [
            {
                "name": g.name,
                "vram_total_mb": g.vram_total_mb,
                "vram_free_mb": g.vram_free_mb,
                "cuda_version": g.cuda_version,
                "driver_version": g.driver_version,
                "compute_capability": g.compute_capability,
                "supports_bf16": g.supports_bf16,
            }
            for g in detected_system.gpus
        ],
        "selected_profile": {
            "name": hardware_profile.name,
            "precision": hardware_profile.precision.value,
            "max_resolution": hardware_profile.max_resolution,
            "default_batch_size": hardware_profile.default_batch_size,
            "max_concurrent_loras": hardware_profile.max_concurrent_loras,
            "cpu_offload": hardware_profile.enable_cpu_offload,
            "quantization": hardware_profile.enable_quantization,
            "multi_model_cache": hardware_profile.enable_multi_model_cache,
            "notes": hardware_profile.notes,
        },
    }


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "pipeline": "mock" if USE_MOCK_PIPELINE else "diffusers",
        "hardware_profile": hardware_profile.name,
    }
