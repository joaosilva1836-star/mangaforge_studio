# MangaForge Studio

Primeiros módulos do MangaForge AI Enterprise: **Hardware Manager**,
**AI Optimizer** e o **Studio** de produção (geração de **personagens
consistentes**, **storyboard**, **páginas de mangá** e **exportação**
em PDF/CBZ/PNG/WEBP). Roda 100% offline.

## Arquitetura (Clean Architecture)

```
mangaforge_studio/
├── domain/          # Entidades puras (Character, Page, Panel, Chapter...) — zero dependências
├── interfaces/      # Contratos abstratos (Repository Pattern, Pipeline)
├── services/        # Lógica de negócio (CharacterService, PageService, StoryboardService, ExportService)
├── hardware/        # Hardware Manager (detecção de GPU/CUDA/VRAM) + AI Optimizer (perfis automáticos)
├── infra/           # Implementações concretas (diffusers/SDXL, JSON repos, compositor de página)
├── api/             # FastAPI + Swagger (REST)
├── frontend/        # Dashboard estático (tema escuro) servido pela API
└── tests/           # Testes de ponta a ponta com pipeline mock (sem GPU)
```

A regra de dependência do Clean Architecture é respeitada: `domain` não
importa nada; `services` só conhece `interfaces`; `infra` e `api`
implementam/conectam essas interfaces. Isso significa que trocar SDXL
por FLUX, ou trocar o repositório JSON por Postgres, não exige tocar em
nenhuma linha de `services/`.

## Rodando sem GPU (desenvolvimento)

Por padrão a API usa `MockPipelineFactory`, que gera imagens sólidas
com o prompt escrito em cima — serve para testar toda a orquestração
(storyboard → página → export) sem gastar VRAM.

Rode a partir da raiz do repositório:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # núcleo + pytest/ruff
uvicorn mangaforge_studio.api.main:app --reload
# Swagger em http://localhost:8000/docs
```

## Rodando com GPU NVIDIA (produção)

1. Instale as deps de GPU: `pip install -e ".[gpu]"` (ou `pip install -r requirements.txt`)
2. Em `api/main.py`, mude `USE_MOCK_PIPELINE = False`
3. Na subida, o **Hardware Manager** detecta a GPU/VRAM e o **AI
   Optimizer** escolhe automaticamente o perfil (8/12/16/24/48 GB),
   a precisão (FP16/BF16) e demais parâmetros — veja `GET /hardware`.
   `DiffusersSDXLPipeline` ativa xFormers quando disponível.

## Já implementado

- **Hardware Manager** (`hardware/detector.py`): detecção de GPU/CUDA/VRAM/RAM/CPU via `nvidia-smi` + `torch.cuda`.
- **AI Optimizer** (`hardware/optimizer.py`): escolha automática de perfil (8/12/16/24/48 GB) e precisão (FP16/BF16).
- **Studio**: personagens consistentes, storyboard → páginas, composição e exportação (PDF/CBZ/PNG/WEBP).
- Endpoint `GET /hardware` expõe o que foi detectado e o perfil escolhido.

## O que ainda falta (próximos módulos da spec)

- **AutoTrainer**: treinamento automático de LoRAs a partir de datasets
- **MangaForge Brain**: agente de orquestração/relatórios/otimização contínua
- **Sistema de Análise**: extração automática de Perfil de Estilo
- Upscale real (Real-ESRGAN), inpainting/outpainting reais (hoje são stubs em `diffusion_pipeline.py`)
- Vetorização para exportação SVG
- FLUX pipeline (stub em `DiffusersPipelineFactory.create`)

## Testando

```bash
pip install -e ".[dev]"
pytest        # roda a suíte (mock pipeline, sem GPU)
ruff check .  # lint
```
