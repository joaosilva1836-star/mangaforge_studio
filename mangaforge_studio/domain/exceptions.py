class MangaForgeError(Exception):
    """Erro base do MangaForge Studio."""


class GenerationError(MangaForgeError):
    """Falha ao gerar uma imagem/asset."""


class NotFoundError(MangaForgeError):
    """Entidade não encontrada no repositório."""


class InvalidStateError(MangaForgeError):
    """Operação inválida para o estado atual da entidade."""


class ExportError(MangaForgeError):
    """Falha ao exportar um capítulo/página."""
