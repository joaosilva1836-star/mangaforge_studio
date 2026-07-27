"""
Garante que as dimensões dos quadros passadas ao pipeline são múltiplas de 8
(SDXL/diffusers exigem isso; layouts em frações como 0.34 geram valores
quebrados que travavam a geração real).
"""
from __future__ import annotations

import pytest

from mangaforge_studio.services.page_service import PageService


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1536 * 0.34, 520),  # 522.24 -> múltiplo de 8 mais próximo
        (1024 * 0.5, 512),
        (1024 * 0.33, 336),  # 337.92 -> 336
        (10, 64),  # abaixo do piso
        (0, 64),
    ],
)
def test_round_to_multiple_of_8(raw: float, expected: int) -> None:
    result = PageService._round_to_multiple_of_8(raw)
    assert result == expected
    assert result % 8 == 0
    assert result >= 64
