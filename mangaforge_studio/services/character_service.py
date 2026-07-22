from __future__ import annotations

import random

from mangaforge_studio.domain.entities import AssetStatus, Character
from mangaforge_studio.domain.exceptions import GenerationError, NotFoundError
from mangaforge_studio.interfaces.pipeline import GenerationRequest, PipelineFactory
from mangaforge_studio.interfaces.repositories import CharacterRepository, StyleProfileRepository


class CharacterService:
    """Cria e mantém personagens consistentes entre gerações.

    Consistência é garantida fixando a mesma seed + o mesmo LoRA/style
    profile para todas as gerações futuras daquele personagem, além de
    reaproveitar imagens de referência como `init_image` quando
    disponíveis.
    """

    def __init__(
        self,
        character_repo: CharacterRepository,
        style_repo: StyleProfileRepository,
        pipeline_factory: PipelineFactory,
        output_dir: str = "./output/characters",
    ):
        self._characters = character_repo
        self._styles = style_repo
        self._pipeline_factory = pipeline_factory
        self._output_dir = output_dir

    def create_character(
        self,
        name: str,
        description: str,
        style_profile_id: str | None = None,
        seed: int | None = None,
    ) -> Character:
        character = Character(
            name=name,
            description=description,
            style_profile_id=style_profile_id,
            seed=seed if seed is not None else random.randint(0, 2**31 - 1),
        )
        return self._characters.save(character)

    def generate_reference_sheet(self, character_id: str, poses: list[str] | None = None) -> Character:
        """Gera uma folha de referência (várias poses) para manter consistência
        visual ao longo do mangá."""
        character = self._characters.get(character_id)
        if character is None:
            raise NotFoundError(f"Character {character_id} não encontrado")

        poses = poses or ["front view, neutral pose", "side view", "3/4 view, action pose"]
        style = self._styles.get(character.style_profile_id) if character.style_profile_id else None
        lora_paths = style.lora_paths if style else []
        base_model = style.base_model if style else "sdxl"

        pipeline = self._pipeline_factory.create(base_model=base_model, lora_paths=lora_paths)

        character.status = AssetStatus.GENERATING
        self._characters.save(character)

        generated_paths = []
        try:
            for pose in poses:
                prompt = f"{character.description}, {pose}, character reference sheet, consistent character design"
                request = GenerationRequest(
                    prompt=prompt,
                    seed=character.seed,
                    lora_paths=lora_paths,
                    width=1024,
                    height=1024,
                )
                result = pipeline.generate(request)
                generated_paths.append(result.image_path)
        except Exception as exc:  # noqa: BLE001
            character.status = AssetStatus.FAILED
            self._characters.save(character)
            raise GenerationError(f"Falha ao gerar reference sheet: {exc}") from exc

        character.reference_image_paths.extend(generated_paths)
        character.status = AssetStatus.READY
        return self._characters.save(character)
