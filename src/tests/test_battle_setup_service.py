from pathlib import Path
from uuid import uuid4

import pytest

from character import CharacterRepository, InternalCharacter
from combat import Position, Team, TerrainType
from ui.services import BattleSetupRequest, BattleSetupService


def test_preset_random_battle_creates_environment_with_spawned_sides() -> None:
    service = BattleSetupService()

    result = service.create_random_battle(
        BattleSetupRequest(
            party_preset="balanced_level_5",
            difficulty="hard",
            map_name="cover_arena",
            enemy_group="auto",
            controller_mode="ai_all",
            seed=42,
        )
    )

    state = result.environment.combat_state
    players = [character for character in state.characters if character.team is Team.PLAYERS]
    enemies = [character for character in state.characters if character.team is Team.ENEMIES]

    assert len(players) == 3
    assert len(enemies) >= 3
    assert result.map_name == "cover_arena"
    assert result.difficulty == "Сложный"
    assert "Party:" in result.summary
    assert "Enemies:" in result.summary
    assert state.grid_map is not None
    assert all(state.grid_map.is_walkable(character.position) for character in state.characters)
    assert all(character.position.x <= 1 for character in players)
    assert all(character.position.x >= state.grid_map.width - 2 for character in enemies)


def test_saved_characters_can_be_used_as_party() -> None:
    repository = CharacterRepository(_repo_dir())
    saved = repository.save_character(_internal_character())
    service = BattleSetupService(repository)

    result = service.create_random_battle(
        BattleSetupRequest(
            saved_character_ids=(saved.id,),
            party_preset="none",
            difficulty="easy",
            map_name="open_field",
            enemy_group="goblin_patrol",
            seed=7,
        )
    )

    player = result.environment.combat_state.characters[0]
    assert player.name == "Saved Fighter"
    assert player.team is Team.PLAYERS
    assert player.speed == 3
    assert player.weapons[0].name == "Longsword"


def test_no_party_selection_is_rejected() -> None:
    service = BattleSetupService()

    with pytest.raises(ValueError, match="Выберите хотя бы одного персонажа"):
        service.create_random_battle(BattleSetupRequest(party_preset="none"))


def test_random_map_is_reproducible_with_seed() -> None:
    service = BattleSetupService()

    first = service.resolve_map_name("random", seed=123)
    second = service.resolve_map_name("random", seed=123)

    assert first == second
    assert first != "random"


def test_map_presets_include_expected_terrain() -> None:
    service = BattleSetupService()

    cover_map = service.create_map("cover_arena")
    difficult_map = service.create_map("difficult_terrain_pass")
    obstacle_map = service.create_map("obstacle_corridor")

    assert _contains_terrain(cover_map, TerrainType.LOW_COVER)
    assert _contains_terrain(difficult_map, TerrainType.DIFFICULT_TERRAIN)
    assert _contains_terrain(obstacle_map, TerrainType.BLOCKED)


def _repo_dir() -> Path:
    path = Path("checkpoints") / f"test_battle_setup_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _internal_character() -> InternalCharacter:
    return InternalCharacter(
        id="saved-fighter",
        name="Saved Fighter",
        class_name="Fighter",
        subclass_name="Champion",
        level=3,
        experience=900,
        race_name="Human",
        role="MELEE_DAMAGE",
        stats={
            "str": 16,
            "dex": 12,
            "con": 14,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        hp=28,
        ac=16,
        speed=30,
        proficiency_bonus=2,
        weapons=(
            {"name": "Longsword", "range": 1, "damage": "1d8", "damage_type": "slashing"},
        ),
        armor={"name": "Chain Mail", "base_ac": 16},
        class_features=("Fighting Style", "Second Wind", "Action Surge"),
        subclass_features=("Improved Critical",),
        race_traits={"size": "Medium", "speed": 30},
    )


def _contains_terrain(grid_map, terrain: TerrainType) -> bool:
    return any(
        grid_map.terrain_at(Position(x, y)) is terrain
        for y in range(grid_map.height)
        for x in range(grid_map.width)
    )
