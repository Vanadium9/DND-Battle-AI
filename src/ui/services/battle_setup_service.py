"""Build random GUI test battles from saved characters and presets."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Iterable

from character import CharacterRepository, InternalCharacter
from combat import (
    Bandit,
    Character,
    ClericLifeSupport,
    CombatEnvironment,
    FireElementalSimple,
    FighterChampionArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    GoblinArcher,
    GoblinMelee,
    GridMap,
    OrcWarrior,
    Position,
    SkeletonArcher,
    Stats,
    Team,
    WeaponAttack,
    WizardEvoker,
    Wolf,
)
from combat.class_features import Resource
from combat.damage import normalize_damage_type_set
from combat.inventory import get_item_definition
from combat.map_config import (
    DEFAULT_MAP_CONFIG_DIR,
    MapConfig,
    MapConfigValidationError,
    load_map_config_by_name,
    map_options,
    normalize_map_key,
)
from combat.race_traits import RaceTraits
from combat.spellcasting import resolve_spell_list


DIFFICULTIES: dict[str, str] = {
    "easy": "Лёгкий",
    "medium": "Средний",
    "hard": "Сложный",
}

ENEMY_GROUPS: dict[str, str] = {
    "auto": "Автоматически по сложности",
    "goblin_patrol": "Goblin patrol",
    "orc_raiders": "Orc raiders",
    "undead_archers": "Undead archers",
    "mixed_enemies": "Mixed enemies",
    "fire_elemental": "Fire elemental",
}

CONTROLLER_MODES: dict[str, str] = {
    "ai_players": "AI управляет игроками",
    "ai_enemies": "AI управляет врагами",
    "ai_all": "AI управляет всеми",
    "manual_players_ai_enemies": "Игрок вручную управляет игроками, AI управляет врагами",
}

PARTY_PRESETS: dict[str, str] = {
    "none": "Выбрать созданных персонажей",
    "fighter_level_1": "Fighter level 1",
    "fighter_archer_level_5": "Fighter Champion Archer level 5",
    "balanced_level_5": "Fighter + Cleric + Wizard level 5",
}


@dataclass(frozen=True)
class BattleSetupRequest:
    """All inputs needed to create a GUI test battle."""

    saved_character_ids: tuple[str, ...] = ()
    party_preset: str = "none"
    difficulty: str = "medium"
    map_name: str = "open_field"
    enemy_group: str = "auto"
    controller_mode: str = "manual_players_ai_enemies"
    seed: int | None = None


@dataclass(frozen=True)
class BattleSetupResult:
    """Created battle plus user-facing metadata for BattleScreen."""

    environment: CombatEnvironment
    party_names: tuple[str, ...]
    enemy_names: tuple[str, ...]
    map_name: str
    difficulty: str
    controller_mode: str
    seed: int | None
    summary: str


class BattleSetupService:
    """Create CombatEnvironment instances for GUI random battles."""

    def __init__(
        self,
        character_repository: CharacterRepository | None = None,
        map_dir: str | Path = DEFAULT_MAP_CONFIG_DIR,
    ) -> None:
        self.character_repository = character_repository or CharacterRepository()
        self.map_dir = Path(map_dir)

    def list_saved_characters(self) -> list[InternalCharacter]:
        return self.character_repository.list_characters()

    def party_presets(self) -> dict[str, str]:
        return dict(PARTY_PRESETS)

    def difficulties(self) -> dict[str, str]:
        return dict(DIFFICULTIES)

    def maps(self) -> dict[str, str]:
        options = map_options(self.map_dir)
        if options:
            options["random"] = "random"
        return options

    def enemy_groups(self) -> dict[str, str]:
        return dict(ENEMY_GROUPS)

    def controller_modes(self) -> dict[str, str]:
        return dict(CONTROLLER_MODES)

    def preview_battle(self, request: BattleSetupRequest) -> str:
        """Return a stable text description without creating an environment."""

        resolved_map = self.resolve_map_name(request.map_name, request.seed)
        party_names = self._party_preview_names(request)
        enemy_names = self._enemy_preview_names(request, len(party_names))
        return _format_summary(
            party_names=party_names,
            enemy_names=enemy_names,
            map_name=resolved_map,
            difficulty=_label(DIFFICULTIES, request.difficulty),
            controller_mode=_label(CONTROLLER_MODES, request.controller_mode),
            seed=request.seed,
        )

    def create_random_battle(self, request: BattleSetupRequest) -> BattleSetupResult:
        """Create a fully initialized CombatEnvironment for the GUI."""

        self._validate_request(request)
        resolved_map_name = self.resolve_map_name(request.map_name, request.seed)
        map_config = self.get_map_config(resolved_map_name)
        grid_map = map_config.to_grid_map()
        party = self._resolve_party(request)
        enemies = self._resolve_enemies(request, len(party))
        self._place_characters(grid_map, party, enemies, map_config)
        environment = CombatEnvironment(
            characters=[*party, *enemies],
            grid_map=grid_map,
            initiative_seed=request.seed,
            log_to_console=False,
        )
        party_names = tuple(character.name for character in party)
        enemy_names = tuple(character.name for character in enemies)
        difficulty_label = _label(DIFFICULTIES, request.difficulty)
        controller_label = _label(CONTROLLER_MODES, request.controller_mode)
        summary = _format_summary(
            party_names=party_names,
            enemy_names=enemy_names,
            map_name=resolved_map_name,
            difficulty=difficulty_label,
            controller_mode=controller_label,
            seed=request.seed,
        )
        return BattleSetupResult(
            environment=environment,
            party_names=party_names,
            enemy_names=enemy_names,
            map_name=resolved_map_name,
            difficulty=difficulty_label,
            controller_mode=request.controller_mode,
            seed=request.seed,
            summary=summary,
        )

    def resolve_map_name(self, map_name: str, seed: int | None = None) -> str:
        normalized = normalize_map_key(map_name)
        if normalized == "random":
            rng = random.Random(seed)
            return rng.choice(tuple(key for key in self.maps() if key != "random"))
        if normalized not in self.maps():
            raise ValueError(f"Unknown map: {map_name}")
        return normalized

    def get_map_config(self, map_name: str) -> MapConfig:
        """Return a validated map config for a selected map name."""

        try:
            return load_map_config_by_name(map_name, self.map_dir)
        except MapConfigValidationError:
            raise

    def create_map(self, map_name: str) -> GridMap:
        return self.get_map_config(map_name).to_grid_map()

    def _validate_request(self, request: BattleSetupRequest) -> None:
        if request.party_preset not in PARTY_PRESETS:
            raise ValueError(f"Unknown party preset: {request.party_preset}")
        if request.difficulty not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {request.difficulty}")
        if request.enemy_group not in ENEMY_GROUPS:
            raise ValueError(f"Unknown enemy group: {request.enemy_group}")
        if request.controller_mode not in CONTROLLER_MODES:
            raise ValueError(f"Unknown controller mode: {request.controller_mode}")
        if normalize_map_key(request.map_name) not in self.maps():
            raise ValueError(f"Unknown map: {request.map_name}")
        if request.party_preset == "none" and not request.saved_character_ids:
            raise ValueError("Выберите хотя бы одного персонажа или готовый party preset.")

    def _resolve_party(self, request: BattleSetupRequest) -> list[Character]:
        if request.party_preset != "none":
            return _party_preset_characters(request.party_preset)

        party: list[Character] = []
        missing: list[str] = []
        for character_id in request.saved_character_ids:
            internal = self.character_repository.get_character(character_id)
            if internal is None:
                missing.append(character_id)
                continue
            party.append(_internal_character_to_combat(internal))
        if missing:
            raise ValueError(f"Saved character not found: {', '.join(missing)}")
        if not party:
            raise ValueError("Выберите хотя бы одного персонажа или готовый party preset.")
        return party

    def _resolve_enemies(
        self,
        request: BattleSetupRequest,
        party_size: int,
    ) -> list[Character]:
        group_key = request.enemy_group
        if group_key == "auto":
            group_key = _auto_enemy_group(request.difficulty, party_size)
        return _enemy_group_characters(group_key)

    def _party_preview_names(self, request: BattleSetupRequest) -> tuple[str, ...]:
        if request.party_preset != "none":
            return tuple(character.name for character in _party_preset_characters(request.party_preset))
        names = []
        for character_id in request.saved_character_ids:
            character = self.character_repository.get_character(character_id)
            names.append(character.name if character is not None else f"{character_id} (missing)")
        return tuple(names)

    def _enemy_preview_names(
        self,
        request: BattleSetupRequest,
        party_size: int,
    ) -> tuple[str, ...]:
        group_key = request.enemy_group
        if group_key == "auto":
            group_key = _auto_enemy_group(request.difficulty, party_size)
        return tuple(character.name for character in _enemy_group_characters(group_key))

    @staticmethod
    def _place_characters(
        grid_map: GridMap,
        party: list[Character],
        enemies: list[Character],
        map_config: MapConfig,
    ) -> None:
        player_positions = _configured_spawn_positions(
            map_config,
            Team.PLAYERS,
            len(party),
        )
        enemy_positions = _configured_spawn_positions(
            map_config,
            Team.ENEMIES,
            len(enemies),
        )
        for character, position in zip(party, player_positions):
            character.team = Team.PLAYERS
            character.position = position
        for character, position in zip(enemies, enemy_positions):
            character.team = Team.ENEMIES
            character.position = position


def _party_preset_characters(preset_key: str) -> list[Character]:
    if preset_key == "fighter_level_1":
        return [FighterLevel1Basic()]
    if preset_key == "fighter_archer_level_5":
        return [FighterChampionArcher()]
    if preset_key == "balanced_level_5":
        return [FighterChampionGreatsword(), ClericLifeSupport(), WizardEvoker()]
    raise ValueError(f"Unknown party preset: {preset_key}")


def _enemy_group_characters(group_key: str) -> list[Character]:
    if group_key == "goblin_patrol":
        return [GoblinMelee(), GoblinArcher()]
    if group_key == "orc_raiders":
        return [OrcWarrior(), GoblinMelee(), Wolf()]
    if group_key == "undead_archers":
        return [SkeletonArcher(), SkeletonArcher()]
    if group_key == "mixed_enemies":
        return [OrcWarrior(), GoblinArcher(), Bandit(), Wolf()]
    if group_key == "fire_elemental":
        return [FireElementalSimple()]
    raise ValueError(f"Unknown enemy group: {group_key}")


def _auto_enemy_group(difficulty: str, party_size: int) -> str:
    if difficulty == "easy":
        return "goblin_patrol" if party_size > 1 else "goblin_patrol"
    if difficulty == "hard":
        return "mixed_enemies" if party_size > 1 else "orc_raiders"
    return "orc_raiders" if party_size > 1 else "goblin_patrol"


def _internal_character_to_combat(internal: InternalCharacter) -> Character:
    stats = Stats(**{ability: int(internal.stats.get(ability, 10)) for ability in _ABILITIES})
    inventory = []
    for item in internal.inventory:
        definition = get_item_definition(str(item.get("name", "")))
        if definition is None:
            continue
        definition.quantity = int(item.get("quantity", definition.quantity))
        inventory.append(definition)
    cantrips, prepared_spells = _split_spells(internal)
    resources = {
        name: Resource(name=name, max_uses=max(1, int(value)), uses_remaining=int(value))
        for name, value in internal.resources.items()
    }
    character = Character(
        name=internal.name,
        hp=internal.hp,
        max_hp=max(1, internal.hp),
        ac=internal.ac,
        position=Position(),
        speed=_grid_speed(internal.speed),
        stats=stats,
        team=Team.PLAYERS,
        class_name=internal.class_name or None,
        subclass_name=internal.subclass_name,
        level=internal.level,
        experience=internal.experience,
        proficiency_bonus=internal.proficiency_bonus,
        race_name=internal.race_name or None,
        race_traits=_race_traits_from_internal(internal),
        role=internal.role,
        resistances=normalize_damage_type_set(internal.resistances),
        immunities=normalize_damage_type_set(internal.immunities),
        vulnerabilities=normalize_damage_type_set(internal.vulnerabilities),
        fighting_style=str(internal.race_traits.get("fighting_style", "")) or None,
        wearing_armor=bool(internal.armor and internal.armor.get("name") != "None"),
        weapons=[_weapon_from_mapping(weapon) for weapon in internal.weapons],
        inventory=inventory,
        prepared_spells=resolve_spell_list(prepared_spells),
        cantrips=resolve_spell_list(cantrips),
        resources=resources,
    )
    if not character.weapons:
        character.weapons.append(_default_weapon())
    return character


def _split_spells(internal: InternalCharacter) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cantrips: list[str] = []
    prepared: list[str] = []
    levels: dict[str, int] = {}
    for spell in internal.spells:
        if isinstance(spell, dict):
            name = str(spell.get("name", ""))
            levels[_key(name)] = int(spell.get("level", 1))
    for spell_name in internal.prepared_spells:
        if levels.get(_key(spell_name), 1) == 0:
            cantrips.append(spell_name)
        else:
            prepared.append(spell_name)
    return tuple(cantrips), tuple(prepared)


def _race_traits_from_internal(internal: InternalCharacter) -> RaceTraits | None:
    if not internal.race_name:
        return None
    return RaceTraits(
        name=internal.race_name,
        ability_score_bonuses=dict(internal.race_traits.get("ability_score_bonuses", {})),
        speed=_grid_speed(int(internal.race_traits.get("speed", internal.speed))),
        size=str(internal.race_traits.get("size", "Medium")),
        darkvision_range=internal.race_traits.get("darkvision_range"),
        skill_proficiencies=tuple(internal.race_traits.get("skill_proficiencies", ())),
        weapon_proficiencies=tuple(internal.race_traits.get("weapon_proficiencies", ())),
        saving_throw_advantages=tuple(
            internal.race_traits.get("saving_throw_advantages", ())
        ),
        damage_resistances=tuple(internal.race_traits.get("damage_resistances", ())),
        special_traits=tuple(internal.race_traits.get("special_traits", ())),
    )


def _weapon_from_mapping(data: dict[str, object]) -> WeaponAttack:
    name = str(data.get("name", "Weapon"))
    ability_score = "dex" if _weapon_range(data) > 1 else "str"
    return WeaponAttack(
        name=name,
        description=str(data.get("description", "")),
        range=_weapon_range(data),
        damage=data.get("damage", "1d6"),
        attack_bonus=int(data.get("attack_bonus", 0)),
        ability_score=str(data.get("ability_score", ability_score)),
        damage_ability_score=str(data.get("damage_ability_score", ability_score)),
        damage_type=data.get("damage_type"),
        two_handed=bool(data.get("two_handed", False)),
        heavy=bool(data.get("heavy", False)),
    )


def _weapon_range(data: dict[str, object]) -> int:
    return max(1, int(data.get("range", 1)))


def _default_weapon() -> WeaponAttack:
    return WeaponAttack(
        name="Unarmed Strike",
        range=1,
        damage=1,
        attack_bonus=0,
        ability_score="str",
        damage_ability_score="str",
        damage_type="bludgeoning",
    )


def _grid_speed(speed: int) -> int:
    speed = int(speed)
    if speed > 10:
        return max(1, speed // 10)
    return max(1, speed)


def _configured_spawn_positions(
    map_config: MapConfig,
    team: Team,
    count: int,
) -> list[Position]:
    if count <= 0:
        return []
    positions = (
        map_config.spawn_zones.players
        if team is Team.PLAYERS
        else map_config.spawn_zones.enemies
    )
    if len(positions) < count:
        raise ValueError("Map has not enough configured spawn cells.")
    return list(positions[:count])


def _format_summary(
    *,
    party_names: Iterable[str],
    enemy_names: Iterable[str],
    map_name: str,
    difficulty: str,
    controller_mode: str,
    seed: int | None,
) -> str:
    return "\n".join(
        (
            f"Party: {', '.join(party_names) or 'не выбрана'}",
            f"Enemies: {', '.join(enemy_names) or 'не выбраны'}",
            f"Map: {map_name}",
            f"Difficulty: {difficulty}",
            f"Controller: {controller_mode}",
            f"Seed: {seed if seed is not None else 'random'}",
        )
    )


def _label(options: dict[str, str], key: str) -> str:
    return options.get(key, key)


def _key(value: object) -> str:
    return str(value).strip().lower()


_ABILITIES = ("str", "dex", "con", "int", "wis", "cha")
