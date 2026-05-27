"""Reusable evaluation encounters for policy benchmarking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from combat.abilities import WeaponAttack
from combat.character_builder import build_character
from combat.environment import CombatEnvironment
from combat.map import GridMap
from combat.models import Character, CombatState, Position, Stats, Team
from combat.monsters import (
    Bandit,
    FireElementalSimple,
    GoblinArcher,
    GoblinMelee,
    OrcWarrior,
    SkeletonArcher,
    Wolf,
)
from combat.presets import (
    ClericLifeSupport,
    FighterChampionArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    WizardEvoker,
)
from combat.terrain import TerrainType


@dataclass(frozen=True)
class EvaluationScenario:
    """A named combat setup used for policy evaluation."""

    name: str
    level: int
    description: str
    factory: Callable[[], CombatState]

    def create_state(self) -> CombatState:
        """Build a fresh combat state for one evaluation episode."""

        return self.factory()

    def create_environment(
        self,
        *,
        use_initiative: bool = True,
        initiative_seed: int | None = None,
        log_to_console: bool = False,
    ) -> CombatEnvironment:
        """Build a fresh combat environment for one evaluation episode."""

        state = self.create_state()
        return CombatEnvironment(
            characters=state.characters,
            grid_map=state.grid_map,
            use_initiative=use_initiative,
            initiative_seed=initiative_seed,
            log_to_console=log_to_console,
        )


def get_evaluation_scenarios() -> tuple[EvaluationScenario, ...]:
    """Return the stable default evaluation scenario list."""

    return EVALUATION_SCENARIOS


def get_evaluation_scenarios_by_level() -> dict[int, tuple[EvaluationScenario, ...]]:
    """Return evaluation scenarios grouped by supported character level."""

    grouped: dict[int, list[EvaluationScenario]] = defaultdict(list)
    for scenario in EVALUATION_SCENARIOS:
        grouped[scenario.level].append(scenario)
    return {
        level: tuple(grouped.get(level, ()))
        for level in range(1, 6)
    }


def get_scenario(name: str) -> EvaluationScenario | None:
    """Return a scenario by exact or case-insensitive name."""

    lookup = _lookup_key(name)
    for scenario in EVALUATION_SCENARIOS:
        if scenario.name == name or _lookup_key(scenario.name) == lookup:
            return scenario
    return None


def scenario_names() -> tuple[str, ...]:
    """Return stable scenario names for CLI choices."""

    return tuple(scenario.name for scenario in EVALUATION_SCENARIOS)


def level_1_fighter_vs_goblin() -> CombatState:
    return CombatState(
        characters=[
            FighterLevel1Basic(Position(0, 2)),
            GoblinMelee(Position(4, 2)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_1_fighter_cleric_vs_two_goblins() -> CombatState:
    return CombatState(
        characters=[
            FighterLevel1Basic(Position(0, 1)),
            _cleric(1, Position(0, 3), name="Cleric Life Level 1"),
            GoblinMelee(Position(5, 1)),
            GoblinArcher(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_2_fighter_cleric_vs_goblin_bandit() -> CombatState:
    return CombatState(
        characters=[
            _fighter(2, Position(0, 1), name="Fighter Level 2"),
            _cleric(2, Position(0, 3), name="Cleric Life Level 2"),
            GoblinMelee(Position(5, 1)),
            Bandit(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_3_fighter_champion_vs_orc() -> CombatState:
    return CombatState(
        characters=[
            _fighter(3, Position(0, 2), name="Fighter Champion Level 3"),
            OrcWarrior(Position(5, 2)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_3_cleric_life_fighter_vs_orc_goblin() -> CombatState:
    return CombatState(
        characters=[
            _cleric(3, Position(0, 1), name="Cleric Life Level 3"),
            _fighter(3, Position(0, 3), name="Fighter Champion Level 3"),
            OrcWarrior(Position(5, 1)),
            GoblinArcher(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_4_fighter_cleric_vs_orc_skeleton() -> CombatState:
    return CombatState(
        characters=[
            _fighter(4, Position(0, 1), name="Fighter Champion Level 4"),
            _cleric(4, Position(0, 3), name="Cleric Life Level 4"),
            OrcWarrior(Position(6, 1)),
            SkeletonArcher(Position(6, 3)),
        ],
        grid_map=GridMap(width=7, height=5),
    )


def level_5_wizard_evoker_vs_three_goblins() -> CombatState:
    return CombatState(
        characters=[
            WizardEvoker(Position(0, 2)),
            GoblinMelee(Position(5, 1)),
            GoblinMelee(Position(5, 2)),
            GoblinArcher(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_5_wizard_evoker_vs_fire_elemental() -> CombatState:
    return CombatState(
        characters=[
            WizardEvoker(Position(0, 2)),
            FireElementalSimple(Position(5, 2)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def level_5_full_party_vs_mixed_enemies() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            ClericLifeSupport(Position(0, 2)),
            WizardEvoker(Position(0, 3)),
            OrcWarrior(Position(7, 1)),
            GoblinArcher(Position(7, 2)),
            Bandit(Position(7, 3)),
            Wolf(Position(6, 2)),
        ],
        grid_map=GridMap(width=8, height=5),
    )


def level_5_ranged_party_cover_map_vs_mixed_enemies() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionArcher(Position(0, 1)),
            _wizard(5, Position(0, 2), name="Wizard Evoker Ranged"),
            _cleric(5, Position(0, 3), name="Cleric Life Ranged"),
            GoblinArcher(Position(7, 1)),
            SkeletonArcher(Position(7, 2)),
            OrcWarrior(Position(7, 3)),
        ],
        grid_map=_cover_map(),
    )


def level_5_melee_party_difficult_terrain_vs_archers() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            _fighter(5, Position(0, 3), name="Fighter Champion Shield"),
            ClericLifeSupport(Position(0, 2)),
            GoblinArcher(Position(7, 1)),
            SkeletonArcher(Position(7, 2)),
            Bandit(Position(7, 3)),
        ],
        grid_map=_difficult_terrain_map(),
    )


EVALUATION_SCENARIOS: tuple[EvaluationScenario, ...] = (
    EvaluationScenario(
        name="Level 1 Fighter vs 1 Goblin",
        level=1,
        description="Single level 1 fighter duel against a melee goblin.",
        factory=level_1_fighter_vs_goblin,
    ),
    EvaluationScenario(
        name="Level 1 Fighter + Cleric vs 2 Goblins",
        level=1,
        description="Two-character level 1 party against melee and ranged goblins.",
        factory=level_1_fighter_cleric_vs_two_goblins,
    ),
    EvaluationScenario(
        name="Level 2 Fighter + Cleric vs Goblin + Bandit",
        level=2,
        description="Level 2 party check for early resources before subclasses.",
        factory=level_2_fighter_cleric_vs_goblin_bandit,
    ),
    EvaluationScenario(
        name="Level 3 Fighter Champion vs Orc",
        level=3,
        description="Champion subclass duel against a durable brute.",
        factory=level_3_fighter_champion_vs_orc,
    ),
    EvaluationScenario(
        name="Level 3 Cleric Life + Fighter vs Orc + Goblin",
        level=3,
        description="Level 3 support and melee pair against mixed enemies.",
        factory=level_3_cleric_life_fighter_vs_orc_goblin,
    ),
    EvaluationScenario(
        name="Level 4 Fighter Champion + Cleric Life vs Orc + Skeleton Archer",
        level=4,
        description="Level 4 ASI-era party against melee pressure and ranged undead.",
        factory=level_4_fighter_cleric_vs_orc_skeleton,
    ),
    EvaluationScenario(
        name="Level 5 Wizard Evoker vs 3 Goblins",
        level=5,
        description="Level 5 evoker AoE benchmark against clustered low-HP enemies.",
        factory=level_5_wizard_evoker_vs_three_goblins,
    ),
    EvaluationScenario(
        name="Level 5 Wizard Evoker vs FireElementalSimple",
        level=5,
        description="Damage-type benchmark against fire immunity and weapon resistance.",
        factory=level_5_wizard_evoker_vs_fire_elemental,
    ),
    EvaluationScenario(
        name="Level 5 Fighter + Cleric + Wizard vs mixed enemies",
        level=5,
        description="Full level 5 party against mixed melee and ranged enemies.",
        factory=level_5_full_party_vs_mixed_enemies,
    ),
    EvaluationScenario(
        name="Level 5 ranged party on map with cover vs mixed enemies",
        level=5,
        description="Ranged level 5 party benchmark with low and high cover.",
        factory=level_5_ranged_party_cover_map_vs_mixed_enemies,
    ),
    EvaluationScenario(
        name="Level 5 melee party on difficult terrain map vs archers",
        level=5,
        description="Melee level 5 party benchmark crossing difficult terrain into archers.",
        factory=level_5_melee_party_difficult_terrain_vs_archers,
    ),
)


def _fighter(
    level: int,
    position: Position,
    *,
    name: str | None = None,
    ranged: bool = False,
) -> Character:
    subclass_name = "Champion" if level >= 3 else None
    if ranged:
        weapon = WeaponAttack(
            name="Longbow",
            description="Long-range bow attack.",
            range=6,
            damage="1d8",
            attack_bonus=0,
            ability_score="dex",
            damage_ability_score="dex",
            damage_type="piercing",
            two_handed=True,
        )
        stats = Stats(str=10, dex=16 + (2 if level >= 4 else 0), con=14, wis=12)
        fighting_style = "Archery"
        hp = 12 + (level - 1) * 7
        ac = 15
    else:
        weapon = WeaponAttack(
            name="Greatsword" if level >= 3 else "Longsword",
            description="Martial melee attack.",
            range=1,
            damage="2d6" if level >= 3 else "1d8",
            attack_bonus=0,
            ability_score="str",
            damage_ability_score="str",
            damage_type="slashing",
            two_handed=level >= 3,
            heavy=level >= 3,
        )
        stats = Stats(str=16 + (2 if level >= 4 else 0), dex=12, con=16, wis=10)
        fighting_style = "Great Weapon Fighting" if level >= 3 else "Defense"
        hp = 13 + (level - 1) * 8
        ac = 16

    return build_character(
        name=name or f"Fighter Level {level}",
        class_name="Fighter",
        subclass_name=subclass_name,
        level=level,
        experience=_xp_for_level(level),
        stats=stats,
        hp=hp,
        max_hp=hp,
        ac=ac,
        speed=3,
        position=position,
        weapons=(weapon,),
        fighting_style=fighting_style,
        wearing_armor=True,
    )


def _cleric(level: int, position: Position, *, name: str | None = None) -> Character:
    hp = 10 + (level - 1) * 7
    return build_character(
        name=name or f"Cleric Life Level {level}",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=level,
        experience=_xp_for_level(level),
        stats=Stats(str=12, dex=10, con=14, wis=16 + (2 if level >= 4 else 0), cha=12),
        hp=hp,
        max_hp=hp,
        ac=16,
        speed=3,
        position=position,
        weapons=(
            WeaponAttack(
                name="Mace",
                description="Simple melee weapon attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="bludgeoning",
            ),
        ),
        wearing_armor=True,
        cantrips=("Sacred Flame", "Spare the Dying"),
        prepared_spells=("Cure Wounds", "Healing Word", "Guiding Bolt", "Bless"),
    )


def _wizard(level: int, position: Position, *, name: str | None = None) -> Character:
    subclass_name = "School of Evocation" if level >= 2 else None
    known_spell_names = ["Magic Missile", "Shield", "Burning Hands", "Fire Bolt", "Ray of Frost"]
    prepared_spell_names = ["Magic Missile", "Shield", "Burning Hands"]
    if level >= 3:
        known_spell_names.append("Scorching Ray")
        prepared_spell_names.append("Scorching Ray")
    if level >= 5:
        known_spell_names.append("Fireball")
        prepared_spell_names.append("Fireball")

    hp = 8 + (level - 1) * 6
    return build_character(
        name=name or f"Wizard Evoker Level {level}",
        class_name="Wizard",
        subclass_name=subclass_name,
        level=level,
        experience=_xp_for_level(level),
        stats=Stats(str=8, dex=14, con=14, int=16 + (2 if level >= 4 else 0), wis=12),
        hp=hp,
        max_hp=hp,
        ac=12,
        speed=3,
        position=position,
        weapons=(
            WeaponAttack(
                name="Quarterstaff",
                description="Simple melee weapon attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="bludgeoning",
            ),
        ),
        cantrips=("Fire Bolt", "Ray of Frost"),
        known_spells=tuple(known_spell_names),
        prepared_spells=tuple(prepared_spell_names),
    )


def _cover_map() -> GridMap:
    normal = TerrainType.NORMAL
    low = TerrainType.LOW_COVER
    high = TerrainType.HIGH_COVER
    return GridMap(
        width=8,
        height=5,
        terrain_grid=(
            (normal, normal, normal, normal, normal, low, normal, normal),
            (normal, low, normal, normal, normal, normal, low, normal),
            (normal, normal, normal, high, normal, normal, normal, normal),
            (normal, low, normal, normal, normal, normal, low, normal),
            (normal, normal, normal, normal, normal, low, normal, normal),
        ),
    )


def _difficult_terrain_map() -> GridMap:
    normal = TerrainType.NORMAL
    difficult = TerrainType.DIFFICULT_TERRAIN
    return GridMap(
        width=8,
        height=5,
        terrain_grid=(
            (normal, normal, difficult, difficult, difficult, normal, normal, normal),
            (normal, normal, difficult, difficult, difficult, normal, normal, normal),
            (normal, normal, difficult, difficult, difficult, normal, normal, normal),
            (normal, normal, difficult, difficult, difficult, normal, normal, normal),
            (normal, normal, difficult, difficult, difficult, normal, normal, normal),
        ),
    )


def _xp_for_level(level: int) -> int:
    return {
        1: 0,
        2: 300,
        3: 900,
        4: 2700,
        5: 6500,
    }[max(1, min(5, int(level)))]


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


__all__ = [
    "EVALUATION_SCENARIOS",
    "EvaluationScenario",
    "get_evaluation_scenarios",
    "get_evaluation_scenarios_by_level",
    "get_scenario",
    "level_1_fighter_cleric_vs_two_goblins",
    "level_1_fighter_vs_goblin",
    "level_2_fighter_cleric_vs_goblin_bandit",
    "level_3_cleric_life_fighter_vs_orc_goblin",
    "level_3_fighter_champion_vs_orc",
    "level_4_fighter_cleric_vs_orc_skeleton",
    "level_5_full_party_vs_mixed_enemies",
    "level_5_melee_party_difficult_terrain_vs_archers",
    "level_5_ranged_party_cover_map_vs_mixed_enemies",
    "level_5_wizard_evoker_vs_fire_elemental",
    "level_5_wizard_evoker_vs_three_goblins",
    "scenario_names",
]
