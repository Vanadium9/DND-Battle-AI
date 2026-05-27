"""Random encounter generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import random

from combat.evaluation_scenarios import (
    level_1_fighter_vs_goblin,
    level_3_cleric_life_fighter_vs_orc_goblin,
    level_3_fighter_champion_vs_orc,
    level_5_full_party_vs_mixed_enemies,
    level_5_melee_party_difficult_terrain_vs_archers,
    level_5_wizard_evoker_vs_three_goblins,
)
from combat.environment import CombatEnvironment
from combat.map import GridMap
from combat.models import Character, CombatState, Position
from combat.monsters import (
    FireElementalSimple,
    GoblinArcher,
    GoblinMelee,
    OrcWarrior,
    SkeletonArcher,
)
from combat.presets import (
    ClericLifeSupport,
    FighterArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    Goblin,
    Orc,
    WizardEvoker,
)
from combat.terrain import TerrainType


CharacterFactory = Callable[[Position], Character]
StateFactory = Callable[[], CombatState]


@dataclass(frozen=True)
class CurriculumStage:
    """One fixed difficulty tier for curriculum learning."""

    level: int
    name: str
    description: str
    factory: StateFactory


@dataclass
class EncounterGenerator:
    """Generate simple random combat encounters."""

    seed: int | None = None
    width: int = 8
    height: int = 8
    curriculum_level: int | None = None
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def reseed(self, seed: int | None) -> None:
        self.seed = seed
        self._random.seed(seed)

    def generate_state(self) -> CombatState:
        if self.curriculum_level is not None:
            return self.generate_curriculum_state(self.curriculum_level)

        player_count = self._random.randint(1, 2)
        enemy_count = self._random.randint(1, 4)

        player_factories = self._random.sample(
            [FighterChampionGreatsword, FighterArcher],
            k=player_count,
        )
        enemy_factories = [
            self._random.choice([Goblin, Orc])
            for _ in range(enemy_count)
        ]

        player_positions = self._random_positions(
            player_count,
            x_values=range(0, self.width // 2),
        )
        enemy_positions = self._random_positions(
            enemy_count,
            x_values=range(self.width // 2, self.width),
        )

        characters = [
            factory(position)
            for factory, position in zip(player_factories, player_positions)
        ]
        characters.extend(
            factory(position)
            for factory, position in zip(enemy_factories, enemy_positions)
        )

        return CombatState(
            characters=characters,
            grid_map=GridMap(width=self.width, height=self.height),
        )

    def set_curriculum_level(self, level: int | None) -> None:
        """Set the active curriculum level, or disable curriculum with None."""

        if level is None:
            self.curriculum_level = None
            return
        self.curriculum_level = clamp_curriculum_level(level)

    def generate_curriculum_state(self, level: int | None = None) -> CombatState:
        """Generate a state for a fixed curriculum difficulty level."""

        selected_level = self.curriculum_level if level is None else level
        stage = get_curriculum_stage(selected_level or 1)
        return stage.factory()

    def generate_curriculum_environment(
        self,
        level: int | None = None,
        use_initiative: bool = True,
        log_to_console: bool = True,
    ) -> CombatEnvironment:
        """Generate an environment for a fixed curriculum difficulty level."""

        combat_state = self.generate_curriculum_state(level)
        return CombatEnvironment(
            characters=combat_state.characters,
            grid_map=combat_state.grid_map,
            use_initiative=use_initiative,
            initiative_seed=self.seed,
            log_to_console=log_to_console,
        )

    def generate_environment(
        self,
        use_initiative: bool = True,
        log_to_console: bool = True,
    ) -> CombatEnvironment:
        combat_state = self.generate_state()
        return CombatEnvironment(
            characters=combat_state.characters,
            grid_map=combat_state.grid_map,
            use_initiative=use_initiative,
            initiative_seed=self.seed,
            log_to_console=log_to_console,
        )

    def generate(
        self,
        as_environment: bool = False,
        use_initiative: bool = True,
        log_to_console: bool = True,
    ) -> CombatState | CombatEnvironment:
        if as_environment:
            return self.generate_environment(
                use_initiative=use_initiative,
                log_to_console=log_to_console,
            )
        return self.generate_state()

    def _random_positions(
        self,
        count: int,
        x_values: range,
    ) -> list[Position]:
        positions = [
            Position(x, y)
            for x in x_values
            for y in range(self.height)
        ]
        self._random.shuffle(positions)
        return positions[:count]


def get_curriculum_stage(level: int) -> CurriculumStage:
    """Return a curriculum stage by clamped difficulty level."""

    selected_level = clamp_curriculum_level(level)
    return CURRICULUM_STAGES[selected_level - 1]


def clamp_curriculum_level(level: int) -> int:
    """Clamp curriculum level to the supported 1-9 range."""

    return max(1, min(MAX_CURRICULUM_LEVEL, int(level)))


def _curriculum_level_2_state() -> CombatState:
    return CombatState(
        characters=[
            FighterLevel1Basic(Position(0, 2)),
            GoblinMelee(Position(5, 1)),
            GoblinArcher(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def _curriculum_resistance_state() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            WizardEvoker(Position(0, 3)),
            FireElementalSimple(Position(7, 2)),
            SkeletonArcher(Position(6, 1)),
        ],
        grid_map=GridMap(width=8, height=5),
    )


def _curriculum_obstacle_cover_state() -> CombatState:
    normal = TerrainType.NORMAL
    blocked = TerrainType.BLOCKED
    low = TerrainType.LOW_COVER
    high = TerrainType.HIGH_COVER
    terrain = (
        (normal, normal, normal, blocked, normal, low, normal, normal),
        (normal, low, normal, blocked, normal, normal, low, normal),
        (normal, normal, normal, high, normal, normal, normal, normal),
        (normal, low, normal, blocked, normal, normal, low, normal),
        (normal, normal, normal, blocked, normal, low, normal, normal),
    )
    return CombatState(
        characters=[
            FighterArcher(Position(0, 1)),
            ClericLifeSupport(Position(0, 3)),
            OrcWarrior(Position(7, 1)),
            GoblinArcher(Position(7, 2)),
            SkeletonArcher(Position(7, 3)),
        ],
        grid_map=GridMap(width=8, height=5, terrain_grid=terrain),
    )


CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        level=1,
        name="1 Fighter level 1 vs 1 Goblin",
        description="Introductory duel with one low-level martial character.",
        factory=level_1_fighter_vs_goblin,
    ),
    CurriculumStage(
        level=2,
        name="1 Fighter level 1 vs 2 Goblins",
        description="Early outnumbered fight with melee and ranged goblins.",
        factory=_curriculum_level_2_state,
    ),
    CurriculumStage(
        level=3,
        name="Fighter level 3 Champion vs Orc",
        description="Subclass fighter against a single durable brute.",
        factory=level_3_fighter_champion_vs_orc,
    ),
    CurriculumStage(
        level=4,
        name="Fighter + Cleric vs Orc + Goblins",
        description="Small party fight with healing and mixed enemies.",
        factory=level_3_cleric_life_fighter_vs_orc_goblin,
    ),
    CurriculumStage(
        level=5,
        name="Wizard scenarios",
        description="Evoker spellcasting and AoE targeting against clustered enemies.",
        factory=level_5_wizard_evoker_vs_three_goblins,
    ),
    CurriculumStage(
        level=6,
        name="mixed party vs mixed enemies",
        description="Full level 5 party against varied monster roles.",
        factory=level_5_full_party_vs_mixed_enemies,
    ),
    CurriculumStage(
        level=7,
        name="enemies with resistances/immunities",
        description="Resistant and immune enemies that require damage-type awareness.",
        factory=_curriculum_resistance_state,
    ),
    CurriculumStage(
        level=8,
        name="maps with obstacles and cover",
        description="Map reasoning with blocked cells, low cover and high cover.",
        factory=_curriculum_obstacle_cover_state,
    ),
    CurriculumStage(
        level=9,
        name="maps with difficult terrain and ranged enemies",
        description="Movement-cost planning against ranged enemies on difficult terrain.",
        factory=level_5_melee_party_difficult_terrain_vs_archers,
    ),
)

MAX_CURRICULUM_LEVEL = len(CURRICULUM_STAGES)


__all__ = [
    "CURRICULUM_STAGES",
    "MAX_CURRICULUM_LEVEL",
    "CurriculumStage",
    "EncounterGenerator",
    "clamp_curriculum_level",
    "get_curriculum_stage",
]
