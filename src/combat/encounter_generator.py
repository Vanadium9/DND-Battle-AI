"""Random encounter generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import random

from combat.evaluation_scenarios import (
    _wizard as wizard_character,
    level_1_fighter_cleric_vs_two_goblins,
    level_1_fighter_vs_goblin,
    level_2_fighter_cleric_vs_goblin_bandit,
    level_3_cleric_life_fighter_vs_orc_goblin,
    level_3_fighter_champion_vs_orc,
    level_4_fighter_cleric_vs_orc_skeleton,
    level_5_full_party_vs_mixed_enemies,
    level_5_melee_party_difficult_terrain_vs_archers,
    level_5_ranged_party_cover_map_vs_mixed_enemies,
    level_5_wizard_evoker_vs_fire_elemental,
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
    phase: str = "general"
    trainable_classes: tuple[str, ...] = ()
    rehearsal_probability: float | None = None


@dataclass
class EncounterGenerator:
    """Generate simple random combat encounters."""

    seed: int | None = None
    width: int = 8
    height: int = 8
    curriculum_level: int | None = None
    curriculum_kind: str = "general"
    rehearsal_probability: float = 0.0
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
        self.curriculum_level = self.clamp_curriculum_level(level)

    @property
    def curriculum_stages(self) -> tuple[CurriculumStage, ...]:
        """Return the stage table selected for this generator."""

        if self.curriculum_kind == "class":
            return CLASS_CURRICULUM_STAGES
        return CURRICULUM_STAGES

    @property
    def max_curriculum_level(self) -> int:
        return len(self.curriculum_stages)

    def clamp_curriculum_level(self, level: int) -> int:
        return max(1, min(self.max_curriculum_level, int(level)))

    def get_curriculum_stage(self, level: int) -> CurriculumStage:
        selected_level = self.clamp_curriculum_level(level)
        return self.curriculum_stages[selected_level - 1]

    def generate_curriculum_state(self, level: int | None = None) -> CombatState:
        """Generate a state for a fixed curriculum difficulty level."""

        selected_level = self.curriculum_level if level is None else level
        current_stage = self.get_curriculum_stage(selected_level or 1)
        selected_stage = self._select_curriculum_stage(current_stage)
        state = selected_stage.factory()
        state.training_classes = selected_stage.trainable_classes
        state.curriculum_source_level = selected_stage.level
        state.curriculum_is_rehearsal = selected_stage.level != current_stage.level
        return state

    def generate_curriculum_environment(
        self,
        level: int | None = None,
        use_initiative: bool = True,
        log_to_console: bool = True,
    ) -> CombatEnvironment:
        """Generate an environment for a fixed curriculum difficulty level."""

        combat_state = self.generate_curriculum_state(level)
        environment = CombatEnvironment(
            characters=combat_state.characters,
            grid_map=combat_state.grid_map,
            use_initiative=use_initiative,
            initiative_seed=self.seed,
            log_to_console=log_to_console,
        )
        environment.combat_state.training_classes = combat_state.training_classes
        environment.combat_state.curriculum_source_level = (
            combat_state.curriculum_source_level
        )
        environment.combat_state.curriculum_is_rehearsal = (
            combat_state.curriculum_is_rehearsal
        )
        return environment

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

    def _select_curriculum_stage(self, current_stage: CurriculumStage) -> CurriculumStage:
        rehearsal_probability = (
            self.rehearsal_probability
            if current_stage.rehearsal_probability is None
            else current_stage.rehearsal_probability
        )
        if (
            self.curriculum_kind != "class"
            or rehearsal_probability <= 0.0
            or current_stage.level <= 1
            or self._random.random() >= rehearsal_probability
        ):
            return current_stage
        candidates = self.curriculum_stages[: current_stage.level - 1]
        return self._random.choice(candidates)


def get_curriculum_stage(level: int) -> CurriculumStage:
    """Return a curriculum stage by clamped difficulty level."""

    selected_level = clamp_curriculum_level(level)
    return CURRICULUM_STAGES[selected_level - 1]


def clamp_curriculum_level(level: int) -> int:
    """Clamp curriculum level to the supported curriculum range."""

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


def _curriculum_wizard_fire_elemental_state() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            WizardEvoker(Position(0, 3)),
            FireElementalSimple(Position(7, 2)),
        ],
        grid_map=GridMap(width=8, height=5),
    )


def _curriculum_wizard_intro_state() -> CombatState:
    return CombatState(
        characters=[
            FighterLevel1Basic(Position(0, 1)),
            wizard_character(1, Position(0, 3), name="Wizard Level 1 Intro"),
            GoblinMelee(Position(5, 2)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def _curriculum_wizard_aoe_basics_state() -> CombatState:
    return CombatState(
        characters=[
            FighterLevel1Basic(Position(0, 1)),
            wizard_character(3, Position(0, 3), name="Wizard Evoker Level 3 Basics"),
            GoblinMelee(Position(5, 1)),
            GoblinMelee(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def _class_cleric_resources_state() -> CombatState:
    return CombatState(
        characters=[
            _fighter_for_class_curriculum(Position(0, 1)),
            _cleric_for_class_curriculum(Position(0, 3)),
            GoblinMelee(Position(5, 1)),
            GoblinMelee(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def _fighter_for_class_curriculum(position: Position) -> Character:
    state = level_2_fighter_cleric_vs_goblin_bandit()
    fighter = next(
        character for character in state.characters
        if character.class_name == "Fighter"
    )
    fighter.position = position
    return fighter


def _cleric_for_class_curriculum(position: Position) -> Character:
    state = level_2_fighter_cleric_vs_goblin_bandit()
    cleric = next(
        character for character in state.characters
        if character.class_name == "Cleric"
    )
    cleric.position = position
    return cleric


def _curriculum_mixed_party_intro_state() -> CombatState:
    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            ClericLifeSupport(Position(0, 2)),
            WizardEvoker(Position(0, 3)),
            OrcWarrior(Position(7, 1)),
            GoblinMelee(Position(7, 3)),
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
        name="Level 1 Fighter + Cleric vs 2 Goblins",
        description="Low-level party fight with early support behavior.",
        factory=level_1_fighter_cleric_vs_two_goblins,
    ),
    CurriculumStage(
        level=3,
        name="Level 2 Fighter + Cleric vs Goblin + Bandit",
        description="Early resources before subclasses against mixed humanoids.",
        factory=level_2_fighter_cleric_vs_goblin_bandit,
    ),
    CurriculumStage(
        level=4,
        name="Fighter level 3 Champion vs Orc",
        description="Subclass fighter against a single durable brute.",
        factory=level_3_fighter_champion_vs_orc,
    ),
    CurriculumStage(
        level=5,
        name="Fighter + Cleric vs Orc + Goblins",
        description="Small party fight with healing and mixed enemies.",
        factory=level_3_cleric_life_fighter_vs_orc_goblin,
    ),
    CurriculumStage(
        level=6,
        name="Level 4 Fighter + Cleric vs Orc + Skeleton Archer",
        description="ASI-era party against brute pressure and ranged undead.",
        factory=level_4_fighter_cleric_vs_orc_skeleton,
    ),
    CurriculumStage(
        level=7,
        name="Wizard intro with Fighter support",
        description="First wizard stage without archer pressure or enemy numbers advantage.",
        factory=_curriculum_wizard_intro_state,
    ),
    CurriculumStage(
        level=8,
        name="Wizard AoE basics",
        description="Low-pressure spellcasting and AoE setup against melee goblins.",
        factory=_curriculum_wizard_aoe_basics_state,
    ),
    CurriculumStage(
        level=9,
        name="mixed party intro",
        description="Full level 5 party against two melee enemies before ranged/kiting pressure.",
        factory=_curriculum_mixed_party_intro_state,
    ),
    CurriculumStage(
        level=10,
        name="ranged party on map with cover",
        description="Ranged party tactics with cover and ranged enemies.",
        factory=level_5_ranged_party_cover_map_vs_mixed_enemies,
    ),
    CurriculumStage(
        level=11,
        name="FireElementalSimple and resistances/immunities",
        description="Resistant and immune enemies that require damage-type awareness.",
        factory=_curriculum_resistance_state,
    ),
    CurriculumStage(
        level=12,
        name="maps with obstacles and cover",
        description="Map reasoning with blocked cells, low cover and high cover.",
        factory=_curriculum_obstacle_cover_state,
    ),
    CurriculumStage(
        level=13,
        name="maps with difficult terrain and ranged enemies",
        description="Movement-cost planning against ranged enemies on difficult terrain.",
        factory=level_5_melee_party_difficult_terrain_vs_archers,
    ),
)

MAX_CURRICULUM_LEVEL = len(CURRICULUM_STAGES)


CLASS_CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        level=1,
        name="Fighter fundamentals",
        description="Level 1 Fighter learns movement, weapon attacks and action economy.",
        factory=level_1_fighter_vs_goblin,
        phase="fighter",
        trainable_classes=("Fighter",),
        rehearsal_probability=0.0,
    ),
    CurriculumStage(
        level=2,
        name="Fighter Champion duel",
        description="Level 3 Champion learns subclass-modified weapon combat against an Orc.",
        factory=level_3_fighter_champion_vs_orc,
        phase="fighter",
        trainable_classes=("Fighter",),
        rehearsal_probability=0.05,
    ),
    CurriculumStage(
        level=3,
        name="Fighter with support ally",
        description="Fighter learns party positioning while an untrained Cleric is heuristic-controlled.",
        factory=level_4_fighter_cleric_vs_orc_skeleton,
        phase="fighter",
        trainable_classes=("Fighter",),
        rehearsal_probability=0.05,
    ),
    CurriculumStage(
        level=4,
        name="Cleric fundamentals",
        description="Level 1 Cleric learns healing and spell action economy with a frozen Fighter ally.",
        factory=level_1_fighter_cleric_vs_two_goblins,
        phase="cleric",
        trainable_classes=("Cleric",),
        rehearsal_probability=0.0,
    ),
    CurriculumStage(
        level=5,
        name="Cleric resources",
        description="Level 2 Cleric learns spell slots and class resources.",
        factory=_class_cleric_resources_state,
        phase="cleric",
        trainable_classes=("Cleric",),
        rehearsal_probability=0.0,
    ),
    CurriculumStage(
        level=6,
        name="Cleric Life support",
        description="Life Cleric learns target selection and support in a mixed encounter.",
        factory=level_3_cleric_life_fighter_vs_orc_goblin,
        phase="cleric",
        trainable_classes=("Cleric",),
        rehearsal_probability=0.05,
    ),
    CurriculumStage(
        level=7,
        name="Wizard fundamentals",
        description="Level 1 Wizard learns cantrips and positioning with a frozen Fighter ally.",
        factory=_curriculum_wizard_intro_state,
        phase="wizard",
        trainable_classes=("Wizard",),
        rehearsal_probability=0.0,
    ),
    CurriculumStage(
        level=8,
        name="Wizard area spells",
        description="Wizard learns spell slots, target cells and basic area damage.",
        factory=_curriculum_wizard_aoe_basics_state,
        phase="wizard",
        trainable_classes=("Wizard",),
        rehearsal_probability=0.0,
    ),
    CurriculumStage(
        level=9,
        name="Wizard damage types",
        description="Wizard learns to avoid ineffective damage against an immune target.",
        factory=_curriculum_wizard_fire_elemental_state,
        phase="wizard",
        trainable_classes=("Wizard",),
        rehearsal_probability=0.05,
    ),
    CurriculumStage(
        level=10,
        name="Party integration",
        description="Fighter, Cleric and Wizard are jointly fine-tuned in one party.",
        factory=_curriculum_mixed_party_intro_state,
        phase="integration",
        trainable_classes=("Fighter", "Cleric", "Wizard"),
        rehearsal_probability=0.10,
    ),
    CurriculumStage(
        level=11,
        name="Party cover tactics",
        description="Joint policy learns ranged combat and cover.",
        factory=level_5_ranged_party_cover_map_vs_mixed_enemies,
        phase="integration",
        trainable_classes=("Fighter", "Cleric", "Wizard"),
        rehearsal_probability=0.10,
    ),
    CurriculumStage(
        level=12,
        name="Party resistance tactics",
        description="Joint policy learns damage-type choices against resistant enemies.",
        factory=_curriculum_resistance_state,
        phase="integration",
        trainable_classes=("Fighter", "Wizard"),
        rehearsal_probability=0.10,
    ),
    CurriculumStage(
        level=13,
        name="Party obstacle tactics",
        description="Joint policy learns blocked cells, cover and ranged pressure.",
        factory=_curriculum_obstacle_cover_state,
        phase="integration",
        trainable_classes=("Fighter", "Cleric"),
        rehearsal_probability=0.10,
    ),
    CurriculumStage(
        level=14,
        name="Party difficult terrain tactics",
        description="Joint policy learns movement costs and engagement against archers.",
        factory=level_5_melee_party_difficult_terrain_vs_archers,
        phase="integration",
        trainable_classes=("Fighter", "Cleric"),
        rehearsal_probability=0.10,
    ),
)

MAX_CLASS_CURRICULUM_LEVEL = len(CLASS_CURRICULUM_STAGES)


__all__ = [
    "CLASS_CURRICULUM_STAGES",
    "CURRICULUM_STAGES",
    "MAX_CLASS_CURRICULUM_LEVEL",
    "MAX_CURRICULUM_LEVEL",
    "CurriculumStage",
    "EncounterGenerator",
    "clamp_curriculum_level",
    "get_curriculum_stage",
]
