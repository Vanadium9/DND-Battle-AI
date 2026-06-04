import pytest
import torch

from agents import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    ActionCategory,
    MainActionType,
    build_action_masks,
    build_fast_training_action_masks,
    decode_action,
    decode_fast_training_action,
    explain_action_mask,
)
from combat import (
    ActionSurgeAction,
    AttackAction,
    CastSpellAction,
    CombatState,
    DashAction,
    EndTurnAction,
    FighterArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    FireElementalSimple,
    Goblin,
    GridMap,
    MoveAction,
    Orc,
    PotionOfHealing,
    Position,
    SecondWindAction,
    ShoveAction,
    Stats,
    Team,
    TerrainType,
    WeaponAttack,
    WizardEvoker,
    build_character,
)


def move_index(position: Position, width: int = 8) -> int:
    return position.y * width + position.x


def make_state() -> CombatState:
    return CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            FighterChampionGreatsword(Position(0, 2)),
            Goblin(Position(1, 0)),
            Orc(Position(7, 7)),
        ],
        grid_map=GridMap(width=8, height=8),
    )


def test_build_action_masks_for_active_actor() -> None:
    state = make_state()

    masks = build_action_masks(state, actor_id=0)

    assert set(masks) == {
        "action_category",
        "main_action_type",
        "target_index",
        "move_index",
        "target_cell_index",
        "direction_index",
        "slot_level",
        "option_index",
    }
    assert masks["action_category"].dtype == torch.bool
    assert masks["action_category"].shape == (ACTION_CATEGORY_COUNT,)
    assert masks["main_action_type"].shape == (MAIN_ACTION_TYPE_COUNT,)
    assert masks["target_index"].shape == (len(state.characters),)
    assert masks["move_index"].shape == (64,)
    assert masks["target_cell_index"].shape == (64,)
    assert masks["direction_index"].shape == (4,)
    assert masks["option_index"].shape[0] >= 8

    assert masks["action_category"][ActionCategory.MAIN_ACTION]
    assert masks["action_category"][ActionCategory.MOVEMENT]
    assert masks["action_category"][ActionCategory.END_TURN]
    assert masks["action_category"][ActionCategory.BONUS_ACTION]
    assert not masks["action_category"][ActionCategory.REACTION]
    assert masks["action_category"][ActionCategory.CLASS_FEATURE]

    assert masks["main_action_type"][MainActionType.ATTACK]
    assert masks["main_action_type"][MainActionType.DASH]
    assert masks["main_action_type"][MainActionType.READY]
    assert not masks["main_action_type"][MainActionType.CAST_SPELL]

    assert not masks["target_index"][0]
    assert masks["target_index"][1]
    assert masks["target_index"][2]
    assert masks["target_index"][3]

    assert masks["option_index"][0]
    assert not masks["move_index"][move_index(Position(0, 0))]
    assert masks["move_index"][move_index(Position(0, 1))]
    assert masks["move_index"][move_index(Position(3, 0))]
    assert not masks["move_index"][move_index(Position(7, 7))]


def test_build_fast_training_action_masks_reduces_common_action_space() -> None:
    state = make_state()

    masks = build_fast_training_action_masks(state, actor_id=0)

    assert masks["action_category"][ActionCategory.MAIN_ACTION]
    assert masks["action_category"][ActionCategory.MOVEMENT]
    assert masks["action_category"][ActionCategory.END_TURN]
    assert masks["main_action_type"][MainActionType.ATTACK]
    assert masks["main_action_type"][MainActionType.DASH]
    assert not masks["main_action_type"][MainActionType.READY]
    assert masks["target_index"][2]


def test_decode_fast_training_action_handles_reduced_space() -> None:
    state = make_state()
    masks = build_fast_training_action_masks(state, actor_id=0)

    attack = decode_fast_training_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.ATTACK,
        target_index=2,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
        masks=masks,
    )
    move = decode_fast_training_action(
        ActionCategory.MOVEMENT,
        MainActionType.ATTACK,
        target_index=0,
        move_index=move_index(Position(0, 1)),
        option_index=0,
        state=state,
        actor_id=0,
        masks=masks,
    )
    dash = decode_fast_training_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.DASH,
        target_index=0,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
        masks=masks,
    )

    assert isinstance(attack, AttackAction)
    assert attack.target_id == 2
    assert isinstance(move, MoveAction)
    assert isinstance(dash, DashAction)


def test_build_action_masks_respect_action_economy() -> None:
    state = make_state()
    actor = state.characters[0]

    actor.action_economy.action_available = False
    masks = build_action_masks(state, actor_id=0)
    assert not masks["main_action_type"][MainActionType.ATTACK]
    assert masks["action_category"][ActionCategory.MOVEMENT]

    actor.action_economy.movement_remaining = 0
    masks = build_action_masks(state, actor_id=0)
    assert not masks["action_category"][ActionCategory.MOVEMENT]
    assert not masks["move_index"].any()
    assert masks["action_category"][ActionCategory.END_TURN]


def test_build_action_masks_mask_non_active_actor() -> None:
    state = make_state()

    masks = build_action_masks(state, actor_id=1)

    assert not masks["action_category"].any()
    assert not masks["main_action_type"].any()
    assert not masks["target_index"].any()
    assert not masks["move_index"].any()
    assert not masks["target_cell_index"].any()
    assert not masks["direction_index"].any()
    assert not masks["option_index"].any()


def test_decode_action_returns_concrete_combat_actions() -> None:
    state = make_state()

    move_action = decode_action(
        ActionCategory.MOVEMENT,
        MainActionType.ATTACK,
        target_index=0,
        move_index=move_index(Position(0, 1)),
        option_index=0,
        state=state,
        actor_id=0,
    )
    attack_action = decode_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.ATTACK,
        target_index=2,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )
    shove_action = decode_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.SHOVE,
        target_index=2,
        move_index=0,
        option_index=1,
        state=state,
        actor_id=0,
    )
    end_turn_action = decode_action(
        ActionCategory.END_TURN,
        MainActionType.ATTACK,
        target_index=0,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )
    bonus_action = decode_action(
        ActionCategory.BONUS_ACTION,
        MainActionType.ATTACK,
        target_index=0,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )
    class_feature_action = decode_action(
        ActionCategory.CLASS_FEATURE,
        MainActionType.ATTACK,
        target_index=0,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )

    assert isinstance(move_action, MoveAction)
    assert move_action.destination == Position(0, 1)
    assert isinstance(attack_action, AttackAction)
    assert attack_action.target_id == 2
    assert attack_action.weapon is not None
    assert attack_action.weapon.name == "Longbow"
    assert isinstance(shove_action, ShoveAction)
    assert shove_action.shove_effect == "push"
    assert isinstance(end_turn_action, EndTurnAction)
    assert isinstance(bonus_action, SecondWindAction)
    assert isinstance(class_feature_action, ActionSurgeAction)


def test_decode_cast_spell_uses_selected_slot_level_for_upcast() -> None:
    state = CombatState(
        characters=[
            WizardEvoker(Position(0, 0)),
            Goblin(Position(3, 0)),
        ],
        grid_map=GridMap(width=6, height=5),
    )
    masks = build_action_masks(state, actor_id=0)

    action = decode_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.CAST_SPELL,
        target_index=1,
        move_index=0,
        option_index=2,
        state=state,
        actor_id=0,
        slot_level=3,
        masks=masks,
    )

    assert isinstance(action, CastSpellAction)
    assert action.spell is not None
    assert action.spell.name == "Magic Missile"
    assert action.cast_level == 3


def test_spell_options_mask_immune_damage_when_better_damage_type_exists() -> None:
    state = CombatState(
        characters=[
            WizardEvoker(Position(0, 2)),
            FireElementalSimple(Position(5, 2)),
        ],
        grid_map=GridMap(width=6, height=5),
    )

    masks = build_action_masks(state, actor_id=0)

    assert masks["main_action_type"][MainActionType.CAST_SPELL]
    action = decode_action(
        ActionCategory.MAIN_ACTION,
        MainActionType.CAST_SPELL,
        target_index=1,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
        masks=masks,
    )

    assert isinstance(action, CastSpellAction)
    assert action.spell is not None
    assert action.spell.name in {"Ray of Frost", "Magic Missile"}


def test_decode_action_rejects_masked_components() -> None:
    state = make_state()

    with pytest.raises(ValueError, match="move_index"):
        decode_action(
            ActionCategory.MOVEMENT,
            MainActionType.ATTACK,
            target_index=0,
            move_index=move_index(Position(7, 7)),
            option_index=0,
            state=state,
            actor_id=0,
        )

    with pytest.raises(ValueError, match="target_index"):
        decode_action(
            ActionCategory.MAIN_ACTION,
            MainActionType.ATTACK,
            target_index=3,
            move_index=0,
            option_index=0,
            state=state,
            actor_id=0,
        )

    state.characters[0].resources["second_wind"].spend()
    with pytest.raises(ValueError, match="BONUS_ACTION|action_category"):
        decode_action(
            ActionCategory.BONUS_ACTION,
            MainActionType.ATTACK,
            target_index=0,
            move_index=0,
            option_index=0,
            state=state,
            actor_id=0,
        )

    state.characters[0].resources["action_surge"].spend()
    with pytest.raises(ValueError, match="CLASS_FEATURE|action_category"):
        decode_action(
            ActionCategory.CLASS_FEATURE,
            MainActionType.ATTACK,
            target_index=0,
            move_index=0,
            option_index=0,
            state=state,
            actor_id=0,
        )


def reason_for(explanations: list[dict[str, object]], action: str) -> str:
    return next(str(item["reason"]) for item in explanations if item["action"] == action)


def test_explain_mask_blocks_extra_attack_for_level_one_fighter() -> None:
    fighter = FighterLevel1Basic(Position(0, 0))
    enemy = Goblin(Position(1, 0))
    state = CombatState(
        characters=[fighter, enemy],
        grid_map=GridMap(width=4, height=4),
    )

    explanations = explain_action_mask(state, actor_id=0)

    assert reason_for(explanations, "ClassFeature:Extra Attack") == "blocked: wrong_level"


def test_explain_mask_blocks_fireball_for_level_one_wizard() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        subclass_name=None,
        level=1,
        stats=Stats(int=16),
    )
    enemy = Goblin(Position(1, 0))
    state = CombatState(
        characters=[wizard, enemy],
        grid_map=GridMap(width=5, height=5),
    )

    explanations = explain_action_mask(state, actor_id=0)

    assert reason_for(explanations, "CastSpell:Fireball") == "blocked: wrong_level"


def test_explain_mask_blocks_zero_quantity_potion() -> None:
    fighter = FighterLevel1Basic(Position(0, 0))
    fighter.hp = 5
    fighter.inventory = [PotionOfHealing(quantity=0)]
    state = CombatState(
        characters=[fighter],
        grid_map=GridMap(width=3, height=3),
    )

    masks = build_action_masks(state, actor_id=0)
    explanations = explain_action_mask(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.USE_OBJECT]
    assert reason_for(explanations, "UseObject:Potion of Healing") == "blocked: no_item_quantity"


def test_explain_mask_blocks_ranged_attack_against_full_cover() -> None:
    bow = WeaponAttack(name="Bow", range=6, damage=1, ability_score="dex")
    archer = FighterArcher(Position(0, 0))
    archer.weapons = [bow]
    target = Goblin(Position(2, 0))
    state = CombatState(
        characters=[archer, target],
        grid_map=GridMap(
            width=4,
            height=3,
            terrain_grid=[
                [TerrainType.NORMAL, TerrainType.HIGH_COVER, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ],
        ),
    )

    masks = build_action_masks(state, actor_id=0)
    explanations = explain_action_mask(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.ATTACK]
    assert reason_for(explanations, "Attack:Bow->Goblin") == "blocked: full_cover"


def test_explain_mask_blocks_blocked_movement_cell() -> None:
    fighter = FighterLevel1Basic(Position(0, 0))
    state = CombatState(
        characters=[fighter],
        grid_map=GridMap(
            width=3,
            height=3,
            terrain_grid=[
                [TerrainType.NORMAL, TerrainType.BLOCKED, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ],
        ),
    )

    masks = build_action_masks(state, actor_id=0)
    explanations = explain_action_mask(state, actor_id=0)

    assert not masks["move_index"][move_index(Position(1, 0), width=3)]
    assert reason_for(explanations, "Move:1,0") == "blocked: blocked_cell"
