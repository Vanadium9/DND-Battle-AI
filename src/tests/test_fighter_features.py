from agents import ActionCategory, build_action_masks, decode_action
from agents.action_space import MainActionType
from combat import (
    ActionSurgeAction,
    AttackAction,
    Character,
    CharacterBuildRequest,
    CombatState,
    FighterChampionArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    GridMap,
    Position,
    SecondWindAction,
    Stats,
    Team,
    WeaponAttack,
    archery_attack_bonus,
    build_character,
    character_has_class_feature,
    critical_hit_threshold,
    weapon_attack_count_for_attack_action,
)


def make_target(hp: int = 30, ac: int = 12) -> Character:
    return Character(
        name="Target",
        hp=hp,
        max_hp=hp,
        ac=ac,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
    )


def test_fighter_progression_levels_one_to_five_and_builder_champion_choice() -> None:
    for level in range(1, 6):
        request = CharacterBuildRequest(
            name=f"Fighter {level}",
            class_name="Fighter",
            subclass_name="Champion" if level >= 3 else None,
            level=level,
            fighting_style="Defense",
            wearing_armor=True,
        )
        fighter = build_character(request)

        assert fighter.level == level
        assert character_has_class_feature(fighter, "Fighting Style")
        assert character_has_class_feature(fighter, "Second Wind")
        assert character_has_class_feature(fighter, "Action Surge") is (level >= 2)
        assert character_has_class_feature(fighter, "Improved Critical") is (level >= 3)
        assert character_has_class_feature(fighter, "Extra Attack") is (level >= 5)


def test_second_wind_is_bonus_action_once_per_combat(monkeypatch) -> None:
    fighter = FighterLevel1Basic()
    fighter.hp = 5
    state = CombatState(characters=[fighter], grid_map=GridMap(width=3, height=3))
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 6)

    masks = build_action_masks(state, actor_id=0)
    decoded = decode_action(
        ActionCategory.BONUS_ACTION,
        MainActionType.ATTACK,
        target_index=0,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )
    result = decoded.execute(state)
    masks_after = build_action_masks(state, actor_id=0)

    assert isinstance(decoded, SecondWindAction)
    assert result.success
    assert fighter.hp == 12
    assert fighter.action_economy.bonus_action_available is False
    assert fighter.resources["second_wind"].uses_remaining == 0
    assert masks["action_category"][ActionCategory.BONUS_ACTION]
    assert not masks_after["action_category"][ActionCategory.BONUS_ACTION]
    assert SecondWindAction(actor_id=0).execute(state).success is False


def test_action_surge_restores_action_once_per_combat() -> None:
    fighter = FighterChampionGreatsword()
    state = CombatState(characters=[fighter], grid_map=GridMap(width=3, height=3))
    fighter.action_economy.action_available = False

    masks = build_action_masks(state, actor_id=0)
    result = ActionSurgeAction(actor_id=0).execute(state)
    masks_after = build_action_masks(state, actor_id=0)
    second_result = ActionSurgeAction(actor_id=0).execute(state)

    assert masks["action_category"][ActionCategory.CLASS_FEATURE]
    assert result.success
    assert fighter.action_economy.action_available is True
    assert fighter.resources["action_surge"].uses_remaining == 0
    assert not masks_after["action_category"][ActionCategory.CLASS_FEATURE]
    assert second_result.success is False


def test_improved_critical_uses_nineteen_to_crit(monkeypatch) -> None:
    weapon = WeaponAttack(
        name="Sword",
        range=1,
        damage="1d6",
        attack_bonus=20,
        damage_ability_score=None,
    )
    fighter = Character(
        name="Champion",
        hp=30,
        max_hp=30,
        ac=16,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        class_name="Fighter",
        subclass_name="Champion",
        level=3,
        fighting_style="Defense",
        wearing_armor=True,
        weapons=[weapon],
    )
    target = make_target(hp=30)
    state = CombatState(characters=[fighter, target], grid_map=GridMap(width=3, height=3))
    rolls = iter([19, 4, 4])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = AttackAction(actor_id=0, target_id=1, weapon=weapon).execute(state)

    assert result.success
    assert critical_hit_threshold(fighter) == 19
    assert "critical hit" in result.description
    assert target.hp == 22


def test_extra_attack_makes_two_weapon_attacks_inside_one_action(monkeypatch) -> None:
    weapon = WeaponAttack(
        name="Sword",
        range=1,
        damage=1,
        attack_bonus=20,
        damage_ability_score=None,
    )
    fighter = Character(
        name="Level 5 Fighter",
        hp=40,
        max_hp=40,
        ac=16,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        class_name="Fighter",
        subclass_name="Champion",
        level=5,
        weapons=[weapon],
    )
    target = make_target(hp=10)
    state = CombatState(characters=[fighter, target], grid_map=GridMap(width=3, height=3))
    rolls = iter([10, 10])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = AttackAction(actor_id=0, target_id=1, weapon=weapon).execute(state)

    assert result.success
    assert weapon_attack_count_for_attack_action(fighter) == 2
    assert "attack 1" in result.description
    assert "attack 2" in result.description
    assert target.hp == 8
    assert fighter.action_economy.action_available is False


def test_defense_and_archery_fighting_styles_apply_to_ac_and_ranged_attack(monkeypatch) -> None:
    basic = FighterLevel1Basic()
    archer = FighterChampionArcher(Position(0, 0))
    target = make_target(hp=20, ac=12)
    state = CombatState(characters=[archer, target], grid_map=GridMap(width=4, height=4))
    rolls = iter([3, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = AttackAction(actor_id=0, target_id=1, weapon=archer.weapons[0]).execute(state)

    assert basic.ac == 17
    assert archery_attack_bonus(archer, archer.weapons[0]) == 2
    assert result.success
    assert "modifier=9" in result.description
    assert target.hp < target.max_hp


def test_great_weapon_fighting_rerolls_low_damage_die(monkeypatch) -> None:
    weapon = WeaponAttack(
        name="Greatsword",
        range=1,
        damage="1d6",
        attack_bonus=20,
        damage_ability_score=None,
        two_handed=True,
    )
    fighter = Character(
        name="Great Weapon Fighter",
        hp=20,
        max_hp=20,
        ac=16,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=1,
        fighting_style="Great Weapon Fighting",
        weapons=[weapon],
    )
    target = make_target(hp=10)
    state = CombatState(characters=[fighter, target], grid_map=GridMap(width=3, height=3))
    rolls = iter([10, 2, 5])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))

    result = AttackAction(actor_id=0, target_id=1, weapon=weapon).execute(state)

    assert result.success
    assert target.hp == 5
