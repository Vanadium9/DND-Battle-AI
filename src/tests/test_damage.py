from agents import (
    ACTOR_FEATURE_SIZE,
    BASE_CHARACTER_FEATURE_SIZE,
    DAMAGE_TYPE_FEATURE_SIZE,
    MAX_NEARBY_CHARACTERS,
    OTHER_CHARACTER_FEATURE_SIZE,
    encode_observation,
)
from combat import (
    AttackAction,
    CastSpellAction,
    Character,
    CombatState,
    DAMAGE_TYPES,
    DamageType,
    FireElementalSimple,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
    WizardEvoker,
    build_character,
)


def damage_index(damage_type: DamageType) -> int:
    return DAMAGE_TYPES.index(damage_type)


def spell_by_name(character: Character, spell_name: str):
    return next(
        spell
        for spell in [*character.cantrips, *character.prepared_spells]
        if spell.name == spell_name
    )


def test_fire_immunity_zeros_fireball_damage(monkeypatch) -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=5,
        prepared_spells=("Fireball",),
        stats=Stats(int=18),
    )
    elemental = FireElementalSimple(Position(3, 0))
    state = CombatState(
        characters=[wizard, elemental],
        grid_map=GridMap(width=6, height=4),
    )
    fireball = spell_by_name(wizard, "Fireball")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = CastSpellAction(
        actor_id=0,
        spell=fireball,
        target_cell=elemental.position,
    ).execute(state)

    assert result.success
    assert elemental.hp == elemental.max_hp
    assert "Fire Elemental Simple: 0 damage" in result.description


def test_resistance_halves_damage(monkeypatch) -> None:
    attacker = Character(
        name="Warrior",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(str=16),
        team=Team.PLAYERS,
        weapons=[
            WeaponAttack(
                name="Longsword",
                range=1,
                damage=10,
                attack_bonus=20,
                ability_score="str",
                damage_ability_score=None,
                damage_type=DamageType.SLASHING,
            )
        ],
    )
    elemental = FireElementalSimple(Position(1, 0))
    state = CombatState(
        characters=[attacker, elemental],
        grid_map=GridMap(width=3, height=3),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = AttackAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert elemental.hp == elemental.max_hp - 5
    assert "for 5 damage" in result.description


def test_vulnerability_doubles_damage(monkeypatch) -> None:
    attacker = Character(
        name="Mage Blade",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        weapons=[
            WeaponAttack(
                name="Frost Dagger",
                range=1,
                damage=7,
                attack_bonus=20,
                damage_ability_score=None,
                damage_type=DamageType.COLD,
            )
        ],
    )
    target = Character(
        name="Cold Weak Target",
        hp=30,
        max_hp=30,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
        vulnerabilities={DamageType.COLD},
    )
    state = CombatState(
        characters=[attacker, target],
        grid_map=GridMap(width=3, height=3),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = AttackAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert target.hp == target.max_hp - 14
    assert "for 14 damage" in result.description


def test_observation_encodes_damage_types_and_target_damage_profile() -> None:
    wizard = WizardEvoker(Position(0, 0))
    elemental = FireElementalSimple(Position(3, 0))
    state = CombatState(
        characters=[wizard, elemental],
        grid_map=GridMap(width=6, height=4),
    )

    observation = encode_observation(state, actor_id=0)
    actor_damage_start = BASE_CHARACTER_FEATURE_SIZE + 18
    enemy_start = ACTOR_FEATURE_SIZE + OTHER_CHARACTER_FEATURE_SIZE * MAX_NEARBY_CHARACTERS
    enemy_profile_start = enemy_start + BASE_CHARACTER_FEATURE_SIZE + 9
    enemy_resistance_start = enemy_profile_start
    enemy_immunity_start = enemy_profile_start + DAMAGE_TYPE_FEATURE_SIZE

    assert observation[actor_damage_start + damage_index(DamageType.FIRE)] == 1
    assert observation[enemy_resistance_start + damage_index(DamageType.SLASHING)] == 1
    assert observation[enemy_resistance_start + damage_index(DamageType.PIERCING)] == 1
    assert observation[enemy_resistance_start + damage_index(DamageType.BLUDGEONING)] == 1
    assert observation[enemy_immunity_start + damage_index(DamageType.FIRE)] == 1
