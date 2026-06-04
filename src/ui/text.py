"""Russian UI labels for internal combat/rules names."""

from __future__ import annotations

import re


LABELS: dict[str, str] = {
    "PLAYERS": "Игроки",
    "ENEMIES": "Враги",
    "players": "Игроки",
    "enemies": "Враги",
    "Movement": "Перемещение",
    "Main Action": "Основное действие",
    "Attack Abilities": "Атакующие способности",
    "Spells": "Заклинания",
    "Inventory": "Инвентарь",
    "Bonus Action": "Бонусное действие",
    "Reaction": "Реакция",
    "End Turn": "Конец хода",
    "Move": "Перемещение",
    "Attack": "Атака",
    "Cast Spell": "Заклинание",
    "Dash": "Рывок",
    "Disengage": "Отход",
    "Dodge": "Уклонение",
    "Help": "Помощь",
    "Hide": "Скрыться",
    "Use Object": "Использовать предмет",
    "Ready": "Подготовить действие",
    "Grapple": "Захват",
    "Shove": "Толчок",
    "Shove Prone": "Толчок: сбить с ног",
    "Shove Push": "Толчок: оттолкнуть",
    "Opportunity Attack": "Провоцированная атака",
    "EndTurn": "Завершить ход",
    "MoveAction": "Перемещение",
    "AttackAction": "Атака",
    "CastSpellAction": "Заклинание",
    "DashAction": "Рывок",
    "DisengageAction": "Отход",
    "DodgeAction": "Уклонение",
    "HelpAction": "Помощь",
    "HideAction": "Скрыться",
    "UseObjectAction": "Предмет",
    "ReadyAction": "Подготовка",
    "GrappleAction": "Захват",
    "ShoveAction": "Толчок",
    "EndTurnAction": "Конец хода",
    "SecondWindAction": "Второе дыхание",
    "ActionSurgeAction": "Всплеск действий",
    "Fire Bolt": "Огненный снаряд",
    "Ray of Frost": "Луч холода",
    "Magic Missile": "Волшебная стрела",
    "Shield": "Щит",
    "Burning Hands": "Пылающие ладони",
    "Scorching Ray": "Палящий луч",
    "Fireball": "Огненный шар",
    "Sacred Flame": "Священное пламя",
    "Cure Wounds": "Лечение ран",
    "Healing Word": "Исцеляющее слово",
    "Guiding Bolt": "Направляющий снаряд",
    "Bless": "Благословение",
    "Spare the Dying": "Уход за умирающим",
    "Potion of Healing": "Зелье лечения",
    "Alchemist Fire": "Алхимический огонь",
    "Bomb": "Бомба",
    "HealerKit": "Набор лекаря",
    "Healer Kit": "Набор лекаря",
    "Longsword": "Длинный меч",
    "Greatsword": "Двуручный меч",
    "Greataxe": "Секира",
    "Shortbow": "Короткий лук",
    "Longbow": "Длинный лук",
    "Light Crossbow": "Лёгкий арбалет",
    "Dagger": "Кинжал",
    "Quarterstaff": "Боевой посох",
    "Mace": "Булава",
    "Claws": "Когти",
    "Bite": "Укус",
    "MELEE_DAMAGE": "Ближний урон",
    "RANGED_DAMAGE": "Дальний урон",
    "TANK": "Защитник",
    "SUPPORT": "Поддержка",
    "CASTER": "Заклинатель",
    "BRUTE_ENEMY": "Грубая сила",
    "SKIRMISHER_ENEMY": "Налётчик",
    "Human": "Человек",
    "Dwarf": "Дварф",
    "Elf": "Эльф",
    "Halfling": "Полурослик",
    "Goblin Melee": "Гоблин-рукопашник",
    "Goblin Archer": "Гоблин-лучник",
    "Goblin": "Гоблин",
    "Orc Warrior": "Орк-воин",
    "Orc": "Орк",
    "Skeleton Archer": "Скелет-лучник",
    "Bandit": "Бандит",
    "Wolf": "Волк",
    "Fire Elemental": "Огненный элементаль",
    "Fire Elemental Simple": "Огненный элементаль",
    "FireElementalSimple": "Огненный элементаль",
    "open_field": "Открытое поле",
    "cover_arena": "Арена с укрытиями",
    "difficult_terrain_pass": "Перевал со сложной местностью",
    "obstacle_corridor": "Коридор с препятствиями",
    "Fighter": "Воин",
    "Cleric": "Жрец",
    "Wizard": "Волшебник",
    "Champion": "Чемпион",
    "Life Domain": "Домен Жизни",
    "School of Evocation": "Школа Воплощения",
    "prone": "лежит",
    "grappled": "в захвате",
    "hidden": "скрыт",
    "dodging": "уклоняется",
    "disengaged": "отошёл",
    "stable": "стабилен",
}

LOG_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("AI action error", "Ошибка действия AI"),
    ("Manual action error", "Ошибка ручного действия"),
    ("Selected action", "Выбранное действие"),
    ("target already defeated", "цель уже побеждена"),
    ("cannot attack missing target", "не может атаковать отсутствующую цель"),
    ("has no available weapon attack", "нет доступной атаки оружием"),
    ("cannot attack", "не может атаковать"),
    ("cannot move to", "не может переместиться в"),
    ("cannot stand up from prone", "не может встать из положения лёжа"),
    ("moves from", "перемещается из"),
    ("to Position", "в Position"),
    ("Movement spent", "Потрачено перемещение"),
    ("movement remaining", "перемещения осталось"),
    ("attacks", "атакует"),
    ("with", "оружием"),
    ("attack", "атака"),
    ("critical hit", "критическое попадание"),
    ("hit", "попадание"),
    ("miss", "промах"),
    ("modifier", "модификатор"),
    ("Total damage", "Итого урона"),
    ("damage", "урон"),
    ("healing", "лечение"),
    ("heals", "лечит"),
    ("casts", "читает заклинание"),
    ("at level", "уровнем"),
    ("after save", "после спасброска"),
    ("Action spent", "Действие потрачено"),
    ("action_available", "основное действие доступно"),
    ("bonus_action_available", "бонусное действие доступно"),
    ("reaction_available", "реакция доступна"),
    ("Round", "Раунд"),
    ("begins", "начинается"),
    ("Active actor", "Активный участник"),
    ("starts turn", "начинает ход"),
    ("skips turn", "пропускает ход"),
    ("is dead", "мёртв"),
    ("is incapacitated", "недееспособен"),
    ("dead", "мёртв"),
    ("incapacitated", "недееспособен"),
    ("Initiative order", "Порядок инициативы"),
    ("rolls initiative", "бросает инициативу"),
    ("d20", "d20"),
    ("DEX", "ЛОВ"),
    ("CON", "ТЕЛ"),
    ("save", "спасбросок"),
    ("failed", "провалил"),
    ("succeeded", "успешно прошёл"),
    ("Opportunity attack", "Провоцированная атака"),
    ("uses", "использует"),
    ("spends", "тратит"),
    ("restores", "восстанавливает"),
    ("Winner", "Победитель"),
    ("none", "нет"),
)


def ru_label(value: object) -> str:
    """Return a Russian display label for a known internal value."""

    text = str(value)
    return LABELS.get(text, text)


def ru_sentence(text: object) -> str:
    """Translate common fragments inside action labels."""

    result = str(text)
    for source, target in sorted(LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(source, target)
    return result


def translate_battle_log(text: object) -> str:
    """Translate common combat-engine log fragments for GUI display."""

    result = ru_sentence(text)
    result = re.sub(r"target (\d+)", r"цель \1", result)
    result = re.sub(r"slot (\d+)", r"ячейка \1", result)
    result = re.sub(r"AC (\d+)", r"КД \1", result)
    result = re.sub(r"vs КД", "против КД", result)
    result = re.sub(r"for ([\d-]+) урон", r"на \1 урона", result)
    result = re.sub(r"for ([\d-]+) damage", r"на \1 урона", result)
    result = re.sub(
        r"Position\(x=([-\d]+), y=([-\d]+)\)",
        r"клетка(\1, \2)",
        result,
    )
    for source, target in LOG_REPLACEMENTS:
        result = result.replace(source, target)
    result = ru_sentence(result)
    result = result.replace("True", "да").replace("False", "нет")
    return result
