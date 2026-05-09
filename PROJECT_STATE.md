1\.
Создай Python-проект для RL-симулятора тактических боёв в стиле D\&D.



Требования:

\- Python 3.11+

\- структура проекта:

&#x20; - src/

&#x20;   - combat/

&#x20;   - agents/

&#x20;   - training/

&#x20;   - configs/

&#x20;   - tests/

&#x20; - scripts/

\- используй PyTorch

\- добавь requirements.txt

\- добавь README.md

\- добавь базовый entrypoint scripts/run\_demo.py



Пока не реализуй PPO. Только структура проекта, базовые импорты и пустые классы.


2.
Реализуй базовые сущности боя в src/combat/.

Нужны:
- Character
- Enemy
- Ability
- WeaponAttack
- SpellAbility
- Condition
- CombatState
- Position
- Team enum: PLAYERS, ENEMIES

Для персонажа добавь:
- name
- hp, max_hp
- ac
- position
- speed
- stats: str, dex, con, int, wis, cha
- team
- alive/dead check
- available abilities

Сделай простые dataclass-модели без сложных D&D-правил.


3.
Добавь GridMap в src/combat/map.py.

Требования:
- прямоугольная карта width x height
- проверка, находится ли клетка в пределах карты
- Manhattan distance
- поиск соседних клеток
- проверка занятости клетки существом
- получение допустимых клеток движения в пределах speed

Пока без препятствий и line of sight.


4.
Реализуй систему действий в src/combat/actions.py.

Нужны действия:
- MoveAction
- AttackAction
- EndTurnAction

Каждое действие должно иметь:
- actor_id
- execute(combat_state)
- is_valid(combat_state)

AttackAction:
- проверяет дистанцию оружия
- бросок атаки: d20 + attack_bonus >= target.ac
- при попадании наносит фиксированный или случайный урон

MoveAction:
- перемещает персонажа в допустимую клетку

Добавь action result с текстовым описанием для логирования.


5.
Добавь механику action economy в src/combat/action_economy.py.

Требования:
- у каждого существа на ход есть:
  - action_available: bool
  - bonus_action_available: bool
  - reaction_available: bool
  - movement_remaining: int
- в начале хода ресурсы хода сбрасываются:
  - action_available = True
  - bonus_action_available = True
  - reaction_available = True
  - movement_remaining = speed
- MoveAction тратит movement_remaining
- AttackAction тратит action_available
- EndTurnAction завершает ход
- bonus_action_available пока должен существовать, но не использоваться
- reaction_available пока должен существовать, но не использоваться

Обнови is_valid() у действий с учётом action economy.
Обнови консольное логирование, чтобы было видно, что потратилось.


6.
Реализуй CombatEnvironment в src/combat/environment.py.

Требования:
- reset()
- step(action)
- get_observation(actor_id)
- get_available_actions(actor_id)
- is_done()
- get_winner()
- turn order по инициативе или фиксированный порядок
- после смерти существо пропускает ход
- бой заканчивается, когда одна команда мертва

Интегрируй action economy:
- при начале хода вызывай reset_turn_resources()
- если action_available = False, нельзя выполнить обычное действие
- персонаж может двигаться и атаковать в одном ходу
- после EndTurnAction ход переходит следующему существу
- если существо больше не может или не хочет действовать, оно завершает ход

Добавь консольное логирование каждого действия.


7.
Создай src/combat/presets.py с тестовыми персонажами.

Добавь:
- FighterChampionGreatsword
- FighterArcher
- Goblin
- Orc

Воин с двуручным мечом:
- высокий STR
- melee attack range 1
- большой урон

Воин-лучник:
- высокий DEX
- ranged attack range 6
- меньший урон

Гоблин:
- слабый melee/ranged враг

Орк:
- больше HP и урона

Добавь функцию create_test_encounter().


8.
Реализуй EncounterGenerator в src/combat/encounter_generator.py.

Требования:
- генерировать случайные простые бои:
  - 1-2 игрока
  - 1-4 врага
  - карта 8x8
  - случайные стартовые позиции без пересечений
- поддерживать seed
- возвращать CombatEnvironment или CombatState

Пока используй только FighterChampionGreatsword, FighterArcher, Goblin и Orc.
