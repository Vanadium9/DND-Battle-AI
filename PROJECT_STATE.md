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
