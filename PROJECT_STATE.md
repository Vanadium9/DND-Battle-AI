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


9.
Реализуй преобразование CombatState в тензор для нейросети.

Файл: src/agents/observation.py

Для плана минимум используй фиксированный вектор.

В observation включи:
- данные текущего актёра
- HP/max HP
- AC
- позицию x,y
- team
- melee/ranged флаги
- до N ближайших союзников
- до N ближайших врагов
- расстояния до них
- жив/мёртв
- action_available
- bonus_action_available
- reaction_available
- movement_remaining / speed

Сделай padding, если существ меньше N.

Добавь функцию encode_observation(state, actor_id) -> torch.Tensor.


10.
Реализуй иерархическое пространство действий для PPO.

Файл: src/agents/action_space.py

План минимум:
- action_type:
  - MOVE
  - MAIN_ACTION_ATTACK
  - END_TURN

Зарезервировать на будущее:
- BONUS_ACTION
- REACTION

Нужно:
- target_index:
  - индекс цели из списка существ
- move_index:
  - индекс клетки из списка допустимых клеток
- build_action_masks(state, actor_id)
- decode_action(action_type, target_index, move_index, state, actor_id)

Action masks должны учитывать:
- action_available
- bonus_action_available
- reaction_available
- movement_remaining

invalid actions должны маскироваться.


11.
Реализуй PPO Actor-Critic модель в src/agents/ppo_model.py.

Требования:
- PyTorch
- общий MLP encoder
- policy heads:
  - action_type_head
  - target_head
  - move_head
- value_head
- поддержка action masking перед softmax
- метод act(observation, masks)
- метод evaluate_actions(observations, actions, masks)

Пока без GNN.


12.
Реализуй PPO trainer в src/training/ppo_trainer.py.

Требования:
- rollout collection
- хранить:
  - observations
  - actions
  - log_probs
  - rewards
  - dones
  - values
  - masks
- считать returns
- считать advantages
- PPO clipped objective
- value loss
- entropy bonus
- gradient clipping
- сохранение модели в checkpoints/

Добавь конфиг гиперпараметров.


13.
Реализуй reward function для D&D-like боя.

Файл: src/combat/rewards.py

Награды:
- + за нанесённый урон врагу
- + за убийство врага
- + за победу
- - за смерть союзника
- - за получение урона
- - за поражение
- небольшой штраф за бесполезный ход
- небольшой штраф за слишком длинный бой

Интегрируй reward в CombatEnvironment.step().


14.
Создай scripts/train_ppo.py.

Требования:
- создаёт EncounterGenerator
- создаёт PPO model
- запускает обучение
- раз в N эпизодов выводит:
  - win rate
  - average reward
  - average episode length
  - action distribution
- сохраняет checkpoints
- поддерживает аргументы:
  - --episodes
  - --seed
  - --checkpoint


15.
Создай scripts/run_demo.py.

Требования:
- загружает trained PPO checkpoint
- создаёт тестовый бой
- запускает бой пошагово
- каждый ход печатает:
  - номер раунда
  - имя актёра
  - HP всех существ
  - выбранное действие
  - результат действия
- в конце печатает победителя

Пока только консольный вывод.


16.
Добавь pytest-тесты.

Проверь:
- движение по карте
- невозможность выйти за границы карты
- атака по цели в радиусе
- запрет атаки по мёртвой цели
- окончание боя
- корректность action masks
- encode_observation возвращает фиксированный размер
- PPO model возвращает валидные действия
- AttackAction тратит action_available
- после атаки нельзя атаковать второй раз без специальной способности
- движение тратит movement_remaining
- после начала нового хода action_available снова True
- bonus_action_available существует и сбрасывается, но пока не используется

Не делай тесты слишком большими.
