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


17.
Переработай систему классов и действий.

Цель:
разделить общие боевые действия, доступные всем существам, и классовые способности.

Новая структура:
- src/combat/common_actions.py
- src/combat/class_features.py
- src/combat/abilities.py

Требования:
- WeaponAttack не должен быть классовой способностью
- любое существо может иметь список оружейных атак
- эффективность атаки зависит от stats, proficiency_bonus, weapon, attack_bonus
- классы дают features/resources, но не владеют базовой логикой AttackAction
- Character должен иметь:
  - class_name optional
  - level
  - proficiency_bonus
  - weapons
  - common_actions
  - class_features
  - resources

Обнови FighterChampionGreatsword, FighterArcher, Goblin и Orc под новую структуру.

Сохрани совместимость с текущим CombatEnvironment.


18.
Реализуй общие боевые действия D&D 5e.

Файл: src/combat/common_actions.py

Добавь действия:
- Attack
- CastSpell
- Dash
- Disengage
- Dodge
- Help
- Hide
- Search
- UseObject
- Ready
- Grapple
- Shove
- Stabilize
- ImprovisedAction
- EndTurn

Упрощённая реализация:
- Attack: оружейная атака по цели
- CastSpell: вызывает SpellAbility, если она есть у существа
- Dash: добавляет movement_remaining += speed и тратит action
- Disengage: ставит флаг disengaged_until_end_of_turn и тратит action
- Dodge: ставит флаг dodging_until_start_of_next_turn и тратит action
- Help: даёт союзнику advantage на следующую атаку по выбранной цели или advantage на проверку
- Hide: делает Stealth check и ставит hidden=True при успехе
- Search: делает Perception или Investigation check
- UseObject: тратит action и вызывает эффект предмета, если предмет реализован
- Ready: сохраняет prepared_action и trigger_description
- Grapple: special melee attack, contested Athletics vs Athletics/Acrobatics
- Shove: special melee attack, contested Athletics vs Athletics/Acrobatics, результат: prone или push на 1 клетку
- Stabilize: Medicine check DC 10 на существе с 0 HP
- ImprovisedAction: placeholder, тратит action и логирует описание
- EndTurn: завершает ход

Пока не реализуй сложные предметы и сложные prepared triggers.
Главная цель — корректная структура, action economy и action masks.


19.
Расширь action economy до полноценной модели D&D-like боя.

Файл: src/combat/action_economy.py

У каждого существа на ход должны быть:
- action_available
- bonus_action_available
- reaction_available
- movement_remaining
- free_object_interaction_available

Добавь состояния хода:
- dodging_until_start_of_next_turn
- disengaged_until_end_of_turn
- hidden
- prone
- grappled
- grappling_target_id
- helped_target_id
- help_against_target_id
- prepared_action
- reaction_used_this_round

Правила:
- основные действия тратят action_available
- бонусные действия тратят bonus_action_available
- реакции тратят reaction_available
- движение тратит movement_remaining
- Dash увеличивает movement_remaining на speed
- Dodge действует до начала следующего хода существа
- Disengage действует до конца текущего хода
- Ready тратит action_available и резервирует reaction_available для срабатывания
- Grapple/Shove считаются частью Attack action
- если существо prone, вставание тратит половину speed
- если grappled, movement_remaining = 0, пока захват не снят

Обнови начало и конец хода в CombatEnvironment.


20.
Обнови иерархическое пространство действий под общие боевые действия.

Файл: src/agents/action_space.py

Новая иерархия:
1. action_category:
   - MAIN_ACTION
   - BONUS_ACTION
   - MOVEMENT
   - REACTION
   - END_TURN

2. main_action_type:
   - ATTACK
   - CAST_SPELL
   - DASH
   - DISENGAGE
   - DODGE
   - HELP
   - HIDE
   - SEARCH
   - USE_OBJECT
   - READY
   - GRAPPLE
   - SHOVE
   - STABILIZE
   - IMPROVISED

3. target_index:
   - цель действия, если нужна

4. move_index:
   - клетка движения, если нужна

5. option_index:
   - вариант действия:
     - weapon index
     - spell index
     - shove mode: prone/push
     - ability check type
     - object index

Action masks должны учитывать:
- action_available
- bonus_action_available
- reaction_available
- movement_remaining
- наличие оружия
- наличие заклинаний
- дистанцию до цели
- line of sight, если реализовано
- жив/мёртв
- prone/grappled/hidden/dodging/disengaged
- spell slots, если spell system уже есть

Важно:
если spell system ещё не реализована, CastSpell должен быть замаскирован.


21.
Расширь observation encoder под общие боевые действия.

Файл: src/agents/observation.py

Добавь признаки текущего актёра:
- action_available
- bonus_action_available
- reaction_available
- movement_remaining / speed
- free_object_interaction_available
- prone
- grappled
- hidden
- dodging
- disengaged
- has_prepared_action
- number_of_weapons
- has_spells
- can_cast_spell
- can_attack
- can_dash
- can_disengage
- can_dodge
- can_hide
- can_help
- can_grapple
- can_shove

Для других существ добавь:
- prone
- grappled
- hidden
- dodging
- distance_to_actor
- in_melee_reach
- can_be_attacked
- can_be_helped_against
- can_be_grappled
- can_be_shoved

Обнови input size PPO модели.


22.
Добавь систему проверок характеристик.

Файл: src/combat/checks.py

Нужно:
- ability_modifier(score)
- roll_d20()
- roll_ability_check(character, ability, proficiency=False, advantage_state="normal")
- roll_contested_check(actor, target, actor_check, target_check_options)

Использовать для:
- Grapple: Athletics vs Athletics или Acrobatics
- Shove: Athletics vs Athletics или Acrobatics
- Hide: Stealth vs passive Perception или заданный DC
- Search: Perception/Investigation check
- Stabilize: Medicine DC 10

Добавь логирование бросков.


23.
Добавь pytest-тесты для общих боевых действий.

Проверь:
- Dash увеличивает movement_remaining и тратит action
- Disengage предотвращает opportunity attack
- Dodge даёт disadvantage атакующим до начала следующего хода
- Help даёт advantage союзнику на следующую атаку
- Hide может установить hidden=True
- Search выполняет проверку и возвращает результат
- Grapple ставит grappled при успешной contested check
- Shove может поставить prone
- Shove может оттолкнуть цель на 1 клетку
- Stabilize работает на существе с 0 HP
- Ready сохраняет prepared_action
- OpportunityAttack тратит reaction
- нельзя выполнить два основных действия без Action Surge
- нельзя выполнить реакцию дважды за раунд


24.
Обнови reward function с учётом новых общих действий.

Файл: src/combat/rewards.py

Добавь награды/штрафы:
- небольшой плюс за успешный Grapple, если цель опасна или рядом с уязвимым союзником
- плюс за успешный Shove prone, если союзники могут атаковать цель
- плюс за Dodge, если после этого атаки по персонажу промахнулись
- плюс за Disengage, если персонаж избежал opportunity attack или вышел из опасной позиции
- плюс за Help, если союзник после этого попал атакой
- штраф за бесполезный Dash без тактического улучшения
- штраф за Hide, если оно не даёт преимущества
- штраф за Ready, если prepared action не сработал
- штраф за UseObject/ImprovisedAction без эффекта

Не делай reward слишком большим: эти награды должны быть меньше, чем победа, убийство врага и предотвращение смерти союзника.


25.
Добавь ruleset registry для проекта.

Цель:
проект должен явно понимать, какие правила D&D-like 5e поддерживаются, а какие нет.

Файлы:
- src/rules/ruleset.py
- src/rules/registry.py
- configs/ruleset_srd5e_minimal.yaml

Требования:
- зафиксировать ruleset_name = "srd5e_minimal_2014"
- supported_levels: 1-5
- supported_classes:
  - Fighter
  - Cleric
  - Wizard
- supported_subclasses:
  - Fighter: Champion
  - Cleric: Life Domain
  - Wizard: School of Evocation
- supported_races:
  - Human
  - Dwarf
  - Elf
  - Halfling
- supported_common_actions:
  - Attack
  - CastSpell
  - Dash
  - Disengage
  - Dodge
  - Help
  - Hide
  - Search
  - UseObject
  - Ready
  - Grapple
  - Shove
  - Stabilize
  - EndTurn
- supported_spell_levels: 0-3
- supported_content_policy:
  - unsupported classes are rejected during import
  - unsupported spells are marked as unavailable
  - unsupported features are saved as notes but not used in combat
  - unsupported races fall back to CustomRace only after user confirmation

Добавь функцию:
- is_supported_content(content_type, name) -> bool
- get_unsupported_reason(content_type, name) -> str

Обнови README:
- проект реализует ограниченное подмножество D&D-like 5e
- цель — корректная работа боевого AI, а не полная цифровая копия D&D.


26.
Реализуй систему инициативы боя.

Файлы:
- src/combat/initiative.py
- src/combat/environment.py
- src/combat/checks.py

Требования:
- в начале боя каждое существо бросает инициативу:
  - d20 + DEX modifier
- результат инициативы сохраняется в CombatState
- порядок ходов определяется по убыванию инициативы
- при равенстве инициативы использовать:
  1. больший DEX modifier
  2. случайный tie-breaker с seed
- CombatEnvironment должен хранить:
  - initiative_order
  - current_turn_index
  - round_number
- после последнего существа в initiative_order начинается новый раунд
- мёртвые / incapacitated существа пропускают ход
- новые состояния начала/конца хода должны работать через initiative order

Добавь логирование:
- броски инициативы
- итоговый порядок ходов
- начало каждого раунда
- текущий активный участник

Добавь тесты:
- порядок инициативы корректен
- tie-breaker воспроизводим при seed
- мёртвое существо пропускает ход
- после полного круга увеличивается round_number.
