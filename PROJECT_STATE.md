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


27.
Добавь систему прогрессии персонажа с 1 по 5 уровень.

Файлы:
- src/rules/progression.py
- src/combat/character.py
- src/character/schema.py

Требования:
- Character должен иметь:
  - level
  - experience
  - proficiency_bonus
  - class_name
  - subclass_name optional
- поддерживать уровни 1-5
- таблица XP thresholds:
  - level 1: 0
  - level 2: 300
  - level 3: 900
  - level 4: 2700
  - level 5: 6500
- proficiency bonus:
  - levels 1-4: +2
  - level 5: +3
- добавить функцию:
  - get_level_for_xp(xp) -> int
  - get_proficiency_bonus(level) -> int
  - can_level_up(character) -> bool
  - apply_level_up(character)
- при повышении уровня пересчитывать:
  - proficiency_bonus
  - class features
  - spell slots, если персонаж spellcaster
  - available actions/features
- в бою уровень не должен автоматически повышаться посреди encounter, только после завершения боя

Добавь тесты:
- XP thresholds работают корректно
- proficiency bonus меняется на 5 уровне
- уровень не превышает 5
- level up обновляет features.


28.
Добавь систему расовых особенностей персонажей.

Файлы:
- src/rules/races.py
- src/combat/race_traits.py
- src/character/schema.py

Требования:
- Character должен иметь:
  - race_name
  - race_traits
- реализовать data-driven RaceDefinition:
  - name
  - ability_score_bonuses
  - speed
  - size
  - darkvision_range optional
  - skill_proficiencies
  - weapon_proficiencies
  - saving_throw_advantages
  - damage_resistances
  - special_traits

Для MVP реализовать:
- Human
- Dwarf
- Elf
- Halfling

Боевые эффекты:
- racial speed влияет на movement_remaining
- racial proficiencies влияют на владение оружием
- racial resistances добавляются в damage system
- darkvision пока хранить как feature, но не использовать без line of sight / lighting
- Halfling Lucky можно сделать stub или упрощённо:
  - reroll natural 1 once per roll
  - включить через feature flag

Не реализуй пока все подрасы.
Добавь CustomRace fallback, но только для импортированных персонажей и с warning.

Добавь тесты:
- race bonuses применяются
- speed берётся из race или character override
- resistance от race работает в damage system.


29.
Добавь систему ASI и черт персонажа.

Файлы:
- src/rules/feats.py
- src/combat/features.py
- src/character/schema.py

Требования:
- Character должен иметь:
  - feats: list
  - ability_score_improvements: list
- на 4 уровне класс может получить ASI или feat
- реализовать универсальную структуру FeatDefinition:
  - name
  - prerequisites
  - stat_bonuses
  - passive_effects
  - active_effects
  - combat_hooks
  - implemented: bool

Реализовать ASI:
- +2 к одной характеристике
- или +1 к двум характеристикам

Для MVP реализовать:
- Ability Score Improvement
- Grappler, если он уже есть в ruleset config
- остальные feats пока не реализовывать

Добавить hooks:
- on_attack_roll
- on_damage_roll
- on_saving_throw
- on_ability_check
- on_turn_start
- on_turn_end

Важно:
- character builder должен показывать только поддержанные feats
- unsupported feats не должны попадать в персонажа через UI/CLI
- если feat не implemented, он не должен влиять на бой и не должен попадать в action masks

Добавь тесты:
- ASI меняет stats
- ASI не позволяет поднять характеристику выше допустимого лимита, если такой лимит задан
- feat prerequisites проверяются
- неimplemented feat не влияет на combat hooks


30.
Переработай классовую систему в data-driven формат.

Файлы:
- src/rules/classes.py
- src/rules/subclasses.py
- src/combat/class_features.py
- src/combat/character_builder.py

Цель:
классы не должны быть набором случайно добавленных действий.
Они должны определяться через progression table по уровням 1-5.

Реализуй:
- ClassDefinition:
  - name
  - hit_die
  - primary_abilities
  - saving_throw_proficiencies
  - armor_proficiencies
  - weapon_proficiencies
  - skill_choices
  - level_features
  - spellcasting_progression optional
  - subclass_level
- SubclassDefinition:
  - name
  - parent_class
  - level_features
- FeatureDefinition:
  - name
  - level
  - action_cost optional
  - resource_cost optional
  - passive_hooks
  - active_action optional
  - description
  - implemented: bool

Требования:
- при создании персонажа class features должны выдаваться по уровню
- subclass выбирается на нужном уровне класса
- unsupported class features должны сохраняться как not_implemented внутри правил, но не должны попадать в action mask
- character builder должен разрешать выбирать только поддержанные классы и подклассы из ruleset registry
- action masks должны учитывать только implemented features
- observation должен получать признаки доступных implemented features

Добавь тесты:
- Fighter level 1 получает только features 1 уровня
- Fighter level 3 получает subclass features
- Wizard level 5 получает доступ к class features до 5 уровня
- unsupported feature не попадает в action mask


31.
Расширь систему карты.

Файлы:
- src/combat/map.py
- src/combat/terrain.py
- src/combat/line_of_sight.py
- src/combat/cover.py
- src/agents/action_space.py
- src/agents/observation.py

Цель:
реализовать базовые механики карты, которых сейчас нет:
- obstacles
- cover
- terrain cost
- полноценная line-of-sight модель

Реализуй типы клеток:
- NORMAL
- DIFFICULT_TERRAIN
- BLOCKED
- LOW_COVER
- HIGH_COVER

GridMap должен поддерживать:
- width
- height
- terrain grid
- проверку walkable / blocked
- movement cost для клетки
- neighbors с учётом terrain cost
- поиск достижимых клеток с учётом movement_remaining и difficult terrain
- проверку occupied cell

Movement:
- NORMAL стоит 1 movement unit
- DIFFICULT_TERRAIN стоит 2 movement units
- BLOCKED недоступен
- occupied клетка недоступна для завершения движения

Line of sight:
- реализовать line_of_sight(start, end)
- использовать алгоритм Bresenham или аналогичный grid raycast
- BLOCKED и HIGH_COVER блокируют line of sight
- LOW_COVER не блокирует line of sight

Cover:
- реализовать get_cover_between(attacker_pos, target_pos)
- варианты:
  - NO_COVER
  - HALF_COVER
  - THREE_QUARTERS_COVER
  - FULL_COVER
- FULL_COVER запрещает ranged attack и большинство targeted spells
- HALF_COVER даёт +2 AC и +2 DEX saves
- THREE_QUARTERS_COVER даёт +5 AC и +5 DEX saves
- cover должен учитываться в AttackAction и saving throws от AoE/заклинаний

Обнови действия:
- MoveAction учитывает terrain cost
- AttackAction проверяет line of sight для ranged attacks
- CastSpell проверяет line of sight для targeted spells
- Hide может требовать cover или blocked line of sight
- Search может обнаруживать hidden target

Обнови action masks:
- нельзя двигаться в BLOCKED
- нельзя выбрать недостижимую клетку
- нельзя ranged attack по цели за FULL_COVER
- нельзя targeted spell без line of sight
- Hide доступен только при наличии укрытия или отсутствия видимости врагов

Обнови observation:
- terrain type вокруг актёра
- наличие cover относительно ближайших врагов
- line of sight до целей
- movement cost до доступных клеток

Добавь тесты:
- BLOCKED клетка недоступна для движения
- DIFFICULT_TERRAIN тратит больше movement
- line_of_sight блокируется препятствием
- LOW_COVER не блокирует line of sight
- FULL_COVER запрещает ranged attack
- HALF_COVER добавляет AC bonus
- Hide маскируется без укрытия

## 32. Запрос: Реализуй Fighter уровней 1-5 по progression table

Файлы:
- src/rules/classes/fighter.py
- src/rules/subclasses/champion.py
- src/combat/class_features.py
- src/combat/presets.py

Fighter progression:
- Level 1:
  - Fighting Style
  - Second Wind
- Level 2:
  - Action Surge
- Level 3:
  - Martial Archetype: Champion
  - Champion feature: Improved Critical
- Level 4:
  - Ability Score Improvement
- Level 5:
  - Extra Attack

Реализовать Fighting Style:
- Archery
- Defense
- Great Weapon Fighting

Реализовать:
- Second Wind:
  - bonus action
  - 1 раз за бой
  - self-heal
- Action Surge:
  - 1 раз за бой
  - восстанавливает action_available
- Improved Critical:
  - critical hit on 19-20
- Extra Attack:
  - две weapon attacks внутри одного Attack action
- Defense:
  - +1 AC, если персонаж носит armor
- Archery:
  - бонус к ranged weapon attack
- Great Weapon Fighting:
  - reroll низкого damage die для двуручного оружия

Важно:
- Attack остаётся common action
- Fighter features только модифицируют common action или добавляют resource activation
- action masks должны учитывать уровень, subclass и доступность ресурсов
- character builder должен уметь создать Fighter 1-5 уровня и выбрать Champion на 3 уровне

Добавь presets:
- FighterChampionGreatsword level 5
- FighterChampionArcher level 5
- FighterLevel1Basic level 1

Добавь тесты всех features.

## 33. Запрос: Реализуй универсальную систему spellcasting progression для уровней 1-5

Файлы:
- src/rules/spellcasting_progression.py
- src/combat/spellcasting.py
- src/combat/resources.py

Требования:
- поддержать cantrips и spell slots уровней 1-3
- spell slots зависят от class level и spellcasting type
- для MVP реализовать:
  - full caster progression для Cleric и Wizard уровней 1-5

Character должен иметь:
- known_spells
- prepared_spells
- cantrips
- spell_slots
- spellcasting_ability
- spell_save_dc
- spell_attack_bonus

Формулы:
- spell_save_dc = 8 + proficiency_bonus + spellcasting ability modifier
- spell_attack_bonus = proficiency_bonus + spellcasting ability modifier

Требования к подготовке заклинаний:
- для MVP можно хранить fixed prepared_spells
- архитектура должна позволять позже выбирать prepared spells
- action masks должны показывать только prepared spells
- cantrips не тратят spell slots
- level 1+ spells тратят spell slots
- upcast поддержать инфраструктурно
- для MVP upcast реализовать только для healing/damage spells

Character builder:
- должен показывать только поддержанные заклинания
- должен запрещать выбирать spell level выше доступного персонажу
- должен валидировать prepared_spells

Добавь тесты:
- Wizard level 5 имеет spell slots до 3 уровня
- Cleric level 3 имеет spell slots до 2 уровня
- cantrip не тратит slot
- spell level 3 нельзя кастовать без slot 3 уровня

## 34. Запрос: Реализуй Cleric уровней 1-5 с Life Domain

Файлы:
- src/rules/classes/cleric.py
- src/rules/subclasses/life_domain.py
- src/combat/spells/cleric_spells.py
- src/combat/presets.py

Cleric progression:
- Level 1:
  - Spellcasting
  - Divine Domain: Life Domain
- Level 2:
  - Channel Divinity
- Level 3:
  - spell slots 2 уровня
- Level 4:
  - Ability Score Improvement
- Level 5:
  - spell slots 3 уровня

Для MVP реализовать:
- Spellcasting через prepared_spells
- Life Domain как subclass
- Channel Divinity resource
- Channel Divinity: Preserve Life как healing action
- Turn Undead сделать stub/not_implemented, если undead system ещё нет

Заклинания:
- Sacred Flame — cantrip
- Cure Wounds — level 1
- Healing Word — level 1, bonus action
- Guiding Bolt — level 1
- Bless — concentration buff, можно stub или простую реализацию
- Spiritual Weapon пока не реализовывать
- Revivify пока не реализовывать

Требования:
- Mace Attack использует common Attack
- Cleric spellcasting ability = WIS
- Cure Wounds требует action
- Healing Word требует bonus_action
- Guiding Bolt требует action
- Sacred Flame требует action и WIS/DEX save по выбранной реализации
- action masks должны учитывать:
  - prepared spell
  - spell slot
  - action economy
  - валидную цель
  - range
  - line of sight
  - cover, если применимо
- healing spells должны быть замаскированы, если нет раненых союзников и сам Cleric не ранен

Character builder:
- должен уметь создать Cleric 1-5 уровня
- должен позволять выбрать Life Domain
- должен показывать только поддержанные cleric spells

Добавь preset:
- ClericLifeSupport level 5

Добавь тесты:
- Healing Word тратит bonus action
- Cure Wounds тратит action
- spell slots тратятся
- Channel Divinity тратит resource
- Cleric level 1 не имеет Channel Divinity

## 35. Запрос: Реализуй Wizard уровней 1-5 с School of Evocation

Файлы:
- src/rules/classes/wizard.py
- src/rules/subclasses/evocation.py
- src/combat/spells/wizard_spells.py
- src/combat/presets.py

Wizard progression:
- Level 1:
  - Spellcasting
  - Arcane Recovery
- Level 2:
  - Arcane Tradition: School of Evocation
  - Evocation feature: Sculpt Spells
- Level 3:
  - spell slots 2 уровня
- Level 4:
  - Ability Score Improvement
- Level 5:
  - spell slots 3 уровня

Для MVP реализовать:
- spellbook как список known_spells
- prepared_spells как фиксированный список
- Arcane Recovery:
  - восстанавливает часть spell slots после боя или short rest
  - для MVP можно сделать reset между боями
- Sculpt Spells:
  - союзники могут быть исключены из AoE Evocation spells
  - работает для Fireball и Burning Hands

Заклинания:
- Fire Bolt — cantrip
- Ray of Frost — cantrip optional
- Magic Missile — level 1
- Shield — reaction, AC bonus до начала следующего хода
- Burning Hands — level 1, cone AoE
- Scorching Ray — level 2 optional
- Fireball — level 3, radius AoE

Требования:
- Quarterstaff Attack использует common Attack
- Wizard spellcasting ability = INT
- Fireball доступен только Wizard level 5+
- action masks должны учитывать:
  - prepared spell
  - spell slot
  - action economy
  - range
  - line of sight
  - cover
  - AoE target
  - наличие врагов в зоне
  - friendly fire и Sculpt Spells

Character builder:
- должен уметь создать Wizard 1-5 уровня
- должен позволять выбрать School of Evocation
- должен показывать только поддержанные wizard spells

Добавь preset:
- WizardEvoker level 5

Добавь тесты:
- Wizard level 1 не может кастовать Fireball
- Wizard level 5 может кастовать Fireball при наличии slot 3
- Shield тратит reaction
- Sculpt Spells защищает союзника от Fireball/Burning Hands

## 36. Запрос: Добавь поддержку AoE targeting для заклинаний и предметов

Файлы:
- src/combat/aoe.py
- src/combat/spellcasting.py
- src/combat/items.py
- src/agents/action_space.py

Реализуй формы:
- RADIUS
- CONE
- LINE

Для плана минимум:
- radius по клеткам Manhattan или Chebyshev distance
- cone можно упростить до направления:
  - UP
  - DOWN
  - LEFT
  - RIGHT
- line — прямая линия по направлению

Требования:
- Fireball использует RADIUS
- Burning Hands использует CONE
- AoE должен находить всех затронутых существ
- friendly fire должен быть включен
- Sculpt Spells может исключать союзников из Evocation AoE
- cover может давать бонус к DEX save против AoE, если это поддерживается текущей реализацией
- action mask должен позволять выбирать target_cell или direction
- reward должен учитывать урон по союзникам как штраф

Добавь логирование всех затронутых целей.
Добавь тесты AoE.

## 37. Запрос: Добавь упрощённую систему concentration

Файлы:
- src/combat/spellcasting.py
- src/combat/conditions.py

Требования:
- SpellAbility может иметь concentration=True
- у персонажа может быть active_concentration_spell
- при касте нового concentration spell старый сбрасывается
- при получении урона персонаж делает CON save:
  - DC = max(10, damage / 2)
- при провале concentration сбрасывается
- логировать начало, замену и потерю концентрации

Для MVP:
- Bless может использовать concentration
- если Bless реализован как stub, концентрационная инфраструктура всё равно должна быть готова

Добавь тесты:
- concentration spell устанавливает active_concentration_spell
- новый concentration spell заменяет старый
- урон вызывает CON save
- провал CON save сбрасывает концентрацию

## 38. Запрос: Добавь систему типов урона, сопротивлений, иммунитетов и уязвимостей

Файлы:
- src/combat/damage.py
- src/combat/entities.py
- src/combat/spellcasting.py
- src/combat/actions.py

DamageType enum:
- SLASHING
- PIERCING
- BLUDGEONING
- FIRE
- COLD
- LIGHTNING
- ACID
- POISON
- NECROTIC
- RADIANT
- FORCE
- PSYCHIC
- THUNDER

У персонажа/врага должны быть:
- resistances: set[DamageType]
- immunities: set[DamageType]
- vulnerabilities: set[DamageType]

Правила:
- immunity = 0 урона
- resistance = половина урона
- vulnerability = двойной урон
- порядок:
  1. базовый урон
  2. модификаторы spell/weapon
  3. resistance/immunity/vulnerability

Обнови:
- WeaponAttack
- SpellAbility
- ItemEffect
чтобы они имели damage_type.

Добавь в observation признаки:
- основные immunities/resistances/vulnerabilities цели
- damage_type доступных действий, если возможно

Добавь тестового врага FireElementalSimple:
- immunity FIRE
- resistance SLASHING, PIERCING, BLUDGEONING

Добавь тесты:
- fire immunity обнуляет Fireball damage
- resistance делит урон
- vulnerability удваивает урон

## 39. Запрос: Добавь расширенный набор базовых врагов

Файл:
- src/combat/presets.py
- src/combat/monsters.py

Добавь:
- GoblinMelee
- GoblinArcher
- OrcWarrior
- SkeletonArcher
- Bandit
- Wolf
- FireElementalSimple

Требования:
- все враги используют common actions
- у каждого врага должны быть:
  - weapons
  - stats
  - hp
  - ac
  - speed
  - challenge_rating
  - xp_value
  - role
- FireElementalSimple должен иметь immunity FIRE
- SkeletonArcher должен иметь immunity или resistance к POISON, если damage system это поддерживает
- Wolf должен иметь melee attack и повышенную скорость
- GoblinArcher должен иметь ranged weapon
- не реализуй сложные легендарные действия

Добавь тесты:
- каждый preset создаётся без ошибок
- у каждого врага есть валидная атака
- у каждого врага есть CR и XP

## 40. Запрос: Добавь систему опыта за победу над врагами

Файлы:
- src/rules/xp.py
- src/combat/monsters.py
- src/combat/environment.py
- src/combat/rewards.py

Требования:
- каждый Enemy/Monster должен иметь:
  - challenge_rating
  - xp_value
- XP начисляется после завершения боя
- XP начисляется персонажам команды победителей
- для MVP делить XP поровну между участниками команды игроков
- не начислять XP нейросети как reward напрямую
- RL reward и D&D XP должны быть разными системами:
  - reward нужен для обучения поведения
  - XP нужен для progression/campaign mode

Добавь CR XP table в config:
- 0: 10
- 1/8: 25
- 1/4: 50
- 1/2: 100
- 1: 200
- 2: 450
- 3: 700
- 4: 1100
- 5: 1800

Добавь функции:
- get_xp_for_cr(cr) -> int
- calculate_encounter_xp(monsters) -> int
- award_party_xp(party, defeated_monsters)

Добавь тесты:
- XP по CR считается корректно
- XP делится между party
- level up после боя возможен
- XP не начисляется при поражении

## 41. Запрос: Добавь систему инвентаря и используемых предметов

Важно:
этот блок реализуется после классов, spellcasting и action economy.

Файлы:
- src/combat/inventory.py
- src/combat/items.py
- src/combat/common_actions.py
- src/character/schema.py

Требования:
- Character должен иметь inventory
- ItemDefinition:
  - name
  - item_type
  - quantity
  - action_cost:
    - ACTION
    - BONUS_ACTION
    - REACTION
    - FREE_INTERACTION
  - target_type:
    - SELF
    - ALLY
    - ENEMY
    - POINT
  - range
  - effect
  - consumable: bool
  - implemented: bool
- UseObject common action должен уметь применять предметы
- action masks должны учитывать:
  - наличие предмета
  - quantity > 0
  - action economy
  - валидность цели
  - range
  - line of sight, если предмет метательный

Для MVP реализовать:
- Potion of Healing:
  - consumable
  - action
  - healing
- Bomb или AlchemistFire как simplified custom item:
  - consumable
  - thrown item
  - fire damage
  - range
  - DEX save или attack roll
- HealerKit:
  - consumable charges
  - stabilize target

Требования:
- после использования consumable quantity уменьшается
- если quantity = 0, item скрывается action mask
- character builder должен позволять добавлять только implemented items
- reward должен учитывать:
  - полезное лечение
  - урон по врагу
  - штраф за урон по союзнику
  - штраф за трату предмета без эффекта

Добавь тесты:
- potion лечит
- potion тратится
- bomb наносит damage
- нельзя использовать предмет без quantity
- UseObject тратит правильный action resource

## 42. Запрос: Обнови action masks с учётом новых систем

Обнови action masks с учётом новых систем:
- levels
- class features
- subclass features
- race traits
- feats
- spell slots
- inventory
- map obstacles
- cover
- terrain cost
- line of sight

Файл:
- src/agents/action_space.py

Action masks должны учитывать:
- уровень персонажа
- реализована ли feature
- выбран ли subclass
- доступен ли class resource
- доступна ли bonus action
- доступна ли reaction
- есть ли нужный spell slot
- prepared ли spell
- есть ли item quantity > 0
- есть ли race/feat combat hook
- достижима ли клетка движения
- не заблокирована ли клетка
- есть ли line of sight
- не находится ли цель за full cover
- есть ли укрытие для Hide

Добавь debug режим:
- explain_action_mask(state, actor_id)

Функция должна возвращать список действий и причины:
- allowed
- blocked: no_action_available
- blocked: no_bonus_action_available
- blocked: no_reaction_available
- blocked: no_spell_slot
- blocked: unsupported_feature
- blocked: wrong_level
- blocked: no_valid_target
- blocked: no_item_quantity
- blocked: blocked_cell
- blocked: unreachable_cell
- blocked: no_line_of_sight
- blocked: full_cover
- blocked: no_cover_to_hide

Добавь тесты:
- Fighter level 1 не имеет Extra Attack
- Wizard level 1 не имеет Fireball
- potion masked при quantity = 0
- ranged attack masked при full cover
- movement masked для blocked cell

## 43. Запрос: Обнови observation encoder под реальные игровые признаки

Файлы:
- src/agents/observation.py
- src/agents/entity_observation.py

Добавь признаки актёра:
- level normalized
- proficiency_bonus normalized
- class_id
- subclass_id
- race_id
- feat flags
- action_available
- bonus_action_available
- reaction_available
- movement_remaining
- class resources:
  - second_wind_available
  - action_surge_available
  - channel_divinity_available
  - arcane_recovery_available
- spell slots current/max levels 1-3
- prepared spell flags
- inventory usable item flags
- current cover status
- terrain around actor
- visible enemies count

Добавь признаки других существ:
- known class/monster role
- challenge_rating normalized
- estimated_xp_value normalized
- resistances/immunities/vulnerabilities
- conditions
- active concentration
- current AC
- current HP ratio
- threat estimate
- line_of_sight_from_actor
- cover_from_actor
- distance_to_actor
- reachable_by_actor

Добавь global features:
- round_number
- initiative_position
- allies_alive
- enemies_alive
- encounter_difficulty_estimate
- map width/height normalized

Обнови input size PPO/GNN моделей и тесты.

## 44. Запрос: Обнови reward function с учётом ограниченных ресурсов, карты и D&D-механик

Файл:
- src/combat/rewards.py

Добавь:
- небольшой штраф за трату spell slot
- штраф выше для более высокого slot level
- небольшой штраф за трату Action Surge
- небольшой штраф за трату Second Wind, если лечение было неэффективным
- штраф за overkill дорогим ресурсом
- штраф за применение урона по immunity
- штраф за применение resisted damage, если были доступные лучшие альтернативы
- бонус за эффективное использование дорогого ресурса, если оно привело к убийству, спасению союзника или победе
- бонус за использование cover, если оно помогло избежать урона
- бонус за Dodge, если после этого атаки по персонажу промахнулись
- бонус за Disengage, если персонаж избежал opportunity attack
- штраф за движение в плохую позицию без тактической выгоды
- штраф за friendly fire
- штраф за трату предмета без эффекта

Важно:
- победа должна быть важнее экономии ресурсов
- предотвращение смерти союзника должно быть важнее штрафа за ресурс
- награды за ресурсы, cover и позиционирование не должны доминировать над основной целью боя

Добавь reward breakdown в лог.
Добавь тесты для основных reward components.

## 45. Запрос: Добавь evaluation scenarios для уровней 1-5

Файлы:
- src/combat/evaluation_scenarios.py
- scripts/evaluate_policy.py

Сценарии:
- Level 1 Fighter vs 1 Goblin
- Level 1 Fighter + Cleric vs 2 Goblins
- Level 3 Fighter Champion vs Orc
- Level 3 Cleric Life + Fighter vs Orc + Goblin
- Level 5 Wizard Evoker vs 3 Goblins
- Level 5 Wizard Evoker vs FireElementalSimple
- Level 5 Fighter + Cleric + Wizard vs mixed enemies
- Level 5 ranged party on map with cover vs mixed enemies
- Level 5 melee party on difficult terrain map vs archers

evaluate_policy.py должен выводить:
- win rate
- average reward
- average XP gained
- average resource usage
- spell slots used by level
- class features usage
- item usage
- cover usage
- average movement cost
- deaths
- action distribution
- masked action statistics

Добавь режим:
- --by-level
который прогоняет сценарии для каждого уровня 1-5.

## 46. Запрос: Добавь curriculum learning в EncounterGenerator и PPO trainer

Файлы:
- src/combat/encounter_generator.py
- src/training/ppo_trainer.py
- configs/train_curriculum.yaml

Уровни сложности:
1. 1 Fighter level 1 vs 1 Goblin
2. 1 Fighter level 1 vs 2 Goblins
3. Fighter level 3 Champion vs Orc
4. Fighter + Cleric vs Orc + Goblins
5. Wizard scenarios
6. mixed party vs mixed enemies
7. enemies with resistances/immunities
8. maps with obstacles and cover
9. maps with difficult terrain and ranged enemies

Trainer должен:
- отслеживать win rate на текущем уровне
- повышать difficulty при достижении threshold
- сохранять current curriculum_level в checkpoint
- логировать переходы между уровнями

Добавь тесты:
- curriculum level повышается при win rate threshold
- curriculum state сохраняется в checkpoint

## 47. Запрос: Переделай observation encoder на entity-based формат

Файлы:
- src/agents/observation.py
- src/agents/entity_observation.py

Вместо одного плоского вектора сделай:
- actor_features
- entities_features
- map_features
- global_features
- entity_mask

Для каждого entity:
- hp ratio
- ac
- position
- team relation
- alive
- distance_to_actor
- prone
- grappled
- hidden
- dodging
- has_reaction
- is_spellcaster
- resistances/immunities/vulnerabilities
- threat estimate
- line_of_sight_from_actor
- cover_from_actor

Map features:
- локальное окно вокруг актёра
- terrain type
- blocked cells
- cover cells
- movement cost map
- visible cells, если реализовано

Global features:
- round number
- initiative position
- map size
- current team
- remaining allies
- remaining enemies

Сохрани совместимость:
- старая MLP модель может использовать flattened version
- новая GNN модель должна использовать entity-based version

Добавь тесты:
- entity mask корректен
- map features имеют стабильный размер
- flatten совместим со старой моделью

## 48. Запрос: Добавь GNN encoder для состояния боя

Файл:
- src/agents/gnn_encoder.py

Требования:
- каждый персонаж/враг — node
- node features берутся из entity-based observation
- edge features:
  - distance
  - same_team
  - enemy_relation
  - can_attack
  - can_help
  - line_of_sight
  - cover_between
- реализуй простой message passing без внешних GNN-библиотек
- поддержи padding и entity_mask

Выход:
- actor_embedding
- pooled_allies_embedding
- pooled_enemies_embedding
- pooled_battle_embedding

Не удаляй старый MLP encoder.

Добавь тесты:
- GNN работает при разном числе существ
- masked entities не влияют на pooled embedding
- выходные размерности стабильны

## 49. Запрос: Создай новую модель GNNPPOActorCritic

Файл:
- src/agents/gnn_ppo_model.py

Требования:
- использовать GNNEncoder
- policy heads:
  - action_category_head
  - main_action_type_head
  - bonus_action_type_head
  - reaction_type_head
  - class_feature_head
  - target_head
  - move_head
  - spell_head
  - slot_level_head
  - item_head
  - option_head
- value_head
- поддержка action masking для всех heads
- act(...)
- evaluate_actions(...)

Важно:
- не удаляй MLP PPO модель
- добавь model_type в конфиг:
  - mlp
  - gnn

Добавь тесты:
- модель возвращает легальное действие при masks
- модель работает с variable entity count
- модель работает с map features

## 50. Запрос: Добавь centralized critic для multi-agent PPO

Файлы:
- src/agents/gnn_ppo_model.py
- src/training/ppo_trainer.py

Идея:
- actor выбирает действие для конкретного существа
- critic оценивает полное состояние боя

Требования:
- actor использует actor-centric observation
- critic использует global/entity observation всего боя
- value_head должен получать pooled_battle_embedding
- PPO trainer должен сохранять:
  - actor_observation
  - critic_observation
  - action
  - log_prob
  - value
  - reward
  - done
  - masks

Добавь флаг:
- centralized_critic: true/false

Добавь тесты:
- trainer работает с centralized_critic=true
- trainer работает с centralized_critic=false

## 51. Запрос: Добавь поддержку разных политик для разных команд и ролей

Файлы:
- src/training/multi_agent.py
- src/training/ppo_trainer.py

Требования:
- одна политика может управлять всеми существами
- отдельная player_policy может управлять игроками
- отдельная enemy_policy может управлять врагами
- возможность rule_based_enemy_policy для baseline
- возможность random_policy для baseline

Добавь role embeddings:
- MELEE_DAMAGE
- RANGED_DAMAGE
- TANK
- SUPPORT
- CASTER
- BRUTE_ENEMY
- SKIRMISHER_ENEMY

Observation/model должны получать role_id.

Добавь тесты:
- player_policy управляет только игроками
- enemy_policy управляет только врагами
- одна общая policy может управлять всеми

## 52. Запрос: Добавь self-play обучение

Файлы:
- src/training/self_play.py
- src/training/ppo_trainer.py
- configs/train_self_play.yaml

Требования:
- текущая policy может играть против копий старых версий policy
- opponent pool хранит checkpoints
- каждые N updates текущая policy добавляется в opponent pool
- opponent выбирается случайно
- можно заморозить enemy_policy
- можно обучать только player side или обе стороны

Логировать:
- текущий opponent checkpoint
- win rate против каждого opponent
- среднюю награду
- action distribution
- resource usage

Добавь тесты:
- opponent pool добавляет checkpoint
- opponent случайно выбирается
- frozen enemy policy не обновляется

## 53. Запрос: Добавь rule-based baseline agents для сравнения с PPO

Файлы:
- src/agents/rule_based.py

Реализуй:
- AggressiveMeleeAgent
- RangedKitingAgent
- SimpleHealerAgent
- SimpleCasterAgent
- CoverAwareRangedAgent
- RandomLegalAgent

Требования:
- агенты должны использовать те же action masks
- не должны выбирать нелегальные действия
- RangedKitingAgent должен учитывать distance и terrain
- CoverAwareRangedAgent должен пытаться занимать cover
- SimpleHealerAgent должен использовать healing spells/items при низком HP союзника
- можно использовать их в evaluate_policy.py
- можно использовать их как opponents в self-play

Добавь тесты:
- каждый baseline agent выбирает legal action
- healer лечит раненого союзника при наличии ресурса
- ranged agent не идёт в melee без необходимости

## 54. Запрос: Добавь формат записи боя BattleReplay

Файлы:
- src/combat/replay.py
- scripts/run_demo.py

BattleReplay должен сохранять каждый шаг боя в JSON:
- round
- turn_index
- initiative_order
- actor
- actor_team
- positions
- hp values
- conditions
- resources
- action
- action_category
- targets
- dice rolls
- damage
- healing
- spell slots spent
- items spent
- deaths
- reward breakdown
- map terrain snapshot или map metadata
- cover/line_of_sight info для действия, если применимо
- winner
- XP gained

Обнови run_demo.py:
- добавить аргумент --save-replay
- сохранять replay в папку replays/

Добавь тесты:
- replay сохраняется
- replay содержит все обязательные поля

## 55. Запрос: Добавь удобный консольный replay viewer

Скрипт:
- scripts/view_replay_console.py

Требования:
- загружать replay JSON
- пошагово выводить состояние боя
- показывать карту ASCII
- отображать:
  - blocked cells
  - difficult terrain
  - cover cells
  - позиции существ
- показывать HP и ресурсы существ
- показывать последнее действие
- показывать initiative order
- поддерживать команды:
  - next
  - prev
  - quit
  - autoplay

Добавь тест на загрузку replay.

## 56. Запрос: Создай основу полноценного desktop GUI для проекта

Технология:
- использовать PySide6
- не использовать GUI для обучения модели
- обучение PPO/GNN остаётся через существующие scripts/train_*.py

Файлы:
- src/ui/
  - app.py
  - main_window.py
  - navigation.py
  - theme.py
  - widgets/
  - screens/
- scripts/run_gui.py

Требования:
- создать главное окно приложения
- добавить боковое или верхнее меню навигации
- пункты меню:
  - Главная
  - Персонажи
  - Создать персонажа
  - Случайный бой
  - Кастомный бой
  - Реплеи
  - Настройки
- использовать QStackedWidget или аналогичный механизм переключения экранов
- добавить базовую тему оформления:
  - спокойные цвета
  - читаемые шрифты
  - единый стиль кнопок
- добавить стартовый экран с кратким описанием проекта:
  - D&D Battle AI
  - использование обученной нейросети для пошаговых боёв
  - обучение запускается отдельно через скрипты

Важно:
- GUI не должен ломать существующие CLI-скрипты
- вся логика боя должна использовать уже существующий combat engine
- не дублировать боевую логику внутри UI

Добавь README-раздел:
- как запустить GUI:
  - python scripts/run_gui.py

## 57. Запрос: Добавь систему хранения созданных персонажей для GUI

Файлы:
- src/character/schema.py
- src/character/io.py
- src/character/validation.py
- src/character/repository.py
- data/characters/

InternalCharacter должен включать:
- id
- name
- class_name
- subclass_name
- level
- experience
- race_name
- role
- stats
- hp
- ac
- speed
- proficiency_bonus
- weapons
- armor
- class_features
- subclass_features
- race_traits
- feats
- spells
- prepared_spells
- spell_slots
- resources
- inventory
- resistances
- immunities
- vulnerabilities

Реализуй:
- CharacterRepository
  - list_characters()
  - get_character(id)
  - save_character(character)
  - delete_character(id)
  - duplicate_character(id)
  - validate_character(character)
- хранение персонажей в JSON-файлах в data/characters/
- автоматическое создание папки data/characters/, если её нет
- генерацию id для персонажа

Важно:
- только внутренний формат персонажей
- никаких LongStoryShort/import URL функций
- validation должна проверять соответствие ruleset registry
- GUI должен использовать CharacterRepository, а не читать файлы напрямую

Добавь тесты:
- персонаж сохраняется
- персонаж загружается
- список персонажей возвращается корректно
- невалидный персонаж не сохраняется без ошибки validation

## 58. Запрос: Реализуй экран просмотра созданных персонажей

Файлы:
- src/ui/screens/character_list_screen.py
- src/ui/widgets/character_card.py

Экран "Персонажи" должен показывать:
- список всех созданных персонажей из CharacterRepository
- карточку персонажа:
  - имя
  - раса
  - класс
  - подкласс
  - уровень
  - HP
  - AC
  - основное оружие
  - роль
- кнопки:
  - Просмотр
  - Редактировать
  - Дублировать
  - Удалить

При открытии персонажа показывать подробную информацию:
- характеристики
- владения
- оружие
- броня
- классовые способности
- расовые особенности
- черты
- заклинания
- ячейки заклинаний
- инвентарь
- сопротивления/иммунитеты/уязвимости

Требования:
- если персонажей нет, показать сообщение "Персонажи ещё не созданы"
- кнопка "Создать персонажа" должна переводить на экран создания
- после удаления или создания список должен обновляться
- удаление должно требовать подтверждения

Важно:
- экран только отображает и управляет персонажами
- боевую логику не реализовывать в этом экране

## 59. Запрос: Создай полноценный GUI-конструктор персонажей

Файлы:
- src/ui/screens/character_builder_screen.py
- src/ui/widgets/stat_editor.py
- src/ui/widgets/spell_selector.py
- src/ui/widgets/inventory_editor.py

Конструктор должен позволять:
- создать нового персонажа
- редактировать существующего персонажа
- выбрать race из поддержанных ruleset:
  - Human
  - Dwarf
  - Elf
  - Halfling
- выбрать class:
  - Fighter
  - Cleric
  - Wizard
- выбрать level 1-5
- выбрать subclass, если уровень позволяет:
  - Fighter: Champion
  - Cleric: Life Domain
  - Wizard: Evocation
- ввести или сгенерировать stats
- применить racial bonuses
- выбрать ASI/feat на 4 уровне
- выбрать fighting style для Fighter
- выбрать prepared spells для Cleric/Wizard
- выбрать weapons
- выбрать armor
- добавить inventory items из supported items
- рассчитать:
  - proficiency_bonus
  - hp
  - ac
  - spell_save_dc
  - spell_attack_bonus
  - spell slots
  - class resources

UI требования:
- использовать пошаговый мастер или вкладки:
  1. Основное
  2. Характеристики
  3. Класс
  4. Заклинания
  5. Экипировка
  6. Инвентарь
  7. Проверка
- на последнем шаге показывать validation errors/warnings
- кнопка "Сохранить" доступна только если персонаж валиден
- builder должен показывать только поддержанные элементы
- нельзя выбрать Fireball для Wizard level 1
- нельзя выбрать unsupported class/race/subclass/spell/item

Важно:
- никакого импорта из внешних сервисов
- все ограничения брать из ruleset registry
- после сохранения пользователь возвращается на экран списка персонажей

Добавь тесты для validation-части, GUI-тесты делать необязательно.

## 60. Запрос: Добавь сервис инференса обученной нейросети для GUI

Файлы:
- src/inference/
  - policy_loader.py
  - battle_ai.py
  - action_selector.py
- src/ui/services/model_service.py

Требования:
- GUI должен уметь загрузить обученный checkpoint
- поддержать model_type:
  - mlp
  - gnn, если реализован
- если checkpoint не выбран, использовать RandomLegalAgent или RuleBasedAgent как fallback
- BattleAIService должен иметь методы:
  - load_checkpoint(path, model_type)
  - is_model_loaded()
  - select_action(combat_state, actor_id)
  - get_policy_name()
- select_action должен использовать:
  - observation encoder
  - action masks
  - PPO/GNN model
  - decode_action

Важно:
- обучение не запускать из GUI
- GUI только использует уже обученную модель
- если модель не загрузилась, показать понятную ошибку
- если модель не поддерживает текущий action space, показать ошибку совместимости

Добавь настройки:
- путь к checkpoint
- model_type
- fallback_agent

Добавь тесты:
- fallback agent выбирает legal action
- несуществующий checkpoint даёт понятную ошибку
- загруженная модель используется для выбора действия, если checkpoint валиден

## 61. Запрос: Реализуй экран запуска случайного тестового боя

Файлы:
- src/ui/screens/random_battle_screen.py
- src/ui/services/battle_setup_service.py

Экран "Случайный бой" должен позволять:
- выбрать party из созданных персонажей
- или использовать готовые preset-персонажи
- выбрать уровень сложности:
  - Лёгкий
  - Средний
  - Сложный
- выбрать карту:
  - open_field
  - cover_arena
  - difficult_terrain_pass
  - obstacle_corridor
  - random
- выбрать врагов:
  - автоматически по сложности
  - или preset enemy group
- выбрать управляющего:
  - AI управляет игроками
  - AI управляет врагами
  - AI управляет всеми
  - игрок вручную управляет игроками, AI управляет врагами

При нажатии "Начать бой":
- создать CombatEnvironment
- загрузить выбранную карту
- расставить персонажей и врагов в spawn zones
- открыть BattleScreen

Требования:
- если персонажи не выбраны, показать ошибку
- если модель не загружена, предупредить, что будет использован fallback agent
- случайная генерация должна поддерживать seed
- отображать краткое описание боя перед запуском:
  - party
  - enemies
  - map
  - difficulty

## 62. Запрос: Реализуй основной экран боя с отрисованной клеточной картой

Файлы:
- src/ui/screens/battle_screen.py
- src/ui/widgets/battle_map_widget.py
- src/ui/widgets/initiative_panel.py
- src/ui/widgets/combat_log_widget.py
- src/ui/widgets/creature_status_panel.py

BattleScreen должен показывать:
- клеточную карту
- токены персонажей и врагов
- HP над токенами или рядом с ними
- текущего активного участника
- порядок инициативы
- панель статуса выбранного существа
- боевой лог
- кнопки управления:
  - Следующий шаг
  - Автобой
  - Пауза
  - Завершить бой
  - Сохранить реплей

Цвета карты:
- normal terrain — салатовый
- blocked — коричневый
- difficult terrain — жёлто-зелёный
- low cover — светло-серый
- high cover — тёмно-серый
- доступные клетки движения — полупрозрачная подсветка
- возможные цели — красная подсветка
- текущий активный токен — обводка

Токены:
- игроки — синий/зелёный круг
- враги — красный/оранжевый круг
- мёртвые существа — затемнённый токен или крест
- если есть portrait/icon, использовать его
- если нет, показывать инициалы

Лог боя должен отображать:
- начало раунда
- активное существо
- выбранное действие
- броски кубов
- урон/лечение
- траты ресурсов
- смерти
- победителя

Важно:
- экран боя использует CombatEnvironment.step()
- не дублировать правила боя в UI
- действия AI выбираются через BattleAIService

## 63. Запрос: Добавь пошаговое управление боем и простейшие анимации

Файлы:
- src/ui/screens/battle_screen.py
- src/ui/widgets/battle_map_widget.py
- src/ui/animations.py

Требования:
- режим "Следующий шаг":
  - выполняет ровно одно действие активного существа
  - обновляет карту, HP, лог и инициативу
- режим "Автобой":
  - автоматически выполняет действия с задержкой
  - задержка настраивается, например 300-1500 мс
- режим "Пауза":
  - останавливает автобой

Простейшие анимации:
- движение:
  - токен плавно перемещается из клетки в клетку
- melee attack:
  - короткое смещение токена к цели и назад
- ranged attack:
  - линия или маленький снаряд от атакующего к цели
- spell:
  - подсветка цели или области AoE
- damage:
  - краткая красная вспышка токена
- healing:
  - краткая зелёная вспышка токена
- death:
  - затемнение токена или появление крестика

Важно:
- анимации не должны менять combat state
- combat state меняется только через CombatEnvironment
- если анимация выключена, бой всё равно должен работать

Добавь настройку:
- animations_enabled: true/false
- animation_speed

## 64. Запрос: Добавь возможность ручного управления персонажами в GUI

Файлы:
- src/ui/screens/battle_screen.py
- src/ui/widgets/action_panel.py
- src/ui/services/manual_action_builder.py

Требования:
если текущим существом управляет игрок вручную:
- показать список доступных действий из action masks
- действия сгруппировать:
  - Movement
  - Main Action
  - Bonus Action
  - Reaction, если применимо
  - End Turn
- для каждого действия показывать только legal options

Ручной выбор должен поддерживать:
- перемещение:
  - пользователь кликает доступную клетку
- атака:
  - пользователь выбирает weapon
  - кликает цель
- spell:
  - выбирает spell
  - выбирает slot level
  - кликает цель или клетку AoE
- item:
  - выбирает item
  - выбирает цель
- class feature:
  - выбирает доступную feature
- End Turn

Важно:
- GUI должен использовать те же action masks, что и нейросеть
- нельзя вручную выбрать нелегальное действие
- если действие требует цель, подсвечивать допустимые цели
- если действие требует клетку, подсвечивать допустимые клетки
- после выбора действие передается в CombatEnvironment.step()

Если manual mode не нужен для текущего боя, этот экран работает только как viewer AI боя.

## 65. Запрос: Интегрируй реплеи в GUI

Файлы:
- src/ui/screens/replay_list_screen.py
- src/ui/screens/replay_viewer_screen.py
- src/combat/replay.py

Экран "Реплеи" должен:
- показывать список replay JSON из папки replays/
- отображать:
  - дата
  - название боя
  - победитель
  - количество раундов
  - участники
- кнопки:
  - Открыть
  - Удалить
  - Переименовать

Replay viewer должен:
- использовать тот же BattleMapWidget, что и BattleScreen
- позволять:
  - следующий шаг
  - предыдущий шаг
  - autoplay
  - pause
  - перейти к началу
  - перейти к концу
- показывать:
  - карту
  - позиции существ
  - HP/resources
  - боевой лог
  - инициативу
  - последнее действие

Важно:
- replay viewer не запускает CombatEnvironment
- он только читает сохранённые состояния из replay JSON
- replay должен отображаться даже без загруженной модели

## 66. Запрос: Реализуй экран настроек GUI

Файлы:
- src/ui/screens/settings_screen.py
- src/ui/settings.py
- data/settings.json

Настройки:
- путь к checkpoint модели
- model_type:
  - mlp
  - gnn
- fallback agent:
  - random legal
  - aggressive melee
  - rule-based
- скорость анимаций
- включить/выключить анимации
- задержка автобоя
- папка персонажей
- папка реплеев
- папка карт
- seed для случайных боёв

Требования:
- настройки сохраняются в data/settings.json
- при запуске GUI настройки загружаются автоматически
- кнопка "Проверить модель":
  - пытается загрузить checkpoint
  - показывает успех или ошибку
- если checkpoint не задан:
  - GUI работает через fallback agent
- обучение модели в настройках не запускать

Добавь понятные сообщения об ошибках:
- checkpoint не найден
- model_type не совпадает
- неподдерживаемый action space
- повреждённый файл настроек

## 67. Запрос: Добавь поддержку конфигураций карт для GUI

Файлы:
- src/combat/map_config.py
- src/ui/widgets/map_preview_widget.py
- maps/

Map JSON должен содержать:
- name
- width
- height
- terrain grid
- spawn zones:
  - players
  - enemies

Terrain values:
- NORMAL
- DIFFICULT_TERRAIN
- BLOCKED
- LOW_COVER
- HIGH_COVER

Добавь примеры карт:
- open_field.json
- cover_arena.json
- difficult_terrain_pass.json
- obstacle_corridor.json

GUI требования:
- в экранах Random Battle и Custom Battle можно выбрать карту
- показывать preview карты перед запуском боя
- отображать цвета terrain так же, как в BattleMapWidget
- проверять валидность карты:
  - размеры корректны
  - terrain grid соответствует размерам
  - spawn zones не находятся на blocked cells
  - spawn zones не выходят за границы

Добавь тесты:
- карта загружается из JSON
- preview получает корректные данные
- invalid map config вызывает validation error

## 68. Запрос: Улучши внешний вид и удобство GUI

Требования:
- добавить единый стиль приложения
- добавить иконки или текстовые эмодзи для:
  - персонажи
  - бой
  - настройки
  - реплеи
- добавить статусную строку:
  - текущая модель
  - fallback agent
  - выбранный бой
- добавить confirmation dialogs:
  - удалить персонажа
  - удалить реплей
  - выйти из активного боя
- добавить error dialogs:
  - невалидный персонаж
  - невалидная карта
  - модель не загружена
  - невозможно выполнить действие
- добавить loading states:
  - загрузка checkpoint
  - создание боя
  - открытие replay
- добавить пустые состояния:
  - нет персонажей
  - нет реплеев
  - нет карт

Важно:
- GUI должен быть удобен для демонстрации проекта
- не добавлять обучение модели в GUI
- не усложнять UI сверх необходимости

## 69. Запрос: Обнови README.md и ROADMAP.md под наличие полноценного GUI

README должен содержать:
- как запустить GUI:
  - python scripts/run_gui.py
- что можно делать в GUI:
  - просматривать персонажей
  - создавать персонажей
  - запускать случайный бой
  - создавать кастомный бой
  - вручную расставлять персонажей и врагов
  - смотреть пошаговый бой на клеточной карте
  - смотреть реплеи
  - выбирать checkpoint обученной модели
- отдельно указать:
  - обучение модели НЕ запускается из GUI
  - обучение запускается через существующие training scripts
- описать папки:
  - data/characters/
  - replays/
  - maps/
  - checkpoints/

ROADMAP должен обновиться:
- GUI:
  - character browser
  - character builder
  - random battle launcher
  - custom battle setup
  - battle viewer
  - replay viewer
- будущие улучшения:
  - drag-and-drop в редакторе боя
  - портреты токенов
  - более красивые анимации
  - полноценный GUI редактор карт
  - расширенный character builder
  - поддержка большего числа классов/рас/заклинаний

Удалить или исправить старые упоминания:
- simple 2D replay viewer как отдельный pygame viewer, если теперь используется PySide6 GUI
- любые упоминания LongStoryShort importer

## 70. Запрос: Исправить создание персонажа и упростить настройки GUI

Проблемы:
- при создании персонажа сыпятся ошибки `stats_changed() only accepts 0 argument(s), 1 given!`
- при изменении инвентаря сыпятся ошибки `inventory_changed() only accepts 0 argument(s), 1 given!`
- нового персонажа нельзя сохранить из-за `id: must not be empty`
- модель не подхватывается в GUI
- из настроек нужно убрать выбор карт, моделей, checkpoint и других важных параметров

Требования:
- исправить Qt-сигналы редакторов характеристик и инвентаря
- id нового персонажа должен генерироваться системой, пользователь не должен его вводить
- настройки GUI не должны позволять пользователю выбирать `model_type`, checkpoint, fallback agent и служебные папки
- модель в GUI должна быть зафиксирована как PPO Actor-Critic с GNN encoder
- если checkpoint несовместим или отсутствует, показывать понятную ошибку и использовать внутренний fallback без ручного выбора

## 71. Запрос: Перевести обучение на GPU и GNN checkpoint

Проблемы:
- файла `checkpoints/gnn_ppo_actor_critic.pt` нет, хотя модель обучалась
- обучение идёт на CPU, а не на GPU

Требования:
- проверить, почему PyTorch не видит CUDA
- настроить установку CUDA-сборки PyTorch, если в окружении стоит CPU-сборка
- обновить `scripts/train_ppo.py`, чтобы он поддерживал `--device auto/cuda/cpu`
- обновить `scripts/train_ppo.py`, чтобы он поддерживал `--model-type gnn/mlp`
- для GNN по умолчанию сохранять checkpoint в `checkpoints/gnn_ppo_actor_critic.pt`
- для MLP по умолчанию сохранять checkpoint в `checkpoints/ppo_actor_critic.pt`

## 72. Запрос: Перевести train_ppo на fixed rollout batches

Цель:
- повысить полезную загрузку GPU на PPO update
- не обновлять модель после каждого отдельного боя

Требования:
- использовать `PPOTrainer.collect_rollout()` вместо цикла `collect_episode() -> update()`
- добавить CLI-параметры:
  - `--updates`
  - `--rollout-steps`
  - `--minibatch-size`
  - `--update-epochs`
  - `--log-interval`
- оставить `--episodes` как совместимый alias для числа update-итераций
- выводить update metrics:
  - rollout steps
  - finished episodes
  - win rate
  - average step reward
  - policy loss
  - value loss
  - entropy
  - loss
  - checkpoint path

## 73. Запрос: Добавить max episode steps для PPO rollout

Проблема:
- при batch-обучении `episodes_finished=0`, `win_rate=0`
- политика может слишком долго не завершать бой, поэтому rollout не даёт завершённых эпизодов

Требования:
- добавить максимальное количество шагов на эпизод внутри `PPOTrainer.collect_rollout`
- при таймауте завершать текущий episode как `done=True`
- сбрасывать среду на новый бой после таймаута
- учитывать таймауты отдельно от побед/поражений
- добавить CLI-параметр `--max-episode-steps`
- выводить `completed` и `timeouts` в training log

## 74. Запрос: Performance-pass перед расширением curriculum

Цель:
- снизить CPU bottleneck перед добавлением полного набора сценариев обучения
- не ломать GUI/evaluation full action masks

Требования:
- добавить быстрый training action mask с ограниченным набором действий для раннего обучения
- оставить полный action mask для GUI и evaluation
- добавить кэш в `GridMap` для повторных LoS/cover/neighbors расчётов
- добавить profiling metrics в PPO rollout:
  - observation
  - mask
  - model_act
  - decode
  - env_step
  - update
- добавить CLI-флаги:
  - `--fast-action-masks`
  - `--profile-training`
- выводить timing breakdown в training log при включённом profiling

## 75. Запрос: Продолжить пошаговую оптимизацию PPO/GNN training

Цель:
- уменьшать лишнюю работу в GNN PPO step-by-step
- оптимизировать обучение постепенно, не ломая GUI/evaluation

Первый шаг:
- передавать masks для всех GNN policy heads
- дополнительные heads, которые пока не используются декодером (`spell_index`, `item_index`, `slot_level`, `reaction_type`, `bonus_action_type`, `class_feature`), должны быть детерминированно замаскированы на noop index 0
- уменьшить лишний entropy/log_prob noise от неиспользуемых heads
- добавить тесты, что GNN rollout содержит masks для всех heads

Второй шаг:
- отключить `validate_args` у `torch.distributions.Categorical` в PPO/GNN моделях
- оставить валидацию действий на уровне masks/decode, но убрать runtime overhead PyTorch distribution validation

Третий шаг:
- использовать CPU raw masks для `decode_action`
- использовать padded GPU masks только для model forward/evaluate
- убрать лишнюю CUDA synchronization при decode
- объединить чтение action indices с GPU в один CPU transfer вместо нескольких `.item()`
- добавить `decode_fast_training_action` для reduced action space, чтобы fast training не проходил через полный D&D decoder

## 76. Запрос: Добавить parallel/vectorized rollout для PPO training

Цель:
- ускорить сбор rollout за счёт нескольких независимых боёв внутри одного PPOTrainer
- уменьшить overhead мелких GPU-вызовов через batched `policy.act`
- сохранить текущий combat engine без multiprocessing и без дублирования правил боя

Требования:
- добавить параметр `num_envs` в PPOTrainer
- добавить CLI-флаг `--num-envs` в `scripts/train_ppo.py`
- при `num_envs > 1` создавать несколько `CombatEnvironment` через `EncounterGenerator`
- собирать rollout из нескольких env round-robin
- если активные политики одинаковые torch-модели, выполнять batched action selection
- считать returns/advantages отдельно по `env_id`, чтобы GAE не смешивал разные бои
- сохранять корректные episode timeout/winner метрики для каждого env
- добавить тесты multi-env rollout и per-env advantage calculation

## 77. Запрос: Оптимизировать fast training action masks после замеров num_envs

Контекст:
- `num_envs=8` ускорил запуск относительно `num_envs=4`
- после batching заметными bottleneck остались `model_act_ms`, `mask_ms`, `observation_ms`
- для fast training полный D&D target validation не нужен на этапе построения маски, потому что combat engine всё равно валидирует действие при выполнении

Требования:
- не менять полный `build_action_masks` для GUI/evaluation
- оптимизировать только `build_fast_training_action_masks`
- заменить повторные `path_movement_cost` для movement candidates на один `movement_costs_from`
- упростить fast attack mask до:
  - actor может потратить Attack action
  - target живой и вражеский
  - target находится в пределах range доступного weapon
- не выполнять в fast attack mask дорогие LoS/cover проверки
- сохранить тесты fast action masks и PPO trainer

## 78. Запрос: Добавить fast observation для warm-up PPO/GNN training

Цель:
- уменьшить CPU bottleneck на observation encoding после оптимизации `num_envs` и fast masks
- сохранить совместимость checkpoint между fast warm-up и full fine-tuning

Требования:
- добавить опциональный режим `fast=True` в `encode_observation` и `encode_entity_observation`
- размер actor/entity/map/global tensors должен остаться тем же, что в полном observation
- в fast observation заменить дорогие признаки на приближения:
  - не считать full LoS/cover для каждого entity
  - не считать full reachable/pathfinding для каждого entity
  - не считать full local map visibility
  - не считать current cover status через всех врагов
- добавить `fast_observation` в `PPOTrainer`
- добавить CLI-флаг `--fast-observation` в `scripts/train_ppo.py`
- полный observation по умолчанию оставить без изменений для GUI/evaluation/full training
- добавить тесты, что fast/full observation имеют одинаковые shapes

## 79. Запрос: Прокачать обучение под все поддержанные классы, врагов и карты

Проблема:
- если сразу обучать на всех классах, врагах, заклинаниях и типах карт, training time резко вырастет
- модель должна получать сложность постепенно, иначе PPO будет собирать шумный опыт и долго не сходиться

Цель:
- сделать staged curriculum для Fighter/Cleric/Wizard, разных врагов, resistances/immunities и карт
- подключить curriculum к `scripts/train_ppo.py`, а не держать его только внутри trainer/generator

Требования:
- расширить `CURRICULUM_STAGES` до последовательной лестницы:
  - level 1 Fighter vs Goblin
  - Fighter + Cleric против Goblins
  - Fighter + Cleric против Goblin + Bandit
  - Fighter Champion против Orc
  - Cleric Life + Fighter против Orc + Goblin
  - level 4 Fighter + Cleric против Orc + Skeleton Archer
  - Wizard Evoker AoE scenarios
  - Wizard vs FireElementalSimple
  - full party vs mixed enemies
  - ranged party on cover map
  - enemies with resistances/immunities
  - obstacle/cover map
  - difficult terrain vs ranged enemies
- обновить `configs/train_curriculum.yaml`
- добавить CLI-флаги:
  - `--curriculum`
  - `--curriculum-config`
  - `--curriculum-level`
  - `--curriculum-max-level`
  - `--curriculum-threshold`
  - `--curriculum-window-size`
- выводить активный curriculum level при запуске
- печатать curriculum transition в training log
- сохранить random training как режим по умолчанию, если `--curriculum` не указан

## 80. Запрос: Исправить падение curriculum training на option_index mask

Проблема:
- при переходе curriculum к более сложным уровням обучение падало с:
  - `ValueError: option_index mask is larger than the model head`
- причина:
  - Wizard/Evoker stages создают `option_index` mask размера 10
  - дефолтная PPO/GNN модель создавалась с `option_count=8`

Требования:
- увеличить дефолтный `option_count` модели так, чтобы он покрывал supported curriculum action options
- не обрезать mask, потому что это скрывает легальные spell/item/weapon options
- добавить regression test:
  - модель из `scripts/train_ppo.py build_model("gnn")` должна покрывать максимальный `option_index` среди всех curriculum stages
- проверить smoke-запуск curriculum level с Wizard

## 81. Запрос: Исправить full training без завершённых эпизодов и resume checkpoint

Проблемы:
- при `num_envs=16`, `rollout_steps=1024`, `max_episode_steps=256` в full training были:
  - `episodes_finished=0`
  - `timeouts=0`
  - деградация reward/entropy
- причина:
  - `episode_steps` хранился локально внутри `collect_rollout`
  - между PPO update-ами счётчик шагов эпизода сбрасывался
  - env никогда не доходил до timeout, если один rollout давал меньше шагов на env, чем `max_episode_steps`
- вторая проблема:
  - `scripts/train_ppo.py` создавал новую модель и не загружал существующий checkpoint перед продолжением обучения

Требования:
- хранить `episode_steps_by_env` как состояние `PPOTrainer`
- сохранять счётчики между `collect_rollout` вызовами
- сбрасывать счётчик только при завершении/timeout/reset конкретного env
- добавить `--no-resume` для явного старта с новой модели
- по умолчанию пытаться загрузить совместимый checkpoint
- восстанавливать model state, optimizer state и curriculum level/state из checkpoint
- если checkpoint несовместим, стартовать fresh и явно указать статус в логе
- добавить regression tests для persistent timeout и checkpoint resume

## 82. Запрос: Обновить README и ROADMAP под актуальное обучение и GUI

Требования:
- восстановить/обновить `README.md`, если файл отсутствует или помечен как удалённый
- переписать `ROADMAP.md` в нормальной русской кодировке
- зафиксировать актуальное состояние проекта:
  - PySide6 GUI для персонажей, боёв, карт и реплеев
  - обучение модели только через CLI, без запуска обучения из GUI
  - фиксированная GNN PPO policy для GUI inference
  - curriculum training, fast warm-up, multi-env rollouts и checkpoint resume
  - persistent `max_episode_steps` timeouts между PPO update-ами
- добавить рекомендуемые команды:
  - быстрый warm-up с `--fast-action-masks`, `--fast-observation`, `--num-envs`, `--no-resume`
  - полное дообучение без fast-флагов
- пояснить:
  - `--num-envs`
  - `--no-resume`
  - `checkpoint_status`
  - что `win_rate` относится к победам команды игроков
  - что fast-режимы не должны заменять финальное обучение
- обновить roadmap по направлениям:
  - GUI
  - combat/rules
  - training/evaluation
  - performance
  - data storage

## 83. Запрос: Исправить обучение wizard/elemental stage и подключить slot level

Проблемы:
- на curriculum level 8 `Wizard vs FireElementalSimple` win_rate падал почти до нуля
- spellcasting в целом работал, но:
  - FireElementalSimple имеет `FIRE immunity`
  - Fireball, Fire Bolt, Scorching Ray наносили 0 урона
  - Magic Missile и Ray of Frost наносили урон
- `GNNPPOActorCritic` имел `slot_level_head`, но:
  - `decode_action(...)` не принимал `slot_level`
  - `PPOTrainer._decode_model_action(...)` отбрасывал `slot_level`
  - upcast через PPO/GNN фактически не использовался
- общий `option_index` пересекался между main spell, reaction spell, item/weapon options, поэтому одна mask не могла идеально запретить Fire Bolt, если Shield reaction использовал тот же index

Требования:
- добавить `slot_level` mask в `build_action_masks`
- передавать `slot_level` из PPO/GNN output в `decode_action`
- передавать `cast_level` в `CastSpellAction`
- для spellcasting декодера:
  - если выбранный spell option бесполезен из-за immunity и есть Force/Cold альтернатива, выбирать первый валидный spell option
  - если выбранная цель/клетка/направление невалидны, выбирать первый валидный вариант для выбранного spell
- обновить GUI/evaluation inference decode, чтобы они тоже передавали `slot_level`
- смягчить curriculum level 8:
  - вместо одиночного Wizard vs FireElementalSimple использовать Wizard + Fighter vs FireElementalSimple
  - evaluation scenario оставить жёстким
- добавить regression tests:
  - Magic Missile декодируется с upcast slot level 3
  - fire spell против FireElementalSimple remap-ится на Ray of Frost или Magic Missile при наличии альтернативы
- прогнать полный pytest
