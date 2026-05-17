# D&D Tactical Combat RL Simulator

Python 3.11+ проект для экспериментов с reinforcement learning в небольшом
D&D-like симуляторе тактического боя.

Сейчас проект включает grid-based симулятор боя, общие D&D боевые действия,
action economy, fixed-vector observations, иерархические PPO action masks,
PyTorch actor-critic модель, PPO trainer, demo scripts и pytest-покрытие
ключевого поведения.

## Структура проекта

```text
src/
  agents/      observation encoder, action space, PPO model
  combat/      combat models, actions, rewards, encounters
  configs/     training/config objects
  tests/       pytest suite
  training/    PPO rollout and training logic
scripts/
  run_demo.py
  train_ppo.py
checkpoints/
requirements.txt
README.md
ROADMAP.md
```

## Общие D&D Боевые Действия

Общие боевые действия реализованы отдельно от классов персонажей в
`src/combat/common_actions.py`.

Сейчас реализованы:

- `AttackAction`
- `CastSpellAction`
- `DashAction`
- `DisengageAction`
- `DodgeAction`
- `HelpAction`
- `HideAction`
- `SearchAction`
- `UseObjectAction`
- `ReadyAction`
- `GrappleAction`
- `ShoveAction`
- `StabilizeAction`
- `ImprovisedAction`
- `OpportunityAttackAction`
- `EndTurnAction`

Оружейные атаки принадлежат существам через список `weapons`. Они не являются
классовыми способностями. Эффективность атаки зависит от характеристик,
proficiency bonus, настроек оружия и attack bonus.

## Action Economy

Каждое существо хранит ресурсы хода в `src/combat/action_economy.py`:

- `action_available`
- `bonus_action_available`
- `reaction_available`
- `movement_remaining`
- `free_object_interaction_available`

Основные действия, такие как Attack, Dash, Dodge, Help, Hide, Search, Ready,
Use Object, Grapple, Shove, Stabilize и Improvised Action, тратят
`action_available`. Движение тратит `movement_remaining`. Реакции тратят
`reaction_available`.

Combat state также хранит временные D&D-like состояния: prone, grappled, hidden,
dodging, disengaged, helped targets, prepared actions и использование реакции.

## Классы, Features И Resources

Классы персонажей сейчас моделируются как metadata плюс definitions для
features/resources. Классы добавляют `class_features` и `resources`, но не
владеют общей логикой Attack, Dash, Dodge, Help, Grapple, Shove и других common
actions.

Например, fighter presets могут иметь ресурсы Action Surge и Second Wind в
данных персонажа, но сама реализация common actions остаётся независимой от
класса.

## Bonus Actions И Reactions

Action economy уже резервирует bonus actions и reactions:

- `bonus_action_available` существует и сбрасывается каждый ход.
- `reaction_available` существует и тратится `OpportunityAttackAction`.
- `ReadyAction` сохраняет prepared action и trigger description.

Сложные классовые bonus actions и более богатые reaction triggers будут
добавлены позже. Текущие PPO action masks резервируют категории bonus action и
reaction, но большинство class-specific вариантов намеренно ещё не реализовано.

## Observations, Action Space И PPO

`src/agents/observation.py` кодирует combat state в fixed-size PyTorch tensor.
Encoder включает ресурсы актёра, состояния, доступность common actions,
ближайших союзников, ближайших врагов, дистанции и признаки targetability.

`src/agents/action_space.py` задаёт иерархическое пространство действий:

- action category
- main action type
- target index
- move index
- option index

`src/agents/ppo_model.py` реализует actor-critic сеть с общим encoder,
отдельными policy heads и action masking.

## Rewards

`src/combat/rewards.py` содержит reward shaping для:

- нанесённого и полученного урона
- убийств и смертей
- победы и поражения
- слишком длинных или бесполезных ходов
- тактических common actions, таких как Grapple, Shove, Dodge, Disengage и Help
- штрафов за low-value Dash, Hide, Ready, Use Object и Improvised actions

Тактические награды намеренно маленькие по сравнению с победой, убийством врага
и предотвращением смерти союзника.

## Текущие Ограничения

- Правила упрощены; это не полная реализация D&D 5e.
- Сложные class-specific bonus actions пока не реализованы.
- Spellcasting существует только как простой путь через `SpellAbility`; full
  spell slots, saves, concentration, areas и spell lists пока не реализованы.
- Items и improvised actions являются упрощёнными placeholders без богатых
  эффектов.
- Ready сохраняет prepared action data, но сложное resolution для trigger ещё
  не реализовано.
- Карты пока не моделируют obstacles, terrain cost, cover или полноценный line
  of sight.
- PPO использует fixed-vector observations вместо graph neural networks.
- Existing checkpoints могут стать несовместимыми при изменении observation или
  policy-head sizes.

## Установка

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Запуск Тестов

```bash
python -m pytest
```

В этом workspace через локальное virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Обучение PPO

```bash
python scripts/train_ppo.py --episodes 100 --seed 0 --checkpoint checkpoints/ppo_actor_critic.pt
```

## Запуск Demo

```bash
python scripts/run_demo.py --checkpoint checkpoints/ppo_actor_critic.pt
```

## Roadmap

См. [ROADMAP.md](ROADMAP.md) с планируемыми работами и текущими приоритетами.
