# Roadmap Проекта

Этот roadmap описывает следующие крупные направления разработки D&D tactical
combat RL simulator.

## Текущее Состояние

- Core combat entities реализованы через dataclasses.
- Общие D&D combat actions реализованы отдельно от class features.
- Классы добавляют features и resources, но не владеют общей логикой Attack,
  Dash, Dodge, Help, Grapple, Shove и похожих действий.
- Action economy отслеживает actions, bonus actions, reactions, movement, free
  object interaction и временные combat states.
- PPO observation encoding, action masks, model, trainer, rewards, scripts и
  tests уже есть.

## Общие D&D Боевые Действия

Реализованные common actions:

- Attack
- Cast Spell
- Dash
- Disengage
- Dodge
- Help
- Hide
- Search
- Use Object
- Ready
- Grapple
- Shove
- Stabilize
- Improvised Action
- Opportunity Attack
- End Turn

Следующие задачи:

- Добавить более богатые action results и event metadata для reward attribution.
- Улучшить target selection и option encoding для non-attack actions.
- Добавить более точную обработку Ready triggers и reaction windows.

## Action Economy

Реализовано:

- Main action availability
- Bonus action availability
- Reaction availability
- Movement budget
- Free object interaction
- Prone, grappled, hidden, dodging, disengaged, helped и prepared-action states

Следующие задачи:

- Добавить Action Surge как явное class feature action.
- Добавить class и monster abilities, которые изменяют обычную action economy.
- Улучшить round-level reaction reset semantics для раундов с несколькими
  существами.

## Bonus Actions И Reactions

Реализовано:

- Bonus action resource существует и сбрасывается.
- Reaction resource существует и тратится opportunity attacks.
- Ready сохраняет prepared action и trigger description.

Запланировано:

- Fighter Second Wind как bonus action.
- Rogue-style Cunning Action actions.
- Monster-specific bonus actions.
- Reaction triggers beyond opportunity attacks.
- PPO masks для конкретных bonus-action и reaction choices.

Сложные class-specific bonus actions будут добавлены позже. Они должны жить как
class features или feature actions, а логика common actions должна оставаться
общей.

## Items И Improvised Actions

Текущее поведение:

- Use Object тратит action и логирует использование объекта.
- Improvised Action тратит action и логирует описание.
- Оба действия намеренно упрощены и пока не применяют богатые item или
  environment effects.

Запланировано:

- Добавить item definitions и effects.
- Добавить consumables, interactables и простые battlefield objects.
- Добавить improvised-action hooks, которые могут создавать явные combat events.

## Текущие Ограничения

- Это упрощённый D&D-like simulator, а не полноценный rules engine.
- Class features и resources присутствуют, но многие class-specific actions пока
  placeholders.
- Full spellcasting, saving throws, spell slots, concentration, areas of effect
  и condition rules пока не реализованы.
- Maps не имеют obstacles, cover, terrain cost или полноценной line-of-sight
  модели.
- Reward shaping эвристический и должен проверяться в процессе обучения.
- Observations являются fixed vectors; graph/entity-based observations остаются
  будущей задачей.
- Existing checkpoints могут потребовать retraining после изменений observation
  или action space.
