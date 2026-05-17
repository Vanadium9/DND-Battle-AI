# Project Roadmap

This roadmap tracks the next major development areas for the D&D tactical combat
RL simulator.

## Current State

- Core combat entities are implemented with dataclasses.
- Common D&D combat actions are implemented separately from class features.
- Classes add features and resources, but they do not own shared logic for
  Attack, Dash, Dodge, Help, Grapple, Shove, and similar actions.
- Action economy tracks actions, bonus actions, reactions, movement, free object
  interaction, and temporary combat states.
- PPO observation encoding, action masks, model, trainer, rewards, scripts, and
  tests are in place.

## Common D&D Combat Actions

Implemented common actions:

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

Next work:

- Add richer action results and event metadata for reward attribution.
- Improve target selection and option encoding for non-attack actions.
- Add more precise handling for Ready triggers and reaction windows.

## Action Economy

Implemented:

- Main action availability
- Bonus action availability
- Reaction availability
- Movement budget
- Free object interaction
- Prone, grappled, hidden, dodging, disengaged, helped, and prepared-action
  states

Next work:

- Add Action Surge as an explicit class feature action.
- Add class and monster abilities that alter the normal action economy.
- Add better round-level reaction reset semantics for multi-creature rounds.

## Bonus Actions And Reactions

Implemented:

- Bonus action resource exists and resets.
- Reaction resource exists and is spent by opportunity attacks.
- Ready stores a prepared action and trigger description.

Planned:

- Fighter Second Wind as a bonus action.
- Rogue-style Cunning Action actions.
- Monster-specific bonus actions.
- Reaction triggers beyond opportunity attacks.
- PPO masks for concrete bonus-action and reaction choices.

Complex class-specific bonus actions will be added later. They should live as
class features or feature actions, while common action logic remains shared.

## Items And Improvised Actions

Current behavior:

- Use Object spends an action and logs the object use.
- Improvised Action spends an action and logs a description.
- Both are intentionally simplified and do not yet apply rich item or
  environment effects.

Planned:

- Add item definitions and effects.
- Add consumables, interactables, and simple battlefield objects.
- Add improvised-action hooks that can produce explicit combat events.

## Current Limitations

- This is a simplified D&D-like simulator, not a complete rules engine.
- Class features and resources are present, but many class-specific actions are
  still placeholders.
- Full spellcasting, saving throws, spell slots, concentration, areas of effect,
  and condition rules are not implemented.
- Maps have no obstacles, cover, terrain cost, or full line-of-sight model.
- Reward shaping is heuristic and should be validated during training.
- Observations are fixed vectors; graph/entity-based observations are future
  work.
- Existing checkpoints may need retraining after observation or action-space
  changes.
