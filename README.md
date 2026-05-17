# D&D Tactical Combat RL Simulator

Python 3.11+ project for experimenting with reinforcement learning in a small
D&D-like tactical combat environment.

The project currently includes a grid-based combat simulator, common D&D combat
actions, action economy, fixed-vector observations, hierarchical PPO action
masks, a PyTorch actor-critic model, a PPO trainer, demo scripts, and pytest
coverage for core behavior.

## Project Structure

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

## Common D&D Combat Actions

Common combat actions are implemented separately from character classes in
`src/combat/common_actions.py`.

Implemented common actions include:

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

Weapon attacks belong to creatures through their `weapons` list. They are not
class features. Attack effectiveness depends on stats, proficiency bonus,
weapon configuration, and attack bonus.

## Action Economy

Each creature tracks turn resources in `src/combat/action_economy.py`:

- `action_available`
- `bonus_action_available`
- `reaction_available`
- `movement_remaining`
- `free_object_interaction_available`

Main actions such as Attack, Dash, Dodge, Help, Hide, Search, Ready, Use Object,
Grapple, Shove, Stabilize, and Improvised Action spend `action_available`.
Movement spends `movement_remaining`. Reactions spend `reaction_available`.

Combat state also tracks temporary D&D-like states such as prone, grappled,
hidden, dodging, disengaged, helped targets, prepared actions, and reaction use.

## Classes, Features, And Resources

Character classes are modeled as metadata plus feature/resource definitions.
Classes add `class_features` and `resources`, but they do not own the shared
logic for Attack, Dash, Dodge, Help, Grapple, Shove, and other common actions.

For example, Fighter presets can have resources such as Action Surge and Second
Wind reserved in their character data, while the common action implementation
remains class-agnostic.

## Bonus Actions And Reactions

The action economy already reserves bonus actions and reactions:

- `bonus_action_available` exists and resets each turn.
- `reaction_available` exists and is spent by `OpportunityAttackAction`.
- `ReadyAction` stores a prepared action and trigger description.

Complex class-specific bonus actions and richer reaction triggers are planned
for later. Current PPO action masks reserve bonus-action and reaction categories,
but most class-specific options are intentionally not implemented yet.

## Observations, Action Space, And PPO

`src/agents/observation.py` encodes combat state into a fixed-size PyTorch
tensor. The encoder includes actor resources, statuses, common-action
availability, nearby allies, nearby enemies, distances, and targetability
features.

`src/agents/action_space.py` defines a hierarchical action space:

- action category
- main action type
- target index
- move index
- option index

`src/agents/ppo_model.py` implements a shared-encoder actor-critic network with
separate policy heads and action masking.

## Rewards

`src/combat/rewards.py` includes reward shaping for:

- damage dealt and taken
- kills and deaths
- victory and defeat
- long or useless turns
- tactical common actions such as Grapple, Shove, Dodge, Disengage, and Help
- penalties for low-value Dash, Hide, Ready, Use Object, and Improvised actions

Tactical action rewards are intentionally small compared with victory, enemy
kills, and preventing allied deaths.

## Current Limitations

- Rules are simplified and are not a complete D&D 5e implementation.
- Complex class-specific bonus actions are not implemented yet.
- Spellcasting exists only as a simple `SpellAbility` path; full spell slots,
  saves, concentration, areas, and spell lists are not implemented.
- Items and improvised actions are simplified placeholders without rich effects.
- Ready stores prepared action data, but complex trigger resolution is not
  implemented.
- Maps do not yet model obstacles, terrain cost, cover, or full line of sight.
- PPO uses fixed-vector observations rather than graph neural networks.
- Existing checkpoints can become incompatible when observation or policy-head
  sizes change.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest
```

On this workspace, using the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Train PPO

```bash
python scripts/train_ppo.py --episodes 100 --seed 0 --checkpoint checkpoints/ppo_actor_critic.pt
```

## Run Demo

```bash
python scripts/run_demo.py --checkpoint checkpoints/ppo_actor_critic.pt
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned work and current priorities.
