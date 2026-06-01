# D&D Battle AI

Python-проект для D&D-like 5e tactical combat simulator с RL/PPO/GNN-инфраструктурой и desktop GUI на PySide6.

Цель проекта - корректная работа боевого AI в пошаговых тактических боях, а не полная цифровая копия D&D. Правила реализуют ограниченное подмножество D&D-like 5e: уровни 1-5, классы Fighter/Cleric/Wizard, базовые расы, common combat actions, карты, spellcasting, ресурсы, предметы, реплеи и evaluation/training scripts.

## Структура

```text
src/
  agents/      observation encoders, action space, PPO/GNN models, baselines
  combat/      combat engine, actions, maps, rewards, replay, presets
  rules/       ruleset registry, classes, races, feats, progression
  training/    PPO, multi-agent, curriculum, self-play
  ui/          PySide6 desktop GUI
configs/       ruleset and training configs
data/
  characters/  saved GUI characters
maps/          JSON map configs
replays/       saved battle replays
checkpoints/   trained model checkpoints
scripts/       CLI entry points
```

## Установка

```bash
python -m venv .venv
pip install -r requirements.txt
```

На Windows из локального venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Запуск GUI

```bash
python scripts/run_gui.py
```

GUI сделан как отдельный PySide6-слой поверх существующего combat engine. В GUI можно:

- просматривать сохранённых персонажей;
- создавать и редактировать персонажей через ruleset-aware character builder;
- запускать случайный бой;
- настраивать кастомный бой и выбирать карту;
- вручную расставлять персонажей и врагов в рамках custom battle flow по мере развития редактора боя;
- смотреть пошаговый бой на клеточной карте;
- вручную управлять персонажами в поддержанном режиме боя;
- смотреть сохранённые реплеи;
- использовать преднастроенную политику PPO Actor-Critic с GNN encoder;
- автоматически переходить на внутренний fallback agent, если фиксированный checkpoint недоступен.

GUI не даёт пользователю выбирать `model_type`, checkpoint, fallback agent и служебные папки, чтобы демонстрационный режим не ломался из-за несовместимой конфигурации.

Обучение модели НЕ запускается из GUI. Обучение остаётся в существующих CLI scripts, например:

```bash
python scripts/train_ppo.py --episodes 100 --seed 0 --checkpoint checkpoints/ppo_actor_critic.pt
```

## Важные папки

- `data/characters/` - JSON-файлы персонажей, созданных в GUI.
- `replays/` - BattleReplay JSON, сохранённые из демо или GUI.
- `maps/` - JSON-конфигурации карт: terrain grid и spawn zones.
- `checkpoints/` - сохранённые PPO/GNN checkpoints для инференса и обучения.

## Карты

Карты задаются JSON-файлами в `maps/`. Конфиг содержит:

- `name`
- `width`
- `height`
- `terrain_grid`
- `spawn_zones.players`
- `spawn_zones.enemies`

Поддержанные terrain values:

- `NORMAL`
- `DIFFICULT_TERRAIN`
- `BLOCKED`
- `LOW_COVER`
- `HIGH_COVER`

В комплекте есть:

- `open_field.json`
- `cover_arena.json`
- `difficult_terrain_pass.json`
- `obstacle_corridor.json`

## Ruleset

Активный ruleset: `srd5e_minimal_2014`.

Поддержано:

- уровни 1-5;
- классы: Fighter, Cleric, Wizard;
- подклассы: Champion, Life Domain, School of Evocation;
- расы: Human, Dwarf, Elf, Halfling;
- spell levels 0-3;
- common actions: Attack, CastSpell, Dash, Disengage, Dodge, Help, Hide, Search, UseObject, Ready, Grapple, Shove, Stabilize, EndTurn.

Классы добавляют features/resources, но не владеют общей логикой Attack/Dash/Dodge/etc. Базовые боевые действия реализованы отдельно от классов. Сложные бонусные действия классов и расширенный контент добавляются постепенно.

## CLI

Тестовый demo:

```bash
python scripts/run_demo.py
```

Сохранить replay из demo:

```bash
python scripts/run_demo.py --save-replay
```

Консольный просмотр replay:

```bash
python scripts/view_replay_console.py replays/example.json
```

Оценка политики:

```bash
python scripts/evaluate_policy.py --by-level
```

## Тесты

```bash
python -m pytest
```

На Windows из локального venv:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Текущие ограничения

- Ruleset намеренно минимальный.
- GUI не запускает обучение.
- Custom battle setup уже использует JSON-карты и preview, но полноценный drag-and-drop редактор боя ещё в roadmap.
- Предметы и improvised actions реализованы упрощённо.
- Некоторые class features/spells сохранены как not implemented и не попадают в action masks.

## Roadmap

См. [ROADMAP.md](ROADMAP.md).
