# D&D Battle AI

D&D Battle AI - учебный проект тактического пошагового боя в стиле D&D-like 5e с агентом обучения с подкреплением. Проект реализует ограниченный ruleset `srd5e_minimal_2014`: цель - корректная работа боевого AI, action economy, карты и тактических решений, а не полная цифровая копия D&D.

## Возможности

- Пошаговый `CombatEnvironment` с инициативой, раундами, action/bonus action/reaction и movement.
- Общие боевые действия: Attack, CastSpell, Dash, Disengage, Dodge, Help, Hide, UseObject, Ready, Grapple, Shove, EndTurn.
- Поддержанные классы 1-5 уровней: Fighter, Cleric, Wizard.
- Поддержанные подклассы: Champion, Life Domain, School of Evocation.
- Поддержанные расы: Human, Dwarf, Elf, Halfling.
- Карты с terrain cost, obstacles, cover и line of sight.
- Damage types, resistances, immunities, vulnerabilities.
- Entity-based observation, GNN encoder и PPO Actor-Critic модель.
- Curriculum training, fast warm-up режимы, multi-env rollout batching и checkpoint resume.
- PySide6 desktop GUI для демонстрации боёв, персонажей и реплеев.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если PyTorch был установлен отдельно под CUDA, используйте ту же виртуальную среду. Проверить CUDA можно так:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Запуск GUI

```powershell
.\.venv\Scripts\python.exe scripts\run_gui.py
```

В GUI можно:

- просматривать созданных персонажей;
- создавать и редактировать персонажей;
- запускать случайный бой;
- настраивать кастомный бой;
- вручную расставлять персонажей и врагов;
- смотреть пошаговый бой на клеточной карте;
- вручную управлять персонажами в поддержанном режиме;
- смотреть реплеи;
- использовать обученную GNN PPO policy для инференса.

Обучение модели из GUI не запускается. GUI использует уже существующий checkpoint или fallback agent, если checkpoint недоступен.

## Обучение

Обучение запускается только через CLI-скрипты. Актуальная модель по умолчанию сохраняется в:

```text
checkpoints/gnn_ppo_actor_critic.pt
```

По умолчанию `scripts/train_ppo.py` пытается продолжить совместимый checkpoint. Для свежего старта используйте `--no-resume`.

Быстрый warm-up до сложных сценариев:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py --updates 300 --rollout-steps 1024 --max-episode-steps 128 --minibatch-size 256 --update-epochs 4 --model-type gnn --device auto --num-envs 16 --fast-action-masks --fast-observation --curriculum --curriculum-max-level 6 --curriculum-window-size 80 --curriculum-threshold 0.85 --profile-training --log-interval 5 --no-resume
```

Полное дообучение на более богатых observation/masks:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo.py --updates 150 --rollout-steps 1024 --max-episode-steps 128 --minibatch-size 256 --update-epochs 4 --model-type gnn --device auto --num-envs 16 --curriculum --curriculum-level 7 --curriculum-max-level 13 --log-interval 5
```

Важные параметры:

- `--device auto` использует CUDA, если доступна.
- `--num-envs` задаёт число независимых combat environments в одном rollout tick. Это не отдельные процессы, а способ собирать batch для GPU эффективнее.
- `--max-episode-steps` ограничивает длину боя. Счётчик шагов сохраняется между PPO update-ами, поэтому зависшие бои должны завершаться timeout.
- `--fast-action-masks` и `--fast-observation` ускоряют warm-up, но скрывают часть тактических признаков. Для финального обучения и оценки их лучше отключать.
- `win_rate` считается как доля завершённых побед команды игроков среди завершённых эпизодов. Timeout не считается победой.
- `checkpoint_status` в стартовом логе показывает, был ли checkpoint загружен или обучение началось fresh.

## Оценка

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_policy.py --by-level
```

Evaluation scenarios покрывают уровни 1-5, одиночные и партийные бои, разные карты, врагов с resistances/immunities и сценарии для Fighter, Cleric, Wizard.

## Структура проекта

- `src/combat/` - combat engine, действия, карты, урон, spellcasting.
- `src/rules/` - ruleset registry, progression, classes, subclasses, races, feats, XP.
- `src/agents/` - observation encoders, action space, PPO/GNN models, rule-based agents.
- `src/training/` - PPO trainer, curriculum, multi-agent и self-play заготовки.
- `src/ui/` - PySide6 GUI.
- `src/character/` - внутренний формат персонажей, validation, repository.
- `configs/` - ruleset/training конфиги.
- `maps/` - JSON-карты.
- `data/characters/` - сохранённые персонажи GUI.
- `checkpoints/` - обученные модели.

## Ограничения

- Реализовано ограниченное подмножество D&D-like 5e, а не весь ruleset.
- Базовые боевые действия реализованы отдельно от классов. Классы добавляют features/resources, но не владеют общей логикой Attack/Dash/Dodge/etc.
- Сложные bonus actions/reactions классов и многие spell interactions пока упрощены или отмечены как not implemented.
- Предметы и импровизированные действия реализованы упрощённо.
- Расовые особенности есть в rules/character system, но training curriculum пока в основном использует class-focused presets.
- GUI предназначен для демонстрации, инференса и просмотра боёв; обучение остаётся CLI-only.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest
```
