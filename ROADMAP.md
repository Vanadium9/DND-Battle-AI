# Roadmap проекта

Roadmap описывает текущее состояние D&D Battle AI и ближайшие направления развития.

## Текущее состояние

- Combat engine с action economy, initiative order, common D&D-like actions и пошаговым `CombatEnvironment`.
- Ruleset registry `srd5e_minimal_2014` для уровней 1-5.
- Data-driven классы, подклассы, расы, feats/ASI, spellcasting progression и inventory.
- Карты через JSON-конфиги в `maps/`: terrain grid, cover, obstacles, spawn zones.
- Replay JSON format и просмотр реплеев.
- PPO/GNN модели, action masks, entity-based observations, baseline agents, curriculum/self-play заготовки.
- PySide6 desktop GUI поверх существующего combat engine.

## GUI

Реализовано:

- character browser;
- character builder;
- random battle launcher;
- custom battle setup;
- battle viewer с клеточной картой;
- manual action controls для поддержанного режима;
- replay viewer;
- settings screen для безопасных UI-настроек и read-only статуса фиксированной GNN PPO политики;
- map preview для JSON-карт;
- status bar, loading/error/empty states и confirmation dialogs.

GUI используется для демонстрации, запуска боёв, просмотра реплеев и инференса обученной политики. Обучение модели в GUI не запускается.

## Бой и правила

Реализовано:

- Attack, Cast Spell, Dash, Disengage, Dodge, Help, Hide, Search, Use Object, Ready, Grapple, Shove, Stabilize, Opportunity Attack, End Turn.
- Action, bonus action, reaction, movement, free object interaction.
- Conditions/flags: prone, grappled, hidden, dodging, disengaged, prepared action, concentration.
- Damage types, resistances, immunities, vulnerabilities.
- Cover, line of sight, difficult terrain and blocked cells.
- Fighter 1-5 Champion, Cleric 1-5 Life Domain, Wizard 1-5 Evocation.

## Training и Evaluation

Реализовано или заложено:

- PPO trainer;
- MLP PPO model;
- GNN PPO model;
- centralized critic flag;
- multi-agent policies;
- self-play opponent pool;
- curriculum config;
- evaluation scenarios by level;
- rule-based baseline agents.

## Будущие улучшения GUI

- drag-and-drop в редакторе боя;
- ручное размещение party/enemies с сохранением custom encounter;
- портреты токенов;
- более красивые анимации атак, заклинаний и перемещения;
- полноценный GUI редактор карт;
- расширенный character builder;
- быстрый импорт/экспорт внутренних JSON-персонажей;
- фильтры и поиск по персонажам, реплеям и картам.

## Будущие улучшения правил

- поддержка большего числа классов, рас, подклассов и заклинаний;
- больше feats и class-specific bonus actions/reactions;
- более точные spell areas, saving throws и concentration effects;
- monster abilities beyond common actions;
- более богатые item effects и battlefield objects;
- расширенная система условий.

## Будущие улучшения AI

- стабилизация action space compatibility между checkpoint версиями;
- расширенные reward breakdown dashboards;
- более сильные rule-based baselines;
- evaluation reports по уровням, картам и ролям;
- tuning curriculum/self-play thresholds;
- улучшение GNN encoder и centralized critic для партий разного размера.

## Удалено/заменено

- Отдельный simple 2D replay viewer на pygame не используется как основной viewer. Реплеи смотрятся через PySide6 GUI и консольный viewer.
- Внешние importer workflows не являются частью текущего GUI. GUI работает с внутренними JSON-персонажами из `data/characters/`.
