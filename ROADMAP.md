# Roadmap проекта

Roadmap фиксирует текущее состояние D&D Battle AI и ближайшие направления развития. Проект развивается как ограниченный D&D-like 5e combat simulator для обучения и демонстрации тактического AI.

## Текущее состояние

- Combat engine с инициативой, action economy, common actions и пошаговым `CombatEnvironment`.
- Ruleset registry `srd5e_minimal_2014` для уровней 1-5.
- Data-driven классы, подклассы, расы, ASI/feats, spellcasting progression и inventory.
- Fighter 1-5 Champion, Cleric 1-5 Life Domain, Wizard 1-5 School of Evocation.
- Базовые враги: Goblin, Orc, Skeleton, Bandit, Wolf, FireElementalSimple.
- Карты через JSON-конфиги в `maps/`: terrain grid, cover, obstacles, spawn zones.
- PPO/GNN модель, action masks, entity-based observations, GNN encoder.
- Curriculum training с уровнями сложности, multi-env rollouts, fast warm-up режимами и checkpoint resume.
- PySide6 desktop GUI поверх существующего combat engine.

## GUI

Реализовано:

- character browser;
- character builder;
- random battle launcher;
- custom battle setup;
- battle viewer с клеточной картой;
- manual action controls для поддержанного режима;
- settings screen с безопасными UI-настройками и read-only статусом фиксированной GNN PPO policy;
- map preview для JSON-карт;
- status bar, loading states, empty states, error dialogs и confirmation dialogs.

GUI используется для демонстрации, запуска боёв, просмотра реплеев и инференса обученной policy. Обучение модели в GUI не запускается.

Ближайшие улучшения GUI:

- drag-and-drop в редакторе боя;
- более удобная ручная расстановка party/enemies;
- портреты токенов;
- более красивые анимации атак, заклинаний и перемещения;
- полноценный GUI-редактор карт;
- расширенный character builder;
- фильтры и поиск по персонажам, реплеям и картам.

## Бой и правила

Реализовано:

- Attack, CastSpell, Dash, Disengage, Dodge, Help, Hide, UseObject, Ready, Grapple, Shove, Opportunity Attack, EndTurn.
- Action, bonus action, reaction, movement, free object interaction.
- Conditions/flags: prone, grappled, hidden, dodging, disengaged, prepared action, concentration.
- Ability checks, contested checks и логирование бросков.
- Damage types, resistances, immunities, vulnerabilities.
- Cover, line of sight, difficult terrain and blocked cells.
- AoE targeting: radius, cone, line.
- Spell slots, cantrips, prepared spells и базовый upcast для healing/damage.

Будущие улучшения правил:

- больше классов, рас, подклассов и заклинаний;
- больше feats и class-specific bonus actions/reactions;
- более точные spell areas, saving throws и concentration effects;
- monster abilities beyond common actions;
- более богатые item effects и battlefield objects;
- расширенная система условий;
- освещение, darkvision и visibility beyond current line of sight.

## Training и Evaluation

Реализовано:

- MLP PPO model;
- GNN PPO Actor-Critic model;
- centralized critic flag;
- PPO trainer с rollout collection и action masking;
- multi-env rollout batching через `--num-envs`;
- persistent `max_episode_steps` timeout между PPO update-ами;
- checkpoint resume по умолчанию и `--no-resume` для fresh start;
- curriculum stages для уровней 1-5 и разных типов encounter;
- fast warm-up режимы `--fast-action-masks` и `--fast-observation`;
- profiling training output: observation, masks, model_act, decode, env_step, update;
- evaluation scenarios by level;
- rule-based baseline agents;
- self-play opponent pool заготовка.

Ближайшие улучшения обучения:

- side-aware CLI режимы: обучать только players, только enemies или обе стороны;
- отдельные player/enemy policy presets для `train_ppo.py`;
- rule-based enemy baseline как стабильный opponent для начального обучения;
- rewards, явно разделённые по стороне боя;
- curriculum gating с минимальным числом update-ов на stage, plateau detection и per-stage metrics;
- более строгая checkpoint metadata/action-space compatibility;
- отдельный `--no-save` или smoke checkpoint mode, чтобы тестовые запуски не перезаписывали основную модель;
- ускорение full observation и full action masks без потери тактических признаков;
- расширенные evaluation reports по уровням, картам, ролям, ресурсам и enemy types.

## Производительность

Текущая стратегия:

- GPU используется для GNN/PPO forward pass и optimization.
- Combat simulation, rules, masks и environment step остаются CPU-bound.
- `--num-envs` повышает загрузку GPU за счёт батчинга нескольких независимых боёв.
- Fast warm-up временно упрощает observation/masks, чтобы быстрее получить базовую policy.
- Финальное дообучение должно идти без fast-флагов, чтобы модель училась на полном состоянии.

Следующие направления оптимизации:

- уменьшить стоимость action mask generation;
- кэшировать статичные признаки карты, ruleset и entity templates;
- ускорить observation encoder для полного режима;
- добавить более дешёвые scripted rollout opponents на ранних curriculum stages;
- профилировать самые дорогие ветки spellcasting/map/line-of-sight.

## Данные и хранение

Используемые папки:

- `data/characters/` - созданные GUI персонажи во внутреннем JSON-формате.
- `maps/` - JSON-конфиги карт.
- `checkpoints/` - обученные модели.
- `configs/` - ruleset и training configs.

Внешние importer workflows сейчас не являются частью GUI. GUI работает с внутренними JSON-персонажами.

## Удалено или заменено

- Упоминания внешнего LongStoryShort importer не относятся к текущему GUI и внутреннему формату персонажей.
