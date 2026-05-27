import torch

from agents import ActionCategory, PPOActorCritic
from agents.entity_observation import COMBAT_ROLE_NAMES, combat_role_name
from combat import CombatEnvironment, FighterArcher, Goblin, GridMap, Position, Team
from configs import PPOConfig
from training import CombatRole, PPOTrainer, RandomPolicy, role_id_for_actor


class RecordingEndTurnPolicy(RandomPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, Team]] = []

    def act(self, observation, masks, **kwargs):
        actor = kwargs["actor"]
        self.calls.append((kwargs["actor_id"], actor.team))
        return {
            "action_category": torch.tensor(int(ActionCategory.END_TURN)),
            "main_action_type": torch.tensor(0),
            "target_index": torch.tensor(0),
            "move_index": torch.tensor(0),
            "option_index": torch.tensor(0),
            "log_prob": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "value": torch.tensor(0.0),
        }


def _trainer(
    *,
    player_policy=None,
    enemy_policy=None,
    shared_policy=None,
) -> PPOTrainer:
    environment = CombatEnvironment(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
        use_initiative=False,
        log_to_console=False,
    )
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(16,))
    config = PPOConfig(
        rollout_steps=2,
        update_epochs=1,
        minibatch_size=2,
    )
    return PPOTrainer(
        environment=environment,
        model=model,
        config=config,
        player_policy=player_policy,
        enemy_policy=enemy_policy,
        shared_policy=shared_policy,
    )


def test_player_policy_controls_only_players() -> None:
    player_policy = RecordingEndTurnPolicy()
    enemy_policy = RecordingEndTurnPolicy()
    trainer = _trainer(player_policy=player_policy, enemy_policy=enemy_policy)

    trainer.collect_episode(max_steps=4)

    assert player_policy.calls
    assert all(team is Team.PLAYERS for _, team in player_policy.calls)


def test_enemy_policy_controls_only_enemies() -> None:
    player_policy = RecordingEndTurnPolicy()
    enemy_policy = RecordingEndTurnPolicy()
    trainer = _trainer(player_policy=player_policy, enemy_policy=enemy_policy)

    trainer.collect_episode(max_steps=4)

    assert enemy_policy.calls
    assert all(team is Team.ENEMIES for _, team in enemy_policy.calls)


def test_shared_policy_controls_all_creatures() -> None:
    shared_policy = RecordingEndTurnPolicy()
    trainer = _trainer(shared_policy=shared_policy)

    trainer.collect_episode(max_steps=4)

    controlled_teams = {team for _, team in shared_policy.calls}
    assert controlled_teams == {Team.PLAYERS, Team.ENEMIES}


def test_role_id_uses_supported_combat_roles() -> None:
    archer = FighterArcher(Position(0, 0))
    goblin = Goblin(Position(1, 0))

    assert combat_role_name(archer) == CombatRole.RANGED_DAMAGE.value
    assert 1 <= role_id_for_actor(goblin) <= len(COMBAT_ROLE_NAMES)
