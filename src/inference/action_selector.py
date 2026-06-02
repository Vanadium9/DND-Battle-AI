"""Action selection helpers for inference policies."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from agents import (
    GNNPPOActorCritic,
    PPOActorCritic,
    build_action_masks,
    decode_action,
    encode_entity_observation,
    encode_observation,
)
from agents.rule_based import RuleBasedAgent
from combat import CombatAction, CombatState
from inference.policy_loader import LoadedPolicy


class ActionSelectionError(RuntimeError):
    """Raised when a policy cannot produce a valid combat action."""


class ActionSpaceCompatibilityError(ActionSelectionError):
    """Raised when the checkpoint heads cannot represent legal current actions."""


def select_action_with_policy(
    policy: LoadedPolicy,
    state: CombatState,
    actor_id: int,
    *,
    deterministic: bool = True,
) -> CombatAction:
    """Select and decode one legal action with a loaded PPO policy."""

    model = policy.model
    masks = build_action_masks(state, actor_id)
    if isinstance(model, PPOActorCritic):
        observation = _fit_observation(encode_observation(state, actor_id), model.observation_size)
        fitted_masks = _fit_mlp_masks(masks, model)
    elif isinstance(model, GNNPPOActorCritic):
        observation = encode_entity_observation(state, actor_id)
        fitted_masks = _fit_gnn_masks(masks, model)
    else:
        raise ActionSelectionError(f"Unsupported loaded model class: {type(model).__name__}")

    with torch.no_grad():
        model_output = model.act(observation, fitted_masks, deterministic=deterministic)
    return decode_policy_output(model_output, state, actor_id)


def select_fallback_action(
    agent: RuleBasedAgent,
    state: CombatState,
    actor_id: int,
    *,
    deterministic: bool = True,
) -> CombatAction:
    """Select and decode one action with a mask-aware fallback agent."""

    actor = state.character_at(actor_id)
    masks = build_action_masks(state, actor_id)
    model_output = agent.act(
        None,
        masks,
        state=state,
        actor_id=actor_id,
        actor=actor,
        deterministic=deterministic,
    )
    return decode_policy_output(model_output, state, actor_id)


def decode_policy_output(
    model_output: Mapping[str, Any],
    state: CombatState,
    actor_id: int,
) -> CombatAction:
    """Decode tensor or scalar policy output into a concrete CombatAction."""

    try:
        return decode_action(
            _output_int(model_output, "action_category"),
            _output_int(model_output, "main_action_type", default=0),
            _output_int(model_output, "target_index", default=0),
            _output_int(model_output, "move_index", default=0),
            _output_int(model_output, "option_index", default=0),
            state,
            actor_id,
            target_cell_index=_optional_output_int(model_output, "target_cell_index"),
            direction_index=_optional_output_int(model_output, "direction_index"),
            slot_level=_optional_output_int(model_output, "slot_level"),
        )
    except ValueError as error:
        raise ActionSelectionError(f"Policy selected an illegal action: {error}") from error


def _fit_mlp_masks(
    masks: Mapping[str, torch.Tensor],
    model: PPOActorCritic,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": _fit_mask(
            masks["action_category"],
            model.action_category_count,
            "action_category",
        ),
        "main_action_type": _fit_mask(
            masks["main_action_type"],
            model.main_action_type_count,
            "main_action_type",
        ),
        "target_index": _fit_mask(masks["target_index"], model.target_count, "target_index"),
        "move_index": _fit_mask(masks["move_index"], model.move_count, "move_index"),
        "option_index": _fit_mask(masks["option_index"], model.option_count, "option_index"),
    }


def _fit_gnn_masks(
    masks: Mapping[str, torch.Tensor],
    model: GNNPPOActorCritic,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": _fit_mask(
            masks["action_category"],
            model.action_category_count,
            "action_category",
        ),
        "main_action_type": _fit_mask(
            masks["main_action_type"],
            model.main_action_type_count,
            "main_action_type",
        ),
        "target_index": _fit_mask(masks["target_index"], model.target_count, "target_index"),
        "move_index": _fit_mask(masks["move_index"], model.move_count, "move_index"),
        "option_index": _fit_mask(masks["option_index"], model.option_count, "option_index"),
    }


def _fit_mask(mask: torch.Tensor, size: int, name: str) -> torch.Tensor:
    prepared = mask.detach().cpu().bool()
    if prepared.ndim != 1:
        raise ActionSpaceCompatibilityError(f"{name} mask must be 1D")
    if prepared.shape[0] > size and prepared[size:].any():
        first_unsupported = int(torch.nonzero(prepared[size:], as_tuple=False)[0].item()) + size
        raise ActionSpaceCompatibilityError(
            f"Model head '{name}' supports {size} choices, "
            f"but current action space allows index {first_unsupported}."
        )
    if prepared.shape[0] == size:
        return prepared
    if prepared.shape[0] > size:
        return prepared[:size]
    padding = torch.zeros(size - prepared.shape[0], dtype=torch.bool)
    return torch.cat((prepared, padding), dim=0)


def _fit_observation(observation: torch.Tensor, expected_size: int) -> torch.Tensor:
    if observation.shape[0] == expected_size:
        return observation
    if observation.shape[0] > expected_size:
        return observation[:expected_size]
    padding = torch.zeros(
        expected_size - observation.shape[0],
        dtype=observation.dtype,
        device=observation.device,
    )
    return torch.cat((observation, padding), dim=0)


def _output_int(
    output: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    if key not in output:
        if default is None:
            raise ActionSelectionError(f"Policy output missing '{key}'")
        return default
    value = output[key]
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _optional_output_int(output: Mapping[str, Any], key: str) -> int | None:
    if key not in output:
        return None
    return _output_int(output, key)


__all__ = [
    "ActionSelectionError",
    "ActionSpaceCompatibilityError",
    "decode_policy_output",
    "select_action_with_policy",
    "select_fallback_action",
]
