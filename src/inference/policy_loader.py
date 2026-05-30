"""Checkpoint loading for GUI inference policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agents import (
    ACTION_CATEGORY_COUNT,
    ACTOR_FEATURE_SIZE,
    MAIN_ACTION_TYPE_COUNT,
    DEFAULT_BONUS_ACTION_TYPE_COUNT,
    DEFAULT_CLASS_FEATURE_COUNT,
    DEFAULT_GNN_POLICY_HIDDEN_SIZE,
    DEFAULT_ITEM_COUNT,
    DEFAULT_MOVE_COUNT,
    DEFAULT_OPTION_COUNT,
    DEFAULT_REACTION_TYPE_COUNT,
    DEFAULT_SLOT_LEVEL_COUNT,
    DEFAULT_SPELL_COUNT,
    DEFAULT_TARGET_COUNT,
    GNNPPOActorCritic,
    PPOActorCritic,
)


SUPPORTED_MODEL_TYPES = {"mlp", "gnn"}


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be loaded for inference."""


class PolicyCompatibilityError(CheckpointLoadError):
    """Raised when a checkpoint does not match the requested model type."""


@dataclass(frozen=True)
class LoadedPolicy:
    """A model loaded from disk with metadata needed by inference."""

    model: nn.Module
    model_type: str
    checkpoint_path: Path

    @property
    def policy_name(self) -> str:
        label = "GNN PPO" if self.model_type == "gnn" else "MLP PPO"
        return f"{label}: {self.checkpoint_path.name}"


def load_policy_checkpoint(
    path: str | Path,
    model_type: str = "mlp",
    *,
    device: str | torch.device = "cpu",
) -> LoadedPolicy:
    """Load an MLP or GNN PPO checkpoint for inference."""

    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise CheckpointLoadError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise CheckpointLoadError(f"Checkpoint path is not a file: {checkpoint_path}")

    normalized_model_type = _normalize_model_type(model_type)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=False,
        )
    except Exception as error:  # pragma: no cover - torch error text is platform-specific
        raise CheckpointLoadError(
            f"Failed to load checkpoint {checkpoint_path}: {error}"
        ) from error

    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    _validate_model_type_matches_state_dict(state_dict, normalized_model_type, checkpoint_path)

    model = (
        _build_gnn_model_from_state_dict(state_dict)
        if normalized_model_type == "gnn"
        else _build_mlp_model_from_state_dict(state_dict)
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise PolicyCompatibilityError(
            f"Checkpoint is incompatible with {normalized_model_type} model: {error}"
        ) from error
    model.to(device)
    model.eval()
    return LoadedPolicy(
        model=model,
        model_type=normalized_model_type,
        checkpoint_path=checkpoint_path,
    )


def _normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise PolicyCompatibilityError("model_type must be 'mlp' or 'gnn'")
    return normalized


def _extract_state_dict(checkpoint: Any, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        candidate = checkpoint.get("model_state_dict", checkpoint)
    else:
        candidate = checkpoint
    if not isinstance(candidate, dict) or not candidate:
        raise CheckpointLoadError(f"Unsupported checkpoint format: {checkpoint_path}")
    if not all(torch.is_tensor(value) for value in candidate.values()):
        raise CheckpointLoadError(f"Checkpoint has no tensor model_state_dict: {checkpoint_path}")
    return candidate


def _validate_model_type_matches_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
    checkpoint_path: Path,
) -> None:
    looks_like_gnn = any(key.startswith("gnn_encoder.") for key in state_dict)
    if model_type == "gnn" and not looks_like_gnn:
        raise PolicyCompatibilityError(
            f"Checkpoint does not contain GNN policy weights: {checkpoint_path}"
        )
    if model_type == "mlp" and looks_like_gnn:
        raise PolicyCompatibilityError(
            f"Checkpoint contains GNN policy weights, but model_type='mlp': {checkpoint_path}"
        )


def _build_mlp_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> PPOActorCritic:
    return PPOActorCritic(
        observation_size=_infer_mlp_observation_size(state_dict),
        target_count=_head_rows(state_dict, "target_head.weight", DEFAULT_TARGET_COUNT),
        move_count=_head_rows(state_dict, "move_head.weight", DEFAULT_MOVE_COUNT),
        option_count=_head_rows(state_dict, "option_head.weight", DEFAULT_OPTION_COUNT),
        action_category_count=_head_rows(
            state_dict,
            "action_category_head.weight",
            ACTION_CATEGORY_COUNT,
        ),
        main_action_type_count=_head_rows(
            state_dict,
            "main_action_type_head.weight",
            MAIN_ACTION_TYPE_COUNT,
        ),
        hidden_sizes=_infer_mlp_hidden_sizes(state_dict),
    )


def _build_gnn_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> GNNPPOActorCritic:
    return GNNPPOActorCritic(
        actor_feature_size=ACTOR_FEATURE_SIZE,
        entity_feature_size=_gnn_input_columns(
            state_dict,
            "gnn_encoder.node_projection.0.weight",
        ),
        map_feature_size=_gnn_input_columns(
            state_dict,
            "map_projection.0.weight",
        ),
        global_feature_size=_gnn_input_columns(
            state_dict,
            "global_projection.0.weight",
        ),
        target_count=_head_rows(state_dict, "target_head.weight", DEFAULT_TARGET_COUNT),
        move_count=_head_rows(state_dict, "move_head.weight", DEFAULT_MOVE_COUNT),
        option_count=_head_rows(state_dict, "option_head.weight", DEFAULT_OPTION_COUNT),
        action_category_count=_head_rows(
            state_dict,
            "action_category_head.weight",
            ACTION_CATEGORY_COUNT,
        ),
        main_action_type_count=_head_rows(
            state_dict,
            "main_action_type_head.weight",
            MAIN_ACTION_TYPE_COUNT,
        ),
        bonus_action_type_count=_head_rows(
            state_dict,
            "bonus_action_type_head.weight",
            DEFAULT_BONUS_ACTION_TYPE_COUNT,
        ),
        reaction_type_count=_head_rows(
            state_dict,
            "reaction_type_head.weight",
            DEFAULT_REACTION_TYPE_COUNT,
        ),
        class_feature_count=_head_rows(
            state_dict,
            "class_feature_head.weight",
            DEFAULT_CLASS_FEATURE_COUNT,
        ),
        spell_count=_head_rows(state_dict, "spell_head.weight", DEFAULT_SPELL_COUNT),
        slot_level_count=_head_rows(
            state_dict,
            "slot_level_head.weight",
            DEFAULT_SLOT_LEVEL_COUNT,
        ),
        item_count=_head_rows(state_dict, "item_head.weight", DEFAULT_ITEM_COUNT),
        gnn_hidden_size=_gnn_hidden_size(state_dict),
        policy_hidden_size=_policy_hidden_size(state_dict),
        message_passing_steps=2,
        context_hidden_sizes=_context_hidden_sizes(state_dict),
    )


def _infer_mlp_observation_size(state_dict: dict[str, torch.Tensor]) -> int:
    if "encoder.0.weight" in state_dict:
        return int(state_dict["encoder.0.weight"].shape[1])
    for head_name in (
        "action_category_head.weight",
        "main_action_type_head.weight",
        "target_head.weight",
    ):
        if head_name in state_dict:
            return int(state_dict[head_name].shape[1])
    raise PolicyCompatibilityError("Cannot infer MLP observation size from checkpoint")


def _infer_mlp_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    hidden_sizes: list[int] = []
    index = 0
    while f"encoder.{index}.weight" in state_dict:
        hidden_sizes.append(int(state_dict[f"encoder.{index}.weight"].shape[0]))
        index += 2
    return tuple(hidden_sizes)


def _gnn_input_columns(state_dict: dict[str, torch.Tensor], key: str) -> int:
    if key not in state_dict:
        raise PolicyCompatibilityError(f"Cannot infer GNN architecture, missing {key}")
    return int(state_dict[key].shape[1])


def _gnn_hidden_size(state_dict: dict[str, torch.Tensor]) -> int:
    if "gnn_encoder.node_projection.0.weight" in state_dict:
        return int(state_dict["gnn_encoder.node_projection.0.weight"].shape[0])
    raise PolicyCompatibilityError("Cannot infer GNN hidden size from checkpoint")


def _policy_hidden_size(state_dict: dict[str, torch.Tensor]) -> int:
    for key in (
        "action_category_head.weight",
        "main_action_type_head.weight",
        "target_head.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return DEFAULT_GNN_POLICY_HIDDEN_SIZE


def _context_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    sizes: list[int] = []
    index = 0
    while f"context_encoder.{index}.weight" in state_dict:
        sizes.append(int(state_dict[f"context_encoder.{index}.weight"].shape[0]))
        index += 2
    if sizes:
        return tuple(sizes[:-1])
    return ()


def _head_rows(
    state_dict: dict[str, torch.Tensor],
    key: str,
    default: int,
) -> int:
    tensor = state_dict.get(key)
    return int(tensor.shape[0]) if torch.is_tensor(tensor) else int(default)
