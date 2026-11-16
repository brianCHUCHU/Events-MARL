from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import torch

from ..tokens import HypernetworkInputs, extract_lidar, lidar_dim
from .base import Requery

FOOD_IN_RANGE_INDEX = 7


def detect_food_in_range(obs_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([agent_obs[:, FOOD_IN_RANGE_INDEX] > 0.5 for agent_obs in obs_list], dim=1)


class NavigationEvents:
    def __init__(self, num_envs: int, num_agents: int, adaptive: bool, obs_dim: int = 0,
                 device: str = "cpu", research=False, interval=None):
        self.adaptive = adaptive
        self.research = research
        self.interval = interval
        self.pending = None
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.already_detected = torch.zeros(num_envs, num_agents, dtype=torch.bool, device=device)

    def prime(self, reset_obs: list[torch.Tensor]) -> None:
        if self.adaptive:
            self.already_detected = detect_food_in_range(reset_obs).clone()

    def after_step(self, env, next_obs, tokens, step: int) -> Requery | None:
        newly = self.newly_detected(next_obs)
        if self.research:
            self.pending = newly
            return None
        if not bool(newly.any()):
            return None
        indices = np.where(newly.reshape(-1).cpu().numpy())[0]
        inputs, entries = self.requery_inputs(next_obs, indices, tokens)
        return Requery(indices=indices, inputs=inputs, entries=entries)

    def newly_detected(self, next_obs: list[torch.Tensor]) -> torch.Tensor:
        if not self.adaptive:
            return torch.zeros_like(self.already_detected)
        current = detect_food_in_range(next_obs)
        newly = current & ~self.already_detected
        self._current = current
        return newly

    def before_step(self, env, obs, tokens, step: int, active_envs=None) -> Requery | None:
        if not self.research or not self.adaptive:
            return None
        if self.interval is not None:
            should = torch.full_like(self.already_detected, step > 0 and step % self.interval == 0)
        else:
            should = self.pending if self.pending is not None else torch.zeros_like(self.already_detected)
        if active_envs is not None:
            should = should & active_envs[:, None]
        self.pending = None
        indices = np.flatnonzero(should.cpu().numpy().reshape(-1))
        if not len(indices):
            return None
        current = jnp.asarray(extract_lidar(obs, lidar_dim(self.obs_dim)).cpu().numpy())
        teams = indices // self.num_agents
        agents = indices % self.num_agents
        contexts = tokens.generation_lidar.at[teams, agents].set(current[teams])
        active_lidar = tokens.lidar.reshape(-1, current.shape[-1]).at[indices].set(current[teams, agents])
        updated = replace(tokens, generation_lidar=contexts, lidar=active_lidar.reshape(tokens.lidar.shape))
        inputs = HypernetworkInputs(lidar=current[teams])
        entries = [{"task": None, "context": None, "lidar": current[team, agent],
                    "team_lidar": current[team],
                    "attention_mask": None if tokens.mask is None else tokens.mask[0],
                    "query_type": "requery", "agent_idx": int(agent)}
                   for team, agent in zip(teams, agents)]
        return Requery(indices=indices, inputs=inputs, entries=entries, tokens=updated,
                       generated_indices=np.arange(len(indices)) * self.num_agents + agents)

    def commit(self) -> None:
        if self.adaptive:
            self.already_detected = self.already_detected | self._current

    def requery_inputs(self, next_obs, indices, tokens) -> tuple[HypernetworkInputs, list[dict]]:
        batch_size = next_obs[0].shape[0] * self.num_agents
        team_lidar = extract_lidar(next_obs, lidar_dim(self.obs_dim)).cpu().numpy()
        updated = team_lidar.reshape(batch_size, -1)
        updated = np.nan_to_num(updated, posinf=1.0, neginf=0.0)
        updated = jnp.asarray(np.clip(updated, -1.0, 1.0))
        subset = updated[indices][:, None, :]

        entries = [{"task": None, "context": None, "lidar": subset[i, 0], "target_nmd": None,
                    "team_lidar": team_lidar[int(idx) // self.num_agents],
                    "attention_mask": None if tokens.mask is None else tokens.mask[0],
                    "query_type": "requery", "hidden_state": None,
                    "agent_idx": int(idx) % self.num_agents}
                   for i, idx in enumerate(indices)]
        return HypernetworkInputs(lidar=subset, mask=None), entries

