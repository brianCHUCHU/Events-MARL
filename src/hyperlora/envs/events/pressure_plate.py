from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..tokens import HypernetworkInputs, pressure_plate_env_context
from .base import Requery

PLATE_TOLERANCE = 0.05


class PressurePlateEvents:
    def __init__(self, config, num_envs: int, num_agents: int):
        self.config = config
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.adaptive = config.model.adaptive_hypernetwork
        self.prev_door_open = torch.zeros(num_envs, dtype=torch.bool)
        self.plate_activation_count = torch.zeros(num_envs, dtype=torch.long)

    def prime(self, reset_obs) -> None:
        return None

    def commit(self) -> None:
        return None

    def after_step(self, env, next_obs, tokens, step: int) -> Requery | None:
        return None

    def _active_plates(self, env) -> torch.Tensor:
        robots = [a for a in env.agents if "ground_robot" in a.name]
        left = torch.zeros(self.num_envs, dtype=torch.bool)
        right = torch.zeros(self.num_envs, dtype=torch.bool)
        radius = env.scenario.plate_radius
        for robot in robots:
            reach = radius + robot.shape.radius + PLATE_TOLERANCE
            left |= torch.linalg.norm(
                robot.state.pos - env.scenario.plate_left.state.pos, dim=-1) < reach
            right |= torch.linalg.norm(
                robot.state.pos - env.scenario.plate_right.state.pos, dim=-1) < reach
        return left.long() + right.long()

    def before_step(self, env, obs, tokens, step: int, active_envs=None) -> Requery | None:
        if self.config.model.query_interval is not None:
            interval = self.config.model.query_interval
            if step == 0 or step % interval:
                return None
            should = torch.ones(self.num_envs, dtype=torch.bool) if active_envs is None else active_envs
            return self._build(env, tokens, should) if bool(should.any()) else None
        if not self.adaptive:
            return None

        current_door_open = env.scenario.door_open
        newly_opened = current_door_open & ~self.prev_door_open

        active = self._active_plates(env)
        second_plate = (active == 2) & (self.plate_activation_count == 1)
        self.plate_activation_count = torch.maximum(self.plate_activation_count, active)

        should_requery = newly_opened | second_plate
        if active_envs is not None:
            should_requery &= active_envs
        requery = None
        if bool(should_requery.any()):
            requery = self._build(env, tokens, should_requery)
        self.prev_door_open = current_door_open.clone()
        return requery

    def _build(self, env, tokens, should_requery: torch.Tensor) -> Requery | None:
        if self.config.research_metrics:
            from jax import numpy as jnp

            teams = np.flatnonzero(should_requery.cpu().numpy())
            context = pressure_plate_env_context(self.config, env, self.num_envs, self.num_agents)
            subset = jnp.asarray(context.cpu().numpy()[teams])
            inputs = replace(tokens, env_context=subset)
            updated = replace(tokens, env_context=tokens.env_context.at[teams].set(subset))
            indices = (teams[:, None] * self.num_agents + np.arange(self.num_agents)).reshape(-1)
            return Requery(indices=indices, inputs=inputs, tokens=updated, scaling=1.0)

        updated = tokens
        if self.config.model.env_context and self.config.model.env_context["door_state"]:
            import jax.numpy as jnp

            context = pressure_plate_env_context(self.config, env, self.num_envs, self.num_agents)
            updated = HypernetworkInputs(
                task=tokens.task, context=tokens.context, lidar=tokens.lidar,
                food_positions=tokens.food_positions, agent_positions=tokens.agent_positions,
                target_nmd=tokens.target_nmd,
                env_context=jnp.asarray(context.cpu().numpy()), mask=tokens.mask)

        flat = np.repeat(should_requery.cpu().numpy(), self.num_agents)
        indices = np.where(flat)[0]
        if len(indices) == 0:
            return Requery(indices=indices, inputs=None, tokens=updated, scaling=1.0)

        batch = self.num_envs * self.num_agents
        subset = {}
        for name in ("task", "context", "env_context"):
            value = getattr(updated, name)
            subset[name] = (None if value is None
                            else value.reshape(batch, -1)[indices][:, None, :])
        inputs = HypernetworkInputs(task=subset["task"], context=subset["context"],
                                    env_context=subset["env_context"], mask=None)
        return Requery(indices=indices, inputs=inputs, entries=[], scaling=1.0, tokens=updated)

