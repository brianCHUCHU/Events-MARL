"""Numeric rollout traces for query-timing and architecture diagnostics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def environment_manifest(config, env):
    return {
        "metrics_schema_version": 2,
        "scenario": config.env.scenario,
        "num_envs": config.env.num_envs,
        "training_horizon": config.ppo.rollout_steps,
        "policy_distribution": "tanh_gaussian",
        "evaluation_action_transform": "tanh_mean",
        "evaluation_seed": config.evaluation.seed,
        "evaluation_rng_isolated": True,
        "configured_fixed_capabilities": dict(config.env.fixed_capabilities),
        "actual_agents_at_initialization": [
            {"name": a.name,
             "max_speed": None if a.max_speed is None else float(a.max_speed),
             "observation_range": (None if a._obs_range is None else float(a._obs_range))}
            for a in env.agents
        ],
        "nmd_reference": ("full_team_initial_contexts_and_previous_valid_observations"
                          if config.env.scenario == "navigation" else None),
        "nmd_action_space": "pre_tanh_gaussian_mean",
    }


class RolloutTrace:
    """Keep team outcomes and timing for all slots and policy inputs for a sample."""

    def __init__(self, scenario, num_envs, num_agents, horizon, sample_offset):
        self.scenario = scenario
        self.num_agents = num_agents
        self.sample = (sample_offset + np.arange(min(4, num_envs))) % num_envs
        self.team_return = np.zeros(num_envs)
        self.length = np.zeros(num_envs, dtype=np.int32)
        self.completed = np.zeros(num_envs, dtype=bool)
        self.queries = np.zeros((horizon, num_envs), dtype=bool)
        self.agent_queries = np.zeros((horizon, num_envs, num_agents), dtype=bool)
        self.agent_detected_steps = np.full((num_envs, num_agents), -1, dtype=np.int32)
        self.agent_collected_steps = np.full((num_envs, num_agents), -1, dtype=np.int32)
        self.agent_last_query = np.zeros((num_envs, num_agents), dtype=np.int32)
        self.query_batch_sizes = np.zeros(horizon, dtype=np.int32)
        self.generated_adapters = np.zeros(horizon, dtype=np.int32)
        self.deployed_adapters = np.zeros(horizon, dtype=np.int32)
        self.observed = np.zeros((horizon, num_envs), dtype=bool)
        self.last_query = np.zeros(num_envs, dtype=np.int32)
        self.milestones = {}
        self.sample_steps = defaultdict(list)
        self.snapshots = defaultdict(list)
        self.wind_totals = defaultdict(lambda: np.zeros(num_envs))
        self.wind_segment_totals = defaultdict(lambda: np.zeros((horizon, num_envs)))
        self.wind_segment_lengths = np.zeros((horizon, num_envs), dtype=np.int32)
        self.wind_targets = np.full(horizon, np.nan, dtype=np.float32)
        self.step_index = 0

    def _save_adapters(self, adapters, step):
        self.snapshots["adapter_step"].append(step)
        for key, matrix in adapters.items():
            shaped = matrix.reshape((-1, self.num_agents, *matrix.shape[1:]))
            self.snapshots[f"adapter_{key}"].append(np.asarray(shaped)[self.sample])

    def initial_adapters(self, adapters):
        self._save_adapters(adapters, 0)

    def step(self, env, obs, actions, rewards, dones, tokens, refresh, adapters,
             means, stds, active, adapter_scale, log_probs=None, values=None):
        t = self.step_index
        done = as_numpy(dones).astype(bool, copy=False)
        if active is None:
            active = ~self.completed
        active = np.asarray(active, dtype=bool)
        self.observed[t] = active
        reward = np.stack([as_numpy(part) for part in rewards], axis=1)
        self.team_return += reward.mean(axis=1) * active
        self.length += active
        self.completed |= done & active

        age_before_query = t - self.last_query
        refreshed = np.zeros_like(active)
        if refresh is not None and len(refresh.indices):
            refreshed[np.asarray(refresh.indices) // self.num_agents] = True
            contexts, slots = refresh.inputs.generation_shape()
            self.query_batch_sizes[t] = contexts
            self.generated_adapters[t] = contexts * slots
            self.deployed_adapters[t] = len(refresh.indices)
            self.queries[t] = refreshed & active
            self.last_query[refreshed] = t
            if refreshed[self.sample].any():
                self._save_adapters(adapters, t)

        scenario = env.scenario
        data = {
            "observations": np.stack([as_numpy(part)[self.sample] for part in obs], axis=1),
            "actions": np.stack([as_numpy(part)[self.sample] for part in actions], axis=1),
            "agent_rewards": reward[self.sample],
            "active": active[self.sample],
            "done": done[self.sample],
            "steps_since_query": age_before_query[self.sample],
            "refresh": refreshed[self.sample],
            "adapter_scale": float(adapter_scale),
        }
        for key, policy_data in (("policy_means", means), ("policy_stds", stds),
                                 ("log_probs", log_probs), ("value_estimates", values)):
            if policy_data is not None:
                data[key] = as_numpy(policy_data).reshape(len(active), self.num_agents, -1)[self.sample]
        for name in tokens.FIELDS:
            token = getattr(tokens, name)
            if token is not None:
                data[f"active_{name}"] = np.asarray(token)[self.sample]

        if self.scenario == "navigation":
            agent_refresh = self.agent_queries[t]
            if refresh is not None and len(refresh.indices):
                flat = agent_refresh.reshape(-1)
                flat[refresh.indices] = True
                agent_refresh &= active[:, None]
            detected = np.stack([as_numpy(part)[:, 7] > 0.5 for part in obs], axis=1)
            collected = np.stack([as_numpy(food.eaten) for food in env.world.landmarks], axis=1)
            self.agent_detected_steps[(self.agent_detected_steps < 0) & detected & active[:, None]] = t
            self.agent_collected_steps[(self.agent_collected_steps < 0) & collected & active[:, None]] = t + 1
            data["agent_steps_since_query"] = (t - self.agent_last_query)[self.sample]
            self.agent_last_query[agent_refresh] = t
            data["agent_refresh"] = agent_refresh[self.sample]
            data["goal_in_range"] = detected[self.sample]
            data["goal_collected"] = collected[self.sample]
            if tokens.generation_lidar is not None:
                data["generation_lidar"] = np.asarray(tokens.generation_lidar)[self.sample]
        elif self.scenario == "pressure_plate":
            positions = np.stack([as_numpy(a.state.pos) for a in env.agents], axis=1)
            velocities = np.stack([as_numpy(a.state.vel) for a in env.agents], axis=1)
            data.update(
                agent_position=positions[self.sample],
                agent_velocity=velocities[self.sample],
                task_stage=as_numpy(scenario.stage)[self.sample],
                assigned_left_agent=as_numpy(scenario.robot_left_idx)[self.sample],
                assigned_right_agent=as_numpy(scenario.robot_right_idx)[self.sample],
                agent_goal_visited=as_numpy(scenario.goal_visitors)[self.sample],
                first_visit_team_bonus=as_numpy(scenario.goal_reward_this_step)[self.sample],
            )
            for name, landmark in (("left_plate", scenario.plate_left),
                                   ("right_plate", scenario.plate_right), ("goal", scenario.goal)):
                distance = np.linalg.norm(positions - as_numpy(landmark.state.pos)[:, None, :], axis=-1)
                data[f"distance_to_{name}"] = distance[self.sample]
            flags = {
                "door_open": as_numpy(scenario.door_open),
                "near_plate": as_numpy(scenario.is_on_left),
                "far_plate": as_numpy(scenario.is_on_right),
                "both_plates": as_numpy(scenario.is_on_left & scenario.is_on_right),
                "any_at_goal": as_numpy(scenario.agents_at_goal).any(axis=1),
                "all_at_goal": done,
            }
            data["agent_at_goal"] = as_numpy(scenario.agents_at_goal)[self.sample]
            for key, flag in flags.items():
                first = self.milestones.setdefault(key, np.full(len(active), -1, dtype=np.int32))
                first[(first < 0) & flag & active] = t + 1
                data[key] = flag[self.sample]
        else:
            positions = np.stack([as_numpy(agent.state.pos) for agent in env.agents], axis=1)
            big = int(np.argmax(scenario.agent_radii))
            small = [i for i in range(self.num_agents) if i != big]
            exposed = np.stack([np.linalg.norm(as_numpy(env.agents[i].gravity), axis=-1)
                                for i in small], axis=1).mean(axis=1)
            wind = np.linalg.norm(as_numpy(scenario.wind), axis=-1)
            wind_data = {
                "energy_reward": np.mean([as_numpy(agent.energy_rew) for agent in env.agents], axis=0),
                "wind_reward": np.mean([as_numpy(agent.wind_rew) for agent in env.agents], axis=0),
                "center_reward": np.mean([as_numpy(agent.center_rew) for agent in env.agents], axis=0),
                "formation_reward": as_numpy(scenario.formation_rew),
                "separation": np.linalg.norm(positions[:, small] - positions[:, big:big + 1],
                                             axis=-1).mean(axis=1),
                "small_agent_shielding": np.where(wind > 0, 1 - exposed / np.maximum(wind, 1e-8), 0),
            }
            self.wind_targets[t] = float(np.asarray(tokens.target_nmd)[0, 0, 0])
            self.wind_segment_lengths[self.last_query, np.arange(len(active))] += active
            for key, value in wind_data.items():
                self.wind_totals[key] += value * active
                self.wind_segment_totals[key][self.last_query, np.arange(len(active))] += value * active
                data[key] = value[self.sample]
        for key, value in data.items():
            self.sample_steps[key].append(np.array(value, copy=True))
        self.step_index += 1

    def episode_arrays(self):
        return {
            "team_return": self.team_return,
            "length": self.length,
            "completed": self.completed,
            "team_queries": 1 + self.queries.sum(axis=0),
            "extra_query_at_step": self.queries,
            "active_at_step": self.observed,
            "physical_query_batch_size": self.query_batch_sizes,
            "generated_agent_adapters_at_step": self.generated_adapters,
            "deployed_agent_adapters_at_step": self.deployed_adapters,
            **{f"first_{key}_step": value for key, value in self.milestones.items()},
            **(self._wind_arrays() if self.wind_totals else {}),
            **({
                "agent_queries": 1 + self.agent_queries.sum(axis=0),
                "agent_query_at_step": self.agent_queries,
                "agent_first_detected_step": self.agent_detected_steps,
                "agent_first_collected_step": self.agent_collected_steps,
            } if self.scenario == "navigation" else {}),
        }

    def _wind_arrays(self):
        values = {}
        for key, value in self.wind_totals.items():
            values[f"mean_{key}"] = value / np.maximum(self.length, 1)
            values[f"segment_mean_{key}"] = (
                self.wind_segment_totals[key] / np.maximum(self.wind_segment_lengths, 1)
            )
        values["segment_active_steps"] = self.wind_segment_lengths
        values["requested_nmd_at_step"] = self.wind_targets
        return values

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            trace_schema_version=np.array(2, dtype=np.int32),
            sample_environment_indices=self.sample,
            **self.episode_arrays(),
            **{f"sample_{key}": np.stack(values) for key, values in self.sample_steps.items()},
            **{key: np.stack(values) for key, values in self.snapshots.items()},
        )
