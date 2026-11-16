from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

METHODS = ("ours", "ps", "dico", "cash")


@dataclass(frozen=True)
class EnvConfig:
    scenario: str
    num_agents: int
    max_agents: int
    num_envs: int
    continuous_actions: bool
    penalise_by_time: bool
    share_reward: bool
    distance_shaping_coef: float
    mixed_agent_training: bool
    min_agents: int
    use_fixed_capabilities: bool
    homogeneous_capabilities: bool
    fixed_capabilities: Mapping[str, list]
    fixed_food_positions: bool
    speed_range: tuple
    lidar_range_range: tuple
    max_speed_range: tuple
    force_multiplier_range: tuple
    scenario_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    policy_hidden_dims: tuple
    critic_hidden_dim: int
    critic_num_layers: int
    use_shared_critic: bool
    log_std_max: float
    log_std_min: float
    min_std: float
    lora_rank: int
    lora_mode: str
    lora_scaling_factor: float
    transformer_dim: int
    transformer_heads: int
    transformer_layers: int
    use_cross_agent_attention: bool
    task_embed_dim: int
    target_nmd_dim: int | None  # Ours only; absent (None) for the other methods
    use_capability_context: bool
    use_lidar_context: bool
    use_positional_context: bool
    positional_encoding_dim: int
    use_onehot_context: bool
    use_agent_position_context: bool
    agent_position_dim: int
    env_context: Mapping[str, bool]
    adaptive_hypernetwork: bool
    gru_hidden_dim: int | None
    fc_dim_size: int | None
    cash_hyper_hidden_dim: int | None
    cash_hyper_layers: int | None
    cash_decoder_hidden_dim: int | None
    cash_use_two_layer_decoder: bool | None
    query_interval: int | None = None


@dataclass(frozen=True)
class PPOConfig:
    num_episodes: int
    rollout_steps: int
    ppo_epochs: int
    ppo_clip_epsilon: float
    entropy_coef: float
    entropy_decay: float
    min_entropy_coef: float
    value_loss_coef: float
    gamma: float
    gae_lambda: float
    reward_scale: float
    advantage_clip_range: float
    normalize_returns: bool
    clip_value_loss: bool
    clip_global_state: bool
    clip_global_state_min: float
    clip_global_state_max: float
    use_minibatches: bool
    num_minibatches: int
    shuffle_minibatches: bool
    max_grad_norm: float
    hypernetwork_grad_norm: float
    critic_grad_norm: float
    adapter_regularization_coef: float


@dataclass(frozen=True)
class NMDConfig:
    """Ours: diversity measured as NMD between hypernetwork behaviours."""
    use_diversity_control: bool
    use_adapter_nmd: bool
    target_nmd: float
    nmd_moving_average_coef: float
    max_diversity_scaling: float
    min_nmd_floor: float
    adapter_nmd_sample_size: int
    target_nmd_list: tuple
    target_nmd_change_interval: int
    target_nmd_mode: str
    target_nmd_sampler: dict


@dataclass(frozen=True)
class SNDConfig:
    """DiCo: diversity measured as SND between agents (Bettini et al.)."""
    use_diversity_control: bool
    target_snd: float
    snd_moving_average_coef: float
    max_diversity_scaling: float
    min_snd_floor: float
    num_snd_samples: int


# The `diversity` section each method takes; PS and CASH have none.
DIVERSITY_SECTIONS = {"ours": NMDConfig, "dico": SNDConfig, "ps": None, "cash": None}

# Fields only some methods take; for every other method the key must be absent and reads as None.
_METHOD_ONLY_FIELDS = {("model", "target_nmd_dim"): ("ours",)}


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    hn_learning_rate: float
    critic_learning_rate: float
    beta1: float
    beta2: float
    eps: float


@dataclass(frozen=True)
class EvalConfig:
    eval_interval: int
    eval_start_episode: int
    num_eval_episodes: int
    max_eval_steps: int
    eval_max_speeds: tuple

    match_training_capabilities: bool
    seed: int = 20261114


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str
    checkpoint_dir: str
    save_interval: int
    experiment_name: str


@dataclass(frozen=True)
class QueryGateConfig:
    mode: str = "disabled"
    coverage_epsilon: float = 0.2
    query_cost: float = 0.01
    learning_rate: float = 0.0003
    entropy_coef: float = 0.01
    hidden_dim: int = 128
    generation_bucket: int = 32


@dataclass(frozen=True)
class Config:
    method: str
    seed: int
    jax_threefry_partitionable: bool
    env: EnvConfig
    model: ModelConfig
    ppo: PPOConfig
    diversity: NMDConfig | SNDConfig | None
    optimizer: OptimizerConfig
    evaluation: EvalConfig
    logging: LoggingConfig
    research_metrics: bool = False
    query_gate: QueryGateConfig = dataclasses.field(default_factory=QueryGateConfig)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if data["diversity"] is None:
            del data["diversity"]
        for (section, key), methods in _METHOD_ONLY_FIELDS.items():
            if self.method not in methods:
                del data[section][key]
        if not self.research_metrics:
            del data["research_metrics"]
        if self.model.query_interval is None:
            del data["model"]["query_interval"]
        if self.query_gate.mode == "disabled":
            del data["query_gate"]
        return data


_SECTIONS = {
    "env": EnvConfig,
    "model": ModelConfig,
    "ppo": PPOConfig,
    "optimizer": OptimizerConfig,
    "evaluation": EvalConfig,
    "logging": LoggingConfig,
}
_REQUIRED_TOP_LEVEL = {"method", "seed", "jax_threefry_partitionable"}
_TOP_LEVEL = _REQUIRED_TOP_LEVEL | {"research_metrics"}
_TUPLE_FIELDS = {"policy_hidden_dims", "speed_range", "lidar_range_range", "eval_max_speeds",
                 "max_speed_range",
                 "force_multiplier_range", "target_nmd_list"}


def _build(section: str, cls, data: Mapping[str, Any], method: str):
    fields = {f.name for f in dataclasses.fields(cls)}
    required = {f.name for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING}
    absent = {key for (sec, key), methods in _METHOD_ONLY_FIELDS.items()
              if sec == section and method not in methods}
    given = set(data)
    if missing := sorted(required - absent - given):
        raise ValueError(f"config section '{section}' is missing required keys: {missing}")
    if unknown := sorted(given - (fields - absent)):
        raise ValueError(f"config section '{section}' has unknown keys for method "
                         f"'{method}': {unknown}")
    values = {k: (tuple(v) if k in _TUPLE_FIELDS and v is not None else v) for k, v in data.items()}
    return cls(**values, **{key: None for key in absent})


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    gate_raw = raw.pop("query_gate", {})
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raw[section] = value
            continue
        if section == "query_gate":
            gate_raw[key] = value
            continue
        if section not in raw:
            raise ValueError(f"override '{dotted}' names an unknown section")
        raw[section][key] = value

    top = {k: v for k, v in raw.items() if k not in _SECTIONS and k != "diversity"}
    if unknown := sorted(set(top) - _TOP_LEVEL):
        raise ValueError(f"config has unknown top-level keys: {unknown}")
    if missing := sorted(_REQUIRED_TOP_LEVEL - set(top)):
        raise ValueError(f"config is missing required top-level keys: {missing}")
    method = top["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if missing := sorted(set(_SECTIONS) - set(raw)):
        raise ValueError(f"config is missing sections: {missing}")

    diversity_cls = DIVERSITY_SECTIONS[method]
    if diversity_cls is None and "diversity" in raw:
        raise ValueError(f"method '{method}' has no diversity measure; remove the "
                         "'diversity' section")
    if diversity_cls is not None and "diversity" not in raw:
        raise ValueError("config is missing sections: ['diversity']")

    if not isinstance(top["jax_threefry_partitionable"], bool):
        raise ValueError("jax_threefry_partitionable must be true or false")

    research_metrics = top.get("research_metrics", False)
    if type(research_metrics) is not bool:
        raise ValueError("research_metrics must be true or false")
    config = Config(
        method=method,
        research_metrics=research_metrics,
        query_gate=_build("query_gate", QueryGateConfig, gate_raw, method),
        seed=int(top["seed"]),
        jax_threefry_partitionable=top["jax_threefry_partitionable"],
        diversity=(None if diversity_cls is None
                   else _build("diversity", diversity_cls, raw["diversity"], method)),
        **{name: _build(name, cls, raw[name], method) for name, cls in _SECTIONS.items()},
    )

    gate = config.query_gate
    if gate.mode not in ("disabled", "learned", "coverage"):
        raise ValueError("query_gate.mode must be disabled, learned or coverage")
    if gate.mode != "disabled":
        if config.ppo.rollout_steps < 2:
            raise ValueError("Learned gate training needs a horizon of at least two steps")
        if not (research_metrics and method == "ours" and config.env.scenario == "pressure_plate"
                and config.model.adaptive_hypernetwork and config.model.query_interval is None):
            raise ValueError("Learned gates require adaptive research Ours Pressure Plate without fixed K")
        for name in ("coverage_epsilon", "query_cost", "learning_rate", "entropy_coef"):
            value = getattr(gate, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"query_gate.{name} must be finite and nonnegative")
        if gate.coverage_epsilon >= 1 or gate.learning_rate == 0:
            raise ValueError("Gate learning_rate must be positive and coverage_epsilon in [0, 1)")
        for name in ("hidden_dim", "generation_bucket"):
            if type(getattr(gate, name)) is not int or getattr(gate, name) < 1:
                raise ValueError(f"query_gate.{name} must be a positive integer")

    if research_metrics and (method != "ours" or
                             config.env.scenario not in ("pressure_plate", "wind_flocking", "navigation")):
        raise ValueError("research_metrics requires Ours on Pressure Plate, Wind Flocking or Navigation")
    interval = config.model.query_interval
    if interval is not None and (not research_metrics or config.env.scenario not in ("pressure_plate", "navigation")
                                 or type(interval) is not int or interval < 1
                                 or not config.model.adaptive_hypernetwork):
        raise ValueError("query_interval requires positive integer, research_metrics and "
                         "an adaptive Pressure Plate or Navigation model")
    if research_metrics and config.env.scenario == "pressure_plate" and (
        config.diversity.use_diversity_control or config.diversity.use_adapter_nmd
    ):
        raise ValueError("Pressure Plate research schedules require the original no-NMD setting")
    if research_metrics:
        if config.env.mixed_agent_training:
            raise ValueError("Research query comparisons currently require fixed-size teams")
        if not config.env.continuous_actions:
            raise ValueError("Research query comparisons currently require continuous actions")
        if config.env.scenario == "pressure_plate" and config.env.scenario_kwargs.get("with_drone", False):
            raise ValueError("Pressure Plate research comparisons currently support ground robots only")
        if (config.diversity.use_adapter_nmd and
                (type(config.diversity.adapter_nmd_sample_size) is not int
                 or config.diversity.adapter_nmd_sample_size < 2)):
            raise ValueError("NMD sampling needs at least two reference adapters")
        for key, value in (("env.num_envs", config.env.num_envs),
                           ("ppo.num_episodes", config.ppo.num_episodes),
                           ("ppo.rollout_steps", config.ppo.rollout_steps),
                           ("ppo.ppo_epochs", config.ppo.ppo_epochs),
                           ("evaluation.num_eval_episodes", config.evaluation.num_eval_episodes),
                           ("evaluation.max_eval_steps", config.evaluation.max_eval_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        count = config.ppo.num_minibatches if config.ppo.use_minibatches else 1
        if type(count) is not int or count < 1 or config.env.num_envs * config.ppo.rollout_steps % count:
            raise ValueError("Training team rows must divide evenly into complete minibatches")
        if type(config.evaluation.seed) is not int or not (
            0 <= config.evaluation.seed <= 2**32 - config.evaluation.num_eval_episodes
        ):
            raise ValueError("evaluation.seed and its batch offsets must fit unsigned 32-bit seeds")
    return config


def dump_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

