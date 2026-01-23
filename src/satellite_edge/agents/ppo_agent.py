"""PPO agent for satellite task scheduling.

Wraps Stable-Baselines3 PPO with hyperparameters tuned for discrete
scheduling on the SatelliteEnv. Based on literature recommendations:
- 37 Implementation Details of PPO (Huang et al.)
- Action masking for infeasible actions (Huang & Ontanon, 2022)
- Separate actor/critic networks for scheduling (MDPI 2025)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
import torch.nn as nn

from satellite_edge.environment.satellite_env import SatelliteEnv, SatelliteConfig, EpisodeConfig


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training on SatelliteEnv.

    Tuned for discrete scheduling (5 actions, 14-dim obs):
    - Higher n_steps for scheduling temporal structure
    - Conservative learning rate with linear annealing
    - Moderate entropy for exploration in small action space
    - Separate actor/critic for value function quality
    """
    # Core PPO
    learning_rate: float = 3e-4
    n_steps: int = 1024           # Full episode per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995          # Long horizon for scheduling
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = True
    ent_coef: float = 0.01        # Prevent collapse in small action space
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03  # Safety valve for custom envs

    # Network architecture
    net_arch_pi: list[int] = field(default_factory=lambda: [64, 64])
    net_arch_vf: list[int] = field(default_factory=lambda: [64, 64])

    # Training
    n_envs: int = 8
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    early_stop_patience: int = 15
    min_evals_before_stop: int = 20

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Environment config
    episode_max_steps: int = 500  # Shorter episodes for faster iteration
    use_subprocess: bool = False  # SubprocVecEnv vs DummyVecEnv


class MetricsCallback(BaseCallback):
    """Callback to track domain-specific metrics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_values: list[float] = []
        self.episode_completions: list[int] = []
        self.episode_drops: list[int] = []

    def _on_step(self) -> bool:
        # Collect info from done episodes
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # SB3 Monitor wrapper provides episode stats
                pass
            if "total_value" in info:
                self.episode_values.append(info["total_value"])
                self.episode_completions.append(info.get("tasks_completed", 0))
                self.episode_drops.append(info.get("tasks_dropped", 0))
        return True


class SatellitePPOAgent:
    """PPO agent for satellite task scheduling.

    Provides a clean interface for training, evaluation, and comparison
    with baseline schedulers.
    """

    def __init__(
        self,
        config: PPOConfig | None = None,
        sat_config: SatelliteConfig | None = None,
        episode_config: EpisodeConfig | None = None,
        log_dir: str | Path = "./logs",
        seed: int = 42,
    ):
        self.config = config or PPOConfig()
        self.sat_config = sat_config or SatelliteConfig()
        self.episode_config = episode_config or EpisodeConfig(
            max_steps=self.config.episode_max_steps
        )
        self.log_dir = Path(log_dir)
        self.seed = seed

        self._model: PPO | None = None
        self._train_env: VecNormalize | None = None
        self._eval_env: VecNormalize | None = None
        self._metrics_callback: MetricsCallback | None = None

    def _make_env_fn(self):
        """Create a factory function for environment instantiation."""
        sat_config = self.sat_config
        episode_config = self.episode_config

        def _factory():
            return SatelliteEnv(
                sat_config=sat_config,
                episode_config=episode_config,
            )

        return _factory

    def _build_envs(self) -> tuple[VecNormalize, VecNormalize]:
        """Build training and evaluation vectorized environments."""
        vec_cls = SubprocVecEnv if self.config.use_subprocess else DummyVecEnv

        train_vec = make_vec_env(
            self._make_env_fn(),
            n_envs=self.config.n_envs,
            seed=self.seed,
            vec_env_cls=vec_cls,
        )
        train_env = VecNormalize(
            train_vec,
            norm_obs=self.config.normalize_obs,
            norm_reward=self.config.normalize_reward,
            clip_obs=self.config.clip_obs,
            clip_reward=self.config.clip_reward,
            gamma=self.config.gamma,
        )

        eval_vec = make_vec_env(
            self._make_env_fn(),
            n_envs=4,
            seed=self.seed + 1000,
            vec_env_cls=DummyVecEnv,
        )
        eval_env = VecNormalize(
            eval_vec,
            norm_obs=self.config.normalize_obs,
            norm_reward=False,  # Raw rewards for eval
            clip_obs=self.config.clip_obs,
            training=False,
        )

        return train_env, eval_env

    def _build_callbacks(self, eval_env: VecNormalize) -> CallbackList:
        """Build training callbacks for evaluation and checkpointing."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.config.early_stop_patience,
            min_evals=self.config.min_evals_before_stop,
            verbose=1,
        )

        eval_freq = max(self.config.eval_freq // self.config.n_envs, 1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.log_dir / "best_model"),
            log_path=str(self.log_dir / "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_callback,
        )

        checkpoint_freq = max(50_000 // self.config.n_envs, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self.log_dir / "checkpoints"),
            save_vecnormalize=True,
        )

        self._metrics_callback = MetricsCallback()

        return CallbackList([eval_callback, checkpoint_callback, self._metrics_callback])

    def train(self, total_timesteps: int | None = None) -> dict[str, Any]:
        """Train the PPO agent.

        Args:
            total_timesteps: Override training duration. If None, uses config.

        Returns:
            Dictionary with training metrics.
        """
        timesteps = total_timesteps or self.config.total_timesteps

        self._train_env, self._eval_env = self._build_envs()
        callbacks = self._build_callbacks(self._eval_env)

        policy_kwargs = {
            "net_arch": dict(
                pi=self.config.net_arch_pi,
                vf=self.config.net_arch_vf,
            ),
            "activation_fn": nn.Tanh,
            "ortho_init": True,
        }

        # Only set tensorboard_log if tensorboard is available
        try:
            import tensorboard  # noqa: F401
            tb_log = str(self.log_dir / "tb")
        except ImportError:
            tb_log = None

        self._model = PPO(
            "MlpPolicy",
            self._train_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            normalize_advantage=self.config.normalize_advantage,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            target_kl=self.config.target_kl,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=0,
            seed=self.seed,
        )

        self._model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        # Collect results
        results = {
            "total_timesteps": timesteps,
            "best_model_path": str(self.log_dir / "best_model" / "best_model.zip"),
        }

        if self._metrics_callback:
            results["episode_values"] = self._metrics_callback.episode_values
            results["episode_completions"] = self._metrics_callback.episode_completions
            results["episode_drops"] = self._metrics_callback.episode_drops

        return results

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict action from observation.

        Args:
            obs: Raw observation from SatelliteEnv.
            deterministic: Whether to use deterministic policy.

        Returns:
            Action index.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first or load().")

        # Normalize observation using training stats
        if self._train_env is not None:
            obs_normalized = self._train_env.normalize_obs(obs)
        else:
            obs_normalized = obs

        action, _ = self._model.predict(obs_normalized, deterministic=deterministic)
        return int(action)

    def save(self, path: str | Path | None = None) -> Path:
        """Save model and normalization statistics."""
        if self._model is None:
            raise RuntimeError("No model to save.")

        save_path = Path(path) if path else self.log_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save(str(save_path / "model"))
        if self._train_env is not None:
            self._train_env.save(str(save_path / "vec_normalize.pkl"))

        return save_path

    def load(self, path: str | Path) -> None:
        """Load a trained model and normalization statistics."""
        load_path = Path(path)

        model_file = load_path / "model.zip"
        if not model_file.exists():
            model_file = load_path / "best_model.zip"

        # Create a dummy env for loading
        env = SatelliteEnv(
            sat_config=self.sat_config,
            episode_config=self.episode_config,
        )

        self._model = PPO.load(str(model_file), env=env)

        # Load normalization stats if available
        norm_file = load_path / "vec_normalize.pkl"
        if norm_file.exists():
            dummy_vec = make_vec_env(
                self._make_env_fn(), n_envs=1, vec_env_cls=DummyVecEnv
            )
            self._train_env = VecNormalize.load(str(norm_file), dummy_vec)
            self._train_env.training = False

    def close(self) -> None:
        """Clean up environments."""
        if self._train_env is not None:
            self._train_env.close()
        if self._eval_env is not None:
            self._eval_env.close()
