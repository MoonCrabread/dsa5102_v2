"""Grouped-action Tetris environment shared by every model in the study.

Each step the agent sees one 13-dimensional feature row per candidate
placement (10 column heights, max height, holes, bumpiness), chooses one of
the 40 (column, rotation) placements, and the piece is hard-dropped.

On top of the Tetris-Gymnasium grouped wrapper, ``AfterstateWrapper``

- reports game-over placements as a full-height board (the library encodes
  them as an empty board, which looks like the best possible outcome), and
- replaces the environment reward (+1 per placement plus width * lines^2)
  with the number of lines cleared (0 for game over), the objective the CE
  models are trained on, and
- adds ``info["candidate_rewards"]``: the immediate reward of every candidate.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Importing envs registers "tetris_gymnasium/Tetris".
from tetris_gymnasium.envs import Tetris  # noqa: F401
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

ENV_ID = "tetris_gymnasium/Tetris"
N_ACTIONS = 40
N_FEATURES = 13


class AfterstateWrapper(gym.Wrapper):
    """Make grouped candidate features safe for afterstate learning."""

    def __init__(self, env):
        super().__init__(env)
        height = env.unwrapped.height
        width = env.unwrapped.width
        self.game_over_features = np.array(
            [height] * width + [height, 0, 0], dtype=np.float32
        )

    def _process(self, observation, info):
        observation = np.array(observation, dtype=np.float32)
        info = dict(info)
        n = observation.shape[0]
        game_over = np.asarray(info.get("game_over_mask", np.zeros(n)), dtype=bool)
        lines = np.asarray(
            info.get("lines_cleared_per_action", np.zeros(n)), dtype=np.int64
        )
        rewards = lines.astype(np.float32)
        rewards[game_over] = 0.0

        observation[game_over] = self.game_over_features
        info["action_mask"] = np.asarray(info["action_mask"], dtype=bool)
        info["game_over_mask"] = game_over
        info["lines_cleared_per_action"] = lines
        info["candidate_rewards"] = rewards
        return observation, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._process(observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(info.get("lines_cleared", 0))
        observation, info = self._process(observation, info)
        return observation, reward, terminated, truncated, info


def make_env(height: int = 20, width: int = 10, render_mode: str | None = None):
    """Create the study environment (default: the standard 10 x 20 board)."""
    base = gym.make(
        ENV_ID, render_mode=render_mode, gravity=False, height=height, width=width
    )
    env = GroupedActionsObservations(
        base,
        observation_wrappers=[FeatureVectorObservation(base)],
        terminate_on_illegal_action=True,
    )
    return AfterstateWrapper(env)
