"""Agents and scoring functions.

Every agent scores the legal candidate placements and picks the best one.
The four studied models differ only in the scoring function and in how its
parameters are obtained:

                    linear              MLP
  policy search     CE  (LinearScorer)   CE  (MLPScorer)
  value learning    linear TD            DQN            (see train_td.py)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from t2x2.env import N_FEATURES

# Fixed input scale: 10 column heights and max height, holes, bumpiness.
FEATURE_SCALE = np.array([20.0] * 11 + [40.0, 40.0], dtype=np.float32)


def legal_actions(info: dict) -> np.ndarray:
    return np.flatnonzero(info["action_mask"])


def scorer_inputs(observation: np.ndarray, info: dict) -> np.ndarray:
    """Scaled afterstate features plus the lines each candidate clears (40, 14)."""
    lines = info["lines_cleared_per_action"].astype(np.float32)[:, None]
    return np.concatenate([observation / FEATURE_SCALE, lines], axis=1)


SCORER_INPUTS = N_FEATURES + 1


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------


class RandomAgent:
    """Uniform over legal placements."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation, info) -> int:
        return int(self.rng.choice(legal_actions(info)))


class MinHeightAgent:
    """Greedy on a single feature: minimize aggregate height."""

    def select_action(self, observation, info) -> int:
        legal = legal_actions(info)
        return int(legal[np.argmin(observation[legal, :10].sum(axis=1))])


class HeuristicAgent:
    """Yiyuan Lee's GA-tuned weights (aggregate height, lines, holes, bumpiness)."""

    def select_action(self, observation, info) -> int:
        legal = legal_actions(info)
        c = observation[legal]
        values = (
            -0.510066 * c[:, :10].sum(axis=1)
            + 0.760666 * info["lines_cleared_per_action"][legal]
            - 0.35663 * c[:, 11]
            - 0.184483 * c[:, 12]
        )
        return int(legal[np.argmax(values)])


# ----------------------------------------------------------------------------
# Policy search (cross-entropy method): parameter vector -> scorer
# ----------------------------------------------------------------------------


class LinearScorer:
    """score = x . w + b over the 14 scorer inputs."""

    n_params = SCORER_INPUTS + 1

    def __init__(self, params: np.ndarray):
        self.w = params[:SCORER_INPUTS]
        self.b = params[SCORER_INPUTS]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b


class MLPScorer:
    """score = w2 . relu(W1 x + b1) + b2, a 14 -> 16 -> 1 network."""

    hidden = 16
    n_params = SCORER_INPUTS * hidden + hidden + hidden + 1

    def __init__(self, params: np.ndarray):
        i, h = SCORER_INPUTS, self.hidden
        self.w1 = params[: i * h].reshape(i, h)
        self.b1 = params[i * h : i * h + h]
        self.w2 = params[i * h + h : i * h + 2 * h]
        self.b2 = params[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x @ self.w1 + self.b1, 0.0) @ self.w2 + self.b2


SCORERS = {"linear": LinearScorer, "mlp": MLPScorer}


class ScorerAgent:
    """Pick the legal candidate with the highest scorer value."""

    def __init__(self, scorer):
        self.scorer = scorer

    def select_action(self, observation, info) -> int:
        legal = legal_actions(info)
        return int(legal[np.argmax(self.scorer(scorer_inputs(observation, info)[legal]))])


# ----------------------------------------------------------------------------
# Value learning: afterstate value networks
# ----------------------------------------------------------------------------


class ValueNetwork(nn.Module):
    """V(afterstate) from the 13 features; linear or a 128-128 MLP."""

    def __init__(self, kind: str):
        super().__init__()
        self.register_buffer(
            "scale", torch.as_tensor(FEATURE_SCALE), persistent=False
        )
        if kind == "linear":
            self.net = nn.Linear(N_FEATURES, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(N_FEATURES, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(f"unknown network kind: {kind}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features / self.scale).squeeze(-1)


class ValueAgent:
    """Pick argmax r(a) + gamma * V(afterstate_a) over legal candidates.

    A game-over afterstate is terminal, so its value is 0 by definition
    rather than estimated by the network.
    """

    def __init__(self, network: ValueNetwork, gamma: float):
        self.network = network
        self.gamma = gamma

    @torch.no_grad()
    def select_action(self, observation, info) -> int:
        legal = legal_actions(info)
        v = self.network(torch.as_tensor(observation[legal])).numpy()
        v[info["game_over_mask"][legal]] = 0.0
        values = info["candidate_rewards"][legal] + self.gamma * v
        return int(legal[np.argmax(values)])
