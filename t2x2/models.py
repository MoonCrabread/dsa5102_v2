"""Load the trained models of the study as ready-to-play agents.

This is the single interface between the trained results and the
visualization tools (``t2x2.demo`` and ``t2x2.animate``). Every controller is
an object with ``select_action(observation, info) -> int`` over the grouped
environment of ``t2x2.env``; to add a new one, add a ``ModelSpec`` below and
a branch in ``load_agent``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from t2x2.policies import (
    SCORERS,
    HeuristicAgent,
    ScorerAgent,
    ValueAgent,
    ValueNetwork,
)

RESULTS = Path(__file__).resolve().parent.parent / "results_10x20"


@dataclass(frozen=True)
class ModelSpec:
    key: str  # folder under results_10x20/
    label: str  # short display name
    method: str  # optimization approach
    function_class: str
    n_params: int
    trained: bool = True


MODELS = {
    spec.key: spec
    for spec in (
        ModelSpec("ce_linear", "CE · LINEAR", "Policy search (CE)", "Linear", 15),
        ModelSpec("ce_mlp", "CE · MLP", "Policy search (CE)", "MLP 14-16-1", 257),
        ModelSpec("linear_td", "LINEAR TD", "Value learning (TD)", "Linear", 14),
        ModelSpec("dqn", "DQN", "Value learning (TD)", "MLP 13-128-128-1", 18_433),
        ModelSpec("heuristic", "HEURISTIC", "Hand-tuned (Lee, 2013)", "Linear", 4, False),
    )
}

# The 2x2 grid of the study, in display order.
GRID = ("ce_linear", "ce_mlp", "linear_td", "dqn")


def run_dir(key: str, seed: int, results: Path = RESULTS) -> Path:
    if MODELS[key].trained:
        return results / key / f"seed_{seed}"
    return results / "baselines" / key


def available(key: str, seed: int = 0, results: Path = RESULTS) -> bool:
    d = run_dir(key, seed, results)
    if not MODELS[key].trained:
        return True
    return (d / "params.npy").exists() or (d / "model.pt").exists()


def load_agent(key: str, seed: int = 0, results: Path = RESULTS):
    """Return the final trained agent of ``key`` for training seed ``seed``."""
    d = run_dir(key, seed, results)
    if key in ("ce_linear", "ce_mlp"):
        params = np.load(d / "params.npy")
        return ScorerAgent(SCORERS[key.removeprefix("ce_")](params))
    if key in ("linear_td", "dqn"):
        config = json.loads((d / "config.json").read_text())
        network = ValueNetwork(config["network"])
        network.load_state_dict(torch.load(d / "model.pt", weights_only=True))
        network.eval()
        return ValueAgent(network, float(config["gamma"]))
    if key == "heuristic":
        return HeuristicAgent()
    raise KeyError(key)


def eval_summary(key: str, seed: int = 0, results: Path = RESULTS) -> dict | None:
    path = run_dir(key, seed, results) / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def eval_game(key: str, seed: int, game_seed: int, results: Path = RESULTS) -> dict | None:
    """The recorded final-evaluation game of this model on ``game_seed``."""
    path = run_dir(key, seed, results) / "final_eval.csv"
    if not path.exists():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) == game_seed:
                return {
                    "lines": int(row["lines"]),
                    "length": int(row["length"]),
                    "capped": row["capped"] == "True",
                }
    return None
