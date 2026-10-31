"""Fixed-seed evaluation shared by all models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from t2x2.env import make_env

# Evaluation seeds never overlap training seeds (training uses < 900_000).
EVAL_SEED_START = 900_000


@dataclass
class Episode:
    seed: int
    lines: int
    length: int
    capped: bool


def run_episode(env, agent, seed: int, cap: int) -> Episode:
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed)
    observation, info = env.reset(seed=seed)
    lines = length = 0
    done = False
    while not done and length < cap:
        observation, _, terminated, truncated, info = env.step(
            agent.select_action(observation, info)
        )
        lines += int(info.get("lines_cleared", 0))
        length += 1
        done = terminated or truncated
    return Episode(seed, lines, length, not done)


def evaluate(agent, episodes: int, cap: int, height: int = 20, env=None) -> list[Episode]:
    own_env = env is None
    env = env or make_env(height=height)
    try:
        seeds = range(EVAL_SEED_START, EVAL_SEED_START + episodes)
        return [run_episode(env, agent, seed, cap) for seed in seeds]
    finally:
        if own_env:
            env.close()


def summarize(episodes: list[Episode]) -> dict:
    lines = np.array([e.lines for e in episodes])
    return {
        "eval_lines_mean": float(lines.mean()),
        "eval_lines_std": float(lines.std()),
        "eval_lines_median": float(np.median(lines)),
        "eval_length_mean": float(np.mean([e.length for e in episodes])),
        "eval_capped": int(sum(e.capped for e in episodes)),
    }


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(episodes[0])))
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)
