"""Evaluate the untrained reference agents (Random, MinHeight, Heuristic).

Usage:
    python -m t2x2.baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from t2x2.evaluate import evaluate, summarize, write_episodes
from t2x2.policies import HeuristicAgent, MinHeightAgent, RandomAgent

AGENTS = {"random": RandomAgent, "min_height": MinHeightAgent, "heuristic": HeuristicAgent}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--eval-cap", type=int, default=5000)
    p.add_argument("--out", type=Path, default=Path("results_10x20"))
    args = p.parse_args()
    for name, cls in AGENTS.items():
        episodes = evaluate(cls(), args.episodes, args.eval_cap, height=args.height)
        run_dir = args.out / "baselines" / name
        write_episodes(run_dir / "final_eval.csv", episodes)
        summary = summarize(episodes)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{name}] {summary}", flush=True)


if __name__ == "__main__":
    main()
