"""How far from death does each model play? Board-shape statistics.

The strong models reach the 5,000-piece cap in most evaluation games, so lines
cleared cannot rank them. This replays the exact final-evaluation games
(same parameters, same seeds, same cap) and records the board after every
placement: maximum column height, mean column height, holes and bumpiness.
A model that keeps the stack low and clean is further from losing.

Usage:
    python -m t2x2.robustness              # replays every game, ~15 min on 10 cores
    python -m t2x2.robustness --plot-only  # redraw from the saved CSVs

Writes results_10x20/robustness/{games.csv, heights.csv, summary.md,
robustness.png}.
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from t2x2.evaluate import EVAL_SEED_START
from t2x2.models import MODELS, RESULTS, eval_game, load_agent

HEIGHT = 20
DANGER = 10  # a stack at least half the board high
MODELS_ORDER = ("ce_mlp", "ce_linear", "dqn", "heuristic", "linear_td")
NAMES = {"ce_mlp": "CE · MLP", "ce_linear": "CE · linear", "dqn": "DQN",
         "heuristic": "Heuristic", "linear_td": "Linear TD"}
SEEDS = (0, 1, 2)
OUT = RESULTS / "robustness"


def play(job):
    """Replay one final-evaluation game and summarize the board after each piece."""
    key, train_seed, game_seed, cap = job
    import torch

    torch.set_num_threads(1)
    from t2x2.env import make_env

    env = make_env()
    agent = load_agent(key, train_seed)
    obs, info = env.reset(seed=game_seed)
    rows, lines, done = [], 0, False
    while not done and len(rows) < cap:
        obs, _, terminated, truncated, info = env.step(agent.select_action(obs, info))
        lines += int(info.get("lines_cleared", 0))
        done = terminated or truncated
        f = info["board"]
        rows.append((f[10], f[:10].mean(), f[11], f[12]))
    env.close()
    if done:  # the losing placement's board is not a real position
        rows = rows[:-1] or rows
    a = np.asarray(rows, dtype=np.float64)
    max_h = a[:, 0]
    return {
        "model": key, "train_seed": train_seed, "game_seed": game_seed,
        "pieces": len(rows) + int(done), "lines": lines, "capped": not done,
        "mean_max_height": max_h.mean(), "p95_max_height": np.percentile(max_h, 95),
        "peak_max_height": max_h.max(), "mean_height": a[:, 1].mean(),
        "mean_holes": a[:, 2].mean(), "mean_bumpiness": a[:, 3].mean(),
        "danger_frac": (max_h >= DANGER).mean(),
        "height_hist": np.bincount(max_h.astype(int), minlength=HEIGHT + 1).tolist(),
    }


def jobs(episodes: int, cap: int):
    for key in MODELS_ORDER:
        seeds = SEEDS if MODELS[key].trained else (0,)
        for s in seeds:
            for g in range(EVAL_SEED_START, EVAL_SEED_START + episodes):
                yield key, s, g, cap


METRICS = [
    ("mean_max_height", "Mean max height", "{:.2f}"),
    ("p95_max_height", "95th pct max height", "{:.1f}"),
    ("peak_max_height", "Peak max height", "{:.1f}"),
    ("mean_height", "Mean column height", "{:.2f}"),
    ("mean_holes", "Mean holes", "{:.2f}"),
    ("mean_bumpiness", "Mean bumpiness", "{:.2f}"),
    ("danger_frac", f"Pieces with max height ≥ {DANGER}", "{:.1%}"),
]


def load_games(path: Path) -> list[dict]:
    with path.open() as f:
        games = list(csv.DictReader(f))
    for g in games:
        for k, v in g.items():
            if k not in ("model", "capped"):
                g[k] = float(v)
        g["capped"] = g["capped"] == "True"
    return games


def write_summary(games: list[dict], out: Path) -> None:
    lines = [
        "Board shape after every placement, over the final-evaluation games "
        f"(same seeds and {int(max(g['pieces'] for g in games)):,}-piece cap as `final_eval.csv`). "
        "Each value is averaged within a game, then over games; lower means further from losing. "
        "Per-seed values are in brackets.",
        "",
        "| Model | Games | Mean lines | " + " | ".join(m[1] for m in METRICS) + " |",
        "|---|---:|---:|" + "---:|" * len(METRICS),
    ]
    for key in MODELS_ORDER:
        mg = [g for g in games if g["model"] == key]
        if not mg:
            continue
        cells = []
        for col, _, fmt in METRICS:
            total = fmt.format(np.mean([g[col] for g in mg]))
            seeds = sorted({int(g["train_seed"]) for g in mg})
            if len(seeds) > 1:
                per = ", ".join(
                    fmt.format(np.mean([g[col] for g in mg if g["train_seed"] == s]))
                    for s in seeds
                )
                total += f" ({per})"
            cells.append(total)
        lines.append(
            f"| {NAMES[key]} "
            f"| {len(mg)} | {np.mean([g['lines'] for g in mg]):.1f} | " + " | ".join(cells) + " |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--cap", type=int, default=5000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        todo = list(jobs(args.episodes, args.cap))
        results = []
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(play, todo, chunksize=1), 1):
                results.append(r)
                if i % 25 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)} games", flush=True)
        results.sort(key=lambda r: (MODELS_ORDER.index(r["model"]), r["train_seed"], r["game_seed"]))

        # Sanity check: the replay must reproduce the recorded evaluation.
        mismatches = 0
        for r in results:
            rec = eval_game(r["model"], r["train_seed"], r["game_seed"])
            if rec and (rec["lines"], rec["length"]) != (r["lines"], r["pieces"]):
                mismatches += 1
        print(f"replays differing from final_eval.csv: {mismatches}", flush=True)

        fields = [k for k in results[0] if k != "height_hist"]
        with (args.out / "games.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        with (args.out / "heights.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "max_height", "pieces"])
            for key in MODELS_ORDER:
                hist = np.sum([r["height_hist"] for r in results if r["model"] == key], axis=0)
                w.writerows([key, h, int(n)] for h, n in enumerate(hist))

    games = load_games(args.out / "games.csv")
    write_summary(games, args.out)
    from t2x2.plot_robustness import plot

    plot(args.out)
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
