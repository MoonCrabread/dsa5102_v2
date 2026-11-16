"""Show how one DQN agent's play improves during training.

Loads checkpoints saved by ``train_td --save-every`` and lets each training
stage play the same evaluation game side by side. Also draws the run's
learning curve with the shown stages marked.

Usage:
    python -m t2x2.train_td --network mlp --seed 0 --name dqn_checkpoints --save-every 2500
    python -m t2x2.animate_training --steps 0 10000 20000 30000 50000

Each shown checkpoint is also evaluated on the standard 50 evaluation games
(cached in <run>/stage_eval.csv); a single game is too noisy to rank stages.

Writes to results_10x20/animations/: dqn_training.mp4, dqn_training.gif and
dqn_training_curve.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from t2x2.animate import Column, side_by_side_frames, write_gif, write_video
from t2x2.evaluate import EVAL_SEED_START
from t2x2.models import RESULTS, load_checkpoint
from t2x2.playback import AgentGame


def read_curve(run: Path) -> dict:
    with (run / "curve.csv").open() as f:
        return {int(r["env_steps"]): r for r in csv.DictReader(f)}


def full_game(agent, seed: int, cap: int) -> tuple[int, int, bool]:
    """Play the whole game without rendering: (lines, pieces, capped)."""
    game = AgentGame(agent, cap)
    game.reset(seed)
    while not game.done:
        game.advance(animate=False)
    game.close()
    return game.lines, game.pieces, game.capped


def _eval_job(job):
    run, step, gamma, seed, cap = job
    import torch

    torch.set_num_threads(1)
    agent = load_checkpoint(Path(run) / "checkpoints" / f"step_{step:06d}.pt", "mlp", gamma)
    return step, seed, *full_game(agent, seed, cap)


def evaluate_stages(run: Path, steps: list, gamma: float, episodes: int, cap: int,
                    workers: int) -> dict:
    """Mean lines of each checkpoint over the standard evaluation games (cached to CSV)."""
    path = run / "stage_eval.csv"
    rows = []
    if path.exists():
        with path.open() as f:
            rows = [r for r in csv.DictReader(f)]
    done = {(int(r["step"]), int(r["seed"])) for r in rows}
    seeds = range(EVAL_SEED_START, EVAL_SEED_START + episodes)
    todo = [(str(run), s, gamma, g, cap) for s in steps for g in seeds if (s, g) not in done]
    if todo:
        with Pool(workers) as pool:
            for step, seed, lines, pieces, capped in pool.imap_unordered(_eval_job, todo):
                rows.append({"step": step, "seed": seed, "lines": lines,
                             "pieces": pieces, "capped": capped})
        rows.sort(key=lambda r: (int(r["step"]), int(r["seed"])))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "seed", "lines", "pieces", "capped"])
            w.writeheader()
            w.writerows(rows)
    result = {}
    for s in steps:
        lines = [int(r["lines"]) for r in rows if int(r["step"]) == s and int(r["seed"]) in seeds]
        capped = [str(r["capped"]) == "True" for r in rows
                  if int(r["step"]) == s and int(r["seed"]) in seeds]
        result[s] = {"mean": float(np.mean(lines)), "std": float(np.std(lines)),
                     "capped": int(sum(capped)), "games": len(lines)}
        print(f"step {s}: {episodes}-game mean {result[s]['mean']:.1f} lines, "
              f"{result[s]['capped']} capped", flush=True)
    return result


def plot_curve(run: Path, steps: list, out: Path, reference: Path | None,
               stages: dict | None = None) -> None:
    from t2x2.plot import BASELINE, MODELS, TEXT, TEXT_2

    color = MODELS["dqn"][1]
    curve = read_curve(run)
    x = np.array(sorted(curve))
    y = np.array([float(curve[s]["eval_lines_mean"]) for s in x])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    if reference is not None and (reference / "curve.csv").exists():
        ref = read_curve(reference)
        rx = np.array(sorted(ref))
        ax.plot(rx, [float(ref[s]["eval_lines_mean"]) for s in rx], color=BASELINE, lw=1.4,
                ls="--", label="DQN seed 0 of the 2×2 study (earlier run, no checkpoints)")
    ax.plot(x, y, color=color, lw=2, label="This run, 10-game eval during training")
    for s in steps:
        v = float(curve[s]["eval_lines_mean"])
        ax.scatter([s], [max(v, 0.3)], s=70, color=color, edgecolor="white", linewidth=1.5, zorder=5)
    if stages:
        sx = [s for s in steps if s in stages]
        sy = [max(stages[s]["mean"], 0.3) for s in sx]
        ax.scatter(sx, sy, s=70, facecolor="white", edgecolor=TEXT, linewidth=1.5, zorder=6,
                   label="Shown checkpoints, 50-game mean")
        for x_, y_ in zip(sx, sy):
            label = "start" if x_ == 0 else f"{x_ // 1000}k"
            value = f"{y_:,.1f}" if y_ < 10 else f"{y_:,.0f}"
            ax.annotate(f"{label}: {value}", (x_, y_), xytext=(10, -14),
                        textcoords="offset points", ha="left", fontsize=9.5, color=TEXT)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 6000)
    ax.set_xlabel("Environment steps during training")
    ax.set_ylabel("Mean lines cleared (log scale)")
    ax.grid(axis="y", which="major")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=TEXT_2)
    ax.set_title("DQN learning curve with the stages shown in the animation",
                 loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"-> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RESULTS / "dqn_checkpoints" / "seed_0")
    p.add_argument("--reference", type=Path, default=RESULTS / "dqn" / "seed_0")
    p.add_argument("--steps", type=int, nargs="+", default=[0, 10_000, 20_000, 30_000, 50_000])
    p.add_argument("--game-seed", type=int, default=EVAL_SEED_START)
    p.add_argument("--pieces", type=int, default=300)
    p.add_argument("--gif-pieces", type=int, default=100)
    p.add_argument("--gif-scale", type=float, default=0.7)
    p.add_argument("--full-cap", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", type=Path, default=RESULTS / "animations")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.run / "config.json").read_text())
    gamma = float(config["gamma"])
    curve = read_curve(args.run)

    def agent(step):
        return load_checkpoint(args.run / "checkpoints" / f"step_{step:06d}.pt", "mlp", gamma)

    stages = evaluate_stages(args.run, args.steps, gamma, args.eval_episodes, args.full_cap,
                             args.workers)
    columns = []
    for s in args.steps:
        curve_mean = float(curve[s]["eval_lines_mean"])
        columns.append(Column(
            f"STEP {s:,}", f"Training curve: {curve_mean:,.1f} lines (10 games)", agent(s),
            f"{args.eval_episodes}-game mean: {stages[s]['mean']:,.1f} lines"))

    header = "ONE DQN AGENT AS TRAINING PROGRESSES"
    sub = (f"Same game (seed {args.game_seed})  ·  checkpoints of one training run  ·  "
           f"first {{n}} pieces  ·  ghost = chosen placement")
    n = len(columns)
    write_video(side_by_side_frames(columns, args.game_seed, args.pieces, header,
                                    sub.format(n=f"{args.pieces:,}"), args.fps * 3),
                args.out / "dqn_training.mp4", args.fps, n)
    write_gif(side_by_side_frames(columns, args.game_seed, args.gif_pieces, header,
                                  sub.format(n=f"{args.gif_pieces:,}"), args.fps * 3),
              args.out / "dqn_training.gif", args.fps, n, args.gif_scale)
    plot_curve(args.run, args.steps, args.out / "dqn_training_curve.png", args.reference, stages)


if __name__ == "__main__":
    main()
