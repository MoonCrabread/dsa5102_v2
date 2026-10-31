"""Noisy cross-entropy method (Szita & Lőrincz, 2006) for a linear or MLP scorer.

Each generation samples parameter vectors from a diagonal Gaussian, plays a
few games with each (all samples share the same game seeds), keeps the elite
fraction, refits the Gaussian to the elites and adds a decreasing extra
variance to avoid premature convergence.

Usage:
    python -m t2x2.train_ce --policy linear --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from t2x2.env import make_env
from t2x2.evaluate import EVAL_SEED_START, evaluate, run_episode, summarize, write_episodes
from t2x2.policies import SCORERS, ScorerAgent

_worker_env = None
_worker_height = 20


def _init_worker(height: int) -> None:
    global _worker_env, _worker_height
    _worker_height = height
    _worker_env = make_env(height=height)


def _play(job):
    """Play the given seeds with one parameter vector; return (lines, steps) per game."""
    policy, params, seeds, cap = job
    agent = ScorerAgent(SCORERS[policy](np.asarray(params)))
    results = [run_episode(_worker_env, agent, seed, cap) for seed in seeds]
    return [(e.lines, e.length) for e in results]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=tuple(SCORERS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--elite-frac", type=float, default=None)
    p.add_argument("--games-per-sample", type=int, default=3)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--init-std", type=float, default=1.0)
    p.add_argument("--noise-start", type=float, default=0.5)
    p.add_argument("--noise-decay", type=float, default=0.02)
    p.add_argument("--train-cap", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    p.add_argument("--eval-cap", type=int, default=5000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results_10x20"))
    args = p.parse_args()
    # Model-specific defaults: the MLP has ~18x more parameters to search.
    defaults = {"linear": (50, 0.2, 30), "mlp": (100, 0.1, 50)}[args.policy]
    if args.population is None:
        args.population = defaults[0]
    if args.elite_frac is None:
        args.elite_frac = defaults[1]
    if args.generations is None:
        args.generations = defaults[2]
    return args


def evaluate_mean(pool, args, mean: np.ndarray, episodes: int) -> dict:
    seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + episodes))
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    jobs = [(args.policy, mean, chunk, args.eval_cap) for chunk in chunks if chunk]
    lines = [l for result in pool.map(_play, jobs) for l, _ in result]
    return {
        "eval_lines_mean": float(np.mean(lines)),
        "eval_lines_std": float(np.std(lines)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.out / f"ce_{args.policy}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    rng = np.random.default_rng(args.seed)
    n_params = SCORERS[args.policy].n_params
    mean = np.zeros(n_params)
    std = np.full(n_params, args.init_std)
    n_elite = max(1, int(round(args.population * args.elite_frac)))
    env_steps = 0
    started = time.time()

    with Pool(args.workers, initializer=_init_worker, initargs=(args.height,)) as pool, (
        run_dir / "curve.csv"
    ).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "generation", "env_steps", "sample_lines_mean", "elite_lines_mean",
                "eval_lines_mean", "eval_lines_std", "mean_std", "seconds",
            ),
        )
        writer.writeheader()
        row = {"generation": 0, "env_steps": 0, **evaluate_mean(pool, args, mean, args.eval_episodes)}
        writer.writerow({**row, "mean_std": float(std.mean()), "seconds": 0.0})
        f.flush()

        for generation in range(1, args.generations + 1):
            samples = mean + std * rng.standard_normal((args.population, n_params))
            # Common game seeds for every sample in this generation.
            seeds = [
                args.seed * 100_000 + generation * 100 + k
                for k in range(args.games_per_sample)
            ]
            jobs = [(args.policy, s, seeds, args.train_cap) for s in samples]
            results = pool.map(_play, jobs)
            fitness = np.array([np.mean([l for l, _ in r]) for r in results])
            env_steps += sum(n for r in results for _, n in r)

            elite = samples[np.argsort(fitness)[-n_elite:]]
            noise = max(args.noise_start - args.noise_decay * (generation - 1), 0.0)
            mean = elite.mean(axis=0)
            std = np.sqrt(elite.var(axis=0) + noise)

            row = {
                "generation": generation,
                "env_steps": env_steps,
                "sample_lines_mean": float(fitness.mean()),
                "elite_lines_mean": float(np.sort(fitness)[-n_elite:].mean()),
                "mean_std": float(std.mean()),
                "seconds": round(time.time() - started, 1),
            }
            if generation % args.eval_every == 0 or generation == args.generations:
                row.update(evaluate_mean(pool, args, mean, args.eval_episodes))
            writer.writerow(row)
            f.flush()
            print(
                f"[ce_{args.policy} seed={args.seed}] gen={generation} "
                f"steps={env_steps} elite={row['elite_lines_mean']:.1f} "
                f"eval={row.get('eval_lines_mean', float('nan')):.1f} "
                f"t={row['seconds']}s",
                flush=True,
            )

    np.save(run_dir / "params.npy", mean)
    final = evaluate(
        ScorerAgent(SCORERS[args.policy](mean)),
        args.final_eval_episodes,
        args.eval_cap,
        height=args.height,
    )
    write_episodes(run_dir / "final_eval.csv", final)
    summary = {**summarize(final), "env_steps": env_steps}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ce_{args.policy} seed={args.seed}] final {summary}", flush=True)


if __name__ == "__main__":
    main()
