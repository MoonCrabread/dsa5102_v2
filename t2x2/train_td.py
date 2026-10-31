"""Afterstate TD learning with a linear (linear TD) or MLP (DQN) value function.

Both models share everything except the network: experience replay, a target
network, epsilon-greedy exploration and the n-step afterstate target

    V(s'_t) <- r_{t+1} + ... + gamma^(n-2) r_{t+n-1}
               + gamma^(n-1) max_a [ r(a) + gamma * V_target(s'_{t+n-1 -> a}) ],

where r_{t+k} are the rewards of the placements actually made, r(a) is the
known immediate reward of each candidate at the bootstrap state, and V is 0
for game-over candidates and after termination (terminal afterstates).
n = 1 is the one-step target. As in Rainbow, the n-step prefix is not
corrected for exploratory actions.
Actions are chosen by argmax_a [ r(a) + gamma * V(s'_a) ].

The reward is the number of lines cleared (see env.py), the CE objective.

Usage:
    python -m t2x2.train_td --network mlp --seed 0 --height 20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from t2x2.env import N_ACTIONS, N_FEATURES, make_env
from t2x2.evaluate import evaluate, summarize, write_episodes
from t2x2.policies import ValueAgent, ValueNetwork, legal_actions


class Replay:
    """Fixed-size ring buffer of n-step afterstate transitions stored as arrays."""

    def __init__(self, size: int):
        self.size, self.n, self.i = size, 0, 0
        self.features = np.zeros((size, N_FEATURES), np.float32)
        self.prefix = np.zeros(size, np.float32)
        self.discount = np.zeros(size, np.float32)
        self.boot_candidates = np.zeros((size, N_ACTIONS, N_FEATURES), np.float32)
        self.boot_mask = np.zeros((size, N_ACTIONS), bool)
        self.boot_rewards = np.zeros((size, N_ACTIONS), np.float32)
        self.boot_game_over = np.zeros((size, N_ACTIONS), bool)

    def add(self, features, prefix, discount, boot):
        i = self.i
        self.features[i] = features
        self.prefix[i] = prefix
        self.discount[i] = discount
        if boot is not None:
            (self.boot_candidates[i], self.boot_mask[i],
             self.boot_rewards[i], self.boot_game_over[i]) = boot
        self.i = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng, batch: int):
        idx = rng.integers(0, self.n, batch)
        return tuple(
            torch.as_tensor(a[idx])
            for a in (self.features, self.prefix, self.discount, self.boot_candidates,
                      self.boot_mask, self.boot_rewards, self.boot_game_over)
        )


class NStepBuilder:
    """Turn the stream of steps into n-step transitions for the replay."""

    def __init__(self, n: int, gamma: float, replay: Replay):
        self.n, self.gamma, self.replay = n, gamma, replay
        self.pending: deque = deque()  # [features, prefix, steps_accumulated]

    def step(self, features, candidates, mask, rewards, game_over, terminated):
        """Record the afterstate just chosen and the state reached after it."""
        self.pending.append([features, 0.0, 0])
        boot = (candidates, mask, rewards, game_over)
        if terminated:
            # Every pending afterstate's future ends here.
            while self.pending:
                f, prefix, _ = self.pending.popleft()
                self.replay.add(f, prefix, 0.0, None)
            return
        # The oldest pending afterstate bootstraps from this state when it has
        # accumulated n - 1 later rewards.
        while self.pending and self.pending[0][2] == self.n - 1:
            f, prefix, k = self.pending.popleft()
            self.replay.add(f, prefix, self.gamma**k, boot)

    def add_reward(self, reward: float) -> None:
        """Add the reward of the placement just made to earlier afterstates.

        Call before ``step`` so the afterstate of this placement is not yet
        pending: its value excludes its own placement's reward.
        """
        for item in self.pending:
            item[1] += self.gamma ** item[2] * reward
            item[2] += 1

    def clear(self) -> None:
        self.pending.clear()


@torch.no_grad()
def td_targets(target, prefix, discount, candidates, mask, rewards, game_over, gamma):
    targets = prefix.clone()
    can_bootstrap = (discount > 0) & mask.any(dim=1)
    if can_bootstrap.any():
        v = target(candidates[can_bootstrap])
        v = v.masked_fill(game_over[can_bootstrap], 0.0)
        values = rewards[can_bootstrap] + gamma * v
        values = values.masked_fill(~mask[can_bootstrap], -torch.inf)
        targets[can_bootstrap] += discount[can_bootstrap] * values.max(dim=1).values
    return targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=("linear", "mlp"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--epsilon-decay-steps", type=int, default=25_000)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    p.add_argument("--eval-cap", type=int, default=5000)
    p.add_argument("--name", default=None, help="Result folder name (default by network)")
    p.add_argument("--out", type=Path, default=Path("results_10x20"))
    args = p.parse_args()
    if args.learning_rate is None:
        args.learning_rate = {"linear": 1e-3, "mlp": 3e-4}[args.network]
    if args.n_step < 1:
        p.error("--n-step must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    name = args.name or {"linear": "linear_td", "mlp": "dqn"}[args.network]
    run_dir = args.out / name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    online = ValueNetwork(args.network)
    target = ValueNetwork(args.network)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = Replay(args.buffer_size)
    builder = NStepBuilder(args.n_step, args.gamma, replay)
    agent = ValueAgent(online, args.gamma)
    env = make_env(height=args.height)
    eval_env = make_env(height=args.height)

    curve_file = (run_dir / "curve.csv").open("w", newline="")
    curve = csv.DictWriter(
        curve_file,
        fieldnames=(
            "env_steps", "eval_lines_mean", "eval_lines_std", "eval_length_mean",
            "td_loss", "mean_value", "epsilon", "episodes", "seconds",
        ),
    )
    curve.writeheader()
    started = time.time()
    losses, values = [], []
    episodes = 0

    def log(step: int, epsilon: float) -> None:
        summary = summarize(evaluate(agent, args.eval_episodes, args.eval_cap, env=eval_env))
        curve.writerow({
            "env_steps": step,
            "eval_lines_mean": summary["eval_lines_mean"],
            "eval_lines_std": summary["eval_lines_std"],
            "eval_length_mean": summary["eval_length_mean"],
            "td_loss": float(np.mean(losses)) if losses else "",
            "mean_value": float(np.mean(values)) if values else "",
            "epsilon": epsilon,
            "episodes": episodes,
            "seconds": round(time.time() - started, 1),
        })
        curve_file.flush()
        print(
            f"[{name} seed={args.seed}] step={step} eval={summary['eval_lines_mean']:.1f} "
            f"loss={np.mean(losses) if losses else float('nan'):.3f} "
            f"V={np.mean(values) if values else float('nan'):.1f} "
            f"eps={epsilon:.2f} t={time.time() - started:.0f}s",
            flush=True,
        )
        losses.clear()
        values.clear()

    observation, info = env.reset(seed=args.seed * 100_000)
    log(0, args.epsilon_start)
    for step in range(1, args.steps + 1):
        fraction = min(step / max(args.epsilon_decay_steps, 1), 1.0)
        epsilon = args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)
        if rng.random() < epsilon:
            action = int(rng.choice(legal_actions(info)))
        else:
            action = agent.select_action(observation, info)

        selected = observation[action].copy()
        observation, reward, terminated, truncated, info = env.step(action)
        # The reward of this placement belongs to the earlier afterstates' futures.
        builder.add_reward(float(reward))
        builder.step(
            selected,
            observation,
            info["action_mask"],
            info["candidate_rewards"],
            info["game_over_mask"],
            terminated,
        )

        if step >= args.learning_starts and replay.n >= args.batch_size:
            feats, prefix, discount, cand, mask, rew, go = replay.sample(rng, args.batch_size)
            predicted = online(feats)
            expected = td_targets(target, prefix, discount, cand, mask, rew, go, args.gamma)
            loss = nn.functional.smooth_l1_loss(predicted, expected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())
            values.append(predicted.mean().item())

        if step % args.target_update == 0:
            target.load_state_dict(online.state_dict())
        if terminated or truncated:
            builder.clear()
            episodes += 1
            observation, info = env.reset(seed=args.seed * 100_000 + episodes)
        if step % args.eval_every == 0 or step == args.steps:
            log(step, epsilon)

    curve_file.close()
    torch.save(online.state_dict(), run_dir / "model.pt")
    final = evaluate(agent, args.final_eval_episodes, args.eval_cap, env=eval_env)
    write_episodes(run_dir / "final_eval.csv", final)
    summary = {**summarize(final), "env_steps": args.steps}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name} seed={args.seed}] final {summary}", flush=True)
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
