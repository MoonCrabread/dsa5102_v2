"""Figure for t2x2.robustness: how close to the top each model keeps the stack."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from t2x2.plot import BASELINE, GRID, MODELS, TEXT, TEXT_2

ORDER = ("ce_mlp", "ce_linear", "dqn", "heuristic", "linear_td")
STYLE = {k: (label, color) for k, (label, color) in MODELS.items()}
STYLE["heuristic"] = ("Heuristic", BASELINE)


def plot(out: Path) -> None:
    hist = defaultdict(lambda: np.zeros(21))
    with (out / "heights.csv").open() as f:
        for row in csv.DictReader(f):
            hist[row["model"]][int(row["max_height"])] = float(row["pieces"])
    games = defaultdict(list)
    with (out / "games.csv").open() as f:
        for row in csv.DictReader(f):
            games[row["model"]].append(row)

    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.75, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])

    # Left: share of placements whose stack reaches at least h rows.
    h = np.arange(21)
    ends = []
    for key in ORDER:
        if key not in hist:
            continue
        label, color = STYLE[key]
        counts = hist[key]
        tail = counts[::-1].cumsum()[::-1] / counts.sum()
        keep = tail > 0
        ls = "--" if key == "heuristic" else "-"
        ax.plot(h[keep], tail[keep], color=color, lw=2, ls=ls, marker="o", ms=3.5)
        last = h[keep][-1]
        ends.append([last, tail[keep][-1], label, color])
    ax.set_yscale("log")
    ax.set_xlim(0, 22.5)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("Stack height h (rows; the board has 20)")
    ax.set_ylabel("Share of placements with max height ≥ h")
    ax.grid(axis="y", which="major")
    ax.axvline(20, color=GRID, lw=1)
    # Direct labels at line ends, nudged apart when they collide.
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][0] == ends[i - 1][0] and ends[i][1] / ends[i - 1][1] < 1.8:
            ends[i][1] = ends[i - 1][1] * 1.8
    for x, y, label, color in ends:
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=TEXT)
    ax.set_title("How often the stack gets high", loc="left", fontsize=12, color=TEXT)

    # Right: per-seed means of two board statistics (strong models only).
    strong = [k for k in ORDER if k != "linear_td" and k in games]
    for row, (col, title) in enumerate((
        ("mean_max_height", "Mean max height (rows)"),
        ("mean_holes", "Mean holes"),
    )):
        bx = fig.add_subplot(gs[row, 1])
        for i, key in enumerate(strong):
            label, color = STYLE[key]
            seeds = defaultdict(list)
            for g in games[key]:
                seeds[g["train_seed"]].append(float(g[col]))
            per_seed = [np.mean(v) for v in seeds.values()]
            overall = np.mean([float(g[col]) for g in games[key]])
            y = len(strong) - 1 - i
            bx.scatter(per_seed, [y] * len(per_seed), s=36, color=color,
                       edgecolor="white", linewidth=1.2, zorder=3)
            bx.plot([overall, overall], [y - 0.32, y + 0.32], color=TEXT, lw=1.6, zorder=4)
        bx.set_yticks(range(len(strong)))
        bx.set_yticklabels([STYLE[k][0] for k in reversed(strong)])
        bx.set_ylim(-0.6, len(strong) - 0.4)
        bx.grid(axis="x")
        lo, hi = bx.get_xlim()
        bx.set_xlim(0, hi * 1.05)
        bx.set_title(title, loc="left", fontsize=11, color=TEXT)
    td = games.get("linear_td")
    note = "Dots: each training seed (Heuristic: one agent).  Tick: mean over all games."
    if td:
        mh = np.mean([float(g["mean_max_height"]) for g in td])
        ho = np.mean([float(g["mean_holes"]) for g in td])
        note += f"\nLinear TD is off this scale: {mh:.1f} rows, {ho:.1f} holes."
    fig.text(0.585, 0.015, note, fontsize=9, color=TEXT_2)

    fig.suptitle("Distance from losing · 10×20 board, final-evaluation games",
                 x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.17)
    fig.savefig(out / "robustness.png", dpi=160)
    plt.close(fig)
