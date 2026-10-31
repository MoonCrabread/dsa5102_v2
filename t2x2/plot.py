"""Plot the 2x2 study results.

Writes <results>/figures/{learning_curves,all_methods,final_performance,grid_2x2}.png and
<results>/summary.md.

Usage:
    python -m t2x2.plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Reference palette (dataviz skill), light mode; one fixed color per model.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
BASELINE = "#8a8984"
MODELS = {
    "ce_linear": ("交叉熵 · 线性", "#2a78d6"),
    "ce_mlp": ("交叉熵 · MLP", "#eb6834"),
    "linear_td": ("线性 TD", "#1baf7a"),
    "dqn": ("DQN · MLP", "#eda100"),
}
BOARD = "10×20"
BASELINES = {"random": "Random", "min_height": "单特征贪心", "heuristic": "Heuristic"}
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "text.color": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10.5,
})


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def seed_dirs(results: Path, model: str) -> list[Path]:
    return sorted(p for p in (results / model).glob("seed_*") if (p / "summary.json").exists())


def final_lines(run: Path) -> np.ndarray:
    return np.array([float(r["lines"]) for r in read_csv(run / "final_eval.csv")])


def capped_count(run: Path) -> int:
    return sum(r["capped"] == "True" for r in read_csv(run / "final_eval.csv"))


def curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in read_csv(run / "curve.csv") if r.get("eval_lines_mean")]
    x = np.array([float(r["env_steps"]) for r in rows])
    y = np.array([float(r["eval_lines_mean"]) for r in rows])
    keep = x > 0
    return x[keep], y[keep]


BASELINE_STYLES = {"heuristic": "--", "min_height": "-.", "random": ":"}
STEPS_LABEL = "训练交互步数 · 放下的方块数,对数刻度"


THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def fmt_lines(row: dict) -> str:
    """Final mean lines."""
    return f"{row['mean']:,.1f}"


def mean_curve(results: Path, model: str):
    """(steps, mean, min, max) over seeds on a common log grid, or None."""
    runs = seed_dirs(results, model)
    if not runs:
        return None
    curves = [curve(r) for r in runs]
    grid = np.geomspace(max(c[0].min() for c in curves), min(c[0].max() for c in curves), 60)
    ys = np.array([np.interp(np.log(grid), np.log(x), y) for x, y in curves])
    return grid, ys.mean(0), ys.min(0), ys.max(0), len(runs)


def draw_baselines(ax, rows: list[dict]) -> None:
    for r in rows:
        if r["key"] in BASELINE_STYLES:
            ax.axhline(r["mean"], color=BASELINE, lw=1.4, ls=BASELINE_STYLES[r["key"]])


def side_legend(fig, rect, rows: list[dict]) -> None:
    """Legend of all methods, best first: line sample and name."""
    lax = fig.add_axes(rect)
    lax.set_axis_off()
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.text(0, 1, "按最终消行排序", fontsize=9.5, color=TEXT_2, va="top")
    ordered = sorted(rows, key=lambda r: r["mean"], reverse=True)
    step = min(0.11, 0.88 / max(len(ordered), 1))
    for i, r in enumerate(ordered):
        y = 0.86 - i * step
        if r["key"] in BASELINE_STYLES:
            lax.plot([0, 0.13], [y, y], color=BASELINE, lw=1.4, ls=BASELINE_STYLES[r["key"]])
        else:
            lax.plot([0, 0.13], [y, y], color=r["color"], lw=2.2)
        lax.text(0.18, y, r["label"], fontsize=9.5, color=TEXT, va="center")


# Where a model's value goes relative to its curve end: (ha, dx, dy) in points.
# CE linear ends just under the CE MLP curve, so its value goes above-left;
# linear TD ends just above the reference lines near 0, so its value goes up.
END_LABEL = {"ce_linear": ("right", -6, 11), "linear_td": ("left", 6, 11)}
# References labelled at the left end of their line (the upper left is empty).
LEFT_LABEL = {"heuristic"}


def label_values(ax, rows: list[dict], ends: dict) -> None:
    """Write each final value next to the end of its line, in the line's color.

    ``ends`` maps a model key to its curve end (x, y); the references are
    labelled at one end of their horizontal line (LEFT_LABEL: left, else
    right), above it, or just outside its right end when another reference
    line is too close above it.
    Call after the layout is final.
    """
    fig = ax.figure
    for r in rows:
        if r["key"] in ends:
            ha, dx, dy = END_LABEL.get(r["key"], ("left", 6, 0))
            ax.annotate(fmt_lines(r), ends[r["key"]], xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=9.5, color=r["color"],
                        annotation_clip=False)
    refs = sorted((r for r in rows if r["key"] in BASELINE_STYLES), key=lambda r: r["mean"])
    gap = 9.5 * 1.4 * fig.dpi / 72
    pixels = [ax.transData.transform((1, r["mean"]))[1] for r in refs]
    for i, r in enumerate(refs):
        below = i + 1 < len(refs) and pixels[i + 1] - pixels[i] < gap
        left = r["key"] in LEFT_LABEL
        if below:  # crowded by the line above: just outside the right end, level with it
            xy, offset, ha, va = (1, r["mean"]), (4, 0), "left", "center"
        elif left:
            xy, offset, ha, va = (0, r["mean"]), (4, 4), "left", "bottom"
        else:
            xy, offset, ha, va = (1, r["mean"]), (-4, 4), "right", "bottom"
        ax.annotate(fmt_lines(r), xy, xycoords=("axes fraction", "data"), xytext=offset,
                    textcoords="offset points", ha=ha, va=va, fontsize=9.5, color=TEXT_2,
                    annotation_clip=False)


def plot_learning_curves(results: Path, out: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    panels = [("策略搜索 · 交叉熵", ["ce_linear", "ce_mlp"]),
              ("价值学习 · TD", ["linear_td", "dqn"])]
    ends = []
    for ax, (title, models) in zip(axes, panels):
        end = {}
        for model in models:
            c = mean_curve(results, model)
            if c is None:
                continue
            grid, mean, lo, hi, n = c
            color = MODELS[model][1]
            if n > 1:
                ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(grid, mean, color=color, lw=2)
            end[model] = (grid[-1], mean[-1])
        ends.append(end)
        draw_baselines(ax, rows)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left", fontsize=12, color=TEXT)
        ax.set_xlabel(STEPS_LABEL)
    axes[0].set_ylabel("评估平均消行数")
    axes[0].yaxis.set_major_formatter(THOUSANDS)
    fig.suptitle(f"学习曲线 · {BOARD} 棋盘", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.subplots_adjust(left=0.06, right=0.79, top=0.84, bottom=0.15, wspace=0.14)
    for ax, end in zip(axes, ends):
        label_values(ax, rows, end)
    side_legend(fig, [0.84, 0.25, 0.15, 0.59], rows)
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def collect(results: Path) -> list[dict]:
    rows = []
    for key, label in BASELINES.items():
        run = results / "baselines" / key
        if (run / "final_eval.csv").exists():
            lines = final_lines(run)
            rows.append({"key": key, "label": label, "kind": "参考", "color": BASELINE,
                         "mean": lines.mean(), "seed_means": [lines.mean()],
                         "episodes": len(lines), "capped": capped_count(run),
                         "env_steps": 0})
    for key, (label, color) in MODELS.items():
        runs = seed_dirs(results, key)
        if not runs:
            continue
        per_seed = [final_lines(r) for r in runs]
        steps = [json.loads((r / "summary.json").read_text())["env_steps"] for r in runs]
        rows.append({"key": key, "label": label, "kind": "模型", "color": color,
                     "mean": float(np.mean(np.concatenate(per_seed))),
                     "seed_means": [float(s.mean()) for s in per_seed],
                     "episodes": int(sum(len(s) for s in per_seed)),
                     "capped": sum(capped_count(r) for r in runs),
                     "env_steps": float(np.mean(steps))})
    return rows


def plot_all_methods(results: Path, out: Path, rows: list[dict]) -> None:
    """All seven methods on one axis: the four models as learning curves and
    the three references as horizontal lines. The y axis is linear up to 10
    lines and logarithmic above, so 0, 8.9 and ~2,000 stay apart."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    end = {}
    for model, (_, color) in MODELS.items():
        c = mean_curve(results, model)
        if c is None:
            continue
        grid, mean, lo, hi, n = c
        if n > 1:
            ax.fill_between(grid, lo, hi, color=color, alpha=0.13, lw=0)
        ax.plot(grid, mean, color=color, lw=2)
        end[model] = (grid[-1], mean[-1])
    draw_baselines(ax, rows)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10, linscale=0.6)
    ax.set_ylim(0, 4000)
    ax.set_yticks([0, 5, 10, 100, 1000, 2000], ["0", "5", "10", "100", "1,000", "2,000"])
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(STEPS_LABEL)
    ax.set_ylabel("评估平均消行数")
    ax.set_title(f"全部 7 个方法 · {BOARD} 棋盘", loc="left", fontsize=13, color=TEXT)
    fig.subplots_adjust(left=0.08, right=0.76, top=0.9, bottom=0.13)
    label_values(ax, rows, end)
    side_legend(fig, [0.81, 0.3, 0.18, 0.58], rows)
    fig.savefig(out / "all_methods.png", dpi=160)
    plt.close(fig)


def plot_final(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda r: r["mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        ax.barh(yi, row["mean"], height=0.6, color=row["color"], edgecolor=SURFACE, lw=2)
        if len(row["seed_means"]) > 1:
            ax.scatter(row["seed_means"], [yi] * len(row["seed_means"]), s=28,
                       color=SURFACE, edgecolor=TEXT, lw=1, zorder=3)
        # Label to the right of the bar and every seed dot; ">=" when games hit the cap.
        right = max([row["mean"], *row["seed_means"]])
        ax.annotate(fmt_lines(row), (right, yi), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=TEXT)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("最终评估平均消行数")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_title(f"最终表现 · {BOARD} 棋盘", loc="left", fontsize=13, color=TEXT)
    ax.set_xlim(0, max(max([r["mean"], *r["seed_means"]]) for r in rows) * 1.2 + 1)
    fig.tight_layout()
    fig.savefig(out / "final_performance.png", dpi=160)
    plt.close(fig)


def plot_grid(rows: list[dict], out: Path) -> None:
    by_key = {r["key"]: r for r in rows}
    layout = [["ce_linear", "ce_mlp"], ["linear_td", "dqn"]]
    values = np.array([[by_key[k]["mean"] if k in by_key else np.nan for k in row]
                       for row in layout])
    vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.grid(False)
    for i, row in enumerate(layout):
        for j, key in enumerate(row):
            r = by_key.get(key)
            level = 0 if r is None else min(int(r["mean"] / vmax * (len(SEQUENTIAL) - 1) + 0.5),
                                            len(SEQUENTIAL) - 1)
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.98, 0.98, color=SEQUENTIAL[level]))
            ink = "white" if level >= 3 else TEXT
            if r is None:
                text = "未运行"
            else:
                spread = (f" ± {np.std(r['seed_means']):.1f}"
                          if len(r["seed_means"]) > 1 else "")
                text = (f"{MODELS[key][0]}\n{fmt_lines(r)}{spread} 行\n"
                        f"训练步数 ≈ {r['env_steps']:,.0f}")
            ax.text(j + 0.49, 1 - i + 0.49, text, ha="center", va="center",
                    fontsize=10.5, color=ink, linespacing=1.5)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.49, 1.49], ["线性", "MLP"])
    ax.set_yticks([1.49, 0.49], ["策略搜索", "价值学习"])
    ax.tick_params(length=0, labelsize=11.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"2×2 框架 · {BOARD} 棋盘",
                 loc="left", fontsize=12.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "grid_2x2.png", dpi=160)
    plt.close(fig)


def write_summary(rows: list[dict], path: Path, note: str = "") -> None:
    lines = ([note, ""] if note else []) + ["| 模型 | 类别 | 平均消行 | seed 均值 | 评估局数 | 碰到上限局数 | 训练交互步数 |",
             "|---|---|---:|---|---:|---:|---:|"]
    order = ["参考", "模型"]
    for r in sorted(rows, key=lambda r: (order.index(r["kind"]), -r["mean"])):
        seeds = ", ".join(f"{m:.1f}" for m in r["seed_means"])
        lines.append(f"| {r['label']} | {r['kind']} | {r['mean']:.1f} | {seeds} | "
                     f"{r['episodes']} | {r['capped']} | {r['env_steps']:,.0f} |")
    if any(r["capped"] for r in rows):
        lines.append("\n碰到上限的局在放满方块上限时被截断,其消行数是下限,平均消行应读作“至少”。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results_10x20"))
    p.add_argument("--board", default="10×20", help="Board size shown in figure titles")
    args = p.parse_args()
    global BOARD
    BOARD = args.board
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(args.results)
    plot_learning_curves(args.results, out, rows)
    plot_all_methods(args.results, out, rows)
    plot_final(rows, out)
    plot_grid(rows, out)
    summary = args.results / "summary.md"
    write_summary(rows, summary, "价值学习(线性 TD、DQN)的奖励 = 消行数,与交叉熵的目标一致。")
    print(summary.read_text())


if __name__ == "__main__":
    main()
