"""Render animations of the final trained models.

Every agent (the four final models of the 2x2 grid and the three baselines)
plays the same evaluation game (same piece sequence), and the tool writes

    results_10x20/animations/<agent>.mp4                full game screen, one agent
    results_10x20/animations/comparison_baselines.mp4   the three baselines side by side
    results_10x20/animations/comparison_rl.mp4          the four RL models side by side
    results_10x20/animations/comparison_*.gif           shorter versions for the README

The strong models survive the whole 5,000-piece game, so the clips show the
first ``--pieces`` pieces; the full-game result is printed under each board.

Usage:
    python -m t2x2.animate
    python -m t2x2.animate --pieces 500 --fps 30 --train-seed 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from t2x2 import ui
from t2x2.env import make_env
from t2x2.evaluate import EVAL_SEED_START
from t2x2.models import BASELINES, GRID, MODELS, RESULTS, eval_game, eval_summary, load_agent
from t2x2.playback import AgentGame


# Per-agent clips: the 2x2 grid and the three baselines.
SHOWN = (*GRID, *BASELINES)


class Video:
    def __init__(self, path: Path, fps: int, size: tuple[int, int]):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if not self.writer.isOpened():  # no H.264 encoder available
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        for _ in range(repeat):
            self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def full_game_note(key: str, train_seed: int, game_seed: int) -> str:
    game = eval_game(key, train_seed, game_seed)
    if game is None:
        return ""
    if game["capped"]:
        return f"Full game: {game['lines']:,} lines (hit cap)"
    return f"Full game: {game['lines']:,} lines, {game['length']:,} pieces"


def model_clip(key, args, out: Path) -> None:
    spec = MODELS[key]
    game = AgentGame(load_agent(key, args.train_seed), cap=args.pieces)
    game.reset(args.game_seed)
    summary = eval_summary(key, args.train_seed)
    side = [
        ("Method", spec.method),
        ("Function", spec.function_class),
        ("Parameters", f"{spec.n_params:,}"),
        ("Training seed", args.train_seed if spec.trained else "—"),
        ("Eval mean", f"{summary['eval_lines_mean']:,.1f} lines" if summary else "—"),
    ]

    def screen(view):
        hud = ui.Hud(
            controller=spec.label,
            side_title="MODEL",
            side=side,
            stats=[("Lines", f"{game.lines:,}"), ("Pieces", f"{game.pieces:,}"),
                   ("Score", f"{game.score:,}"), ("Game seed", args.game_seed),
                   ("Clip", f"first {args.pieces:,}")],
            controls=[("FINAL", "trained model") if spec.trained else ("REFERENCE", "hand-tuned weights"),
                      ("GREEDY", "one-piece lookahead"),
                      ("GHOST", "= chosen placement")],
            corner=f"Training seed {args.train_seed}" if spec.trained else "Untrained reference",
            next_note="The agent sees only the current piece",
        )
        return ui.render_game(game.base, view, hud)

    video = Video(out / f"{key}.mp4", args.fps, (ui.CANVAS_W, ui.CANVAS_H))
    try:
        while not game.done:
            for view in game.advance():
                video.write(screen(view))
        last = screen(game.view())
        video.write(last, args.fps // 2)
        title = "CLIP ENDS" if game.capped else "GAME OVER"
        rows = [("Lines", f"{game.lines:,}"), ("Pieces", f"{game.pieces:,}"),
                ("Score", f"{game.score:,}")]
        note = full_game_note(key, args.train_seed, args.game_seed)
        video.write(ui.render_game_over(last, rows, title, note=note), args.fps * 3)
    finally:
        video.close()
        game.close()
    print(f"[{key}] {game.pieces} pieces, {game.lines} lines -> {out / f'{key}.mp4'}", flush=True)


@dataclass
class Column:
    """One board of a side-by-side clip: an agent and its labels."""

    title: str
    subtitle: str
    agent: object
    note: str = ""


def side_by_side_frames(columns: list, game_seed: int, pieces: int, header: str,
                        subheader: str, end_hold: int):
    """Yield (frame, hold) with every column playing the same game in lockstep."""
    games = [AgentGame(c.agent, cap=pieces) for c in columns]
    for g in games:
        g.reset(game_seed)
    env = games[0].base

    def render(views):
        cols = []
        for c, g, v in zip(columns, games, views):
            cols.append({
                "title": c.title, "subtitle": c.subtitle, "view": v,
                "lines": f"{g.lines:,}", "pieces": f"{g.pieces:,}",
                "status": "GAME OVER" if g.done and not g.capped else None, "note": c.note,
            })
        return ui.render_comparison(env, cols, header, subheader)

    try:
        while not all(g.done for g in games):
            # Pieces whose frames run out early hold their last frame.
            clips = [g.advance() or [g.view()] for g in games]
            for i in range(max(len(c) for c in clips)):
                yield render([c[min(i, len(c) - 1)] for c in clips]), 1
        yield render([g.view() for g in games]), end_hold
    finally:
        for g in games:
            g.close()


def write_video(frames, path: Path, fps: int, columns: int) -> None:
    video = Video(path, fps, (ui.comparison_width(columns), ui.CMP_H))
    try:
        for frame, hold in frames:
            video.write(frame, hold)
    finally:
        video.close()
    print(f"-> {path}", flush=True)


def color_swatch(env, width: int, cell: int) -> np.ndarray:
    """Every piece color as drawn (block, ghost, flash), for building a GIF palette."""
    colors = ui.palette_of(env)[2:]
    img = ui.background(width, 3 * cell)
    img[:] = (38, 28, 22)
    for i, color in enumerate(colors):
        ui.draw_block(img, i * cell, 0, cell, color)
        ui.draw_ghost_block(img, i * cell, cell, cell, color)
        ui.draw_block(img, i * cell, 2 * cell, cell, color)
        roi = img[2 * cell :, i * cell : (i + 1) * cell]
        roi[:] = (roi * 0.25 + np.array(ui.WHITE) * 0.75).astype(np.uint8)
    return img


def write_gif(frames, path: Path, fps: int, columns: int, scale: float) -> None:
    images, durations = [], []
    w, h = int(ui.comparison_width(columns) * scale), int(ui.CMP_H * scale)
    cell = max(8, int(ui.CMP_CELL * scale))
    palette = None  # one shared palette lets the GIF store only changed regions
    for frame, hold in frames:
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if palette is None:
            env = make_env()
            swatch = color_swatch(env.unwrapped, 7 * 3 * cell, 3 * cell)
            env.close()
            sample = np.vstack([small, cv2.resize(swatch, (w, 9 * cell),
                                                  interpolation=cv2.INTER_NEAREST)])
            palette = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)).quantize(
                colors=128, method=Image.Quantize.MEDIANCUT)
        rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        images.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(int(1000 * hold / fps))
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    print(f"-> {path} ({len(images)} frames, {path.stat().st_size / 1e6:.1f} MB)", flush=True)


# Two side-by-side clips: (file name, agents, header).
COMPARISONS = (
    ("comparison_baselines", BASELINES, "THREE BASELINES ON THE SAME GAME"),
    ("comparison_rl", GRID, "FOUR REINFORCEMENT LEARNING MODELS ON THE SAME GAME"),
)


def comparison_frames(args, keys, header: str, pieces: int):
    """The given agents side by side on the same game."""
    columns = [
        Column(MODELS[k].label, f"{MODELS[k].method} · {MODELS[k].function_class}",
               load_agent(k, args.train_seed), full_game_note(k, args.train_seed, args.game_seed))
        for k in keys
    ]
    trained = any(MODELS[k].trained for k in keys)
    who = f"final models, training seed {args.train_seed}" if trained else "untrained reference agents"
    sub = (f"Game seed {args.game_seed}  ·  {who}  ·  "
           f"first {pieces:,} pieces  ·  ghost = chosen placement")
    return side_by_side_frames(columns, args.game_seed, pieces, header, sub, args.fps * 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(SHOWN), choices=list(MODELS))
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--game-seed", type=int, default=EVAL_SEED_START)
    p.add_argument("--pieces", type=int, default=300, help="pieces per video clip")
    p.add_argument("--gif-pieces", type=int, default=100, help="pieces in the GIF")
    p.add_argument("--gif-scale", type=float, default=0.7)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-comparison", action="store_true")
    p.add_argument("--out", type=Path, default=RESULTS / "animations")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        model_clip(key, args, args.out)
    if not args.no_comparison:
        for name, keys, header in COMPARISONS:
            n = len(keys)
            write_video(comparison_frames(args, keys, header, args.pieces),
                        args.out / f"{name}.mp4", args.fps, n)
            write_gif(comparison_frames(args, keys, header, args.gif_pieces),
                      args.out / f"{name}.gif", args.fps, n, args.gif_scale)


if __name__ == "__main__":
    main()
