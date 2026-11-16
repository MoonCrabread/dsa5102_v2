"""Render animations of the final trained models.

For each model of the 2x2 grid it plays the same evaluation game (same piece
sequence) with the final parameters and writes

    results_10x20/animations/<model>.mp4   full game screen, one model
    results_10x20/animations/comparison.mp4  the four models side by side
    results_10x20/animations/comparison.gif  a shorter version for the README

The strong models survive the whole 5,000-piece game, so the clips show the
first ``--pieces`` pieces; the full-game result is printed under each board.

Usage:
    python -m t2x2.animate
    python -m t2x2.animate --pieces 500 --fps 30 --train-seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from t2x2 import ui
from t2x2.env import make_env
from t2x2.evaluate import EVAL_SEED_START
from t2x2.models import GRID, MODELS, RESULTS, eval_game, eval_summary, load_agent
from t2x2.playback import AgentGame


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
        ("Training seed", args.train_seed),
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
            controls=[("FINAL", "trained model"), ("GREEDY", "one-piece lookahead"),
                      ("GHOST", "= chosen placement")],
            corner=f"Training seed {args.train_seed}",
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


def comparison_frames(args, pieces: int):
    """Yield (frame, hold) for the side-by-side clip of the 2x2 grid."""
    games = {}
    for key in GRID:
        games[key] = AgentGame(load_agent(key, args.train_seed), cap=pieces)
        games[key].reset(args.game_seed)
    notes = {k: full_game_note(k, args.train_seed, args.game_seed) for k in GRID}
    env = games[GRID[0]].base
    header = "FINAL MODELS ON THE SAME GAME"
    sub = (f"Game seed {args.game_seed}  ·  training seed {args.train_seed}  ·  "
           f"first {pieces:,} pieces  ·  ghost = chosen placement")

    def columns(views):
        cols = []
        for key in GRID:
            g, spec = games[key], MODELS[key]
            status = "GAME OVER" if g.done and not g.capped else None
            cols.append({
                "title": spec.label,
                "subtitle": f"{spec.method} · {spec.function_class}",
                "view": views[key], "lines": f"{g.lines:,}", "pieces": f"{g.pieces:,}",
                "status": status, "note": notes[key],
            })
        return cols

    try:
        while not all(g.done for g in games.values()):
            # Pieces whose frames run out early hold their last frame.
            clips = {k: g.advance() or [g.view()] for k, g in games.items()}
            for i in range(max(len(c) for c in clips.values())):
                views = {k: c[min(i, len(c) - 1)] for k, c in clips.items()}
                yield ui.render_comparison(env, columns(views), header, sub), 1
        final = {k: g.view() for k, g in games.items()}
        yield ui.render_comparison(env, columns(final), header, sub), args.fps * 3
    finally:
        for g in games.values():
            g.close()


def comparison_video(args, out: Path) -> None:
    video = Video(out / "comparison.mp4", args.fps, (ui.CMP_W, ui.CMP_H))
    try:
        for frame, hold in comparison_frames(args, args.pieces):
            video.write(frame, hold)
    finally:
        video.close()
    print(f"[comparison] -> {out / 'comparison.mp4'}", flush=True)


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


def comparison_gif(args, out: Path) -> None:
    frames, durations = [], []
    w, h = int(ui.CMP_W * args.gif_scale), int(ui.CMP_H * args.gif_scale)
    cell = max(8, int(ui.CMP_CELL * args.gif_scale))
    palette = None  # one shared palette lets the GIF store only changed regions
    for frame, hold in comparison_frames(args, args.gif_pieces):
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if palette is None:
            env = make_env()
            sample = np.vstack([small, cv2.resize(color_swatch(env.unwrapped, 7 * 3 * cell, 3 * cell),
                                                  (w, 9 * cell), interpolation=cv2.INTER_NEAREST)])
            env.close()
            palette = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)).quantize(
                colors=128, method=Image.Quantize.MEDIANCUT)
        rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(int(1000 * hold / args.fps))
    path = out / "comparison.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    print(f"[comparison] {len(frames)} frames -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(GRID), choices=list(MODELS))
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
        comparison_video(args, args.out)
        comparison_gif(args, args.out)


if __name__ == "__main__":
    main()
