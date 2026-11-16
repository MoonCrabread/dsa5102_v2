"""Interactive Tetris demo: play yourself, or watch any trained model play.

Usage:
    python -m t2x2.demo                  # start menu
    python -m t2x2.demo --model dqn      # jump straight into a model
    python -m t2x2.demo --train-seed 2   # use the models of training seed 2

Human keys:  A/D move · S down · W/E rotate · SPACE hard drop · Q hold
Model keys:  +/- speed · F turbo
Both:        P pause · R restart · ESC menu
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from t2x2 import ui
from t2x2.evaluate import EVAL_SEED_START
from t2x2.models import MODELS, available, eval_summary, load_agent
from t2x2.playback import AgentGame
from tetris_gymnasium.envs import Tetris

WINDOW = "DSA5102 - Tetris RL Project"
CONTROLLERS = ("human", "ce_linear", "ce_mlp", "linear_td", "dqn", "heuristic")
# Seconds per animation frame; the last level is turbo (no animation).
SPEEDS = (0.30, 0.15, 0.07, 0.03, 0.0)
SPEED_NAMES = ("×0.5", "×1", "×2", "×5", "TURBO")
ENTER = (10, 13)
ESC = 27


def menu_items(train_seed: int) -> list:
    items = [("HUMAN", "You play", True)]
    for key in CONTROLLERS[1:]:
        spec = MODELS[key]
        summary = eval_summary(key, train_seed)
        desc = f"{spec.method} · {spec.function_class}"
        if summary:
            desc += f" · {summary['eval_lines_mean']:,.0f} lines"
        items.append((spec.label, desc, available(key, train_seed)))
    return items


# ---------------------------------------------------------------------------
# Human play
# ---------------------------------------------------------------------------


class HumanGame:
    KEYS = {
        ord("a"): "move_left",
        ord("d"): "move_right",
        ord("s"): "move_down",
        ord("w"): "rotate_counterclockwise",
        ord("e"): "rotate_clockwise",
        ord(" "): "hard_drop",
        ord("q"): "swap",
    }
    CONTROLS = [("A/D", "Move"), ("S", "Down"), ("W/E", "Rotate"), ("SPACE", "Drop"),
                ("Q", "Hold"), ("P", "Pause"), ("R", "Restart"), ("ESC", "Menu")]

    def __init__(self):
        self.env = Tetris(render_mode=None, gravity=True)

    def reset(self, seed: int) -> None:
        self.env.reset(seed=seed)
        self.seed = seed
        self.lines = self.pieces = self.score = self.steps = 0
        self.done = False

    def key(self, key: int) -> None:
        name = self.KEYS.get(key)
        if name is None or self.done:
            return
        before = self.env.board.copy()
        _, _, terminated, truncated, info = self.env.step(getattr(self.env.actions, name))
        lines = int(info.get("lines_cleared", 0))
        self.lines += lines
        self.score += self.env.score(lines)
        self.pieces += int(not np.array_equal(before, self.env.board) or lines > 0)
        self.steps += 1
        self.done = bool(terminated or truncated)

    def frame(self, episode: int) -> np.ndarray:
        hud = ui.Hud(
            controller="HUMAN",
            stats=[("Score", self.score), ("Lines", self.lines), ("Pieces", self.pieces),
                   ("Moves", self.steps), ("Episode", episode)],
            controls=self.CONTROLS,
            corner=f"Game seed {self.seed}",
        )
        return ui.render_game(self.env, ui.board_view(self.env), hud)

    def results(self) -> list:
        return [("Score", self.score), ("Lines", self.lines), ("Pieces", self.pieces),
                ("Moves", self.steps)]


# ---------------------------------------------------------------------------
# Model play
# ---------------------------------------------------------------------------


class ModelGame:
    CONTROLS = [("P", "Pause"), ("+/-", "Speed"), ("F", "Turbo"), ("R", "Next game"),
                ("ESC", "Menu")]

    def __init__(self, key: str, train_seed: int, cap: int):
        self.model_key, self.spec = key, MODELS[key]
        self.game = AgentGame(load_agent(key, train_seed), cap)
        self.env = self.game.base
        summary = eval_summary(key, train_seed)
        self.side = [
            ("Method", self.spec.method),
            ("Function", self.spec.function_class),
            ("Parameters", f"{self.spec.n_params:,}"),
            ("Training seed", train_seed if self.spec.trained else "—"),
            ("Eval mean", f"{summary['eval_lines_mean']:,.1f} lines" if summary else "—"),
        ]
        self.speed = 1
        self.queue: list = []
        self.next_time = 0.0
        self.current = None

    def reset(self, seed: int) -> None:
        self.game.reset(seed)
        self.queue, self.current = [], self.game.view()

    @property
    def done(self) -> bool:
        return self.game.done and not self.queue

    def key(self, key: int) -> None:
        if key in (ord("+"), ord("=")):
            self.speed = min(self.speed + 1, len(SPEEDS) - 1)
        elif key in (ord("-"), ord("_")):
            self.speed = max(self.speed - 1, 0)
        elif key == ord("f"):
            self.speed = len(SPEEDS) - 1 if self.speed < len(SPEEDS) - 1 else 1

    def tick(self) -> None:
        now = time.monotonic()
        if now < self.next_time:
            return
        if SPEEDS[self.speed] == 0.0:  # turbo: many pieces per frame, no animation
            deadline = now + 0.03
            while not self.game.done and time.monotonic() < deadline:
                self.game.advance(animate=False)
            self.queue, self.current = [], self.game.view()
            return
        if not self.queue and not self.game.done:
            self.queue = self.game.advance()
        if self.queue:
            self.current = self.queue.pop(0)
        else:
            self.current = self.game.view()
        self.next_time = now + SPEEDS[self.speed]

    def frame(self, episode: int) -> np.ndarray:
        g = self.game
        hud = ui.Hud(
            controller=self.spec.label,
            side_title="MODEL",
            side=self.side,
            stats=[("Lines", f"{g.lines:,}"), ("Pieces", f"{g.pieces:,}"),
                   ("Score", f"{g.score:,}"), ("Speed", SPEED_NAMES[self.speed]),
                   ("Episode", episode)],
            controls=self.CONTROLS,
            corner=f"Game seed {g.seed}  ·  cap {g.cap:,} pieces",
            next_note="The agent sees only the current piece",
        )
        return ui.render_game(self.env, self.current or g.view(), hud)

    def results(self) -> list:
        g = self.game
        return [("Lines", f"{g.lines:,}"), ("Pieces", f"{g.pieces:,}"),
                ("Score", f"{g.score:,}")]

    def note(self) -> str:
        return f"Stopped at the {self.game.cap:,}-piece cap" if self.game.capped else ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=CONTROLLERS, help="skip the menu")
    p.add_argument("--train-seed", type=int, default=0, help="which trained seed to load")
    p.add_argument("--cap", type=int, default=5000, help="max pieces per model game")
    args = p.parse_args()

    items = menu_items(args.train_seed)
    selected, state, game, episode = 0, "menu", None, 1
    games: dict = {}

    def start(index: int):
        nonlocal game, state, episode
        key = CONTROLLERS[index]
        if key not in games:
            games[key] = HumanGame() if key == "human" else ModelGame(key, args.train_seed, args.cap)
        game, episode, state = games[key], 1, "playing"
        game.reset(EVAL_SEED_START)

    def restart():
        nonlocal episode, state
        episode += 1
        game.reset(EVAL_SEED_START + episode - 1)
        state = "playing"

    if args.model:
        start(CONTROLLERS.index(args.model))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ui.CANVAS_W, ui.CANVAS_H)
    try:
        while True:
            if state == "menu":
                frame = ui.render_menu(items, selected)
            else:
                if state == "playing" and isinstance(game, ModelGame):
                    game.tick()
                    if game.done:
                        state = "game_over"
                frame = game.frame(episode)
                if state == "paused":
                    frame = ui.render_pause(frame)
                elif state == "game_over":
                    note = game.note() if isinstance(game, ModelGame) else ""
                    frame = ui.render_game_over(
                        frame, game.results(), note=note,
                        footer=("R / ENTER   Play Again", "ESC   Back to Menu"),
                    )
            cv2.imshow(WINDOW, frame)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue

            if state == "menu":
                if key == ESC:
                    break
                if key in (ord("w"), ord("k")):
                    selected = (selected - 1) % len(items)
                elif key in (ord("s"), ord("j")):
                    selected = (selected + 1) % len(items)
                elif ord("1") <= key < ord("1") + len(items):
                    selected = key - ord("1")
                elif key in ENTER or key == ord(" "):
                    if items[selected][2]:
                        start(selected)
                continue

            if key == ESC:
                state = "menu"
            elif state == "paused":
                if key == ord("p"):
                    state = "playing"
                elif key == ord("r"):
                    restart()
            elif state == "game_over":
                if key == ord("r") or key in ENTER:
                    restart()
            elif key == ord("p"):
                state = "paused"
            elif key == ord("r"):
                restart()
            else:
                game.key(key)
                if isinstance(game, HumanGame) and game.done:
                    state = "game_over"
    finally:
        for g in games.values():
            if isinstance(g, ModelGame):
                g.game.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
