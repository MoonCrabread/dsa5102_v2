"""Step an agent through a game and describe each placement as board frames.

Shared by the interactive demo and the offline animations, so both show
exactly the same thing: for every piece, (1) the piece at the spawn point
with the agent's chosen landing spot as a ghost, (2) the piece rotated and
moved above that spot, (3) a flash of the rows it clears, if any.
"""

from __future__ import annotations

import numpy as np

from t2x2.env import make_env
from t2x2.ui import BoardView, Placement


def plan(env, action: int) -> tuple[Placement, bool]:
    """Where ``action`` puts the active piece (mirrors GroupedActionsObservations.step)."""
    column, rotations = divmod(int(action), 4)
    t = env.active_tetromino
    for _ in range(rotations):
        t = env.rotate(t)
    x = column + env.padding - env.active_tetromino.matrix.shape[1] // 2
    y = int(env.y)
    if env.collision(t, x, y):
        return Placement(t, x, y), True
    while not env.collision(t, x, y + 1):
        y += 1
    return Placement(t, x, y), False


def filled_rows(env, board: np.ndarray) -> tuple:
    """Indices of the full rows of a padded board, in visible coordinates."""
    visible = env.crop_padding(board)
    return tuple(int(r) for r in np.flatnonzero((visible > 0).all(axis=1)))


class AgentGame:
    """One game of ``agent`` on the study environment, piece by piece."""

    def __init__(self, agent, cap: int = 5000):
        self.env = make_env()
        self.base = self.env.unwrapped
        self.agent = agent
        self.cap = cap

    def reset(self, seed: int) -> None:
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset(seed)
        self.obs, self.info = self.env.reset(seed=seed)
        self.seed = seed
        self.lines = self.pieces = self.score = 0
        self.done = self.capped = False

    def view(self) -> BoardView:
        b = self.base
        board = b.crop_padding(b.board)
        if self.done or b.active_tetromino is None:
            return BoardView(board, game_over=self.done and not self.capped)
        return BoardView(board, Placement(b.active_tetromino, int(b.x), int(b.y)))

    def advance(self, animate: bool = True) -> list[BoardView]:
        """Play one piece. Returns the frames that show it (empty if not ``animate``)."""
        if self.done:
            return []
        b = self.base
        action = self.agent.select_action(self.obs, self.info)
        frames = []
        if animate:
            target, game_over = plan(b, action)
            board = b.crop_padding(b.board)
            spawn = Placement(b.active_tetromino, int(b.x), int(b.y))
            above = Placement(target.tetromino, target.x, int(b.y))
            frames = [BoardView(board, spawn, target), BoardView(board, above, target)]
            if not game_over:
                landed = b.project_tetromino(target.tetromino, target.x, target.y)
                rows = filled_rows(b, landed)
                if rows:
                    lit = BoardView(b.crop_padding(landed), flash_rows=rows)
                    dim = BoardView(b.crop_padding(landed))
                    frames += [lit, dim, lit]
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        lines = int(self.info.get("lines_cleared", 0))
        self.lines += lines
        self.score += b.score(lines)
        self.pieces += 1
        if terminated or truncated:
            self.done = True
        elif self.pieces >= self.cap:
            self.done = self.capped = True
        return frames

    def close(self) -> None:
        self.env.close()
