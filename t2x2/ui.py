"""Rendering for the Tetris demo and the model animations.

Adapted from the DSA5102 ``pretty_tetris_v3`` interface (OpenCV + Pillow).
Everything here only reads the environment; it never changes the game state.
All colors are BGR, as OpenCV expects.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 1100, 820
CELL = 28
BOARD_X, BOARD_Y = 410, 130

BG_TOP = (46, 36, 24)
BG_BOTTOM = (24, 17, 11)
PANEL = (58, 47, 33)
PANEL_LIGHT = (68, 57, 42)
BORDER = (105, 92, 72)
GRID = (65, 57, 45)
ACCENT = (235, 196, 72)
ACCENT_SOFT = (190, 160, 74)
WHITE = (245, 248, 252)
TEXT = (226, 232, 240)
MUTED = (156, 170, 186)
GOOD = (120, 210, 120)
BAD = (95, 95, 235)

# ---------------------------------------------------------------------------
# Fonts and text
# ---------------------------------------------------------------------------

_FONTS = {
    True: [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ],
    False: [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ],
}
_font_cache: dict = {}


def get_font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _font_cache:
        path = next((p for p in _FONTS[bold] if os.path.exists(p)), None)
        _font_cache[key] = (
            ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
        )
    return _font_cache[key]


class Text:
    """Collect text, then draw it all in one Pillow pass (OpenCV has no TTF)."""

    def __init__(self):
        self.items = []

    def add(self, text, x, y, size=20, color=TEXT, bold=False, align="left"):
        self.items.append((str(text), x, y, size, color, bold, align))

    def apply(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, x, y, size, (b, g, r), bold, align in self.items:
            font = get_font(size, bold)
            box = draw.textbbox((0, 0), text, font=font)
            w = box[2] - box[0]
            x = x - w // 2 if align == "center" else x - w if align == "right" else x
            draw.text((x, y), text, font=font, fill=(r, g, b))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Background, panels, blocks
# ---------------------------------------------------------------------------

_backgrounds: dict = {}


def background(w: int = CANVAS_W, h: int = CANVAS_H) -> np.ndarray:
    if (w, h) not in _backgrounds:
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        rows = np.array(BG_TOP, np.float32) * (1 - t) + np.array(BG_BOTTOM, np.float32) * t
        _backgrounds[w, h] = np.repeat(rows[:, None, :], w, axis=1).astype(np.uint8)
    return _backgrounds[w, h].copy()


def draw_panel(img, x, y, w, h, accent=False, fill=PANEL):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)
    if accent:
        cv2.line(img, (x + 1, y + 1), (x + w - 1, y + 1), ACCENT, 3)


def draw_block(img, x, y, size, color):
    pad = 2 if size >= 18 else 1
    x1, y1, x2, y2 = x + pad, y + pad, x + size - pad, y + size - pad
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    bright = tuple(min(255, int(c) + 55) for c in color)
    dark = tuple(max(0, int(c) - 65) for c in color)
    cv2.line(img, (x1, y1), (x2, y1), bright, 2)
    cv2.line(img, (x1, y1), (x1, y2), bright, 2)
    cv2.line(img, (x1, y2), (x2, y2), dark, 2)
    cv2.line(img, (x2, y1), (x2, y2), dark, 2)
    if size >= 18:
        inner = tuple(min(255, int(c) + 18) for c in color)
        cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), inner, 1)


def draw_ghost_block(img, x, y, size, color):
    """Translucent landing preview."""
    pad = 4 if size >= 18 else 3
    x1, y1, x2, y2 = x + pad, y + pad, x + size - pad, y + size - pad
    roi = img[y1 : y2 + 1, x1 : x2 + 1]
    roi[:] = (roi * 0.82 + np.array(color) * 0.18).astype(np.uint8)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def bgr(rgb) -> tuple:
    r, g, b = rgb
    return int(b), int(g), int(r)


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """A tetromino at (x, y) in padded board coordinates."""

    tetromino: object
    x: int
    y: int


@dataclass
class BoardView:
    """Everything needed to draw one board frame.

    ``board`` is the visible board (padding cropped) of pixel ids, ``piece``
    the falling piece, ``ghost`` the landing preview (for an agent: the
    placement it has chosen) and ``flash_rows`` the rows being cleared.
    """

    board: np.ndarray
    piece: Placement | None = None
    ghost: Placement | None = None
    flash_rows: tuple = ()
    game_over: bool = False


def board_view(env, ghost: Placement | None = "drop") -> BoardView:
    """The current state of a ``Tetris`` env (unwrapped) as a BoardView."""
    board = env.crop_padding(env.board)
    t = env.active_tetromino
    if t is None or env.game_over:
        return BoardView(board, game_over=env.game_over)
    piece = Placement(t, int(env.x), int(env.y))
    if ghost == "drop":
        y = int(env.y)
        if env.collision(t, env.x, y):
            ghost = None
        else:
            while not env.collision(t, env.x, y + 1):
                y += 1
            ghost = Placement(t, int(env.x), y)
    return BoardView(board, piece, ghost)


def _cells(p: Placement, padding: int):
    for r, c in np.argwhere(p.tetromino.matrix > 0):
        yield p.y + r, p.x - padding + c


def draw_board(img, view: BoardView, palette, padding, x0=BOARD_X, y0=BOARD_Y, cell=CELL):
    rows, cols = view.board.shape
    w, h = cols * cell, rows * cell
    cv2.rectangle(img, (x0 - 8, y0 - 8), (x0 + w + 8, y0 + h + 8), ACCENT_SOFT, 2)
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + w + 4, y0 + h + 4), (20, 13, 10), -1)
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * cell, y0 + r * cell
            cv2.rectangle(img, (x, y), (x + cell, y + cell), (38, 28, 22), -1)
            cv2.rectangle(img, (x, y), (x + cell, y + cell), GRID, 1)
            pid = int(view.board[r, c])
            if pid > 0:
                draw_block(img, x, y, cell, palette[pid])
    for r in view.flash_rows:
        y = y0 + r * cell
        roi = img[y : y + cell, x0 : x0 + w]
        roi[:] = (roi * 0.25 + np.array(WHITE) * 0.75).astype(np.uint8)
    for p, drawer in ((view.ghost, draw_ghost_block), (view.piece, draw_block)):
        if p is None:
            continue
        color = bgr(p.tetromino.color_rgb)
        for r, c in _cells(p, padding):
            if 0 <= r < rows and 0 <= c < cols:
                drawer(img, x0 + c * cell, y0 + r * cell, cell, color)


def draw_preview(img, tetromino, x, y, w, h, cell=21):
    if tetromino is None:
        return
    occupied = np.argwhere(tetromino.matrix > 0)
    (r0, c0), (r1, c1) = occupied.min(axis=0), occupied.max(axis=0)
    shape = tetromino.matrix[r0 : r1 + 1, c0 : c1 + 1]
    sx = x + (w - shape.shape[1] * cell) // 2
    sy = y + (h - shape.shape[0] * cell) // 2
    color = bgr(tetromino.color_rgb)
    for r, c in np.argwhere(shape > 0):
        draw_block(img, sx + c * cell, sy + r * cell, cell, color)


def palette_of(env) -> list:
    return [bgr(p.color_rgb) for p in env.pixels]


# ---------------------------------------------------------------------------
# Main game screen
# ---------------------------------------------------------------------------


@dataclass
class Hud:
    """What the side panels show.

    ``side`` lists (label, value) rows for the top-left panel; when it is
    None that panel shows the HOLD piece instead (human play).
    """

    controller: str
    stats: list
    controls: list
    side_title: str = "HOLD"
    side: list | None = None
    subtitle: str = "POLICY SEARCH VS VALUE LEARNING  ·  10 × 20"
    corner: str = ""
    next_note: str = ""
    banner: str = ""
    extra: dict = field(default_factory=dict)


def render_game(env, view: BoardView, hud: Hud) -> np.ndarray:
    """The full 1100 x 820 game screen for a ``Tetris`` env (unwrapped)."""
    img = background()
    text = Text()

    text.add("TETRIS RL", 42, 24, 37, WHITE, True)
    text.add(hud.subtitle, 44, 68, 15, MUTED, True)
    cv2.rectangle(img, (767, 28), (1055, 68), PANEL_LIGHT, -1)
    cv2.rectangle(img, (767, 28), (1055, 68), BORDER, 1)
    text.add(f"CONTROLLER  {hud.controller}", 911, 37, 17, WHITE, True, "center")
    text.add(hud.corner, 1055, 82, 14, MUTED, False, "right")

    text.add("GAME BOARD", BOARD_X, 98, 16, MUTED, True)
    draw_board(img, view, palette_of(env), env.padding)
    if hud.banner:
        cv2.rectangle(img, (BOARD_X, 382), (BOARD_X + 280, 438), (20, 13, 10), -1)
        cv2.rectangle(img, (BOARD_X, 382), (BOARD_X + 280, 438), ACCENT_SOFT, 1)
        text.add(hud.banner, BOARD_X + 140, 395, 24, WHITE, True, "center")

    # Top-left: HOLD (human) or MODEL card (agent).
    x, y, w, h = 45, 130, 295, 195
    draw_panel(img, x, y, w, h, True)
    text.add(hud.side_title, x + 20, y + 17, 22, WHITE, True)
    if hud.side is None:
        held = env.holder.get_tetrominoes()
        if held:
            draw_preview(img, held[0], x + 15, y + 48, w - 30, h - 60, 23)
        else:
            text.add("Empty", x + w // 2, y + 104, 18, MUTED, False, "center")
    else:
        yy = y + 56
        for label, value in hud.side:
            text.add(label, x + 22, yy, 15, MUTED)
            text.add(value, x + w - 20, yy - 1, 16, WHITE, True, "right")
            yy += 27

    # GAME INFO
    x, y, w, h = 45, 350, 295, 335
    draw_panel(img, x, y, w, h, True)
    text.add("GAME INFO", x + 20, y + 18, 22, WHITE, True)
    yy = y + 75
    for label, value in hud.stats:
        text.add(label, x + 25, yy, 18, MUTED)
        text.add(value, x + 260, yy - 2, 24, WHITE, True, "right")
        cv2.line(img, (x + 22, yy + 34), (x + w - 22, yy + 34), BORDER, 1)
        yy += 52

    # NEXT
    x, y, w, h = 755, 130, 300, 555
    draw_panel(img, x, y, w, h, True)
    text.add("NEXT", x + 20, y + 17, 22, WHITE, True)
    py = y + 58
    for i, pid in enumerate(env.queue.get_queue()[:4]):
        draw_preview(img, env.tetrominoes[int(pid)], x + 20, py, w - 40, 95, 20)
        if i < 3:
            cv2.line(img, (x + 30, py + 98), (x + w - 30, py + 98), BORDER, 1)
        py += 118
    if hud.next_note:
        text.add(hud.next_note, x + w // 2, y + h - 30, 13, MUTED, False, "center")

    # Controls bar
    draw_panel(img, 45, 720, 1010, 70)
    slot = 1010 // max(1, len(hud.controls))
    for i, (key, desc) in enumerate(hud.controls):
        cx = 45 + slot * i + slot // 2
        text.add(key, cx - 6, 742, 16, WHITE, True, "right")
        text.add(desc, cx + 6, 743, 15, MUTED)

    return text.apply(img)


# ---------------------------------------------------------------------------
# Start menu and overlays
# ---------------------------------------------------------------------------


def render_menu(items: list, selected: int) -> np.ndarray:
    """Start screen. ``items`` are (name, description, enabled) tuples."""
    img = background()
    text = Text()
    blocks = [(408, 70, 0), (450, 70, 0), (492, 70, 0), (492, 28, 0),
              (534, 70, 1), (534, 28, 1), (576, 70, 1)]
    for x, y, k in blocks:
        draw_block(img, x + 40, y, 42, ((50, 210, 255), (240, 90, 190))[k])
    text.add("TETRIS", CANVAS_W // 2, 140, 58, WHITE, True, "center")
    text.add("REINFORCEMENT LEARNING PROJECT", CANVAS_W // 2, 210, 25, ACCENT, True, "center")
    text.add("Policy search vs value learning  ·  linear vs MLP  ·  10 × 20 board",
             CANVAS_W // 2, 255, 18, MUTED, False, "center")
    text.add("CHOOSE A CONTROLLER", CANVAS_W // 2, 300, 15, MUTED, True, "center")

    x, w, h, y = 250, 600, 46, 330
    for i, (name, desc, enabled) in enumerate(items):
        on = i == selected
        draw_panel(img, x, y, w, h, False, PANEL_LIGHT if on else PANEL)
        if on:
            cv2.rectangle(img, (x, y), (x + w, y + h), ACCENT, 2)
        color = WHITE if enabled else MUTED
        text.add(f"{i + 1}", x + 22, y + 12, 18, ACCENT if on else MUTED, True)
        text.add(name, x + 55, y + 11, 19, color, True)
        text.add(desc if enabled else "not trained yet", x + w - 20, y + 14, 15, MUTED, False, "right")
        y += h + 8

    pulse = (np.sin(time.time() * 3.2) + 1) / 2
    b = int(170 + 70 * pulse)
    text.add("W / S or 1–6 to choose  ·  ENTER to start", CANVAS_W // 2, y + 18, 20,
             (b, b, 70), True, "center")
    text.add("ESC  Quit", CANVAS_W // 2, y + 56, 16, MUTED, False, "center")
    return text.apply(img)


def darken(img: np.ndarray) -> np.ndarray:
    return (img * 0.35).astype(np.uint8)


def render_pause(img: np.ndarray) -> np.ndarray:
    img = darken(img)
    text = Text()
    draw_panel(img, 350, 290, 400, 205, True)
    text.add("PAUSED", CANVAS_W // 2, 325, 38, WHITE, True, "center")
    text.add("P  Resume", CANVAS_W // 2, 392, 19, TEXT, True, "center")
    text.add("R  Restart     ESC  Menu", CANVAS_W // 2, 438, 16, MUTED, False, "center")
    return text.apply(img)


def render_game_over(img, rows: list, title="GAME OVER", footer=(), note="") -> np.ndarray:
    img = darken(img)
    text = Text()
    h = 160 + 45 * len(rows) + 33 * len(footer) + (30 if note else 0)
    y0 = (CANVAS_H - h) // 2 - 10
    draw_panel(img, 315, y0, 470, h, True)
    text.add(title, CANVAS_W // 2, y0 + 40, 40, WHITE, True, "center")
    text.add("FINAL RESULTS", CANVAS_W // 2, y0 + 102, 16, ACCENT, True, "center")
    yy = y0 + 150
    for label, value in rows:
        text.add(label, 405, yy, 18, MUTED)
        text.add(value, 690, yy - 2, 23, WHITE, True, "right")
        yy += 45
    if note:
        text.add(note, CANVAS_W // 2, yy - 5, 14, MUTED, False, "center")
        yy += 30
    for i, line in enumerate(footer):
        text.add(line, CANVAS_W // 2, yy + 10 + 33 * i, 17 if i == 0 else 15,
                 WHITE if i == 0 else MUTED, i == 0, "center")
    return text.apply(img)


# ---------------------------------------------------------------------------
# Side-by-side comparison of several agents
# ---------------------------------------------------------------------------

CMP_CELL = 22
CMP_COL_W = 250
CMP_GAP, CMP_H = 20, 720


def comparison_width(columns: int) -> int:
    return columns * CMP_COL_W + (columns + 1) * CMP_GAP


def render_comparison(env, views: list, header: str, subheader: str) -> np.ndarray:
    """One column per agent: title, board and live counters.

    ``views`` holds dicts with keys title, subtitle, view, lines, pieces, status.
    ``env`` supplies the palette and padding (identical for every game).
    """
    width = comparison_width(len(views))
    img = background(width, CMP_H)
    text = Text()
    text.add(header, 40, 22, 30, WHITE, True)
    text.add(subheader, 42, 62, 15, MUTED, True)
    palette, padding = palette_of(env), env.padding
    gap = CMP_GAP
    for i, v in enumerate(views):
        x = gap + i * (CMP_COL_W + gap)
        draw_panel(img, x, 100, CMP_COL_W, 600, True)
        text.add(v["title"], x + CMP_COL_W // 2, 114, 20, WHITE, True, "center")
        text.add(v["subtitle"], x + CMP_COL_W // 2, 141, 13, MUTED, False, "center")
        bx = x + (CMP_COL_W - 10 * CMP_CELL) // 2
        draw_board(img, v["view"], palette, padding, bx, 178, CMP_CELL)
        status = v.get("status")
        if status:
            cv2.rectangle(img, (bx, 360), (bx + 220, 410), (20, 13, 10), -1)
            cv2.rectangle(img, (bx, 360), (bx + 220, 410), BAD if status == "GAME OVER" else ACCENT_SOFT, 1)
            text.add(status, bx + 110, 373, 20, WHITE, True, "center")
        text.add("Lines", x + 22, 640, 16, MUTED)
        text.add(v["lines"], x + 105, 636, 22, WHITE, True, "right")
        text.add("Pieces", x + 130, 640, 16, MUTED)
        text.add(v["pieces"], x + CMP_COL_W - 18, 636, 22, WHITE, True, "right")
        if v.get("note"):
            text.add(v["note"], x + CMP_COL_W // 2, 674, 13, MUTED, False, "center")
    return text.apply(img)
