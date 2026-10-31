"""Pechi as half-block pixel art, drawn from assets/pechi.svg and animated in layers.

Each terminal cell shows two pixels, so the 26 by 28 grid fills 26 by 14 cells.
"""

from functools import lru_cache
from typing import NamedTuple

from rich.style import Style
from rich.text import Text

PALETTE = {
    "k": "#3B2C26",  # mask and ears
    "f": "#D9A66A",  # fawn coat
    "l": "#E7BE8A",  # light fawn
    "c": "#F4E2C6",  # cream chest
    "i": "#E9A99B",  # inner ear
    "b": "#EE9C8E",  # blush
    "e": "#2A1D18",  # eye
    "w": "#FFF8EE",  # eye highlight
    "n": "#17100D",  # nose
    "h": "#8C5A3C",  # harness
    "t": "#E8C468",  # tag
    "p": "#EE8E8E",  # tongue
    "T": "#9CCBE8",  # tear
}
OUTLINE_LIGHT = "#4B3527"  # outline on pale themes
OUTLINE_DARK = "#F4E2C6"  # outline on dark themes, or Pechi would vanish into them

WIDTH = 26
BLANK = "." * WIDTH
BASE = (
    "......oo..........oo......",
    ".....okko........okko.....",
    ".....okko........okko.....",
    "....okiiko......okiiko....",
    "....okiiko......okiiko....",
    "....okiiiko....okiiiko....",
    "...okkiiiko....okiiikko...",
    "...okkiiiooooooooiiikko...",
    "...okkiioffffffffoiikko...",
    "..ooooofffffkkfffffooooo..",
    ".offffffffffkkffffffffffo.",
    ".offffffffffffffffffffffo.",
    ".offffkkffffffffffkkffffo.",
    "offffffffffffffffffffffffo",
    "offffffffffffffffffffffffo",
    "offffffffffkkkkffffffffffo",
    "offfffffffkkkkkkfffffffffo",
    "offbbbfffkkkkkkkkfffbbbffo",
    "offbbbfffkkknnkkkfffbbbffo",
    ".offffffkkknnnnkkkffffffo.",
    ".offffffkkkkkkkkkkffffffo.",
    "..offffffkkkkkkkkffffffo..",
    "...ooofffffkkkkfffffooo...",
    ".....ohhhhhhtthhhhhho.....",
    ".....offffcttttcffffo.....",
    ".....offffccccccffffo.....",
    ".....ollloccccccolllo.....",
    "......oooo......oooo......",
)
EYE_LEFT, EYE_RIGHT, EYE_TOP = 5, 18, 14
EYES = {
    "open": ("ewe", "eee", "eee"),
    "blink": ("fff", "eee", "fff"),
    "happy": ("fef", "efe", "fff"),
    "cry": ("ewe", "eee", "eTe"),
    "look": ("fwe", "fee", "fee"),
    "love": ("pfp", "ppp", "fpf"),
    "wide": ("www", "wew", "www"),
    "cross": ("kkk", "ewe", "eee"),
}
MOUTHS = {
    "smile": ((20, 10, "l"), (20, 15, "l"), (21, 11, "llll")),
    "open": ((20, 11, "pppp"), (21, 11, "pppp")),
    "tongue": ((20, 11, "pppp"), (21, 11, "pppp"), (22, 12, "pp")),
    "wow": ((20, 12, "pp"), (21, 12, "pp")),
    "frown": ((20, 11, "llll"), (21, 10, "l"), (21, 15, "l")),
}
TAILS = (
    ((24, 21, "oo"), (25, 21, "ffko"), (26, 21, "ooo")),
    ((23, 22, "ooo"), (24, 21, "ffko"), (25, 21, "fo")),
    ((22, 22, "oo"), (23, 21, "oko"), (24, 21, "fo"), (25, 21, "fo")),
)


class Pose(NamedTuple):
    eyes: str = "open"
    mouth: str = "smile"
    tail: int = 0
    bob: int = 0  # 1 sinks the head onto the body, which reads as breathing


def stamp(grid: list[list[str]], layer: tuple) -> None:
    for y, x, row in layer:
        grid[y][x : x + len(row)] = row


def pixels(pose: Pose) -> tuple[str, ...]:
    grid = [list(row) for row in BASE]
    for offset, row in enumerate(EYES[pose.eyes]):
        y = EYE_TOP + offset
        stamp(grid, ((y, EYE_LEFT, row), (y, EYE_RIGHT, row)))
    stamp(grid, MOUTHS[pose.mouth])
    stamp(grid, TAILS[pose.tail])
    rows = ["".join(row) for row in grid]
    if pose.bob:  # the head drops one pixel and the row above the harness tucks away
        rows = [BLANK, *rows[:22], *rows[23:]]
    return tuple(rows)


@lru_cache(maxsize=256)
def render(pose: Pose, dark: bool = False) -> Text:
    """Two pixel rows become one terminal row of half blocks."""
    colors = {**PALETTE, "o": OUTLINE_DARK if dark else OUTLINE_LIGHT}
    rows = pixels(pose)
    art = Text(no_wrap=True, end="")
    for y in range(0, len(rows), 2):
        for top, bottom in zip(rows[y], rows[y + 1]):
            upper, lower = colors.get(top), colors.get(bottom)
            if upper and lower:
                art.append("▀", Style(color=upper, bgcolor=lower))
            elif upper:
                art.append("▀", Style(color=upper))
            elif lower:
                art.append("▄", Style(color=lower))
            else:
                art.append(" ")
        if y + 2 < len(rows):
            art.append("\n")
    return art
