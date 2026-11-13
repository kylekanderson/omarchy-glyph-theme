"""Hand-authored silhouettes for the Corvid wallpapers.

Every shape is a list of polygons in a normalised 0..100 coordinate space
(x right, y down). Polygons are unioned when rasterised, so overlapping parts
(body + legs + branch) are fine.
"""


from pathlib import Path

ART = Path(__file__).parent / "art"


def load_grid(name, scale=1):
    """A '#'/'.' text mask from src/art, optionally scaled up by whole cells."""
    import numpy as np

    rows = (ART / name).read_text().split()
    grid = np.array([[c == "#" for c in row] for row in rows])
    return np.kron(grid, np.ones((scale, scale), dtype=bool)) if scale > 1 else grid


def mirror(shape):
    """Flip a shape horizontally (a raven facing the other way)."""
    return [[(100 - x, y) for x, y in reversed(poly)] for poly in shape]


def _symmetric(half):
    """Close a left/right-symmetric outline from its right half (x >= 50)."""
    return half + [(100 - x, y) for x, y in reversed(half[1:-1])]


# A raven perched, facing right. Heavy beak, round head, a chunky body on a
# diagonal and a long wedge tail sweeping down to the bottom-left.
PERCH = [
    [
        (95.5, 22.5),  # beak tip
        (91, 18.0),
        (86, 15.5),
        (82, 13.5),  # forehead
        (77, 11.8),  # crown
        (72, 12.0),
        (68, 13.8),
        (65, 16.5),  # back of head
        (62, 20.5),  # nape
        (57, 24.0),
        (50, 28.5),
        (43, 34.0),
        (37, 40.0),
        (32, 46.5),
        (28, 53.0),
        (24, 60.0),
        (19, 68.0),
        (13, 77.0),
        (8, 85.0),  # tail tip (upper)
        (10, 88.0),
        (14, 88.5),  # tail tip (lower)
        (21, 82.0),
        (28, 75.5),
        (34, 71.0),  # under-tail
        (40, 69.0),  # vent
        (47, 68.5),
        (54, 66.5),  # belly
        (60, 62.5),
        (65, 57.0),
        (69, 50.5),  # chest
        (72, 43.0),
        (74, 36.0),
        (76, 31.0),  # shaggy throat
        (78.5, 27.5),
        (82, 25.5),  # chin
        (88, 24.5),
        (95.5, 22.5),
    ],
    # legs
    [(50.5, 67.0), (53.0, 66.5), (53.0, 79.0), (50.5, 79.0)],
    [(56.5, 64.5), (59.0, 63.5), (59.5, 79.0), (57.0, 79.0)],
]

# Toe line / perch. Kept separate so it can be drawn with its own glyphs.
PERCH_FEET_Y = 79.0
BRANCH = [
    [(18, 79.0), (92, 80.5), (92, 83.0), (18, 82.5)],
]

def _feather(base, angle, length, width):
    """A tapered primary feather growing from `base` at `angle` degrees (0 = right)."""
    import math

    bx, by = base
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    px, py = -dy, dx
    tx, ty = bx + dx * length, by + dy * length
    hw, tw = width / 2, width * 0.32
    return [
        (bx + px * hw, by + py * hw),
        (tx + px * tw, ty + py * tw),
        (tx + dx * width * 0.35, ty + dy * width * 0.35),
        (tx - px * tw, ty - py * tw),
        (bx - px * hw, by - py * hw),
    ]


# Splayed primaries on the right wing: (base, angle, length).
_PRIMARIES = [
    ((85.0, 34.6), -12, 13.5),
    ((85.0, 39.6), 2, 13.0),
    ((84.0, 44.6), 16, 12.0),
    ((82.5, 49.4), 30, 11.0),
    ((80.0, 53.6), 44, 9.5),
]
_RIGHT_FEATHERS = [_feather(b, a, ln, 4.4) for b, a, ln in _PRIMARIES]

# A raven gliding overhead, seen from below: splayed primaries, wedge tail,
# heavy head and beak at the top.
GLIDE = [
    _symmetric(
        [
            (50.0, 21.0),  # beak tip
            (51.8, 22.2),
            (53.6, 23.6),  # head
            (54.6, 26.0),
            (53.8, 28.6),  # neck
            (56.0, 31.2),  # shoulder / leading edge
            (62.0, 29.4),
            (69.0, 27.4),  # wrist, pushed forward
            (76.0, 28.0),
            (82.0, 30.0),
            (87.0, 32.6),  # hand
            (87.5, 40.0),
            (85.5, 48.0),
            (82.0, 55.0),
            (78.0, 57.5),  # trailing edge, bulging at the secondaries
            (71.0, 59.5),
            (64.0, 60.5),
            (57.0, 59.8),
            (59.0, 66.0),  # wedge tail
            (60.0, 72.5),
            (57.5, 79.0),
            (50.0, 84.0),  # tail point
        ]
    ),
    *_RIGHT_FEATHERS,
    *mirror(_RIGHT_FEATHERS),
]
