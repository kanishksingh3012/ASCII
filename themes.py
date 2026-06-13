"""
Theme presets.

Each theme is a small dict so it's trivial for the UI to list, describe,
and override. Monochrome themes use (bg, fg). The True Color engine uses
bg as a backdrop and inherits fg from each source pixel.
"""

THEMES: dict[str, dict] = {
    "Monochrome Dark": {
        "bg": (10, 10, 10),
        "fg": (230, 230, 230),
        "mode": "mono",
        "description": "Classic light-on-dark. Safe, timeless, always reads well.",
        "invert_luminance": False,
    },
    "Monochrome Light": {
        "bg": (245, 245, 245),
        "fg": (18, 18, 18),
        "mode": "mono",
        "description": "Newsprint-style dark-on-light. Great for print exports.",
        "invert_luminance": True,
    },
    "Matrix Green": {
        "bg": (0, 8, 0),
        "fg": (0, 255, 70),
        "mode": "mono",
        "description": "Phosphor green on black. Wake up, Neo.",
        "invert_luminance": False,
    },
    "Amber Terminal": {
        "bg": (16, 10, 0),
        "fg": (255, 176, 0),
        "mode": "mono",
        "description": "Vintage VT100 amber glow.",
        "invert_luminance": False,
    },
    "Synthwave Pink": {
        "bg": (18, 0, 40),
        "fg": (255, 60, 220),
        "mode": "mono",
        "description": "Neon magenta over deep indigo. Retro-futurist nights.",
        "invert_luminance": False,
    },
    "True Color": {
        "bg": (0, 0, 0),
        "fg": None,
        "mode": "color",
        "description": "Every character inherits the RGB of the source pixel.",
        "invert_luminance": False,
    },
}
