"""Camera database for synthetic image generation.

Each entry is keyed by "Manufacturer — Model" and contains:
    width_px, height_px  : sensor dimensions in pixels
    pixel_um             : pixel pitch in micrometres
    full_well_e          : full-well capacity in electrons
    read_noise_e         : typical read noise in electrons (at lowest gain)
"""

from __future__ import annotations

CAMERAS: dict[str, dict] = {
    # ── ZWO ASI ────────────────────────────────────────────────────────────
    "ZWO — ASI6200MM Pro": {
        "width_px": 9576, "height_px": 6388,
        "pixel_um": 3.76, "full_well_e": 51_000, "read_noise_e": 1.5,
    },
    "ZWO — ASI2600MM Pro": {
        "width_px": 6248, "height_px": 4176,
        "pixel_um": 3.76, "full_well_e": 51_000, "read_noise_e": 1.5,
    },
    "ZWO — ASI1600MM Pro": {
        "width_px": 4656, "height_px": 3520,
        "pixel_um": 3.80, "full_well_e": 20_000, "read_noise_e": 1.8,
    },
    "ZWO — ASI294MM Pro": {
        "width_px": 4144, "height_px": 2822,
        "pixel_um": 4.63, "full_well_e": 63_700, "read_noise_e": 1.8,
    },
    "ZWO — ASI183MM Pro": {
        "width_px": 5496, "height_px": 3672,
        "pixel_um": 2.40, "full_well_e": 15_000, "read_noise_e": 1.6,
    },
    "ZWO — ASI533MM Pro": {
        "width_px": 3008, "height_px": 3008,
        "pixel_um": 3.76, "full_well_e": 50_000, "read_noise_e": 1.5,
    },
    "ZWO — ASI2400MC Pro": {
        "width_px": 6072, "height_px": 4042,
        "pixel_um": 5.94, "full_well_e": 75_000, "read_noise_e": 2.0,
    },
    "ZWO — ASI071MC Pro": {
        "width_px": 4944, "height_px": 3284,
        "pixel_um": 4.78, "full_well_e": 46_000, "read_noise_e": 2.5,
    },
    "ZWO — ASI174MM Mini": {
        "width_px": 1936, "height_px": 1216,
        "pixel_um": 5.86, "full_well_e": 32_000, "read_noise_e": 7.0,
    },
    # ── QHY ────────────────────────────────────────────────────────────────
    "QHY — QHY600M": {
        "width_px": 9576, "height_px": 6388,
        "pixel_um": 3.76, "full_well_e": 80_000, "read_noise_e": 1.4,
    },
    "QHY — QHY268M": {
        "width_px": 6280, "height_px": 4210,
        "pixel_um": 3.76, "full_well_e": 51_000, "read_noise_e": 1.45,
    },
    "QHY — QHY294M": {
        "width_px": 4164, "height_px": 2796,
        "pixel_um": 4.63, "full_well_e": 80_000, "read_noise_e": 2.6,
    },
    "QHY — QHY183M": {
        "width_px": 5544, "height_px": 3694,
        "pixel_um": 2.40, "full_well_e": 15_000, "read_noise_e": 1.65,
    },
    "QHY — QHY163M": {
        "width_px": 4656, "height_px": 3522,
        "pixel_um": 3.80, "full_well_e": 20_000, "read_noise_e": 3.0,
    },
    "QHY — QHY533M": {
        "width_px": 3008, "height_px": 3008,
        "pixel_um": 3.76, "full_well_e": 50_000, "read_noise_e": 1.5,
    },
    "QHY — QHY485C": {
        "width_px": 3848, "height_px": 2168,
        "pixel_um": 2.90, "full_well_e": 20_000, "read_noise_e": 1.4,
    },
    "QHY — QHY128C": {
        "width_px": 4688, "height_px": 3520,
        "pixel_um": 5.97, "full_well_e": 70_000, "read_noise_e": 3.0,
    },
    # ── Player One ──────────────────────────────────────────────────────────
    "Player One — Apollo-M Max": {
        "width_px": 9576, "height_px": 6388,
        "pixel_um": 3.76, "full_well_e": 51_000, "read_noise_e": 1.5,
    },
    "Player One — Poseidon-M": {
        "width_px": 9576, "height_px": 6388,
        "pixel_um": 3.76, "full_well_e": 51_000, "read_noise_e": 1.3,
    },
    "Player One — Saturn-M Pro": {
        "width_px": 3008, "height_px": 3008,
        "pixel_um": 3.76, "full_well_e": 50_000, "read_noise_e": 1.5,
    },
    "Player One — Neptune-M Pro": {
        "width_px": 2712, "height_px": 1538,
        "pixel_um": 2.90, "full_well_e": 18_000, "read_noise_e": 0.7,
    },
    "Player One — Mars-C Pro": {
        "width_px": 3840, "height_px": 2160,
        "pixel_um": 2.90, "full_well_e": 25_000, "read_noise_e": 1.0,
    },
    "Player One — Uranus-C Pro": {
        "width_px": 6072, "height_px": 4042,
        "pixel_um": 5.94, "full_well_e": 75_000, "read_noise_e": 2.5,
    },
    "Player One — Mercury-M": {
        "width_px": 1920, "height_px": 1080,
        "pixel_um": 2.90, "full_well_e": 15_000, "read_noise_e": 0.8,
    },
}

# Default selection shown on first open
DEFAULT_CAMERA = "ZWO — ASI2600MM Pro"
