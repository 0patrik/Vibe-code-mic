#!/usr/bin/env python3
"""Generate a macOS .icns app icon with a stylized microphone design."""

import math
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 1024
OUTPUT_PNG = os.path.join(BASE_DIR, "icon_1024.png")
OUTPUT_ICNS = os.path.join(BASE_DIR, "app_icon.icns")
ICONSET_DIR = os.path.join(BASE_DIR, "app_icon.iconset")


def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_rounded_rect(draw, bbox, radius, fill):
    """Draw a filled rounded rectangle."""
    x0, y0, x1, y1 = bbox
    # Four corners
    draw.ellipse([x0, y0, x0 + 2 * radius, y0 + 2 * radius], fill=fill)
    draw.ellipse([x1 - 2 * radius, y0, x1, y0 + 2 * radius], fill=fill)
    draw.ellipse([x0, y1 - 2 * radius, x0 + 2 * radius, y1], fill=fill)
    draw.ellipse([x1 - 2 * radius, y1 - 2 * radius, x1, y1], fill=fill)
    # Connecting rectangles
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)


def generate_icon():
    """Generate the 1024x1024 icon PNG."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # --- Background: rounded rectangle with gradient ---
    color_top = (0x1A, 0x1A, 0x2E)     # #1a1a2e
    color_bottom = (0x6C, 0x34, 0x83)   # #6c3483

    corner_radius = int(SIZE * 0.22)  # macOS-style rounding
    margin = 2

    # Build gradient by drawing horizontal lines clipped to rounded rect shape
    # First, create a mask for the rounded rect
    mask = Image.new("L", (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    draw_rounded_rect(mask_draw, [margin, margin, SIZE - margin, SIZE - margin],
                      corner_radius, fill=255)

    # Create gradient image
    grad = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    grad_draw = ImageDraw.Draw(grad)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        c = lerp_color(color_top, color_bottom, t)
        grad_draw.line([(0, y), (SIZE - 1, y)], fill=(*c, 255))

    # Apply mask
    img.paste(grad, mask=mask)
    draw = ImageDraw.Draw(img)

    # --- Microphone body (rounded rectangle with semicircle top) ---
    cx, cy = SIZE // 2, SIZE // 2
    mic_w = int(SIZE * 0.18)   # half-width of mic body
    mic_h = int(SIZE * 0.28)   # height of rectangular part
    mic_top = cy - int(SIZE * 0.22)
    mic_bottom = mic_top + mic_h
    mic_color = (255, 255, 255, 240)

    # Semicircle top of microphone
    draw.ellipse([cx - mic_w, mic_top - mic_w, cx + mic_w, mic_top + mic_w],
                 fill=mic_color)

    # Rectangular body
    body_radius = int(mic_w * 0.15)
    draw_rounded_rect(draw, [cx - mic_w, mic_top, cx + mic_w, mic_bottom],
                      body_radius, fill=mic_color)

    # --- Microphone cradle arc ---
    cradle_w = int(SIZE * 0.24)
    cradle_top = mic_top + int(mic_h * 0.3)
    cradle_bottom = mic_bottom + int(SIZE * 0.10)
    arc_thickness = int(SIZE * 0.025)

    # Draw the cradle as a thick arc (bottom half of an ellipse)
    for offset in range(arc_thickness):
        draw.arc(
            [cx - cradle_w - offset, cradle_top - offset,
             cx + cradle_w + offset, cradle_bottom + offset],
            start=0, end=180,
            fill=mic_color, width=2
        )

    # --- Stand (vertical line below cradle) ---
    stand_top = cradle_bottom - int(SIZE * 0.01)
    stand_bottom = stand_top + int(SIZE * 0.10)
    stand_half_w = int(SIZE * 0.015)
    draw_rounded_rect(draw, [cx - stand_half_w, stand_top, cx + stand_half_w, stand_bottom],
                      stand_half_w, fill=mic_color)

    # --- Base (horizontal bar) ---
    base_y = stand_bottom
    base_half_w = int(SIZE * 0.09)
    base_h = int(SIZE * 0.025)
    base_radius = base_h // 2
    draw_rounded_rect(draw, [cx - base_half_w, base_y, cx + base_half_w, base_y + base_h],
                      base_radius, fill=mic_color)

    # --- Sound wave arcs on the sides ---
    wave_color_1 = (180, 200, 255, 130)
    wave_color_2 = (180, 200, 255, 80)
    wave_cy = mic_top + mic_h // 2  # center vertically on mic body
    wave_thickness = int(SIZE * 0.018)

    for i, (radius_mult, color) in enumerate([(0.28, wave_color_1), (0.36, wave_color_2)]):
        r = int(SIZE * radius_mult)
        # Right side arcs
        bbox = [cx - r, wave_cy - r, cx + r, wave_cy + r]
        # We draw arcs only on the right side (roughly -50 to 50 degrees)
        # Right
        draw.arc(bbox, start=-40, end=40, fill=color, width=wave_thickness)
        # Left
        draw.arc(bbox, start=140, end=220, fill=color, width=wave_thickness)

    # --- Small glow/highlight on mic top ---
    highlight_r = int(mic_w * 0.35)
    highlight_cy = mic_top - int(mic_w * 0.15)
    highlight = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(highlight)
    h_draw.ellipse(
        [cx - highlight_r, highlight_cy - highlight_r,
         cx + highlight_r, highlight_cy + highlight_r],
        fill=(255, 255, 255, 60)
    )
    img = Image.alpha_composite(img, highlight)

    img.save(OUTPUT_PNG, "PNG")
    print(f"Generated {OUTPUT_PNG}")
    return img


def create_iconset():
    """Create .iconset folder with all required sizes and convert to .icns."""
    if os.path.exists(ICONSET_DIR):
        shutil.rmtree(ICONSET_DIR)
    os.makedirs(ICONSET_DIR)

    # Required icon sizes for macOS .iconset
    # Format: (filename, pixel_size)
    sizes = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_64x64.png", 64),
        ("icon_64x64@2x.png", 128),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]

    base_img = Image.open(OUTPUT_PNG)

    for filename, px in sizes:
        resized = base_img.resize((px, px), Image.LANCZOS)
        out_path = os.path.join(ICONSET_DIR, filename)
        resized.save(out_path, "PNG")

    print(f"Created iconset at {ICONSET_DIR}")

    # Convert to .icns using iconutil
    if os.path.exists(OUTPUT_ICNS):
        os.remove(OUTPUT_ICNS)

    result = subprocess.run(
        ["iconutil", "-c", "icns", ICONSET_DIR, "-o", OUTPUT_ICNS],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"iconutil error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated {OUTPUT_ICNS}")

    # Clean up
    shutil.rmtree(ICONSET_DIR)
    if os.path.exists(OUTPUT_PNG):
        os.remove(OUTPUT_PNG)
    print("Cleaned up temporary files.")


if __name__ == "__main__":
    generate_icon()
    create_iconset()
    print("Done!")
