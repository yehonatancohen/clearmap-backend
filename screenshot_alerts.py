"""
Alert Screenshot Generator -- captures the clearmap in both themes,
overlays the logo and an active-alerts legend, crops to square, and saves locally.

Usage:
    python screenshot_alerts.py [--url URL] [--output DIR]

Requires: playwright, Pillow
One-time setup: playwright install chromium
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont

import requests as http_requests


# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_URL = "https://clearmap.co.il"
VIEWPORT_SIZE = 900  # square viewport
# Logo directory — works locally (relative path to clear-map) and in Docker (bundled)
_LOCAL_LOGO_DIR = Path(__file__).parent.parent.parent / "clear-map" / "public"
_DOCKER_LOGO_DIR = Path(__file__).parent / "public"
LOGO_DIR = _LOCAL_LOGO_DIR if _LOCAL_LOGO_DIR.exists() else _DOCKER_LOGO_DIR
OUTPUT_DIR = Path(__file__).parent / "screenshots"

# Firebase REST API for fetching active alerts
FIREBASE_DB_URL = "https://clear-map-f20d0-default-rtdb.europe-west1.firebasedatabase.app"
FIREBASE_ALERTS_PATH = "/public_state/active_alerts.json"

# ── Legend config ───────────────────────────────────────────────────────────

# Status → (color, Hebrew label) — matching the BOTTOM PANEL in IntelBanner.tsx
LEGEND_ITEMS = [
    ("alert",          (255,  42,  42), "התרעות ירי רקטות וטילים"),
    ("uav",            (224,  64, 251), "התרעות חדירת כלי טיס עוין"),
    ("terrorist",      (255,   0,  85), "חדירת מחבלים"),
    ("pre_alert",      (255, 106,   0), "התרעות מוקדמות"),
    ("after_alert",    (255,  80,  80), "להישאר בממ\"ד"),
]


def _bidi_text(text: str) -> str:
    """Convert Hebrew (RTL) text to visual order based on Pillow's capabilities.
    
    If Pillow has libraqm (Linux/Docker), it natively supports RTL, so we return the
    text unchanged (or it gets double-reversed into gibberish).
    If Pillow lacks libraqm (Windows dev), we use python-bidi to visually reorder it.
    """
    from PIL import features
    if features.check('raqm'):
        return text  # Native shaping handles it
        
    try:
        from bidi.algorithm import get_display
        return get_display(text, base_dir='R')
    except ImportError:
        pass

    # Basic fallback for Windows without python-bidi
    import re
    tokens = re.findall(r'\S+|\s+', text)
    visual = []
    for token in reversed(tokens):
        if token.isspace() or token.isdigit() or re.match(r'^[0-9\W]+$', token):
            visual.append(token)
        else:
            visual.append(token[::-1])
    return "".join(visual)

def _load_hebrew_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a Hebrew-capable font."""
    candidates = [
        "public/hebrew_font.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # Linux / Docker
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def fetch_active_statuses() -> tuple[set[str], dict[str, int]]:
    """Fetch currently active alert statuses and counts from Firebase REST API.

    Returns (set_of_statuses, {status: count}).
    """
    try:
        resp = http_requests.get(
            f"{FIREBASE_DB_URL}{FIREBASE_ALERTS_PATH}",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return set(), {}
        counts: dict[str, int] = {}
        for v in data.values():
            if isinstance(v, dict):
                s = v.get("status", "alert")
                counts[s] = counts.get(s, 0) + 1
        return set(counts.keys()), counts
    except Exception as e:
        print(f"  [warn] Could not fetch active statuses: {e}")
        return set(), {}


def draw_legend(img: Image.Image, active_statuses: set[str], theme: str,
                counts: dict[str, int] | None = None) -> Image.Image:
    """Draw a small legend overlay in the bottom-left corner showing active alert types.

    Only draws legend items whose status is in active_statuses.
    When counts is provided, the count is shown next to each label.
    """
    items = [(color, label, status) for status, color, label in LEGEND_ITEMS
             if status in active_statuses]

    if not items:
        return img  # No active alerts → no legend

    img = img.copy().convert("RGBA")
    w, h = img.size

    # Font sizing — relative to image
    font_size = max(18, int(w * 0.024))
    font = _load_hebrew_font(font_size)
    count_font = _load_hebrew_font(int(font_size * 0.9))
    dot_radius = max(5, int(font_size * 0.4))
    row_height = int(font_size * 1.8)
    padding = int(w * 0.02)
    inner_pad = int(w * 0.012)
    count_pad = int(w * 0.008)  # gap between count badge and label

    # Measure text widths to determine legend box size
    dummy_draw = ImageDraw.Draw(img)
    max_text_w = 0
    for _, label, status in items:
        cnt = counts.get(status, 0) if counts else 0
        if status == "after_alert":
            display = f"{cnt} מקומות להישאר במרחב מוגן" if cnt else "מקומות להישאר במרחב מוגן"
        else:
            display = f"{cnt} {label}" if cnt else label
            
        visual_label = _bidi_text(display)
        # Handle native raqm bbox measuring
        from PIL import features
        kwargs = {"direction": "rtl", "language": "he"} if features.check('raqm') else {}
        bbox = dummy_draw.textbbox((0, 0), visual_label, font=font, **kwargs)
        text_w = bbox[2] - bbox[0]
        max_text_w = max(max_text_w, text_w)

    legend_w = dot_radius * 2 + inner_pad + max_text_w + padding * 2
    legend_h = row_height * len(items) + padding * 2

    # Position: bottom-left corner
    lx = padding
    ly = h - legend_h - padding

    # Draw semi-transparent dark background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    bg_color = (20, 20, 30, 180) if theme == "dark" else (30, 30, 40, 170)
    # Rounded rectangle
    overlay_draw.rounded_rectangle(
        [lx, ly, lx + legend_w, ly + legend_h],
        radius=int(w * 0.012),
        fill=bg_color,
    )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Draw each legend row (RTL: dot on right, text to its left)
    for i, (color, label, status) in enumerate(items):
        row_y = ly + padding + i * row_height

        # Dot position — right side of the legend box
        dot_cx = lx + legend_w - padding - dot_radius
        dot_cy = row_y + row_height // 2

        draw.ellipse(
            [dot_cx - dot_radius, dot_cy - dot_radius,
             dot_cx + dot_radius, dot_cy + dot_radius],
            fill=(*color, 255),
        )

        # Build display text with count
        cnt = counts.get(status, 0) if counts else 0
        if status == "after_alert":
            display = f"{cnt} מקומות להישאר במרחב מוגן" if cnt else "מקומות להישאר במרחב מוגן"
        else:
            display = f"{cnt} {label}" if cnt else label
            
        visual_label = _bidi_text(display)
        from PIL import features
        kwargs = {"direction": "rtl", "language": "he"} if features.check('raqm') else {}
        bbox = draw.textbbox((0, 0), visual_label, font=font, **kwargs)
        text_w = bbox[2] - bbox[0]
        text_x = dot_cx - dot_radius - inner_pad - text_w
        text_y = row_y + (row_height - font_size) // 2
        
        draw.text((text_x, text_y), visual_label, fill=(255, 255, 255, 230), font=font, **kwargs)

    return img


def draw_uav_disclaimer(img: Image.Image, theme: str) -> Image.Image:
    """Draw a disclaimer centered at the bottom about UAV predictions."""
    w, h = img.size
    
    text = "* שימו לב: מיקומי כלי הטיס הם בגדר השערת המערכת בלבד ואין להתבסס עליהם."
    visual_text = _bidi_text(text)
    
    font_size = max(15, int(w * 0.022))
    font = _load_hebrew_font(font_size)
    padding = int(w * 0.012)
    
    # Measure text width
    dummy_draw = ImageDraw.Draw(img)
    from PIL import features
    kwargs = {"direction": "rtl", "language": "he"} if features.check('raqm') else {}
    bbox = dummy_draw.textbbox((0, 0), visual_text, font=font, **kwargs)
    text_w = bbox[2] - bbox[0]
    text_h = int(font_size * 1.2)
    
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2
    
    # Position: top left part of the screen to avoid overlapping with logo
    lx = padding * 2
    ly = padding * 2
    
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    bg_color = (20, 20, 30, 200) if theme == "dark" else (30, 30, 40, 190)
    overlay_draw.rounded_rectangle(
        [lx, ly, lx + box_w, ly + box_h],
        radius=int(w * 0.01),
        fill=bg_color,
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img)
    
    text_x = lx + padding
    text_y = ly + padding + (box_h - padding*2 - text_h) // 2
    
    draw.text((text_x, text_y), visual_text, fill=(253, 224, 71, 230), font=font, **kwargs) # Yellow-300
    
    return img


def hide_ui_overlays(page):
    """Inject CSS to hide all UI overlay elements, keeping only the map + polygons."""
    page.evaluate("""
        const style = document.createElement('style');
        style.id = 'screenshot-hide-ui';
        style.textContent = `
            /* Hide all glass overlays (top bar, bottom bar, about, legend, etc.) */
            .glass-overlay,
            [class*="absolute top-3"],
            [class*="absolute bottom-4"],
            [class*="absolute top-16"],
            [class*="absolute bottom-16"],
            [class*="z-[1000]"],
            [class*="z-[1001]"],
            [class*="z-[1002]"],
            [class*="z-[2000]"] {
                display: none !important;
            }
            /* Hide leaflet controls */
            .leaflet-control-container {
                display: none !important;
            }
        `;
        document.head.appendChild(style);
    """)


def show_ui_overlays(page):
    """Remove the injected CSS to show UI again."""
    page.evaluate("""
        const style = document.getElementById('screenshot-hide-ui');
        if (style) style.remove();
    """)


def switch_theme(page, target_theme: str):
    """Switch the map theme by clicking UI buttons."""
    # First make sure UI is visible
    show_ui_overlays(page)
    time.sleep(0.3)

    try:
        # Click the about/logo button (first button in the top bar)
        logo_btn = page.locator('button').first
        logo_btn.click()
        time.sleep(0.5)

        # Click the appropriate theme button
        if target_theme == "light":
            page.locator('text=בוקר').click()
        else:
            page.locator('text=לילה').click()

        time.sleep(1)

        # Close the about panel by clicking the logo again
        logo_btn.click()
        time.sleep(0.3)
    except Exception as e:
        print(f"  [warn] Could not switch theme via UI: {e}")


def capture_screenshot(page, theme: str, output_dir: Path) -> Path:
    """Take a screenshot of just the map area."""
    raw_path = output_dir / f"raw_{theme}.png"
    page.screenshot(path=str(raw_path), full_page=False)
    return raw_path


def overlay_logo_and_crop(screenshot_path: Path, logo_path: Path, output_path: Path,
                          size: int, active_statuses: set[str] | None = None,
                          theme: str = "dark",
                          counts: dict[str, int] | None = None) -> Path:
    """Overlay logo in top-right corner, add legend with counts, and crop to square."""
    img = Image.open(screenshot_path).convert("RGBA")

    # Center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Resize to target size
    img = img.resize((size, size), Image.LANCZOS)

    # Load and resize logo
    if logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA")

        # Scale logo to ~25% of image width
        logo_w = int(size * 0.25)
        logo_h = int(logo.height * (logo_w / logo.width))
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        # Position in top-right corner with padding
        padding = int(size * 0.03)
        x = size - logo_w - padding
        y = padding

        # Paste logo with alpha compositing
        img.paste(logo, (x, y), logo)
    else:
        print(f"  [warn] Logo not found: {logo_path}")

    # Draw legend overlay if we have active statuses
    if active_statuses:
        img = draw_legend(img, active_statuses, theme, counts=counts)
        
        # Add UAV disclaimer if UAVs are present
        if "uav" in active_statuses:
            img = draw_uav_disclaimer(img, theme)

    # Convert to RGB and save
    img = img.convert("RGB")
    img.save(str(output_path), "PNG", quality=95)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Capture alert map screenshots")
    parser.add_argument("--url", default=DEFAULT_URL, help="Map URL to capture")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--output-file", help="Specific output file path for a single capture")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark", help="Target theme when using --output-file")
    parser.add_argument("--size", type=int, default=1080, help="Output square size in pixels")
    parser.add_argument("--no-legend", action="store_true", help="Skip legend overlay")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Fetch active alert statuses for the legend
    active_statuses: set[str] = set()
    status_counts: dict[str, int] = {}
    if not args.no_legend:
        print("[+] Fetching active alert statuses...")
        active_statuses, status_counts = fetch_active_statuses()
        if active_statuses:
            print(f"  Active types: {', '.join(sorted(active_statuses))}")
            print(f"  Counts: {status_counts}")
        else:
            print("  No active alerts — legend will be skipped.")

    print("[MAP] Alert Screenshot Generator")
    print(f"  URL: {args.url}")
    if args.output_file:
        print(f"  Output File: {args.output_file}")
    else:
        print(f"  Output Dir: {output_dir}")
    print(f"  Size: {args.size}x{args.size}")
    print()

    # Disable heavy browser features to save CPU inside the VM
    chromium_args = [
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-extensions",
        "--mute-audio",
        "--disable-background-networking",
        "--disable-site-isolation-trials",
        "--js-flags=--max-old-space-size=512"
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=chromium_args)
        context = browser.new_context(
            viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
            device_scale_factor=1.5,  # Reduced from 2.0 to save memory footprint
        )
        page = context.new_page()

        if args.output_file:
            print(f"[+] Capturing single '{args.theme}' theme to {args.output_file}...")
            try:
                # Use load or domcontentloaded instead of networkidle, as networkidle can hang 
                # indefinitely if the page has constant background polling or websockets.
                page.goto(args.url, wait_until="load", timeout=45000)
                page.wait_for_selector(".leaflet-container", timeout=15000)
                time.sleep(3)  # Let map render
                
                # Switch theme if needed
                if args.theme == "light":
                    switch_theme(page, "light")
                    time.sleep(3)
                    
                hide_ui_overlays(page)
                time.sleep(0.5)
                
                raw_path = capture_screenshot(page, args.theme, output_dir)
                logo_path = LOGO_DIR / f"logo-{args.theme}-theme.png"
                final_path = Path(args.output_file)
                overlay_logo_and_crop(raw_path, logo_path, final_path, args.size,
                                      active_statuses=active_statuses, theme=args.theme,
                                      counts=status_counts)
                print(f"  [OK] Saved: {final_path}")
                raw_path.unlink(missing_ok=True)
                browser.close()
                return final_path
            except Exception as e:
                print(f"  [ERR] Playwright screenshot failed: {e}")
                browser.close()
                sys.exit(1)
            
        else:
            # Default behavior: capture both themes to the output dir
            # -- Dark theme screenshot -------------------------------------------
            print("[+] Capturing dark theme...")
            page.goto(args.url, wait_until="networkidle")
    
            # Wait for map tiles and polygons to render
            page.wait_for_selector(".leaflet-container", timeout=15000)
            time.sleep(3)  # Let initial fitBounds or default zoom settle
    
            print("  [+] Adjusting zoom level...")
            page.mouse.move(VIEWPORT_SIZE / 2, VIEWPORT_SIZE / 2)
            page.mouse.wheel(0, 0)  # Scroll up to zoom in (less aggressive)
            time.sleep(2)  # Let tiles load after zoom
    
            # Hide UI overlays
            hide_ui_overlays(page)
            time.sleep(0.5)
    
            dark_raw = capture_screenshot(page, "dark", output_dir)
    
            dark_logo = LOGO_DIR / "logo-dark-theme.png"
            dark_output = output_dir / f"alert_dark_{timestamp}.png"
            overlay_logo_and_crop(dark_raw, dark_logo, dark_output, args.size,
                                  active_statuses=active_statuses, theme="dark",
                                  counts=status_counts)
            print(f"  [OK] Saved: {dark_output}")
    
            # -- Light theme screenshot ------------------------------------------
            print("[+] Capturing light theme...")
            switch_theme(page, "light")
            time.sleep(3)  # Let tiles fully reload
    
            # Re-hide UI overlays
            hide_ui_overlays(page)
            time.sleep(0.5)
    
            light_raw = capture_screenshot(page, "light", output_dir)
    
            light_logo = LOGO_DIR / "logo-light-theme.png"
            light_output = output_dir / f"alert_light_{timestamp}.png"
            overlay_logo_and_crop(light_raw, light_logo, light_output, args.size,
                                  active_statuses=active_statuses, theme="light",
                                  counts=status_counts)
            print(f"  [OK] Saved: {light_output}")
    
            # Cleanup raw files
            dark_raw.unlink(missing_ok=True)
            light_raw.unlink(missing_ok=True)
    
            browser.close()
            print()
            print(f"[DONE] Screenshots saved to {output_dir}")
            return dark_output, light_output


def quick_capture_and_send(bot_token: str, chat_id: str,
                           caption: str = "", url: str = DEFAULT_URL,
                           size: int = 1080) -> bool:
    """Capture a dark-theme screenshot with legend and send via Telegram Bot API.

    Returns True on success, False on failure.
    """
    try:
        active_statuses, status_counts = fetch_active_statuses()
        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        chromium_args = [
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--disable-extensions",
            "--mute-audio",
            "--disable-background-networking",
            "--disable-site-isolation-trials",
            "--js-flags=--max-old-space-size=512"
        ]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=chromium_args)
            context = browser.new_context(
                viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
                device_scale_factor=1.5,
            )
            page = context.new_page()
            page.goto(url, wait_until="load", timeout=30000)
            page.wait_for_selector(".leaflet-container", timeout=20000)
            time.sleep(3)

            hide_ui_overlays(page)
            time.sleep(0.5)

            raw_path = capture_screenshot(page, "dark", output_dir)
            browser.close()

        final_path = output_dir / "send_latest.png"
        dark_logo = LOGO_DIR / "logo-dark-theme.png"
        overlay_logo_and_crop(
            raw_path, dark_logo, final_path, size,
            active_statuses=active_statuses, theme="dark",
            counts=status_counts,
        )
        raw_path.unlink(missing_ok=True)

        # Send via Telegram Bot API
        tg_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(final_path, "rb") as photo:
            resp = http_requests.post(
                tg_url,
                data={"chat_id": chat_id, "caption": caption or "📸 Alert Screenshot"},
                files={"photo": ("alert_screenshot.png", photo, "image/png")},
                timeout=30,
            )
        if resp.ok:
            print(f"  [OK] Screenshot sent to Telegram chat {chat_id}")
            return True
        else:
            print(f"  [ERR] Telegram send failed: {resp.text[:200]}")
            return False

    except Exception as e:
        print(f"  [ERR] Screenshot send error: {e}")
        return False


if __name__ == "__main__":
    main()
