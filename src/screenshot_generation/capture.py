"""
Manages an Xvfb virtual display and captures real pixels of the rendered
spreadsheet window, including the surrounding application chrome (menu bar,
toolbar, sheet tabs, cell reference box) -- not just the cell grid. This
matters because a model trained only on tightly-cropped grids will fail on
real-world screenshots, which almost always include surrounding UI.

Approach:
  1. Start Xvfb at the config's target resolution (simulates different
     monitor sizes -- see screenshot_variation.yaml `display.resolution_options`)
  2. Launch/attach to soffice pointed at this display
  3. Use `xdotool` to find the window and optionally resize it (simulating
     a non-maximized window, which happens often in real screenshots)
  4. Use `import` (ImageMagick) or `scrot` to capture window pixels
  5. Apply post-capture crop per cfg.crop_* fractions (simulates scroll
     cutting off part of the sheet)
  6. Apply OS window chrome overlay (see os_chrome_overlays.py, not yet
     built -- for v1, real GNOME/KDE chrome comes from Xvfb + the actual
     window manager theme; Windows/macOS chrome requires either a VM or a
     post-hoc composited overlay, flagged as a follow-up)
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from PIL import Image

from variation_sampler import ScreenshotConfig


class VirtualDisplay:
    """Owns both the Xvfb X server and a minimal EWMH-compliant window
    manager (fluxbox) on the same display. Xvfb alone provides no window
    manager, so it never advertises _NET_ACTIVE_WINDOW support -- any
    `xdotool windowactivate` call against a bare Xvfb display (see
    capture.py's own maximize_or_resize_window and
    render_onlyoffice.py's apply_config_onlyoffice) fails with "Your
    windowmanager claims not to support _NET_ACTIVE_WINDOW". fluxbox is
    started here, once, right after Xvfb comes up, so every caller of this
    context manager gets a display that window activation actually works
    on -- fixing this in one place rather than at each xdotool call site."""

    def __init__(self, resolution: str, display_num: int = 99):
        self.resolution = resolution
        self.display_num = display_num
        self._proc: subprocess.Popen | None = None
        self._wm_proc: subprocess.Popen | None = None

    def __enter__(self):
        w, h = self.resolution.split("x")
        self._proc = subprocess.Popen(
            ["Xvfb", f":{self.display_num}", "-screen", "0", f"{w}x{h}x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # let Xvfb come up before anything tries to connect

        display = f":{self.display_num}"
        self._wm_proc = subprocess.Popen(
            ["fluxbox"],
            env={"DISPLAY": display},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # let fluxbox register its EWMH hints (_NET_ACTIVE_WINDOW etc.)
        return display

    def __exit__(self, *exc):
        # Tear down fluxbox first -- it's a client of the Xvfb server, so
        # stopping it before its server keeps shutdown ordering clean, though
        # nothing downstream currently depends on that ordering specifically.
        if self._wm_proc:
            self._wm_proc.terminate()
            try:
                self._wm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._wm_proc.kill()
                self._wm_proc.wait(timeout=5)
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate() didn't work within 5s -- force it. Without this,
                # a hung Xvfb process is orphaned and leaks memory across every
                # resolution group processed by this shard for the rest of the run.
                self._proc.kill()
                self._proc.wait(timeout=5)


def find_window_id(display: str, name_hint: str = "LibreOffice Calc") -> str:
    out = subprocess.run(
        ["xdotool", "search", "--name", name_hint],
        env={"DISPLAY": display},
        capture_output=True,
        text=True,
    )
    window_ids = out.stdout.strip().split("\n")
    if not window_ids or window_ids == [""]:
        raise RuntimeError(f"No window found matching '{name_hint}' on {display}")
    return window_ids[0]


def maximize_or_resize_window(display: str, window_id: str, maximize: bool = True) -> None:
    env = {"DISPLAY": display}
    if maximize:
        subprocess.run(["xdotool", "windowactivate", window_id], env=env)
        subprocess.run(["wmctrl", "-i", "-r", window_id, "-b", "add,maximized_vert,maximized_horz"], env=env)
    else:
        # non-maximized window: common in real screenshots, especially on macOS
        subprocess.run(["xdotool", "windowsize", window_id, "1400", "900"], env=env)
        subprocess.run(["xdotool", "windowmove", window_id, "80", "60"], env=env)
    time.sleep(0.3)


def capture_window(display: str, window_id: str, out_path: Path) -> Path:
    subprocess.run(
        ["import", "-window", window_id, str(out_path)],
        env={"DISPLAY": display},
        check=True,
    )
    return out_path


def apply_scroll_crop(image_path: Path, cfg: ScreenshotConfig, out_path: Path) -> Path:
    """Crops the raw window capture per the sampled crop_* fractions,
    simulating a screenshot taken mid-scroll that cuts off part of the sheet."""
    img = Image.open(image_path)
    w, h = img.size
    left = int(w * cfg.crop_left_pct)
    right = int(w * (1 - cfg.crop_right_pct))
    top = int(h * cfg.crop_top_pct)
    bottom = int(h * (1 - cfg.crop_bottom_pct))
    cropped = img.crop((left, top, right, bottom))
    cropped.save(out_path)
    return out_path


def capture_screenshot(
    cfg: ScreenshotConfig,
    display: str,
    window_name_hint: str,
    raw_out_path: Path,
    final_out_path: Path,
) -> Path:
    window_id = find_window_id(display, window_name_hint)
    maximize_or_resize_window(display, window_id, maximize=(cfg.crop_top_pct == 0.0))
    capture_window(display, window_id, raw_out_path)
    apply_scroll_crop(raw_out_path, cfg, final_out_path)
    return final_out_path
