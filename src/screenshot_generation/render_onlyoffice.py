"""
Drives OnlyOffice Desktop Editors via xdotool keyboard/window automation to
render the "excel" software variant with real GUI rendering, rather than
falling back to LibreOffice or a synthetic HTML/CSS mockup.

IMPORTANT CAVEAT, read before debugging this module: unlike
render_libreoffice.py, there's no programmatic scripting bridge (no
UNO-equivalent) for OnlyOffice Desktop Editors. Every action here is
simulated keyboard/mouse input via xdotool, which means it's inherently
more fragile than the LibreOffice path -- window titles, menu structure,
and keyboard shortcuts below were written from OnlyOffice's documented
defaults but NOT verified against a running instance (this was written in
an environment without a GUI or the ability to install OnlyOffice). Expect
to spend time confirming/adjusting:
  - WINDOW_TITLE_HINT actually matches your installed version's window title
  - keyboard shortcuts below match your installed version (OnlyOffice has
    changed some shortcuts across major versions)
  - timing (the sleep() calls) -- tuned conservatively-slow as a starting
    point, tighten once you've confirmed correctness, since this runs
    per-document and adds up across a full shard

Coverage vs. ScreenshotConfig is intentionally partial: OnlyOffice's UI
doesn't expose every option LibreOffice's UNO API does (e.g. no scripted
column-width mode, no scripted conditional-formatting type selection).
Applied: zoom, gridlines toggle, cell selection/highlight. NOT applied
(silently ignored for this software path): column_width_mode,
row_height_mode, font overrides, conditional_formatting, filter_dropdowns,
frozen_panes. If eval shows the Excel-variant screenshots need more visual
variety, this is the place to extend -- each additional config dimension
is another menu-navigation or shortcut sequence to work out and verify.

Process model differs from LibreOffice's too: there's no long-lived
"desktop" object to load successive documents into. Each document gets its
own OnlyOffice process, launched with the CSV path as a CLI argument and
killed after capture. This sidesteps the LibreOffice UNO memory-leak
concern (see render_libreoffice.py) entirely, at the cost of higher
per-document startup overhead (a fresh app launch each time vs. reusing a
running instance).
"""

from __future__ import annotations

import subprocess
import time

from variation_sampler import ScreenshotConfig

ONLYOFFICE_BINARY = "desktopeditors"  # confirmed via official install docs (guides.onlyoffice.com) -- NOT "DesktopEditors", which is the raw binary under /opt/onlyoffice/ and fails if invoked directly outside the wrapper script that sets up its library paths
# UNVERIFIED -- confirm against your installed version's actual window
# title (e.g. `xdotool search --name ""` under the same DISPLAY while a
# document is open, to list all window titles present).
WINDOW_TITLE_HINT = "ONLYOFFICE"
APP_STARTUP_WAIT_S = 4.0  # Electron-based UI is slower to cold-start than soffice
POST_ACTION_SETTLE_S = 0.3


def launch_onlyoffice(display: str, csv_path: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [ONLYOFFICE_BINARY, csv_path],
        env={"DISPLAY": display},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(APP_STARTUP_WAIT_S)
    return proc


def find_onlyoffice_window(display: str, timeout_s: float = 15.0) -> str:
    """Polls for the OnlyOffice window to appear, since cold-start time
    varies. Raises if it never shows up within timeout_s -- treat this as a
    signal WINDOW_TITLE_HINT needs correcting for your installed version,
    not necessarily a transient failure."""
    start = time.time()
    while time.time() - start < timeout_s:
        result = subprocess.run(
            ["xdotool", "search", "--name", WINDOW_TITLE_HINT],
            env={"DISPLAY": display},
            capture_output=True,
            text=True,
        )
        window_ids = [w for w in result.stdout.strip().split("\n") if w]
        if window_ids:
            return window_ids[0]
        time.sleep(0.5)
    # Dump whatever windows *did* exist on the display, titles included, so
    # the failure is self-diagnosing -- no need to re-run this by hand under
    # a manual Xvfb+xdotool session just to see what the real title is.
    seen = subprocess.run(
        ["xdotool", "search", "--name", "", "getwindowname", "%@"],
        env={"DISPLAY": display},
        capture_output=True,
        text=True,
    )
    seen_titles = seen.stdout.strip() or "(no windows found at all -- check the app launched)"
    raise RuntimeError(
        f"No OnlyOffice window matching '{WINDOW_TITLE_HINT}' appeared within {timeout_s}s "
        f"on {display} -- WINDOW_TITLE_HINT likely needs correcting for your installed version.\n"
        f"Windows actually present on {display}:\n{seen_titles}"
    )


def _xdotool_key(display: str, window_id: str, keys: str) -> None:
    subprocess.run(
        ["xdotool", "key", "--window", window_id, keys],
        env={"DISPLAY": display},
        check=True,
    )
    time.sleep(POST_ACTION_SETTLE_S)


def apply_config_onlyoffice(display: str, window_id: str, cfg: ScreenshotConfig) -> None:
    """Applies the subset of ScreenshotConfig that OnlyOffice's UI exposes
    via keyboard shortcut. See module docstring for what's NOT covered."""
    subprocess.run(["xdotool", "windowactivate", window_id], env={"DISPLAY": display}, check=True)
    time.sleep(POST_ACTION_SETTLE_S)

    _apply_zoom(display, window_id, cfg.zoom_pct)

    if not cfg.gridlines:
        _toggle_gridlines(display, window_id)

    if cfg.cell_highlight:
        _select_cell_range(display, window_id, cfg.n_selected_cells)


def _apply_zoom(display: str, window_id: str, zoom_pct: int) -> None:
    """OnlyOffice's default zoom is 100%. Ctrl+= / Ctrl+- step by roughly
    10% per press in the desktop editor -- UNVERIFIED exact step size,
    confirm and adjust n_steps calculation if it drifts from 100% + n*10%.
    Reset to 100% first (Ctrl+0, if supported) so repeated calls are
    idempotent rather than compounding from whatever zoom the previous
    document was left at."""
    _xdotool_key(display, window_id, "ctrl+0")  # reset to 100%, UNVERIFIED shortcut
    delta = zoom_pct - 100
    step_key = "ctrl+plus" if delta > 0 else "ctrl+minus"
    n_steps = round(abs(delta) / 10)
    for _ in range(n_steps):
        _xdotool_key(display, window_id, step_key)


def _toggle_gridlines(display: str, window_id: str) -> None:
    """No confirmed direct keyboard shortcut for this in OnlyOffice as of
    writing -- routes through the View menu via Alt-key mnemonics.
    UNVERIFIED: exact mnemonic sequence depends on the installed version's
    menu layout and language. If this doesn't land on the right menu item,
    capture a screenshot mid-sequence (remove the final Return) to see
    where the mnemonic navigation actually ends up, then adjust."""
    _xdotool_key(display, window_id, "alt+v")  # View menu, UNVERIFIED mnemonic
    _xdotool_key(display, window_id, "g")  # "Gridlines" toggle, UNVERIFIED mnemonic
    _xdotool_key(display, window_id, "Return")


def _select_cell_range(display: str, window_id: str, n_cells: int) -> None:
    """Selects a small range starting at A1 by holding Shift and pressing
    arrow keys -- avoids needing to know current cursor position, at the
    cost of the highlighted range always starting at the top-left rather
    than a random position like the LibreOffice path achieves. Good enough
    for "a cell selection is visible in the screenshot" but less varied
    than render_libreoffice.py's random-position selection."""
    side = max(1, round(n_cells**0.5))
    for _ in range(side - 1):
        _xdotool_key(display, window_id, "shift+Right")
    for _ in range(side - 1):
        _xdotool_key(display, window_id, "shift+Down")


def close_onlyoffice(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    except Exception:
        pass
    # Clean up any orphaned child processes from OnlyOffice/desktopeditors
    subprocess.run(["pkill", "-9", "-f", ONLYOFFICE_BINARY], capture_output=True)
