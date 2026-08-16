"""
Loads a CSV into a running LibreOffice Calc instance (via UNO) and applies
a sampled ScreenshotConfig: zoom, gridlines, frozen panes, column widths,
fonts, cell selection, conditional formatting, filter dropdowns.

Requires `soffice --headless --accept="socket,host=localhost,port=2002;urp;"`
running in the background (see generate_dataset.py, which manages the
soffice process lifecycle) and Xvfb providing a virtual display (see
capture.py) since headless UNO can still render to an X display for
screenshotting purposes -- we intentionally do NOT use --headless alone,
because we need actual pixels, not just document manipulation.

This module deliberately does the DOCUMENT-STATE side only. Capture.py
does the PIXEL side. Keeping them separate makes it easy to swap in
render_excel.py later without touching capture logic.
"""

from __future__ import annotations

import uno
from com.sun.star.beans import PropertyValue

from variation_sampler import ScreenshotConfig


def _make_prop(name: str, value) -> PropertyValue:
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def connect_to_soffice(host: str = "localhost", port: int = 2002):
    """Connects to an already-running soffice --accept socket. Raises if
    soffice isn't up yet -- caller (generate_dataset.py) is responsible for
    starting it and retrying the connection with backoff."""
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    ctx = resolver.resolve(
        f"uno:socket,host={host},port={port};urp;StarOffice.ComponentContext"
    )
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    return desktop, ctx


def load_csv(desktop, csv_path: str):
    """Opens a CSV with import filter options forcing comma delimiter and
    UTF-8, so headers with non-ASCII characters (e.g. Norwegian æøå in
    data.norge.no sources) render correctly.

    FilterName is REQUIRED here, not optional -- without it, LibreOffice
    falls back to its own format auto-detection, and because this pipeline
    runs without --headless (an actual window is needed for screenshotting),
    ambiguous auto-detection can pop the interactive "Text Import" dialog
    instead of applying FilterOptions silently. That dialog blocks forever
    waiting for a click that will never come -- a process hang with near-zero
    CPU usage, easy to mistake for something else stuck. Explicit FilterName
    bypasses auto-detection entirely, so this can't happen."""
    url = f"file://{csv_path}"
    filter_name = _make_prop("FilterName", "Text - txt - csv (StarCalc)")
    filter_options = _make_prop(
        "FilterOptions", "44,34,76,1,,0,true,true,true"  # comma-sep, UTF-8, detect quotes
    )
    hidden = _make_prop("Hidden", False)  # must be visible for screenshotting
    doc = desktop.loadComponentFromURL(url, "_blank", 0, (filter_name, filter_options, hidden))
    return doc


def apply_config(doc, cfg: ScreenshotConfig) -> None:
    """Applies the sampled visual configuration to the loaded document."""
    sheet = doc.Sheets.getByIndex(0)
    controller = doc.CurrentController

    # Zoom
    controller.ZoomValue = cfg.zoom_pct

    # Gridlines
    doc.CurrentController.ActiveSheet.IsGridVisible = cfg.gridlines

    # Frozen panes
    if cfg.frozen_rows or cfg.frozen_cols:
        controller.freezeAtPosition(cfg.frozen_cols, cfg.frozen_rows)

    # Column width
    used = sheet.createCursor()
    used.gotoEndOfUsedArea(False)
    n_cols = used.RangeAddress.EndColumn + 1
    n_rows = used.RangeAddress.EndRow + 1

    columns = sheet.Columns
    if cfg.column_width_mode == "auto_fit":
        for c in range(n_cols):
            columns.getByIndex(c).OptimalWidth = True
    elif cfg.column_width_mode == "manual_narrow":
        for c in range(n_cols):
            columns.getByIndex(c).Width = 1200  # 1/100 mm; induces "###" truncation
    else:  # manual_wide
        for c in range(n_cols):
            columns.getByIndex(c).Width = 4500

    # Row height
    rows = sheet.Rows
    if cfg.row_height_mode == "manual":
        for r in range(min(n_rows, 500)):  # cap to avoid pathological render time
            rows.getByIndex(r).Height = 600

    # Font
    data_range = sheet.getCellRangeByPosition(0, 0, n_cols - 1, n_rows - 1)
    data_range.CharFontName = cfg.font_family
    data_range.CharHeight = cfg.font_size_pt

    # Cell selection / highlight
    if cfg.cell_highlight and n_rows > 0 and n_cols > 0:
        import random

        r0 = random.randint(0, max(0, n_rows - 1))
        c0 = random.randint(0, max(0, n_cols - 1))
        r1 = min(n_rows - 1, r0 + int(cfg.n_selected_cells**0.5))
        c1 = min(n_cols - 1, c0 + int(cfg.n_selected_cells**0.5))
        sel_range = sheet.getCellRangeByPosition(c0, r0, c1, r1)
        controller.select(sel_range)

    # Conditional formatting
    if cfg.conditional_formatting:
        _apply_conditional_formatting(sheet, n_rows, n_cols, cfg.conditional_formatting)

    # Filter dropdowns
    if cfg.filter_dropdowns and n_rows > 0 and n_cols > 0:
        header_range = sheet.getCellRangeByPosition(0, 0, n_cols - 1, n_rows - 1)
        doc.DatabaseRanges.addNewByName(
            "AutoFilterRange", header_range.RangeAddress
        )
        doc.DatabaseRanges.getByName("AutoFilterRange").AutoFilter = True

    # Scroll position (partial crop is done at the pixel level in capture.py
    # since UNO scroll APIs are unreliable for precise fractional cropping --
    # cfg.crop_* is passed through and applied there instead)


def _apply_conditional_formatting(sheet, n_rows: int, n_cols: int, cf_type: str) -> None:
    """Applies a color scale / data bar to a random numeric-looking column.
    Best-effort: skipped silently if the range has no numeric data, since
    conditional formatting on all-text columns is a no-op visually anyway."""
    from com.sun.star.sheet.ConditionFormatType import COLORSCALE, DATABAR

    numeric_col = _find_first_numeric_column(sheet, n_rows, n_cols)
    if numeric_col is None:
        return

    cond_range = sheet.getCellRangeByPosition(numeric_col, 1, numeric_col, n_rows - 1)
    cond_formats = sheet.ConditionalFormats
    fmt_type = COLORSCALE if cf_type == "color_scale" else DATABAR
    cond_formats.createByRange(cond_range.RangeAddress, fmt_type)


def _find_first_numeric_column(sheet, n_rows: int, n_cols: int) -> int | None:
    for c in range(n_cols):
        cell = sheet.getCellByPosition(c, 1)  # row 1 (skip header row 0)
        if cell.getType() != 0:  # 0 == EMPTY, non-zero includes VALUE
            try:
                float(cell.getString())
                return c
            except ValueError:
                continue
    return None


def close_doc(doc) -> None:
    doc.close(False)
