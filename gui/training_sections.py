"""Small widget builders shared by the training tab."""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget


def inline_row(
    *widgets,
    spacing: int = 4,
    stretch_first: bool = False,
    stretch_last: bool = False,
) -> QWidget:
    """Return a compact horizontal row for form-layout controls."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    last_index = len(widgets) - 1
    for index, widget in enumerate(widgets):
        stretch = int((stretch_first and index == 0) or (stretch_last and index == last_index))
        layout.addWidget(widget, stretch=stretch)
    return row
