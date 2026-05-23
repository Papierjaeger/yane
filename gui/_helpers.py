"""Shared GUI helper functions used across panels and tabs."""
from __future__ import annotations
from PySide6.QtWidgets import QLabel, QFrame


def _label(text: str, obj_name: str = "") -> QLabel:
    lbl = QLabel(text)
    if obj_name:
        lbl.setObjectName(obj_name)
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.Shape.HLine)
    return f
