"""Shared GUI helper functions used across panels and tabs."""
from __future__ import annotations
from PySide6.QtWidgets import (
    QLabel, QFrame, QWidget, QVBoxLayout, QFormLayout, QPushButton,
)


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


class CollapsibleGroup(QWidget):
    """A titled section that can be expanded/collapsed by clicking the header."""

    _BTN_QSS = (
        "QPushButton {"
        "  text-align: left; padding: 5px 8px;"
        "  font-weight: bold; font-size: 13px; color: #a6adc8;"
        "  background: #313244; border: 1px solid #313244;"
        "  border-radius: 6px;"
        "}"
        "QPushButton:hover { background: #3b3d54; }"
    )
    _BTN_OPEN_QSS = (
        "QPushButton {"
        "  text-align: left; padding: 5px 8px;"
        "  font-weight: bold; font-size: 13px; color: #a6adc8;"
        "  background: #313244; border: 1px solid #313244;"
        "  border-radius: 6px 6px 0 0;"
        "}"
        "QPushButton:hover { background: #3b3d54; }"
    )
    _PANEL_QSS = (
        "QWidget#collapsiblePanel {"
        "  border: 1px solid #313244; border-top: none;"
        "  border-radius: 0 0 6px 6px;"
        "}"
    )

    def __init__(self, title: str, use_form: bool = True,
                 collapsed: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setFlat(False)
        self._btn.clicked.connect(self._toggle)
        outer.addWidget(self._btn)

        self._panel = QWidget()
        self._panel.setObjectName("collapsiblePanel")
        self._panel.setStyleSheet(self._PANEL_QSS)
        if use_form:
            self._inner: QFormLayout | QVBoxLayout = QFormLayout(self._panel)
            self._inner.setSpacing(4)
        else:
            self._inner = QVBoxLayout(self._panel)
        self._inner.setContentsMargins(8, 6, 8, 8)
        outer.addWidget(self._panel)

        self._update_state()

    def _update_state(self) -> None:
        arrow = "▶" if self._collapsed else "▼"
        self._btn.setText(f"  {arrow}  {self._title}")
        self._btn.setStyleSheet(
            self._BTN_QSS if self._collapsed else self._BTN_OPEN_QSS
        )
        self._panel.setVisible(not self._collapsed)

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._update_state()

    def addRow(self, *args) -> None:
        self._inner.addRow(*args)  # type: ignore[union-attr]

    def addWidget(self, widget: QWidget) -> None:
        self._inner.addWidget(widget)

    def setToolTip(self, tip: str) -> None:  # noqa: N802
        self._btn.setToolTip(tip)
