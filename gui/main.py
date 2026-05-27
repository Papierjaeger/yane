"""Entry point — run with: python -m yane.gui"""
import sys


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from yane.gui.window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("YANE")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
