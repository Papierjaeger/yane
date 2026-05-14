"""Launch the YANE GUI. Run this file directly from VSCode (F5) or terminal."""
import sys
import os

# Add the parent of this directory (i.e. the folder containing the yane/ package) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yane.gui.main import main
main()
