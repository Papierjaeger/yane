"""Shared NeuroEvolution state instance for the YANE API server.

Imported by both server.py and all route modules to avoid circular imports.
"""
from yane.neuro_evolution import NeuroEvolution

state: NeuroEvolution = NeuroEvolution()
