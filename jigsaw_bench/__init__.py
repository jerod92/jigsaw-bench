"""Jigsaw puzzle benchmark package."""

from .cuts import (
    PuzzleCuts,
    generate_cuts,
    generate_grid_anchors,
    DEFAULT_PROFILE,
)
from .puzzle import Puzzle, Piece, generate_puzzle
from .shuffle import shuffle_pieces, ShuffleLayout
from .environment import JigsawEnvironment, ActionPoint
from .benchmark import benchmark_model, BenchmarkResult
from .llm_interface import LLMCursorInterface, benchmark_llm

__all__ = [
    "PuzzleCuts",
    "generate_cuts",
    "generate_grid_anchors",
    "DEFAULT_PROFILE",
    "Puzzle",
    "Piece",
    "generate_puzzle",
    "shuffle_pieces",
    "ShuffleLayout",
    "JigsawEnvironment",
    "ActionPoint",
    "benchmark_model",
    "BenchmarkResult",
    "LLMCursorInterface",
    "benchmark_llm",
]
