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
from .benchmark import benchmark_model, BenchmarkResult, piece_errors, piecewise_score
from .gif_utils import record_rollout, save_gif
from .llm_interface import LLMCursorInterface, benchmark_llm
from .geo_model import (
    GeoObservation,
    geo_observation,
    piece_geo_features,
    GEO_PIECE_DIM,
    GEO_CURSOR_DIM,
)

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
    "piece_errors",
    "piecewise_score",
    "record_rollout",
    "save_gif",
    "LLMCursorInterface",
    "benchmark_llm",
    "GeoObservation",
    "geo_observation",
    "piece_geo_features",
    "GEO_PIECE_DIM",
    "GEO_CURSOR_DIM",
]
