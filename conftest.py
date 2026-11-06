import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--replay-dataset-root",
        type=Path,
        default=None,
        help="Enable dataset replay tests using this root containing rand_obj and rand_full.",
    )
    parser.addoption(
        "--replay-n-envs",
        type=int,
        default=5,
        help="Number of parallel environments used by dataset replay tests.",
    )
