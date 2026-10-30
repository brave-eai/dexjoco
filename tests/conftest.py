import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--replay-n-envs",
        type=int,
        default=4,
        help="Number of parallel environments used by dataset replay tests.",
    )
