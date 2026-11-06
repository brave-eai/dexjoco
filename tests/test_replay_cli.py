"""Check replay CLI summaries and failure exit status without running physics."""

import numpy as np
import pytest

from dexjoco import replay_dataset


@pytest.mark.parametrize("success", [[True, True], [True, False]])
def test_replay_cli_summary(monkeypatch, capsys, success):
    monkeypatch.setattr(
        replay_dataset.tyro, "cli", lambda function: np.asarray(success, dtype=bool)
    )
    if all(success):
        replay_dataset.cli()
    else:
        with pytest.raises(SystemExit) as error:
            replay_dataset.cli()
        assert error.value.code == 1
    output = capsys.readouterr().out
    assert f"Replay success: {sum(success)}/{len(success)}" in output
    if not all(success):
        assert "Failed episode indices: [1]" in output
