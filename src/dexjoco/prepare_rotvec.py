"""Prepare a LeRobot dataset with rotation-vector actions as the training target."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tyro


def _replace_columns(table: pa.Table, renames: dict[str, str]) -> pa.Table:
    replaced = set(renames.values())
    columns = [name for name in table.column_names if name not in replaced]
    return table.select(columns).rename_columns([renames.get(name, name) for name in columns])


def prepare_rotvec(dataset_dir: Path, output_dir: Path) -> None:
    """Copy a v1.1 task dataset and use its action_rotvec column as action.

    Args:
        dataset_dir: Local LeRobot dataset containing action and action_rotvec.
        output_dir: New directory for the training dataset.
    """
    if output_dir.resolve().is_relative_to(dataset_dir.resolve()):
        raise ValueError("The training dataset must be outside the source dataset")

    info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_dir / "meta/stats.json").read_text(encoding="utf-8"))
    if "action_rotvec" not in info["features"] or "action_rotvec" not in stats:
        raise ValueError("The source dataset must contain action_rotvec and its statistics")
    info["features"]["action"] = info["features"].pop("action_rotvec")
    stats["action"] = stats.pop("action_rotvec")

    # Training uses the canonical action key, while DexJoCo replay expects the source dataset.
    shutil.copytree(
        dataset_dir, output_dir, ignore=shutil.ignore_patterns("replay_initial_values.jsonl")
    )

    for path in sorted((output_dir / "data").rglob("*.parquet")):
        table = pq.ParquetFile(path).read()
        table = _replace_columns(table, {"action_rotvec": "action"})
        metadata = dict(table.schema.metadata or {})
        if b"huggingface" in metadata:
            hf_metadata = json.loads(metadata[b"huggingface"])
            features = hf_metadata["info"]["features"]
            features["action"] = features.pop("action_rotvec")
            hf_metadata.pop("fingerprint", None)
            metadata[b"huggingface"] = json.dumps(hf_metadata).encode()
            table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)

    for path in sorted((output_dir / "meta/episodes").rglob("*.parquet")):
        table = pq.ParquetFile(path).read()
        renames = {
            name: name.replace("stats/action_rotvec/", "stats/action/", 1)
            for name in table.column_names
            if name.startswith("stats/action_rotvec/")
        }
        pq.write_table(_replace_columns(table, renames), path)

    (output_dir / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    (output_dir / "meta/stats.json").write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def cli() -> None:
    tyro.cli(prepare_rotvec)
