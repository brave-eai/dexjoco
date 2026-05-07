# Asset Bundles

Dexjoco keeps code and task configuration in Git. The entire `dexjoco_sim/envs/xmls` tree and selected GeoRT runtime assets are distributed separately as GitHub release assets so the repository stays lighter and avoids GitHub's large-file limits.

The asset manifest lives at [`assets_manifest.json`](../assets_manifest.json), and the downloader lives at [`scripts/download_assets.py`](../scripts/download_assets.py).

## Recommended Usage

Before running simulation tasks:

```bash
python scripts/download_assets.py --bundle sim-xmls
```

Before using GeoRT-based teleoperation:

```bash
python scripts/download_assets.py --bundle geort-runtime-assets
```

If you omit `--bundle`, the downloader fetches every bundle listed in the manifest.

## Default Release Location

By default the downloader looks for assets under:

```text
https://github.com/brave-eai/dexjoco/releases/download/assets-v1/
```

You can override that with either `--repo/--tag` or `--base-url`.

Examples:

```bash
python scripts/download_assets.py --repo brave-eai/dexjoco --tag assets-v1
python scripts/download_assets.py --base-url https://github.com/brave-eai/dexjoco/releases/download/assets-v1
```

## Bundle Overview

- `sim-xmls`:
  The full `dexjoco_sim/envs/xmls` tree, including XML files, meshes, and textures.
- `geort-runtime-assets`:
  Dexjoco's default GeoRT checkpoints plus the MediaPipe hand landmark model.

## Maintainer Notes

The release asset archives are prepared outside the Git repository. In the current local layout they were generated into:

```text
/home/eai/project/dexjoco/release_assets/
```

For a new public release:

1. Build or refresh the archives.
2. Upload the archive files named in `assets_manifest.json` to a GitHub release tag such as `assets-v1`.
3. If any archive changes, update its `sha256` and `size_bytes` in `assets_manifest.json`.

Downloaded asset files are ignored by Git via `.gitignore`, so restoring them locally should not dirty the repository.
