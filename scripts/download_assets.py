#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "assets_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download large Dexjoco assets that are kept out of Git."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to assets_manifest.json",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        dest="bundles",
        default=[],
        help="Bundle name to download. Repeat to select multiple bundles. Defaults to all bundles.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository in owner/name form. Defaults to the manifest's default_repo.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="GitHub release tag that hosts the asset archives. Defaults to the manifest's default_tag.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Direct base URL for release assets. Overrides --repo and --tag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if all files in a bundle already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available bundles and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing files.",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_complete(bundle):
    required_paths = bundle.get("required_paths", bundle.get("members", []))
    return all((REPO_ROOT / member).exists() for member in required_paths)


def resolve_base_url(args, manifest):
    if args.base_url:
        return args.base_url.rstrip("/")
    repo = args.repo or manifest["default_repo"]
    tag = args.tag or manifest["default_tag"]
    return f"https://github.com/{repo}/releases/download/{tag}"


def safe_extract(tar: tarfile.TarFile, destination: Path):
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(f"Refusing to extract outside repository: {member.name}")
    tar.extractall(destination)


def download_bundle(bundle, base_url, force=False, dry_run=False):
    if bundle_complete(bundle) and not force:
        print(f"[skip] {bundle['name']}: all member files already exist")
        return

    url = f"{base_url}/{bundle['archive_name']}"
    target_root = (REPO_ROOT / bundle["extract_to"]).resolve()
    print(f"[download] {bundle['name']}: {url}")
    if dry_run:
        return

    with tempfile.TemporaryDirectory(prefix="dexjoco-assets-") as tmpdir:
        archive_path = Path(tmpdir) / bundle["archive_name"]
        try:
            with urllib.request.urlopen(url) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

        archive_sha = compute_sha256(archive_path)
        if archive_sha != bundle["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {bundle['archive_name']}: "
                f"expected {bundle['sha256']}, got {archive_sha}"
            )

        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract(tar, target_root)

        restored_count = len(bundle.get("members", []))
        print(f"[ok] {bundle['name']}: restored {restored_count} path entries")


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    bundles = manifest["bundles"]
    bundle_map = {bundle["name"]: bundle for bundle in bundles}

    if args.list:
        for bundle in bundles:
            optional = "optional" if bundle.get("optional", False) else "required"
            print(f"{bundle['name']} ({optional})")
            print(f"  {bundle['description']}")
            print(f"  archive: {bundle['archive_name']}")
        return 0

    requested = args.bundles or [bundle["name"] for bundle in bundles]
    missing = [name for name in requested if name not in bundle_map]
    if missing:
        print(f"Unknown bundle(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    base_url = resolve_base_url(args, manifest)
    print(f"Using asset base URL: {base_url}")
    for name in requested:
        download_bundle(bundle_map[name], base_url, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
