"""Serializable provenance for source artifacts and analysis configuration."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess


def artifact_hash(path: str | Path) -> dict:
    """Hash a source file in bounded blocks; absent requested files are errors."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def capture_provenance(*, sources: list[str | Path] = (), config: str | Path | None = None,
                        seeds: list[int] = (), packages: list[str] = (), git_directory: str | Path | None = None) -> dict:
    """Record available provenance; missing Git or distribution metadata is nonfatal."""
    commit = None
    if git_directory is not None:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_directory,
                                    capture_output=True, text=True, timeout=5, check=False)
            commit = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "config": artifact_hash(config) if config is not None else None,
            "source_artifacts": [artifact_hash(path) for path in sources],
            "python_version": platform.python_version(), "package_versions": versions,
            "random_seeds": list(seeds)}
