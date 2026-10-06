"""Repository identity: keeps one tier of eXCore in one repository.

Every tier (27B, 35B, 70B) is its own repository with its own track. The danger is two tiers
ending up in one repository (a push into the wrong repo, a merge of two histories). That leaves
files from both behind: a second track's config, manifests or registry. ``check_identity``
compares everything that names a track or a repository, so a mixed-up repository fails CI and
the evaluator bot refuses to run in it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import yaml


def _track_of_yaml(path: Path) -> str | None:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    return doc.get("track") if isinstance(doc, dict) else None


def check_identity(cfg: Mapping, repo_root: str | Path, github_repository: str | None = None) -> list[str]:
    """Problems found (empty = this repository is exactly one tier)."""
    root = Path(repo_root)
    track = cfg["track"]
    problems: list[str] = []

    expected = (cfg.get("identity") or {}).get("repository")
    if not expected:
        problems.append("configs/hpc_cpu.yaml has no identity.repository")
    elif github_repository is not None and github_repository.split("/")[-1] != expected:
        problems.append(f"this repository is {github_repository!r} but the track {track} belongs to {expected!r}")

    lock = root / "configs" / "sources.lock.json"
    if lock.exists():
        try:
            t = json.loads(lock.read_text(encoding="utf-8")).get("track")
        except (OSError, json.JSONDecodeError):
            t = "unreadable"
        if t != track:
            problems.append(f"configs/sources.lock.json is for track {t!r}, not {track!r}")

    spectrum = root / "results" / "core_spectrum" / "spectrum.json"
    if spectrum.exists():
        try:
            t = json.loads(spectrum.read_text(encoding="utf-8")).get("track")
        except (OSError, json.JSONDecodeError):
            t = "unreadable"
        if t != track:
            problems.append(f"results/core_spectrum/spectrum.json is for track {t!r}, not {track!r}")

    default_cfg = root / "configs" / "hpc_cpu.yaml"
    for path in sorted((root / "configs").rglob("*.y*ml")) if (root / "configs").is_dir() else []:
        if path == default_cfg:
            continue
        t = _track_of_yaml(path)
        if t is not None and t != track:
            problems.append(f"{path.relative_to(root)} is a config for track {t!r}, not {track!r}")

    for folder in ("manifests", "experiments"):
        base = root / folder
        for path in sorted(base.rglob("*.y*ml")) if base.is_dir() else []:
            t = _track_of_yaml(path)
            if t is not None and t != track:
                problems.append(f"{path.relative_to(root)} is for track {t!r}, not {track!r}")
    return problems
