import json
import shutil
from pathlib import Path

import pytest

from excore.cli import main
from excore.identity import check_identity

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def repo(tmp_path):
    """A minimal copy of this repository's identity-bearing files."""
    for rel in ("configs/hpc_cpu.yaml", "configs/sources.lock.json", "results/core_spectrum/spectrum.json",
                "experiments/base_variants/V0_baseline.yaml", "experiments/base_variants/V1_mlp_focused.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    (tmp_path / "manifests").mkdir()
    return tmp_path


def test_this_repository_is_exactly_one_tier(real_cfg):
    assert check_identity(real_cfg, ROOT, "ikeige-coder/eXCore-v2") == []
    assert check_identity(real_cfg, ROOT) == []                       # no GitHub name known (local use)


def test_wrong_repository_name_is_caught(real_cfg):
    problems = check_identity(real_cfg, ROOT, "ikeige-coder/eXCore")
    assert len(problems) == 1 and "belongs to 'eXCore-v2'" in problems[0]


def test_missing_identity_is_caught(real_cfg, repo):
    cfg = {k: v for k, v in real_cfg.items() if k != "identity"}
    assert "no identity.repository" in " ".join(check_identity(cfg, repo))


def test_a_merged_in_second_tier_is_caught(real_cfg, repo):
    assert check_identity(real_cfg, repo) == []
    old = repo / "configs/tracks/cpu-01-27b.yaml"
    old.parent.mkdir()
    old.write_text("track: CPU-27B\nmodel: {}\n")
    (repo / "manifests/from-the-other-repo.yaml").write_text("schema: excore/manifest@1\ntrack: CPU-35B\nname: x\ndefault: Q4_K\n")
    (repo / "experiments/base_variants/V9.yaml").write_text("track: CPU-27B\n")
    problems = " | ".join(check_identity(real_cfg, repo))
    assert "configs/tracks/cpu-01-27b.yaml is a config for track 'CPU-27B'" in problems
    assert "manifests/from-the-other-repo.yaml is for track 'CPU-35B'" in problems
    assert "experiments/base_variants/V9.yaml" in problems


def test_a_foreign_lock_or_registry_is_caught(real_cfg, repo):
    for rel in ("configs/sources.lock.json", "results/core_spectrum/spectrum.json"):
        doc = json.loads((repo / rel).read_text())
        doc["track"] = "CPU-35B"
        (repo / rel).write_text(json.dumps(doc))
    problems = check_identity(real_cfg, repo)
    assert len(problems) == 2 and all("CPU-35B" in p for p in problems)
    (repo / "results/core_spectrum/spectrum.json").write_text("not json")
    assert any("unreadable" in p for p in check_identity(real_cfg, repo))


def test_cli_exit_codes(capsys):
    assert main(["identity", "--root", str(ROOT), "--repository", "ikeige-coder/eXCore-v2"]) == 0
    assert "identity ok: track CPU-70B" in capsys.readouterr().out
    assert main(["identity", "--root", str(ROOT), "--repository", "ikeige-coder/eXCore"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_the_bot_refuses_to_run_in_the_wrong_repository(monkeypatch):
    from argparse import Namespace

    from evaluator.bot_engine import build_engine

    monkeypatch.setenv("GITHUB_REPOSITORY", "ikeige-coder/eXCore")
    args = Namespace(config=None, repo_root=str(ROOT), pr=7, branch="main", dry_run=True)
    with pytest.raises(SystemExit, match="identity check failed"):
        build_engine(args)
