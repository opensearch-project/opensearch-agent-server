# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for agents.skill_loader.

Covers bundled + user-path discovery, first-seen-wins dedup, ignored-dir
skipping, soft failure on unreadable SKILL.md, and the user-path resolution
(env var, tilde expansion, default dir). All filesystem access is scoped to
tmp_path; HOME and the bundled resource are redirected so the developer's real
skills never load.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.skill_loader import (
    _ENV_VAR,
    _is_in_ignored_dir,
    _load_user_configured_paths,
    load_all_skills,
)

pytestmark = pytest.mark.unit


def _set_fake_home(monkeypatch, home: Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _write_skill(skill_dir: Path, name: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.fixture
def bundled_dir(tmp_path, monkeypatch):
    fake_bundled = tmp_path / "bundled"
    fake_bundled.mkdir()
    monkeypatch.setattr(
        "agents.skill_loader._bundled_skills_resource", lambda: fake_bundled
    )
    _set_fake_home(monkeypatch, tmp_path / "scratch-home")
    monkeypatch.delenv(_ENV_VAR, raising=False)
    return fake_bundled


class TestLoadAllSkills:
    def test_loads_bundled_skill(self, bundled_dir):
        _write_skill(bundled_dir / "s1", "alpha")
        skills = load_all_skills()
        assert [s.name for s in skills] == ["alpha"]

    def test_empty_bundled_returns_empty(self, bundled_dir):
        assert load_all_skills() == []

    def test_missing_bundled_dir_warns_and_continues(self, tmp_path, monkeypatch):
        missing = tmp_path / "does-not-exist"
        monkeypatch.setattr(
            "agents.skill_loader._bundled_skills_resource", lambda: missing
        )
        _set_fake_home(monkeypatch, tmp_path / "scratch-home")
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert load_all_skills() == []

    def test_none_bundled_resource_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agents.skill_loader._bundled_skills_resource", lambda: None
        )
        _set_fake_home(monkeypatch, tmp_path / "scratch-home")
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert load_all_skills() == []

    def test_user_path_via_env_var(self, bundled_dir, tmp_path, monkeypatch):
        user_dir = tmp_path / "user"
        _write_skill(user_dir / "u1", "user-skill")
        monkeypatch.setenv(_ENV_VAR, str(user_dir))
        names = {s.name for s in load_all_skills()}
        assert "user-skill" in names

    def test_bundled_takes_precedence_on_duplicate(
        self, bundled_dir, tmp_path, monkeypatch
    ):
        _write_skill(bundled_dir / "b", "dup")
        user_dir = tmp_path / "user"
        _write_skill(user_dir / "u", "dup")
        monkeypatch.setenv(_ENV_VAR, str(user_dir))
        skills = load_all_skills()
        # dedup by name; only the bundled one survives.
        assert [s.name for s in skills] == ["dup"]

    def test_ignored_dir_is_skipped(self, bundled_dir):
        _write_skill(bundled_dir / "node_modules" / "pkg", "should-skip")
        _write_skill(bundled_dir / "real", "keep")
        names = {s.name for s in load_all_skills()}
        assert names == {"keep"}

    def test_unreadable_skill_is_soft_skipped(self, bundled_dir):
        _write_skill(bundled_dir / "good", "good")
        _write_skill(bundled_dir / "bad", "bad")

        real_from_file = __import__("strands", fromlist=["Skill"]).Skill.from_file

        def flaky(path):
            if path.name == "bad":
                raise ValueError("corrupt frontmatter")
            return real_from_file(path)

        with patch("agents.skill_loader.Skill.from_file", side_effect=flaky):
            names = {s.name for s in load_all_skills()}
        assert names == {"good"}


class TestIsInIgnoredDir:
    def test_ignored_name(self, tmp_path):
        root = tmp_path
        skill_md = tmp_path / "node_modules" / "pkg" / "SKILL.md"
        assert _is_in_ignored_dir(skill_md, root) is True

    def test_dotdir(self, tmp_path):
        skill_md = tmp_path / ".hidden" / "SKILL.md"
        assert _is_in_ignored_dir(skill_md, tmp_path) is True

    def test_not_ignored(self, tmp_path):
        skill_md = tmp_path / "normal" / "SKILL.md"
        assert _is_in_ignored_dir(skill_md, tmp_path) is False

    def test_not_relative_returns_false(self, tmp_path):
        skill_md = Path("/some/other/place/SKILL.md")
        assert _is_in_ignored_dir(skill_md, tmp_path) is False


class TestLoadUserConfiguredPaths:
    def test_env_var_splits_on_pathsep(self, monkeypatch, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        monkeypatch.setenv(_ENV_VAR, f"{a}{os.pathsep}{b}{os.pathsep}  ")
        paths = _load_user_configured_paths()
        assert len(paths) == 2

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        _set_fake_home(monkeypatch, tmp_path)
        monkeypatch.setenv(_ENV_VAR, "~/skills-here")
        paths = _load_user_configured_paths()
        assert str(tmp_path) in str(paths[0])

    def test_default_dir_when_env_unset_and_missing(self, monkeypatch, tmp_path):
        _set_fake_home(monkeypatch, tmp_path / "empty-home")
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert _load_user_configured_paths() == []

    def test_default_dir_when_present(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        default_skills = home / ".config" / "opensearch-agent-server" / "skills"
        default_skills.mkdir(parents=True)
        _set_fake_home(monkeypatch, home)
        monkeypatch.delenv(_ENV_VAR, raising=False)
        paths = _load_user_configured_paths()
        assert len(paths) == 1
