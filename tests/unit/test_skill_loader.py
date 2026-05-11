"""Unit tests for agents.skill_loader."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from agents import skill_loader
from agents.skill_loader import _ENV_VAR, load_all_skills


def _set_fake_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point ~ at a test-controlled location on both POSIX and Windows."""
    monkeypatch.setenv("HOME", str(home))
    # Windows expanduser() uses USERPROFILE, not HOME.
    monkeypatch.setenv("USERPROFILE", str(home))


def _write_skill(skill_dir: Path, name: str, description: str | None = None) -> None:
    """Create a minimal valid SKILL.md at skill_dir."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    description = description or f"Test skill {name}"
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.fixture
def bundled_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated bundled-skills directory + isolated HOME.

    Redirects ``_bundled_skills_path`` to a fresh tmp dir so tests don't
    load the real bundled skills. Also points HOME at a scratch directory
    so the default ``~/.config/opensearch-agent-server/skills/`` location
    doesn't accidentally pick up the developer's real user skills.
    """
    fake_bundled = tmp_path / "bundled"
    fake_bundled.mkdir()
    monkeypatch.setattr(
        "agents.skill_loader._bundled_skills_path", lambda: fake_bundled
    )
    _set_fake_home(monkeypatch, tmp_path / "scratch-home")
    monkeypatch.delenv(_ENV_VAR, raising=False)
    return fake_bundled


class TestBundledOnly:
    """Bundled-skills path when no user config or default dir is present."""

    def test_loads_bundled_skills_when_no_env_var(self, bundled_dir: Path) -> None:
        _write_skill(bundled_dir / "alpha", "alpha")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["alpha"]

    def test_empty_bundled_dir_returns_empty_list(self, bundled_dir: Path) -> None:
        skills = load_all_skills()

        assert skills == []

    def test_missing_bundled_dir_warns_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If bundled path returns None, loader returns []."""
        monkeypatch.setattr("agents.skill_loader._bundled_skills_path", lambda: None)
        _set_fake_home(monkeypatch, tmp_path / "scratch-home")
        monkeypatch.delenv(_ENV_VAR, raising=False)

        skills = load_all_skills()

        assert skills == []

    def test_none_bundled_path_and_missing_user_dir_is_silent_no_op(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When _bundled_skills_path() returns None, it contributes no path
        to search — so we should NOT see a path_missing event for it, and
        the loader should simply return []."""
        monkeypatch.setattr("agents.skill_loader._bundled_skills_path", lambda: None)
        _set_fake_home(monkeypatch, tmp_path / "scratch-home")
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with caplog.at_level(logging.INFO, logger=skill_loader.logger.name):
            skills = load_all_skills()

        assert skills == []
        assert not any(
            getattr(r, "event", None) == "skill_loader.path_missing"
            for r in caplog.records
        )


class TestEnvVarPaths:
    """AG_UI_SKILL_PATHS — os.pathsep-separated paths (POSIX ``:``, Windows ``;``)."""

    def test_single_path_loads(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "beta", "beta")
        monkeypatch.setenv(_ENV_VAR, str(user_skills))

        skills = load_all_skills()

        assert [s.name for s in skills] == ["beta"]

    def test_multiple_paths_colon_separated(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dir_a = tmp_path / "skills-a"
        dir_b = tmp_path / "skills-b"
        _write_skill(dir_a / "alpha", "alpha")
        _write_skill(dir_b / "beta", "beta")
        monkeypatch.setenv(_ENV_VAR, f"{dir_a}{os.pathsep}{dir_b}")

        skills = load_all_skills()

        assert sorted(s.name for s in skills) == ["alpha", "beta"]

    def test_bundled_plus_env_paths_combine(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_skill(bundled_dir / "bundled-one", "bundled-one")
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "user-one", "user-one")
        monkeypatch.setenv(_ENV_VAR, str(user_skills))

        skills = load_all_skills()

        assert sorted(s.name for s in skills) == ["bundled-one", "user-one"]

    def test_tilde_expansion(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """~/foo in the env var expands to the user's home dir."""
        _set_fake_home(monkeypatch, tmp_path)
        _write_skill(tmp_path / "home-skills" / "delta", "delta")
        monkeypatch.setenv(_ENV_VAR, "~/home-skills")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["delta"]

    def test_empty_entries_are_ignored(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leading/trailing/double separators shouldn't confuse the parser."""
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "gamma", "gamma")
        sep = os.pathsep
        monkeypatch.setenv(_ENV_VAR, f"{sep}{user_skills}{sep}{sep}")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["gamma"]

    def test_nested_skill_discovery(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rglob finds SKILL.md at arbitrary depth.

        Mirrors how opensearch-agent-skills nests skills 3 levels deep.
        """
        user_root = tmp_path / "user-skills"
        _write_skill(user_root / "observability" / "log-analytics", "log-analytics")
        _write_skill(user_root / "search" / "launchpad", "launchpad")
        monkeypatch.setenv(_ENV_VAR, str(user_root))

        skills = load_all_skills()

        assert sorted(s.name for s in skills) == ["launchpad", "log-analytics"]

    def test_nonexistent_path_is_skipped(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "beta", "beta")
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv(_ENV_VAR, f"{user_skills}{os.pathsep}{missing}")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["beta"]

    def test_semicolon_separator_when_os_pathsep_is_semicolon(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates Windows: os.pathsep=';' must split on ';' not ':'.

        Regression guard for the original hardcoded-':' bug that broke
        Windows CI.
        """
        dir_a = tmp_path / "skills-a"
        dir_b = tmp_path / "skills-b"
        _write_skill(dir_a / "alpha", "alpha")
        _write_skill(dir_b / "beta", "beta")
        monkeypatch.setattr(os, "pathsep", ";")
        monkeypatch.setenv(_ENV_VAR, f"{dir_a};{dir_b}")

        skills = load_all_skills()

        assert sorted(s.name for s in skills) == ["alpha", "beta"]


class TestDefaultUserDir:
    """Falls back to ~/.config/opensearch-agent-server/skills/ when env var unset."""

    def test_default_dir_loaded_when_env_unset(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        _set_fake_home(monkeypatch, fake_home)
        default_dir = fake_home / ".config" / "opensearch-agent-server" / "skills"
        _write_skill(default_dir / "default-skill", "default-skill")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["default-skill"]

    def test_env_var_overrides_default_dir(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When env var is set, default dir is NOT also scanned."""
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        _set_fake_home(monkeypatch, fake_home)
        default_dir = fake_home / ".config" / "opensearch-agent-server" / "skills"
        _write_skill(default_dir / "default-skill", "default-skill")

        explicit = tmp_path / "explicit-skills"
        _write_skill(explicit / "explicit", "explicit")
        monkeypatch.setenv(_ENV_VAR, str(explicit))

        skills = load_all_skills()

        # Only the explicit path should load, not the default dir.
        assert [s.name for s in skills] == ["explicit"]

    def test_missing_default_dir_is_no_op(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env var + no default dir → only bundled skills load."""
        _write_skill(bundled_dir / "alpha", "alpha")
        # fake-home exists but has no .config/opensearch-agent-server/skills dir.
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        _set_fake_home(monkeypatch, fake_home)

        skills = load_all_skills()

        assert [s.name for s in skills] == ["alpha"]


class TestCollisions:
    """Name-based deduplication — bundled wins over user."""

    def test_bundled_wins_over_user_on_name_collision(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_skill(bundled_dir / "shared", "shared", description="Bundled")
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "shared", "shared", description="User")
        monkeypatch.setenv(_ENV_VAR, str(user_skills))

        skills = load_all_skills()

        assert len(skills) == 1
        assert skills[0].description == "Bundled"

    def test_duplicate_emits_warning_log_with_skill_name(
        self,
        bundled_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Operators rely on `skill_loader.duplicate` at WARN to find shadowed
        user skills. Keep the event code and level stable."""
        _write_skill(bundled_dir / "shared", "shared", description="Bundled")
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "shared", "shared", description="User")
        monkeypatch.setenv(_ENV_VAR, str(user_skills))

        with caplog.at_level(logging.WARNING, logger=skill_loader.logger.name):
            load_all_skills()

        duplicate_records = [
            r
            for r in caplog.records
            if getattr(r, "event", None) == "skill_loader.duplicate"
        ]
        assert duplicate_records, "expected a skill_loader.duplicate log event"
        assert duplicate_records[0].levelno == logging.WARNING
        assert getattr(duplicate_records[0], "skill_name", None) == "shared"


class TestFailureModes:
    """Bad configuration never crashes the loader."""

    def test_invalid_skill_md_is_skipped(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SKILL.md with broken frontmatter is skipped; other skills still load."""
        _write_skill(bundled_dir / "alpha", "alpha")

        user_skills = tmp_path / "user-skills"
        broken = user_skills / "broken"
        broken.mkdir(parents=True)
        (broken / "SKILL.md").write_text("this is not valid frontmatter\n")
        _write_skill(user_skills / "beta", "beta")

        monkeypatch.setenv(_ENV_VAR, str(user_skills))

        skills = load_all_skills()

        assert sorted(s.name for s in skills) == ["alpha", "beta"]
