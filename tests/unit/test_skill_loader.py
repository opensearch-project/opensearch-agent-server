"""Unit tests for agents.skill_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.skill_loader import _ENV_VAR, load_all_skills


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
    monkeypatch.setenv("HOME", str(tmp_path / "scratch-home"))
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
        monkeypatch.setattr(
            "agents.skill_loader._bundled_skills_path", lambda: None
        )
        monkeypatch.setenv("HOME", str(tmp_path / "scratch-home"))
        monkeypatch.delenv(_ENV_VAR, raising=False)

        skills = load_all_skills()

        assert skills == []


class TestEnvVarPaths:
    """AG_UI_SKILL_PATHS — colon-separated paths."""

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
        monkeypatch.setenv(_ENV_VAR, f"{dir_a}:{dir_b}")

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
        """~/foo in the env var expands via $HOME."""
        monkeypatch.setenv("HOME", str(tmp_path))
        _write_skill(tmp_path / "home-skills" / "delta", "delta")
        monkeypatch.setenv(_ENV_VAR, "~/home-skills")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["delta"]

    def test_empty_entries_are_ignored(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leading/trailing/double colons shouldn't confuse the parser."""
        user_skills = tmp_path / "user-skills"
        _write_skill(user_skills / "gamma", "gamma")
        monkeypatch.setenv(_ENV_VAR, f":{user_skills}::")

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
        monkeypatch.setenv(_ENV_VAR, f"{user_skills}:{missing}")

        skills = load_all_skills()

        assert [s.name for s in skills] == ["beta"]


class TestDefaultUserDir:
    """Falls back to ~/.config/opensearch-agent-server/skills/ when env var unset."""

    def test_default_dir_loaded_when_env_unset(
        self, bundled_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
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
        monkeypatch.setenv("HOME", str(fake_home))
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
        monkeypatch.setenv("HOME", str(fake_home))

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
