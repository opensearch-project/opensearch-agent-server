"""Skill discovery — combines bundled and user-configured skill sources.

Loads skills from two layers:

1. Bundled skills — shipped inside the ``agents`` package. Always loaded.
   Located via :mod:`importlib.resources` so they work for both editable
   installs (``pip install -e .``) and wheel installs
   (``pip install opensearch-agent-server``).
2. User skills — either a list of paths in the ``AG_UI_SKILL_PATHS`` env
   var (``os.pathsep``-separated, PATH-style: ``:`` on POSIX, ``;`` on
   Windows), or, if that's unset, the default location
   ``~/.config/opensearch-agent-server/skills/``.

First-seen-wins deduplication by skill name: bundled skills take precedence
over user skills, so a user cannot silently override a skill this repo
ships.

All failure modes are soft: missing paths and unreadable SKILL.md files are
logged and skipped. The agent never fails to start because of
external-skill misconfiguration.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from strands import Skill

from utils.logging_helpers import get_logger, log_info_event, log_warning_event

logger = get_logger(__name__)

_ENV_VAR = "AG_UI_SKILL_PATHS"
_BUNDLED_SKILLS_PACKAGE = "agents"
_BUNDLED_SKILLS_RESOURCE = "skills"
_DEFAULT_USER_SKILLS_DIR = "~/.config/opensearch-agent-server/skills"


def load_all_skills() -> list[Skill]:
    """Load skills from the bundled path and any user-configured paths.

    Returns:
        List of :class:`~strands.Skill` instances. Deduplicated by skill
        name; the first occurrence wins, so bundled skills take precedence
        over user skills.
    """
    search_paths: list[Path] = []

    bundled = _bundled_skills_path()
    if bundled is not None:
        search_paths.append(bundled)

    search_paths.extend(_load_user_configured_paths())

    skills: list[Skill] = []
    seen_names: set[str] = set()

    for root in search_paths:
        if not root.exists():
            log_info_event(
                logger,
                f"Skill path not present, skipping: {root}",
                "skill_loader.path_missing",
                skill_path=str(root),
            )
            continue

        for skill_md in root.rglob("SKILL.md"):
            try:
                skill = Skill.from_file(skill_md.parent)
            except Exception as e:
                log_warning_event(
                    logger,
                    f"Failed to load skill at {skill_md}: {e}",
                    "skill_loader.load_failed",
                    skill_path=str(skill_md),
                    error=str(e),
                )
                continue

            if skill.name in seen_names:
                log_warning_event(
                    logger,
                    f"Duplicate skill name '{skill.name}': keeping the earlier "
                    f"version, skipping {skill_md}. To use this version instead, "
                    f"rename its 'name' field in the frontmatter.",
                    "skill_loader.duplicate",
                    skill_name=skill.name,
                    skill_path=str(skill_md),
                )
                continue

            seen_names.add(skill.name)
            skills.append(skill)
            log_info_event(
                logger,
                f"Loaded skill: {skill.name}",
                "skill_loader.loaded",
                skill_name=skill.name,
                skill_path=str(skill_md),
            )

    return skills


def _bundled_skills_path() -> Path | None:
    """Return the filesystem path to the bundled skills directory.

    Uses :mod:`importlib.resources` so bundled skills are discoverable for
    both editable installs and wheel installs. Returns ``None`` if the
    resource can't be located (shouldn't happen in a correctly-packaged
    install; present as a safety net).
    """
    try:
        resource = files(_BUNDLED_SKILLS_PACKAGE) / _BUNDLED_SKILLS_RESOURCE
        return Path(str(resource))
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def _load_user_configured_paths() -> list[Path]:
    """Return user skill paths.

    Priority:
    1. If ``AG_UI_SKILL_PATHS`` is set, return its entries (split on
       ``os.pathsep``: ``:`` on POSIX, ``;`` on Windows).
    2. Otherwise, return the default user skills dir if it exists.
    3. Otherwise, return an empty list.
    """
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        # os.pathsep is ":" on POSIX, ";" on Windows.
        # Needed because Windows paths contain ":" (e.g. C:\path).
        return [
            Path(entry).expanduser().resolve()
            for entry in env_value.split(os.pathsep)
            if entry.strip()
        ]

    default_dir = Path(_DEFAULT_USER_SKILLS_DIR).expanduser()
    return [default_dir] if default_dir.exists() else []
