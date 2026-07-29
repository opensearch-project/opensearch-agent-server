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
from importlib.resources import as_file, files
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
    skills: list[Skill] = []
    seen_names: set[str] = set()

    # Bundled skills. ``as_file()`` extracts to a temp dir if the package
    # is imported from a zipped wheel, so rglob works regardless of
    # packaging. The context manager cleans up the temp dir after we
    # finish scanning.
    bundled_resource = _bundled_skills_resource()
    if bundled_resource is not None:
        with as_file(bundled_resource) as bundled_path:
            _collect_skills_from_root(bundled_path, skills, seen_names)

    # User-configured skills.
    for root in _load_user_configured_paths():
        _collect_skills_from_root(root, skills, seen_names)

    return skills


def _collect_skills_from_root(
    root: Path, skills: list[Skill], seen_names: set[str]
) -> None:
    """Scan ``root`` for ``SKILL.md`` files and append new skills to the list.

    Mutates ``skills`` and ``seen_names`` in place so bundled-then-user
    dedup works across calls. Skips common non-skill directories (``.git``,
    ``node_modules``, hidden dotdirs, etc.) to avoid walking large unrelated
    trees when a user points ``AG_UI_SKILL_PATHS`` at a git repo root.
    """
    if not root.exists():
        log_info_event(
            logger,
            f"Skill path not present, skipping: {root}",
            "skill_loader.path_missing",
            skill_path=str(root),
        )
        return

    for skill_md in root.rglob("SKILL.md"):
        if _is_in_ignored_dir(skill_md, root):
            continue

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


# Directory names rglob should skip when searching for SKILL.md files.
# Anything in this set, plus any dotdir (".git", ".cache", ...), is
# excluded. Applied only to path components *below* the skill root so we
# don't reject a skill root itself when it happens to be named ".skills".
_IGNORED_DIR_NAMES = frozenset({"node_modules", "__pycache__", ".venv"})


def _is_in_ignored_dir(skill_md: Path, root: Path) -> bool:
    """Return True if any path component between ``root`` and ``skill_md``
    is a dotdir or in the ignored-name set."""
    try:
        relative = skill_md.relative_to(root)
    except ValueError:
        return False
    for part in relative.parts[:-1]:  # exclude the file name itself
        if part.startswith(".") or part in _IGNORED_DIR_NAMES:
            return True
    return False


def _bundled_skills_resource():
    """Return the bundled skills ``Traversable``, or ``None`` if missing.

    Uses :mod:`importlib.resources` so bundled skills are discoverable for
    both editable installs and wheel installs (including zipped wheels).
    The returned object must be passed to :func:`importlib.resources.as_file`
    to get a usable filesystem path.
    """
    try:
        return files(_BUNDLED_SKILLS_PACKAGE) / _BUNDLED_SKILLS_RESOURCE
    except (ModuleNotFoundError, FileNotFoundError, TypeError, OSError):
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
