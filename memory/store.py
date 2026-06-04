#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Store — JSON-backed persistence for LinkCommand.

Provides a :class:`Store` facade over three JSON files:

``user_profile.json``   Onboarding answers + demographic info.
``post_history.json``   Generation + publish events with counters.
``preferences.json``    UI / behaviour toggles (AI mode, default style…).

All mutating methods are auto-persisting.  Data paths are resolved relative
to the project root (parent of this ``memory/`` directory).
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent

PROFILE_PATH  = PROJECT_ROOT / "memory" / "user_profile.json"
HISTORY_PATH  = PROJECT_ROOT / "memory" / "post_history.json"
PREFS_PATH    = PROJECT_ROOT / "memory" / "preferences.json"

# Force UTF-8 on Windows consoles that default to cp1252.
if sys.platform == "win32":                     # pragma: no cover
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stdin .reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _read(path: Path, default: Any) -> Any:
    """Return parsed JSON or *default* when the file is absent / corrupt."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def _write(path: Path, data: Any) -> None:
    """Atomic-ish write: dump to a temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE: dict[str, Any] = {
    "onboarded": False,
    "name": "",
    "role": "",
    "field": "",
    "interests": [],
    "tone": "professional but approachable",
    "goals": [],
    "posting_frequency": "3x/week",
    "completed_at": None,
}

_DEFAULT_PREFS: dict[str, Any] = {
    "ai_enabled": False,
    "auto_suggest_hashtags": True,
    "default_style": "tip",
    "show_emojis": True,
    "daily_reminder": False,
}

_DEFAULT_HISTORY: dict[str, Any] = {
    "posts": [],
    "generation_count": 0,
    "publish_count": 0,
    "streak": 0,
    "last_post_date": None,
    "last_generation_date": None,
}


# ---------------------------------------------------------------------------
# Store facade
# ---------------------------------------------------------------------------

class Store:
    """Facade over the three JSON data files.

    Attributes are lazily loaded on first access and cached for the life of
    the :class:`Store` instance.  Every mutating method persists
    automatically; callers never need to call ``save`` explicitly.
    """

    def __init__(self) -> None:
        self._profile: dict[str, Any] = _read(PROFILE_PATH, dict(_DEFAULT_PROFILE))
        self._prefs: dict[str, Any]   = _read(PREFS_PATH,   dict(_DEFAULT_PREFS))
        self._history: dict[str, Any] = _read(HISTORY_PATH,  dict(_DEFAULT_HISTORY))

    # -- profile ------------------------------------------------------------

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile

    @property
    def onboarded(self) -> bool:
        return bool(self._profile.get("onboarded"))

    def save_profile(self, **updates: Any) -> None:
        """Merge *updates* into the profile and persist."""
        self._profile.update(updates)
        _write(PROFILE_PATH, self._profile)

    def set_profile_field(self, field: str, value: Any) -> None:
        self._profile[field] = value
        _write(PROFILE_PATH, self._profile)

    def complete_onboarding(self) -> None:
        self._profile["onboarded"]    = True
        self._profile["completed_at"] = date.today().isoformat()
        _write(PROFILE_PATH, self._profile)

    # -- preferences --------------------------------------------------------

    @property
    def prefs(self) -> dict[str, Any]:
        return self._prefs

    def save_prefs(self, **updates: Any) -> None:
        self._prefs.update(updates)
        _write(PREFS_PATH, self._prefs)

    def set_pref(self, key: str, value: Any) -> None:
        self._prefs[key] = value
        _write(PREFS_PATH, self._prefs)

    @property
    def ai_enabled(self) -> bool:
        return bool(self._prefs.get("ai_enabled"))

    # -- history / stats ----------------------------------------------------

    @property
    def history(self) -> dict[str, Any]:
        return self._history

    @property
    def stats(self) -> dict[str, Any]:
        """Derived, read-only statistics computed from history."""
        h = self._history
        return {
            "posts_generated":   h.get("generation_count", 0),
            "posts_published":   h.get("publish_count", 0),
            "streak":            h.get("streak", 0),
            "last_post_date":    h.get("last_post_date"),
            "last_generation":   h.get("last_generation_date"),
        }

    def _update_streak(self, today: str) -> None:
        """Helper to compute streak based on daily activity (generation or publishing)."""
        dates = []
        if self._history.get("last_generation_date"):
            dates.append(self._history["last_generation_date"])
        if self._history.get("last_post_date"):
            dates.append(self._history["last_post_date"])

        if dates:
            last_act = max(dates)
            if last_act == today:
                # Already had activity today, streak is maintained but not incremented
                return
            try:
                gap = (date.fromisoformat(today) - date.fromisoformat(last_act)).days
                if gap == 1:
                    self._history["streak"] = self._history.get("streak", 0) + 1
                else:
                    self._history["streak"] = 1
            except ValueError:
                self._history["streak"] = 1
        else:
            self._history["streak"] = 1

    def record_generation(self, post_id: int, title: str,
                          source: str = "ai") -> None:
        """Append a generation event and bump counters + streak."""
        today = date.today().isoformat()
        self._update_streak(today)

        posts: list[dict[str, Any]] = self._history.setdefault("posts", [])
        posts.append({
            "id":       post_id,
            "title":    title,
            "source":   source,
            "action":   "generated",
            "date":     today,
            "datetime": datetime.now().isoformat(),
        })

        self._history["generation_count"]    = self._history.get("generation_count", 0) + 1
        self._history["last_generation_date"] = today

        _write(HISTORY_PATH, self._history)

    def record_publish(self, post_id: int, title: str) -> None:
        """Append a publish event and bump the published counter."""
        today = date.today().isoformat()
        self._update_streak(today)

        posts: list[dict[str, Any]] = self._history.setdefault("posts", [])
        posts.append({
            "id":       post_id,
            "title":    title,
            "action":   "published",
            "date":     today,
            "datetime": datetime.now().isoformat(),
        })
        self._history["publish_count"]  = self._history.get("publish_count", 0) + 1
        self._history["last_post_date"] = today
        _write(HISTORY_PATH, self._history)

    def get_recent_topics(self, limit: int = 10) -> list[str]:
        """Return titles from the most recent generation actions."""
        posts = self._history.get("posts", [])
        gens  = [p for p in posts if p.get("action") == "generated"]
        return [p["title"] for p in gens[-limit:]]

    def save_all(self) -> None:
        """Force-persist all three files (useful after bulk updates)."""
        _write(PROFILE_PATH, self._profile)
        _write(PREFS_PATH,   self._prefs)
        _write(HISTORY_PATH, self._history)
