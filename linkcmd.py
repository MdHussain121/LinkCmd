#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinkCommand — Rich UI Edition.

A feature-complete CLI manager for creating, scheduling, and tracking LinkedIn
posts, powered by the NVIDIA NIM (Llama) API for AI-assisted generation.

Key features
------------
* Rich-powered TUI  with panels, tables, progress spinners, and markdown.
* First-run onboarding that persists every answer to ``memory/`` via
  :class:`memory.store.Store`.
* AI post generation with full profile context (name, role, field, interests,
  tone) forwarded to NIM so the output sounds like *you*.
* Manual drafting with hashtag suggestions.
* Draft library, weekly planner, stats dashboard, and a settings screen.
* Dual mode -- when the API key is missing or the user opts out, the tool
  degrades gracefully to a pure manual draft manager.

Dependencies
------------
pip install rich openai   (openai is a transitive dep of nim_client)

File layout (all paths resolved relative to this script's directory)
--------------------------------------------------------------------
memory/user_profile.json   Onboarding answers.
memory/post_history.json   AI generation log + published counts.
memory/preferences.json    UI / engagement flags.
posts.json                 All drafts and published posts.
config.json                API key placeholder (legacy; actually stored in
                           nim_config.json).
templates.json             Hand-written post templates for inspiration.
nim_config.json            NIM API credentials and profile_context for the
                           :class:`nim_client.NIMClient`.

Usage
-----
    # Start interactive menu:
    python linkcmd.py

    # Direct CLI (power-user mode):
    python linkcmd.py generate --topic "AI" --style tip
    python linkcmd.py list
    python linkcmd.py view 3
    python linkcmd.py publish 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Rich imports
# ---------------------------------------------------------------------------
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from memory.store import Store
from nim_client import NIMClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
POSTS_FILE     = _SCRIPT_DIR / "posts.json"
TEMPLATES_FILE = _SCRIPT_DIR / "templates.json"
CONFIG_FILE    = _SCRIPT_DIR / "nim_config.json"
DATA_DIR       = _SCRIPT_DIR

# Force UTF-8 on cp1252 Windows consoles.
if sys.platform == "win32":                     # pragma: no cover
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stdin .reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Console instance (all output flows through here)
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Colour palette constants (used in escape-free Markdown / Text builders)
# ---------------------------------------------------------------------------
C_HEAD  = "cyan"        # section headings
C_SUCC  = "green"       # success indicators
C_WARN  = "yellow"      # warnings
C_AI    = "magenta"     # anything touching the AI
C_DIM   = "dim"         # secondary info
C_ERR   = "red"         # errors

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_posts() -> list[dict]:
    """Return the list of posts, or [] if the file does not yet exist or is empty."""
    if not POSTS_FILE.exists():
        return []
    if POSTS_FILE.stat().st_size == 0:
        return []
    try:
        with open(POSTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            console.print(f"[{C_ERR}]CRITICAL ERROR: posts.json format is invalid (not a list).[/{C_ERR}]")
            sys.exit(1)
        return data
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[{C_ERR}]CRITICAL ERROR: Failed to load posts.json: {e}[/{C_ERR}]")
        console.print(f"[{C_WARN}]To prevent data loss, operations have been aborted. Check file permissions or corruption.[/{C_WARN}]")
        sys.exit(1)


def _save_posts(posts: list[dict]) -> None:
    POSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(posts, fh, indent=2, ensure_ascii=False, default=str)


def _next_id(posts: list[dict]) -> int:
    return max((int(p.get("id", 0)) for p in posts), default=0) + 1


def _load_nim_client(store: Store) -> Optional[NIMClient]:
    """Instantiate an NIM client if AI is enabled and credentials are present."""
    if not store.prefs.get("ai_enabled"):
        return None
    try:
        return NIMClient(str(CONFIG_FILE))
    except (ImportError, ValueError, Exception):
        return None


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

ONBOARD_QUESTIONS: list[tuple[str, str, str]] = [
    ("name",         "What's your full name?",                      ""),
    ("role",         "What's your current role? (e.g. Backend Engineer)", ""),
    ("field",        "What industry / field are you in? (e.g. cloud infrastructure)", ""),
    ("interests",    "What are your interests? (comma-separated)",  ""),
    ("tone",         "What's your preferred writing tone?          (professional, casual, witty, etc.)", "professional but approachable"),
    ("goals",        "What are your LinkedIn goals?              (networking, hiring, thought leadership, etc.)", ""),
    ("posting_frequency", "How often do you want to post?          (daily, 3x/week, weekly)", "3x/week"),
]


def _run_onboarding(store: Store) -> None:
    """Walk the user through first-run setup and persist answers."""
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to LinkCommand![/bold cyan]\n\n"
            "Let's set up your profile so AI-generated posts feel personal.\n"
            "You can always change these later in Settings.",
            border_style=C_HEAD,
            padding=(1, 4),
        )
    )
    console.print()

    answers: dict[str, Any] = {}
    for field, question, default in ONBOARD_QUESTIONS:
        answer = Prompt.ask(
            f"[{C_HEAD}]{question}[/{C_HEAD}]",
            default=default or None,
            show_default=bool(default),
            console=console,
        ).strip()
        if field == "interests":
            answer = [x.strip() for x in answer.split(",") if x.strip()]
        elif field == "goals":
            answer = [x.strip() for x in answer.split(",") if x.strip()]
        answers[field] = answer

    store.save_profile(**answers)
    store.complete_onboarding()
    console.print()
    console.print(f"[{C_SUCC}]Profile saved!  You're all set, {answers.get('name', 'friend')}.[/{C_SUCC}]")


# ---------------------------------------------------------------------------
# Greeting & stats banner
# ---------------------------------------------------------------------------

def _greet(store: Store, nim_client: Optional[NIMClient]) -> None:
    """Display a personalised greeting and at-a-glance stats."""
    profile = store.profile
    name    = profile.get("name", "there")
    stats   = store.stats

    ai_label = (
        f"[{C_AI}]AI [{C_SUCC}]ON[/{C_SUCC}][/{C_AI}]"
        if (nim_client is not None)
        else f"[{C_WARN}]AI [red]OFF[/red][/{C_WARN}]"
    )

    greeting = Text()
    greeting.append(f"Hey {name}", style=f"bold {C_HEAD}")
    greeting.append("  |  ", style=C_DIM)
    greeting.append(ai_label, style=C_DIM)

    top = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    top.add_column("greeting", justify="left", no_wrap=False)
    top.add_row(greeting)
    console.print(Panel(top, box=box.SIMPLE_HEAD, padding=(0, 1)))

    # Stats row
    stats_table = Table(
        show_header=True,
        header_style=f"bold {C_HEAD}",
        box=box.MINIMAL,
        expand=True,
        padding=(0, 2),
    )
    stats_table.add_column("", justify="center")
    stats_table.add_column("", justify="center")
    stats_table.add_column("", justify="center")
    stats_table.add_column("", justify="center")
    stats_table.add_row(
        "[bold]Posts generated[/bold]",
        "[bold]Published[/bold]",
        "[bold]Streak[/bold]",
        "[bold]Last generated[/bold]",
    )
    stats_table.add_row(
        str(stats["posts_generated"]),
        str(stats["posts_published"]),
        f"{stats['streak']} days",
        stats["last_generation"] or "—",
    )
    console.print(stats_table)
    console.print()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_generate(store: Store, nim_client: Optional[NIMClient],
                  topic: str = "", style: str = "") -> None:
    """AI-generate a new LinkedIn post and save it as a draft."""
    if nim_client is None:
        console.print(f"[{C_ERR}]AI mode is not enabled.[/{C_ERR}]  Toggle it on in Settings.")
        return

    # Augment client with current profile context.
    nim_client.set_profile_context(
        name=store.profile.get("name", ""),
        role=store.profile.get("role", ""),
        field=store.profile.get("field", ""),
        interests=store.profile.get("interests", []),
        tone=store.profile.get("tone", ""),
    )

    # Interactive prompting when run from menu (no CLI flags).
    if not topic:
        topic = Prompt.ask(
            "[bold magenta]Topic[/bold magenta]",
            default=None,
            show_default=False,
            console=console,
        ).strip() or ""
    if not style:
        style_choices = ["tip", "story", "opinion", "question"]
        style = Prompt.ask(
            "[bold magenta]Style[/bold magenta]",
            choices=style_choices,
            default=store.prefs.get("default_style", "tip"),
            console=console,
        )

    console.print()
    with console.status(
        f"[bold {C_AI}]Thinking about \"{topic or 'a great topic'}\"…[/bold {C_AI}]",
        spinner="dots",
    ) as status:
        _time.sleep(0.4)         # give the spinner a moment to render
        try:
            result = nim_client.generate_post(
                topic=topic or None,
                style=style or None,
            )
        except Exception as exc:
            console.print(f"[{C_ERR}]Generation failed: {exc}[/{C_ERR}]")
            return

    if not result or "error" in result or str(result.get("content", "")).startswith("ERROR"):
        console.print(f"[{C_ERR}]Generation error:[/{C_ERR}]\n  {result}")
        return

    # Persist the new draft.
    posts = _load_posts()
    now   = date.today().isoformat()
    pid   = _next_id(posts)

    post: dict[str, Any] = {
        "id":              pid,
        "title":           result.get("title", f"AI Post #{pid}"),
        "content":         result.get("content", ""),
        "status":          "draft",
        "created_at":      now,
        "published_at":    None,
        "source":          "ai_generated",
        "topic":           topic,
        "style":           style,
        "hashtags":        result.get("hashtags", []),
        "image_prompts":   result.get("image_prompts", []),
    }
    posts.append(post)
    _save_posts(posts)
    store.record_generation(pid, post["title"])

    # Render result.
    console.print()
    body = "\n".join(
        result.get("content", "").split("\n")[:20]
    )
    console.print(
        Panel(
            Markdown(body),
            title=f"[bold {C_AI}] {post['title']}[/bold {C_AI}]",
            subtitle=f"  #{pid}  ·  {now}  ·  draft",
            border_style=C_AI,
            padding=(1, 2),
        )
    )

    if post.get("hashtags"):
        console.print("  Hashtags:", " ".join(f"[cyan]#{t}[/cyan]" for t in post["hashtags"]))
    if post.get("image_prompts"):
        console.print()
        console.print(f"[{C_DIM}]Image prompts on file — view from the Drafts screen.[/{C_DIM}]")

    console.print()
    console.print(f"[{C_SUCC}]Saved as draft #{pid}.[/{C_SUCC}]")


def _cmd_manual_write(store: Store) -> None:
    """Let the user type a post from scratch."""
    console.print(Panel("[bold]Write a new post[/bold]", border_style=C_HEAD))
    console.print()

    title = Prompt.ask("Title (for your reference)", console=console).strip()
    if not title:
        console.print(f"[{C_WARN}]Title required — cancelling.[/{C_WARN}]")
        return

    console.print(
        "\nWrite your post below.  Press Enter on an empty line when done.\n"
    )
    lines: list[str] = []
    while True:
        line = Prompt.ask("", default="", console=console, show_default=False)
        if not line and not lines:
            continue          # skip leading blank line
        if not line and lines:
            break
        lines.append(line)

    content = "\n".join(lines).strip()
    if not content:
        console.print(f"[{C_WARN}]No content — aborting.[/{C_WARN}]")
        return

    # Simple keyword-based hashtag suggestions.
    suggested = _suggest_hashtags(content)
    hashtags: list[str] = []
    if suggested:
        console.print(f"\n[{C_DIM}]Suggested hashtags:[/{C_DIM}]  "
                      + " ".join(f"[cyan]#{t}[/cyan]" for t in suggested))
        if Confirm.ask("Add these hashtags?", default=True, console=console):
            hashtags = suggested

    posts = _load_posts()
    now   = date.today().isoformat()
    pid   = _next_id(posts)
    post: dict[str, Any] = {
        "id":           pid,
        "title":        title,
        "content":      content,
        "status":       "draft",
        "created_at":   now,
        "published_at": None,
        "source":       "manual",
        "hashtags":     hashtags,
        "image_prompts": [],
    }
    posts.append(post)
    _save_posts(posts)
    store.record_generation(pid, title, source="manual")

    console.print()
    _display_post_panel(post)
    console.print(f"\n[{C_SUCC}]Saved as draft #{pid}.[/{C_SUCC}]")


def _cmd_view_drafts(store: Store) -> None:
    """Browse drafts in a table; select one to view."""
    posts = _load_posts()
    drafts = [p for p in posts if p.get("status") != "published"]
    if not drafts:
        console.print(f"\n[{C_DIM}]No drafts yet.  Use 'Generate' or 'Write Manually' to create one.[/{C_DIM}]")
        return

    draft_table = Table(
        title="[bold cyan]Your Drafts[/bold cyan]",
        box=box.ROUNDED,
        padding=(0, 2),
        expand=True,
    )
    draft_table.add_column("ID", justify="right", style="bold", width=5)
    draft_table.add_column("Title", width=36)
    draft_table.add_column("Created", width=12)
    draft_table.add_column("Status", justify="center", width=8)
    draft_table.add_column("Src", justify="center", width=8)

    for p in sorted(drafts, key=lambda x: x.get("created_at", ""), reverse=True):
        status_icon = "📝" if p.get("status") == "draft" else "✅"
        source_icon = "🤖" if p.get("source") == "ai_generated" else (
            "🔄" if p.get("source") == "ai_variation" else "✏️"
        )
        draft_table.add_row(
            str(p.get("id", "?")),
            Text(p.get("title", "Untitled"), overflow="ellipsis", no_wrap=True),
            p.get("created_at", "—"),
            status_icon,
            source_icon,
        )

    console.print()
    console.print(draft_table)

    pid_str = Prompt.ask(
        "\nEnter a [bold]draft ID[/bold] to view (or press Enter to cancel)",
        default="",
        console=console,
        show_default=False,
    ).strip()
    if not pid_str:
        return
    try:
        pid = int(pid_str)
    except ValueError:
        console.print(f"[{C_ERR}]Invalid ID.[/{C_ERR}]")
        return
    post = next((p for p in posts if int(p.get("id", -1)) == pid), None)
    if not post:
        console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
        return
    _display_post_panel(post)

    # Offer quick actions.
    actions_table = Table(show_header=False, box=None, expand=False, padding=(0, 1))
    actions_table.add_column("key", style="bold cyan")
    actions_table.add_column("action")
    actions_table.add_row("p", "Mark as published")
    actions_table.add_row("d", "Delete this post")
    
    nim_client = _load_nim_client(store)
    if nim_client is not None:
        actions_table.add_row("i", "AI Improve Post")
        actions_table.add_row("v", "AI Variations")
        actions_table.add_row("g", "AI Image Options")
        
    actions_table.add_row("Enter", "Back")
    console.print(actions_table)

    choices = ["p", "d", ""]
    if nim_client is not None:
        choices.extend(["i", "v", "g"])

    choice = Prompt.ask("Action", choices=choices, default="", console=console)
    if choice == "p":
        _cmd_publish(post, store)
    elif choice == "d":
        _cmd_delete(post, store)
    elif choice == "i" and nim_client is not None:
        _cmd_improve_post(post, store, nim_client)
    elif choice == "v" and nim_client is not None:
        _cmd_generate_variations(post, store, nim_client)
    elif choice == "g" and nim_client is not None:
        _cmd_image_options(post, store, nim_client)


def _cmd_weekly_planner(store: Store, interactive: bool = True) -> None:
    """Show how many posts this week (Mon–Sun) — useful for pacing."""
    posts = _load_posts()
    today  = date.today()
    current_week_start = today - timedelta(days=today.weekday())
    selected_week_start = current_week_start

    while True:
        week_dates = [selected_week_start + timedelta(days=i) for i in range(7)]

        # Build a day → list-of-posts map for the selected week
        def _week_map(start: date) -> dict[str, list[dict]]:
            return {
                (start + timedelta(days=i)).isoformat(): [
                    p for p in posts
                    if p.get("created_at") == (start + timedelta(days=i)).isoformat()
                ]
                for i in range(7)
            }

        this_week_map = _week_map(selected_week_start)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        table = Table(
            title=f"[bold {C_HEAD}]Weekly Planner — week of {selected_week_start.isoformat()}[/bold {C_HEAD}]",
            box=box.ROUNDED,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Day", style="bold")
        table.add_column("Date", style=C_DIM)
        table.add_column("Posts", justify="center")
        table.add_column("Published", justify="center")
        table.add_column("Notes", style=C_DIM)

        for i, day_date in enumerate(week_dates):
            ds       = day_date.isoformat()
            day_posts = this_week_map.get(ds, [])
            drafts   = [p for p in day_posts if p.get("published_at") is None]
            pub      = [p for p in day_posts if p.get("published_at") is not None]
            count    = len(day_posts)
            pub_n    = len(pub)
            
            note_text = ""
            if day_posts:
                note_text = f"{day_posts[0].get('title', 'Untitled')[:40]}…"
                
            table.add_row(
                day_names[i],
                ds[5:],      # MM-DD
                f"{count} {'📝' if drafts else ''}",
                str(pub_n),
                note_text,
            )

        if interactive:
            console.clear()
            console.print()
        console.print(table)

        # Frequencies from profile.
        freq = store.profile.get("posting_frequency", "3x/week").lower()
        target_map = {"daily": 7, "3x/week": 3, "weekly": 1}
        target = target_map.get(freq, 3)
        current_week_count = sum(len(v) for v in this_week_map.values())

        if selected_week_start == current_week_start:
            remaining_days = 7 - today.weekday() - 1   # days left incl. today
            if remaining_days > 0 and current_week_count < target:
                need = target - current_week_count
                console.print(
                    f"\n[{C_WARN}]Goal: {target} posts/week ({freq}).[/{C_WARN}]  "
                    f"{current_week_count} done.  "
                    f"~[bold]{need}[/bold] more needed over the next {remaining_days} day(s). "
                    f"Use [bold magenta]Generate[/bold magenta] to create one!"
                )
        else:
            console.print(f"\n[{C_HEAD}]Goal: {target} posts/week ({freq}).[/{C_HEAD}]  "
                          f"{current_week_count} posts generated in this week.")
        console.print()

        if not interactive:
            break

        console.print("[bold]Options:[/bold]")
        console.print("  [bold cyan]p[/bold cyan] : Previous Week")
        console.print("  [bold cyan]n[/bold cyan] : Next Week")
        console.print("  [bold cyan]t[/bold cyan] : Jump to Current Week")
        console.print("  [bold cyan]Enter[/bold cyan] : Back to main menu")

        nav = Prompt.ask("\nChoose option", choices=["p", "n", "t", ""], default="", console=console).strip().lower()
        if not nav:
            break
        elif nav == "p":
            selected_week_start -= timedelta(days=7)
        elif nav == "n":
            selected_week_start += timedelta(days=7)
        elif nav == "t":
            selected_week_start = current_week_start


def _cmd_stats(store: Store, nim_client: Optional[NIMClient]) -> None:
    """Show a full stats dashboard."""
    stats = store.stats
    posts = _load_posts()
    profile = store.profile

    # --- Stats panel ------------------------------------------------
    stats_panel = Table(
        show_header=False, box=box.ROUNDED, expand=True, padding=(1, 2),
    )
    stats_panel.add_column("metric", style=C_HEAD)
    stats_panel.add_column("value", justify="right", style="bold white")
    stats_panel.add_row("Posts generated",   str(stats["posts_generated"]))
    stats_panel.add_row("Posts published",   str(stats["posts_published"]))
    stats_panel.add_row("Current streak",    f"{stats['streak']} days")
    stats_panel.add_row("Drafts on file",    str(sum(1 for p in posts if p.get("published_at") is None)))
    stats_panel.add_row("Published total",   str(sum(1 for p in posts if p.get("published_at") is not None)))

    console.print()
    console.print(Panel(stats_panel, title=f"[bold {C_HEAD}]Your Stats[/bold {C_HEAD}]", border_style=C_HEAD))

    # --- Recent generations from memory history --------------------
    recent = store.get_recent_topics(limit=8)
    if recent:
        recent_table = Table(
            title=f"[bold]Recent AI Generations[/bold]",
            box=box.MINIMAL,
            expand=True,
        )
        recent_table.add_column("#", justify="right", style=C_DIM, width=4)
        recent_table.add_column("Title", style=C_AI)
        for idx, title in enumerate(reversed(recent), 1):
            recent_table.add_row(str(idx), Text(title))
        console.print(recent_table)

    # --- Profile summary -------------------------------------------
    profile_table = Table(
        title="[bold]Profile[/bold]",
        box=box.MINIMAL,
        expand=True,
    )
    profile_table.add_column("Field", style=C_HEAD)
    profile_table.add_column("Value")
    for k in ("name", "role", "field", "tone", "posting_frequency"):
        profile_table.add_row(k.replace("_", " ").capitalize(),
                              profile.get(k, "") or "[dim]—[/dim]")
    interests = profile.get("interests", [])
    profile_table.add_row("Interests", ", ".join(interests) if interests else "[dim]—[/dim]")
    goals = profile.get("goals", [])
    profile_table.add_row("Goals", ", ".join(goals) if goals else "[dim]—[/dim]")
DEFAULT_NIM_CONFIG = {
    "api_key": "",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "text_model": "meta/llama-3.3-70b-instruct",
    "vision_model": "meta/llama-3.2-90b-vision-instruct",
    "image_model": "stable-diffusion-xl",
    "profile_context": {
        "name": "",
        "role": "",
        "field": "",
        "interests": [],
        "recent_work": [],
        "tone": "professional but approachable"
    }
}


def _load_nim_config() -> dict:
    if not CONFIG_FILE.exists():
        return dict(DEFAULT_NIM_CONFIG)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
            if not isinstance(cfg, dict):
                return dict(DEFAULT_NIM_CONFIG)
            return cfg
    except Exception:
        return dict(DEFAULT_NIM_CONFIG)


def _save_nim_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)


def _cmd_settings(store: Store, nim_client: Optional[NIMClient]) -> None:
    """Interactive settings / profile and AI editor."""
    while True:
        console.clear()

        # --- Profile section ----
        p = store.profile
        profile_table = Table(
            title="[bold cyan]Profile Settings[/bold cyan]",
            box=box.ROUNDED,
            expand=True,
        )
        profile_table.add_column("Key", style="bold cyan", width=5)
        profile_table.add_column("Field", style=C_HEAD, width=20)
        profile_table.add_column("Current value")

        fields_meta = [
            ("name",             p.get("name", ""),               False),
            ("role",             p.get("role", ""),               False),
            ("field",            p.get("field", ""),              False),
            ("tone",             p.get("tone", ""),               False),
            ("interests",        ", ".join(p.get("interests", [])), True),
            ("goals",            ", ".join(p.get("goals", [])),    True),
            ("posting_frequency",p.get("posting_frequency", ""),   False),
        ]
        for idx, (fname, val, _is_list) in enumerate(fields_meta, 1):
            display = val if val else f"[{C_DIM}]not set[/{C_DIM}]"
            profile_table.add_row(str(idx), fname.replace("_", " ").capitalize(), display)
        console.print(profile_table)
        console.print()

        # --- AI Settings section ----
        cfg = _load_nim_config()
        ai_enabled = store.prefs.get("ai_enabled", False)

        ai_table = Table(
            title="[bold magenta]AI Settings[/bold magenta]",
            box=box.ROUNDED,
            expand=True,
        )
        ai_table.add_column("Key", style="bold magenta", width=5)
        ai_table.add_column("Setting", style=C_HEAD, width=20)
        ai_table.add_column("Current value")

        ai_status = f"[{C_SUCC}]ON[/{C_SUCC}]" if ai_enabled else f"[{C_ERR}]OFF[/{C_ERR}]"
        ai_table.add_row("A", "AI Enabled Mode", ai_status)

        raw_key = cfg.get("api_key", "")
        masked_key = f"{raw_key[:10]}...{raw_key[-4:]}" if len(raw_key) > 14 else (f"[{C_DIM}]not set[/{C_DIM}]" if not raw_key else "***")
        ai_table.add_row("K", "NVIDIA NIM API Key", masked_key)
        ai_table.add_row("T", "Text Model ID", cfg.get("text_model", "meta/llama-3.3-70b-instruct"))
        ai_table.add_row("V", "Vision Model ID", cfg.get("vision_model", "meta/llama-3.2-90b-vision-instruct"))
        ai_table.add_row("I", "Image Model ID", cfg.get("image_model", "stable-diffusion-xl"))
        console.print(ai_table)
        console.print()

        # Actions Info
        console.print("[bold]Available Options:[/bold]")
        console.print("  [bold cyan]1-7[/bold cyan]     : Edit corresponding Profile field")
        console.print("  [bold magenta]A[/bold magenta]       : Toggle AI Generation Mode")
        console.print("  [bold magenta]K[/bold magenta]       : Set NIM API Key")
        console.print("  [bold magenta]T/V/I[/bold magenta]   : Edit Text/Vision/Image Model IDs")
        console.print("  [bold cyan]b[/bold cyan]       : Back to main menu\n")

        choice = Prompt.ask("Select an option", console=console).strip().lower()
        if choice == "b":
            break

        if choice == "a":
            new_status = not ai_enabled
            store.set_pref("ai_enabled", new_status)
            status_str = "enabled" if new_status else "disabled"
            console.print(f"[{C_SUCC}]AI mode has been {status_str}![/{C_SUCC}]")
            _time.sleep(1.0)
            continue

        if choice == "k":
            new_key = Prompt.ask("Enter your NVIDIA NIM API Key (nvapi-...)", console=console).strip()
            if new_key:
                cfg["api_key"] = new_key
                _save_nim_config(cfg)
                console.print(f"[{C_SUCC}]API Key saved successfully.[/{C_SUCC}]")
                _time.sleep(1.0)
            continue

        if choice == "t":
            new_val = Prompt.ask("Enter Text Model ID", default=cfg.get("text_model"), console=console).strip()
            if new_val:
                cfg["text_model"] = new_val
                _save_nim_config(cfg)
                console.print(f"[{C_SUCC}]Text Model ID updated.[/{C_SUCC}]")
                _time.sleep(1.0)
            continue

        if choice == "v":
            new_val = Prompt.ask("Enter Vision Model ID", default=cfg.get("vision_model"), console=console).strip()
            if new_val:
                cfg["vision_model"] = new_val
                _save_nim_config(cfg)
                console.print(f"[{C_SUCC}]Vision Model ID updated.[/{C_SUCC}]")
                _time.sleep(1.0)
            continue

        if choice == "i":
            new_val = Prompt.ask("Enter Image Model ID", default=cfg.get("image_model"), console=console).strip()
            if new_val:
                cfg["image_model"] = new_val
                _save_nim_config(cfg)
                console.print(f"[{C_SUCC}]Image Model ID updated.[/{C_SUCC}]")
                _time.sleep(1.0)
            continue

        # Check if index 1-7
        field_idx = None
        try:
            field_idx = int(choice) - 1
        except ValueError:
            console.print(f"[{C_ERR}]Invalid selection.[/{C_ERR}]")
            _time.sleep(1.0)
            continue

        if not (0 <= field_idx < len(fields_meta)):
            console.print(f"[{C_ERR}]Selection out of range.[/{C_ERR}]")
            _time.sleep(1.0)
            continue

        field, _, _ = fields_meta[field_idx]
        new_val = Prompt.ask(
            f"New value for [bold]{field}[/bold]",
            default=store.profile.get(field, ""),
            console=console,
        ).strip()
        if field in ("interests", "goals"):
            new_val = [x.strip() for x in new_val.split(",") if x.strip()]
        store.set_profile_field(field, new_val)

        # Sync back to nim_config.json profile_context as well
        cfg.setdefault("profile_context", {})[field] = new_val
        _save_nim_config(cfg)
        console.print(f"[{C_SUCC}]Updated {field}.[/{C_SUCC}]")
        _time.sleep(1.0)


def _display_post_panel(post: dict) -> None:
    status_icon = "✅" if post.get("published_at") else "📝"
    footer_parts: list[str] = [
        f"#{post.get('id', '?')}",
        post.get("created_at", "—"),
    ]
    if post.get("published_at"):
        footer_parts.append(f"published {post['published_at']}")
    footer = "  ·  ".join(footer_parts)
    body_text = post.get("content", "(empty)")
    console.print(
        Panel(
            Markdown(body_text),
            title=f"[bold]{status_icon}  {post.get('title', 'Untitled')}[/bold]",
            subtitle=f"  {footer}  ",
            border_style=C_HEAD if post.get("published_at") else C_AI,
            padding=(1, 2),
        )
    )
    tags = post.get("hashtags", [])
    if tags:
        console.print("  " + " ".join(f"[cyan]#{t}[/cyan]" for t in tags))


def _cmd_publish(post: dict,  store: Store) -> None:
    posts = _load_posts()
    pid = int(post.get("id", 0))
    for p in posts:
        if int(p.get("id", -1)) == pid:
            p["published_at"] = date.today().isoformat()
            p["status"] = "published"
            break
    _save_posts(posts)
    store.record_publish(pid, post.get("title", ""))
    console.print(f"[{C_SUCC}]Post #{pid} marked as published![/{C_SUCC}]")


def _cmd_delete(post: dict, store: Store) -> None:
    if not Confirm.ask(f"Delete post #{post.get('id')} '{post.get('title')}'?", console=console):
        console.print("Cancelled.")
        return
    posts = _load_posts()
    pid   = int(post.get("id", 0))
    posts = [p for p in posts if int(p.get("id", -1)) != pid]
    _save_posts(posts)
    console.print(f"[{C_ERR}]Post #{pid} deleted.[/{C_ERR}]")

def _cmd_improve_post(post: dict, store: Store, nim_client: NIMClient) -> None:
    """Improve the post draft using AI, displaying recommendations and comparing versions."""
    console.print()
    with console.status(
        f"[bold {C_AI}]Analyzing draft and gathering recommendations...[/bold {C_AI}]",
        spinner="dots",
    ):
        try:
            result = nim_client.improve_post(post["content"])
        except Exception as exc:
            console.print(f"[{C_ERR}]Improvement call failed: {exc}[/{C_ERR}]")
            return

    if not result or "error" in result:
        console.print(f"[{C_ERR}]Could not improve post:[/{C_ERR}] {result.get('error', 'Unknown error')}")
        if "raw_response" in result:
            console.print(Panel(result["raw_response"], title="Raw AI Response"))
        return

    score = result.get("score", "N/A")
    suggestions = result.get("suggestions", [])
    improved_content = result.get("improved_post", "")

    # Display score and suggestions
    console.print(Panel(f"[bold]Engagement Score Suggestion: {score}/10[/bold]", border_style=C_HEAD))
    
    if suggestions:
        s_table = Table(title="[bold cyan]Improvement Suggestions[/bold cyan]", box=box.ROUNDED, expand=True)
        s_table.add_column("Area", style="bold magenta", width=12)
        s_table.add_column("Current", style=C_ERR, width=25)
        s_table.add_column("Suggestion", style=C_SUCC, width=25)
        s_table.add_column("Reason", style=C_DIM)
        for s in suggestions:
            s_table.add_row(
                str(s.get("area", "N/A")),
                str(s.get("current", "N/A")),
                str(s.get("suggestion", "N/A")),
                str(s.get("reason", "N/A")),
            )
        console.print(s_table)
        console.print()

    # Show Side-by-Side/Sequential Comparison
    comparison = Table(show_header=True, box=box.ROUNDED, expand=True)
    comparison.add_column("Original Draft", style=C_DIM)
    comparison.add_column("AI Improved Version", style="white")
    comparison.add_row(Markdown(post["content"]), Markdown(improved_content))
    console.print(comparison)
    console.print()

    # Offer replacement or new draft options
    console.print("[bold]Options:[/bold]")
    console.print("  [bold cyan]1[/bold cyan]: Replace current draft content with improved version")
    console.print("  [bold cyan]2[/bold cyan]: Save improved version as a new draft")
    console.print("  [bold cyan]3[/bold cyan]: Cancel (keep current draft as-is)")
    
    action = Prompt.ask("Choose action", choices=["1", "2", "3"], default="3", console=console)
    if action == "1":
        posts = _load_posts()
        pid = int(post.get("id", 0))
        for p in posts:
            if int(p.get("id", -1)) == pid:
                p["content"] = improved_content
                if result.get("hashtags"):
                    p["hashtags"] = result.get("hashtags")
                break
        _save_posts(posts)
        console.print(f"[{C_SUCC}]Draft #{pid} updated successfully![/{C_SUCC}]")
    elif action == "2":
        posts = _load_posts()
        new_pid = _next_id(posts)
        now = date.today().isoformat()
        new_post = {
            "id": new_pid,
            "title": f"Improved: {post.get('title')}",
            "content": improved_content,
            "status": "draft",
            "created_at": now,
            "published_at": None,
            "source": "ai_variation",
            "hashtags": result.get("hashtags", post.get("hashtags", [])),
            "image_prompts": post.get("image_prompts", []),
        }
        posts.append(new_post)
        _save_posts(posts)
        store.record_generation(new_pid, new_post["title"], source="ai_variation")
        console.print(f"[{C_SUCC}]Saved as new draft #{new_pid}![/{C_SUCC}]")


def _cmd_generate_variations(post: dict, store: Store, nim_client: NIMClient) -> None:
    """Generate and display alternative angles of the post, allowing saving them."""
    console.print()
    with console.status(
        f"[bold {C_AI}]Generating alternative variations...[/bold {C_AI}]",
        spinner="dots",
    ):
        try:
            variations = nim_client.generate_variations(post["content"])
        except Exception as exc:
            console.print(f"[{C_ERR}]Variations call failed: {exc}[/{C_ERR}]")
            return

    if not variations or (isinstance(variations, list) and len(variations) == 0):
        console.print(f"[{C_ERR}]No variations generated.[/{C_ERR}]")
        return

    # Check for error
    if isinstance(variations, dict) and "error" in variations:
        console.print(f"[{C_ERR}]Could not generate variations:[/{C_ERR}] {variations.get('error')}")
        return

    console.print(f"\n[bold {C_HEAD}]Generated Variations[/bold {C_HEAD}]")
    for idx, var in enumerate(variations, 1):
        style = var.get("style", "N/A")
        title = var.get("title", f"Variation {idx}")
        content = var.get("content", "")
        
        console.print(
            Panel(
                Markdown(content),
                title=f"[bold {C_AI}]Variation {idx} ({style}) — {title}[/bold {C_AI}]",
                border_style=C_AI,
                padding=(1, 2)
            )
        )

    # Ask to save one of them
    choices = [str(i) for i in range(1, len(variations) + 1)] + [""]
    save_idx_str = Prompt.ask(
        "\nEnter variation number (1-2) to save as a draft (or Enter to cancel)",
        choices=choices,
        show_choices=False,
        default="",
        console=console
    ).strip()

    if not save_idx_str:
        return

    idx = int(save_idx_str) - 1
    selected_var = variations[idx]
    
    posts = _load_posts()
    new_pid = _next_id(posts)
    now = date.today().isoformat()
    new_post = {
        "id": new_pid,
        "title": selected_var.get("title") or f"Var: {post.get('title')}",
        "content": selected_var.get("content", ""),
        "status": "draft",
        "created_at": now,
        "published_at": None,
        "source": "ai_variation",
        "hashtags": post.get("hashtags", []),
        "image_prompts": post.get("image_prompts", []),
    }
    posts.append(new_post)
    _save_posts(posts)
    store.record_generation(new_pid, new_post["title"], source="ai_variation")
    console.print(f"[{C_SUCC}]Variation #{idx+1} successfully saved as draft #{new_pid}![/{C_SUCC}]")


def _cmd_image_options(post: dict, store: Store, nim_client: NIMClient) -> None:
    """Generate image prompts for a post and let the user generate a local PNG."""
    console.print()
    with console.status(
        f"[bold {C_AI}]Creating creative image prompts...[/bold {C_AI}]",
        spinner="dots",
    ):
        try:
            prompts = nim_client.generate_image_prompts(post["content"])
        except Exception as exc:
            console.print(f"[{C_ERR}]Image prompts call failed: {exc}[/{C_ERR}]")
            return

    if not prompts or (isinstance(prompts, list) and len(prompts) == 0):
        console.print(f"[{C_ERR}]No image prompts generated.[/{C_ERR}]")
        return

    if isinstance(prompts, dict) and "error" in prompts:
        console.print(f"[{C_ERR}]Could not generate prompts:[/{C_ERR}] {prompts.get('error')}")
        return

    console.print(f"\n[bold {C_HEAD}]Suggested AI Image Prompts[/bold {C_HEAD}]")
    for idx, prompt_text in enumerate(prompts, 1):
        console.print(f"  [bold cyan]{idx}.[/bold cyan] {prompt_text}\n")

    if not Confirm.ask("Would you like to generate a local image from one of these prompts?", default=False, console=console):
        # Save prompts to draft instead
        if Confirm.ask("Save these prompts to the draft's record?", default=True, console=console):
            posts = _load_posts()
            pid = int(post.get("id", 0))
            for p in posts:
                if int(p.get("id", -1)) == pid:
                    p["image_prompts"] = prompts
                    break
            _save_posts(posts)
            console.print(f"[{C_SUCC}]Image prompts saved to draft #{pid}.[/{C_SUCC}]")
        return

    choices = [str(i) for i in range(1, len(prompts) + 1)]
    prompt_choice = Prompt.ask("Choose prompt (1-3)", choices=choices, console=console)
    selected_prompt = prompts[int(prompt_choice) - 1]

    filename = Prompt.ask("Save image as (e.g. post_image.png)", default=f"post_{post.get('id')}.png", console=console).strip()
    if not filename:
        filename = f"post_{post.get('id')}.png"

    save_path = _SCRIPT_DIR / filename

    console.print()
    with console.status(
        f"[bold {C_AI}]Calling Stable Diffusion via NVIDIA NIM...[/bold {C_AI}]",
        spinner="dots",
    ):
        try:
            res = nim_client.generate_image(selected_prompt, str(save_path))
        except Exception as exc:
            console.print(f"[{C_ERR}]Image generation call failed: {exc}[/{C_ERR}]")
            return

    if res.get("success"):
        console.print(f"[{C_SUCC}]Success! Image generated and saved to:[/{C_SUCC}] [bold cyan]{save_path}[/bold cyan]")
    else:
        console.print(f"[{C_ERR}]Image generation failed:[/{C_ERR}] {res.get('error', 'Unknown error')}")
        if "note" in res:
            console.print(f"\n[{C_WARN}]Note:[/{C_WARN}]\n{res['note']}")


def _cmd_analyze_profile(store: Store, nim_client: NIMClient, default_image_path: Optional[str] = None) -> None:
    """Analyze a LinkedIn profile screenshot using Llama Vision and display detailed recommendations."""
    console.print(Panel("[bold]LinkedIn Profile Screenshot Analyzer[/bold]", border_style=C_HEAD))
    console.print()

    if default_image_path:
        image_path = default_image_path
    else:
        image_path = Prompt.ask(
            "Enter the path to your LinkedIn profile screenshot (PNG/JPG/WEBP)",
            console=console,
        ).strip()
    if not image_path:
        console.print(f"[{C_WARN}]No path entered — cancelling.[/{C_WARN}]")
        return

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        console.print(f"[{C_ERR}]File not found at: {image_path}[/{C_ERR}]")
        return

    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        console.print(f"[{C_ERR}]Unsupported image format. Please use PNG, JPG, JPEG, WEBP, or GIF.[/{C_ERR}]")
        return

    console.print()
    with console.status(
        f"[bold {C_AI}]Uploading and analyzing profile screenshot via Llama Vision...[/bold {C_AI}]",
        spinner="dots",
    ):
        try:
            result = nim_client.analyze_profile_screenshot(str(path))
        except Exception as exc:
            console.print(f"[{C_ERR}]Analysis failed: {exc}[/{C_ERR}]")
            return

    if not result or "error" in result:
        console.print(f"[{C_ERR}]Could not analyze profile:[/{C_ERR}] {result.get('error', 'Unknown error')}")
        if "raw_response" in result:
            console.print(Panel(result["raw_response"], title="Raw AI Response"))
        return

    # Visual layout of analysis results
    overall_score = result.get("overall_score", 0.0)
    score_pct = int(float(overall_score) * 10)
    # Simple ASCII bar
    score_bar = "█" * (score_pct // 10) + "░" * (10 - (score_pct // 10))
    console.print(
        Panel(
            f"[bold magenta]Overall Profile Score: {overall_score}/10[/bold magenta]\n[cyan]{score_bar}[/cyan]",
            border_style=C_AI,
            expand=True
        )
    )
    console.print()

    sections = result.get("sections", {})
    if sections:
        table = Table(title="[bold cyan]Section Breakdown[/bold cyan]", box=box.ROUNDED, expand=True)
        table.add_column("Section", style="bold", width=15)
        table.add_column("Score", justify="right", style="bold magenta", width=6)
        table.add_column("Observations", width=30)
        table.add_column("Recommended Fix", style=C_SUCC)

        for sec_key, sec_data in sections.items():
            sec_name = sec_key.replace("_", " ").capitalize()
            table.add_row(
                sec_name,
                f"{sec_data.get('score', 'N/A')}/10",
                str(sec_data.get("note", "")),
                str(sec_data.get("suggestion", ""))
            )
        console.print(table)
        console.print()

    priorities = result.get("top_priorities", [])
    if priorities:
        console.print(
            Panel(
                "\n".join(f"• {p}" for p in priorities),
                title="[bold red]🚨 Top Priorities[/bold red]",
                border_style="red",
                expand=True
            )
        )
        console.print()

    wins = result.get("quick_wins", [])
    if wins:
        console.print(
            Panel(
                "\n".join(f"• {w}" for w in wins),
                title="[bold green]⚡ Quick Wins (Under 5 mins)[/bold green]",
                border_style="green",
                expand=True
            )
        )
        console.print()

    # Offer to update profile
    if Confirm.ask("Would you like to update your LinkCommand profile settings based on these findings?", default=True, console=console):
        while True:
            console.print("\n[bold]Which field would you like to update?[/bold]")
            console.print("  [bold cyan]1[/bold cyan]: Name")
            console.print("  [bold cyan]2[/bold cyan]: Role / Headline")
            console.print("  [bold cyan]3[/bold cyan]: Industry / Field")
            console.print("  [bold cyan]4[/bold cyan]: Preferred Tone")
            console.print("  [bold cyan]5[/bold cyan]: Goals")
            console.print("  [bold cyan]b[/bold cyan]: Back to main menu")

            field_choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "b"], default="b", console=console)
            if field_choice == "b":
                break

            field_map = {
                "1": "name",
                "2": "role",
                "3": "field",
                "4": "tone",
                "5": "goals",
            }
            field = field_map[field_choice]
            current_val = store.profile.get(field, "")
            
            new_val = Prompt.ask(
                f"Enter new value for [bold]{field}[/bold]",
                default=str(current_val),
                console=console
            ).strip()

            if field == "goals":
                new_val = [x.strip() for x in new_val.split(",") if x.strip()]

            store.set_profile_field(field, new_val)
            
            # Sync back to nim_config.json profile_context as well
            cfg = _load_nim_config()
            cfg.setdefault("profile_context", {})[field] = new_val
            _save_nim_config(cfg)
            
            # Sync in-memory nim_client profile if initialized
            if nim_client is not None:
                nim_client.profile[field] = new_val

            console.print(f"[{C_SUCC}]Updated {field} successfully![/{C_SUCC}]")


def _suggest_hashtags(content: str) -> list[str]:
    """Extract hashtag candidates from keywords in the content using whole-word boundaries."""
    import re
    tag_map: dict[str, str] = {
        "ai": "AI", "ml": "MachineLearning", "python": "Python",
        "javascript": "JavaScript", "react": "React", "code": "Programming",
        "learn": "ContinuousLearning", "career": "CareerGrowth",
        "team": "Teamwork", "lead": "Leadership",
        "product": "ProductManagement", "startup": "Startups",
        "data": "DataScience", "cloud": "CloudComputing",
        "devops": "DevOps", "design": "Design",
        "remote": "RemoteWork", "interview": "JobSearch",
        "hiring": "Hiring", "project": "ProjectManagement",
        "software": "SoftwareDevelopment", "engineering": "Engineering",
        "growth": "GrowthMindset", "manager": "Management",
    }
    content_lower = content.lower()
    found: list[str] = []
    for kw, tag in tag_map.items():
        # Use regex word boundaries (\b) to prevent false substring matches (e.g. matching "ai" in "training")
        if re.search(r'\b' + re.escape(kw) + r'\b', content_lower) and tag not in found:
            found.append(tag)
        if len(found) >= 5:
            break
    return found


# ---------------------------------------------------------------------------
# Weekly planner helper
# ---------------------------------------------------------------------------

def _posterize(post: dict) -> str:
    """Return a compact one-liner summary for planner display."""
    title   = post.get("title", "?")
    snippet = (title[:30] + "…") if len(title) > 30 else title
    state   = "✅" if post.get("published_at") else "📝"
    return f"{state} {snippet}"


# ---------------------------------------------------------------------------
# Templates command
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def _cmd_help() -> None:
    """Display a full help / navigation guide."""
    console.clear()
    help_text = (
        "[bold cyan]LinkCommand — Help & Guide[/bold cyan]\n\n"
        "[bold]Navigating Menus[/bold]\n"
        "  Type the option number and press Enter.\n"
        "  Press Enter on an empty prompt to cancel/go back.\n"
        "  Type [bold]b[/bold] at the Settings editor to go back.\n\n"

        "[bold]Option 1 — Generate (AI)[/bold]\n"
        "  Enter a topic (e.g. \"AI agents\") and choose a style:\n"
        "    [magenta]tip[/magenta]       Short, actionable advice\n"
        "    [magenta]story[/magenta]     Personal anecdote with a lesson\n"
        "    [magenta]opinion[/magenta]   Strong take / discussion starter\n"
        "    [magenta]question[/magenta]  Open question meant to spark comments\n"
        "  The output is saved as a draft. Browse it in [bold]View Drafts[/bold].\n\n"

        "[bold]Option 2 — Write Manually[/bold]\n"
        "  Compose a post from scratch. Suggested hashtags are offered\n"
        "  automatically based on your content. Save and move on.\n\n"

        "[bold]Option 3 — View Drafts[/bold]\n"
        "  See every draft in a sortable table. Select a draft ID to:\n"
        "    [cyan]p[/cyan]  Mark as published\n"
        "    [cyan]d[/cyan]  Delete the post\n"
        "    [cyan]Enter[/cyan]  Back to the list\n\n"

        "[bold]Option 4 — Weekly Planner[/bold]\n"
        "  A Mon-Sun grid showing how many posts you have each day.\n"
        "  A nudge appears if you are behind your posting-frequency goal\n"
        "  (set during onboarding).\n\n"

        "[bold]Option 5 — Stats[/bold]\n"
        "  Dashboard showing: posts generated, published, current streak\n"
        "  (consecutive days with at least one activity), draft count, and\n"
        "  your profile summary.\n\n"

        "[bold]Option 6 — Templates[/bold]\n"
        "  Six fill-in-the-blank structures. Read the template, then use\n"
        "  [bold]Write Manually[/bold] to turn it into a real post.\n\n"

        "[bold]Option 7 — Settings[/bold]\n"
        "  Two sections:\n"
        "    [bold]Profile[/bold]    Edit name, role, industry, interests, tone, goals,\n"
        "                 and posting frequency.\n"
        "    [bold]AI Settings[/bold] Toggle AI on/off, change your API key,\n"
        "                 and swap models (see below).\n\n"

        "[bold]Changing Your API Key or Models[/bold]\n"
        f"  Go to [bold]Settings (7) → AI Settings[/bold].\n"
        "  You will see these options:\n"
        f"    [magenta]A[/magenta]  API Key — paste your NVIDIA NIM key\n"
        f"    [magenta]M[/magenta]  Text Model — ID of the Llama model for generation\n"
        f"    [magenta]V[/magenta]  Vision Model — ID for image/screenshot analysis\n"
        f"    [magenta]I[/magenta]  Image Model — ID for AI image generation\n"
        "  Changes are saved automatically to nim_config.json.\n\n"
        f"  [bold dim]Common model IDs:[/bold dim]\n"
        "    [cyan]meta/llama-3.3-70b-instruct[/cyan]  — default text (balanced)\n"
        "    [cyan]meta/llama-3.1-405b-instruct[/cyan]  — highest quality text\n"
        "    [cyan]meta/llama-3.3-8b-instruct[/cyan]    — fast, lightweight text\n"
        "    [cyan]meta/llama-3.2-90b-vision-instruct[/cyan]  — default vision\n"
        "    [cyan]meta/llama-3.2-11b-vision-instruct[/cyan]  — fast vision\n"
        "    [cyan]stable-diffusion-xl[/cyan]          — default image gen\n"
        "    [cyan]stable-diffusion-3-medium[/cyan]    — up-to-date SD image gen\n\n"

        "[bold]Power Mode (CLI)[/bold]\n"
        "  Skip the menu entirely:\n"
        "    python linkcmd.py generate   --topic \"AI\" --style tip\n"
        "    python linkcmd.py list\n"
        "    python linkcmd.py view       --id 3\n"
        "    python linkcmd.py publish    --id 2\n"
        "    python linkcmd.py templates\n"
        "    python linkcmd.py week\n"
        "    python linkcmd.py stats\n"
        "    python linkcmd.py settings\n\n"

        "[bold]Tip[/bold]\n"
        "  You can also set your API key via the environment variable\n"
        f"  [cyan]NIM_API_KEY[/cyan] — it overrides the value in nim_config.json.\n"
        f"  Example (PowerShell): [cyan]$env:NIM_API_KEY = 'sk-...'[/cyan]\n"
    )

    from rich.markdown import Markdown
    console.print(Panel(Markdown(help_text), title="[bold cyan]Help[/bold cyan]", border_style=C_HEAD, padding=(1, 2), expand=True))
    Prompt.ask("\nPress Enter to go back", default="", console=console)


def _cmd_templates(store: Optional[Store] = None) -> None:
    if not TEMPLATES_FILE.exists():
        console.print(f"\n[{C_WARN}]No templates.json found.[/{C_WARN}]  The file should live next to this script.")
        return
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as fh:
        try:
            templates: list[dict] = json.load(fh)
        except json.JSONDecodeError:
            console.print(f"[{C_ERR}]templates.json is corrupted.[/{C_ERR}]")
            return

    if not templates:
        console.print(f"\n[{C_DIM}]Template file is empty.[/{C_DIM}]")
        return

    layout = Table(title="[bold cyan]Post Templates[/bold cyan]",
                   box=box.ROUNDED, expand=True, padding=(0, 1))
    layout.add_column("#", justify="right", style="bold", width=4)
    layout.add_column("Name", style=C_HEAD)
    layout.add_column("Category", style=C_DIM)
    for i, t in enumerate(templates, 1):
        layout.add_row(str(i), t.get("name", "?"), t.get("category", "—"))
    console.print()
    console.print(layout)

    choice = Prompt.ask("\nSelect a template (or Enter to cancel)", console=console).strip()
    if not choice:
        return
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(templates)):
            raise ValueError
    except ValueError:
        console.print(f"[{C_ERR}]Invalid selection.[/{C_ERR}]")
        return

    tmpl = templates[idx]
    prompt_tmpl = tmpl.get("prompt", "")
    console.print()
    console.print(
        Panel(
            Markdown(prompt_tmpl),
            title=f"[bold]{tmpl.get('name', '?')}[/bold] — {tmpl.get('category', '')}",
            border_style=C_HEAD,
            padding=(1, 2),
        )
    )

    if not Confirm.ask("\nWould you like to draft a post using this template now?", default=True, console=console):
        return

    import re
    placeholders = re.findall(r'\{([a-zA-Z0-9_]+)\}', prompt_tmpl)
    seen = set()
    unique_placeholders = [x for x in placeholders if not (x in seen or seen.add(x))]

    content = prompt_tmpl
    if unique_placeholders:
        console.print(f"\n[bold {C_HEAD}]Template Wizard — Fill in the blanks:[/bold {C_HEAD}]")
        values = {}
        for placeholder in unique_placeholders:
            label = placeholder.replace("_", " ").capitalize()
            val = Prompt.ask(f"  {label}", console=console).strip()
            while not val:
                console.print(f"  [{C_WARN}]Value cannot be empty.[/{C_WARN}]")
                val = Prompt.ask(f"  {label}", console=console).strip()
            values[placeholder] = val

        # substitute values
        for k, v in values.items():
            content = content.replace("{" + k + "}", v)

    console.print("\n[bold cyan]Preview of your drafted post:[/bold cyan]")
    console.print(Panel(content, border_style=C_SUCC, padding=(1, 2)))

    if not Confirm.ask("Would you like to save this draft?", default=True, console=console):
        console.print("Draft discarded.")
        return

    title = Prompt.ask("Enter a reference title for this draft", default=f"{tmpl.get('name', 'Template')} Draft", console=console).strip()
    if not title:
        title = f"{tmpl.get('name', 'Template')} Draft"

    suggested = _suggest_hashtags(content)
    hashtags = []
    if suggested:
        console.print(f"\n[{C_DIM}]Suggested hashtags:[/{C_DIM}]  "
                      + " ".join(f"[cyan]#{t}[/cyan]" for t in suggested))
        if Confirm.ask("Add these hashtags to the post?", default=True, console=console):
            hashtags = suggested

    posts = _load_posts()
    pid   = _next_id(posts)
    now   = date.today().isoformat()
    post: dict[str, Any] = {
        "id":           pid,
        "title":        title,
        "content":      content,
        "status":       "draft",
        "created_at":   now,
        "published_at": None,
        "source":       "template",
        "hashtags":     hashtags,
        "image_prompts": [],
    }
    posts.append(post)
    _save_posts(posts)

    if store is not None:
        store.record_generation(pid, title, source="template")

    console.print(f"\n[{C_SUCC}]Draft #{pid} successfully saved![/{C_SUCC}]")


# ---------------------------------------------------------------------------
# CLI-args power-user helpers
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="linkcmd.py",
        description="LinkCommand — Rich CLI",
        add_help=True,
    )
    p.add_argument("command", nargs="?", default=None,
                   help="Sub-command: generate | list | view | publish | "
                        "templates | week | stats | settings | write | help | "
                        "improve | variations | image | analyze-profile")
    p.add_argument("--topic", "-t", default="",
                   help="Topic for AI generation")
    p.add_argument("--style", "-s", default="",
                   choices=["tip", "story", "opinion", "question", ""],
                   help="Writing style for AI generation")
    p.add_argument("--id", type=int, default=None,
                   help="Post ID (for view / publish / delete / improve / variations / image)")
    p.add_argument("--image", default="",
                   help="Image path for profile screenshot analysis")
    p.add_argument("--non-interactive", action="store_true",
                   help="Skip prompts; use defaults")
    return p


# ---------------------------------------------------------------------------
# Main interactive menu
# ---------------------------------------------------------------------------

def _interactive_menu(store: Store, nim_client: Optional[NIMClient]) -> None:
    """Main loop — rendered with Rich each iteration."""
    while True:
        console.clear()
        _greet(store, nim_client)

        ai_block = ""
        if nim_client is not None:
            ai_block = (
                f"[bold {C_AI}]🤖 AI Commands[/bold {C_AI}]\n"
                f"  [magenta]1.[/magenta]  🤖  AI Generate\n"
                f"  [magenta]2.[/magenta]  📝  Write Manually\n"
            )
        else:
            ai_block = (
                f"[bold {C_WARN}]✏️  Manual Mode[/bold {C_WARN}]  "
                f"(toggle AI on in Settings to unlock generation)\n"
                f"  [yellow]1.[/yellow]  🤖  AI Generate  [dim](requires AI)[/dim]\n"
                f"  [yellow]2.[/yellow]  📝  Write Manually\n"
            )

        profile_block = (
            f"  10. 🔍  Analyze Profile Screenshot\n"
            if nim_client is not None
            else f"  10. 🔍  Analyze Profile Screenshot  [dim](requires AI)[/dim]\n"
        )

        menu_text = (
            f"{ai_block}"
            f"\n[bold {C_HEAD}]Library & Tools[/bold {C_HEAD}]\n"
            f"  3.  📋  View Drafts\n"
            f"  4.  📅  Weekly Planner\n"
            f"  5.  📊  My Stats\n"
            f"  6.  📋  Templates\n"
            f"{profile_block}"
            f"\n[bold {C_HEAD}]Other[/bold {C_HEAD}]\n"
            f"  9.  ✏️   Help / How to use\n"
            f"\n[bold {C_HEAD}]Settings[/bold {C_HEAD}]\n"
            f"  7.  ⚙️   Settings\n"
            f"  8.  🚪  Exit\n"
        )
        console.print(
            Panel(
                menu_text,
                title="[bold cyan]LinkCommand[/bold cyan]",
                border_style=C_HEAD,
                padding=(1, 3),
            )
        )

        # Dynamic choices list for Prompt.
        choices = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        label = "Choose"
        try:
            raw = Prompt.ask(
                label,
                choices=choices,
                show_choices=False,
                console=console,
            )
        except Exception:
            # Rich may raise on bad terminal; fall back.
            raw = console.input(f"{label} (1-10): ").strip()
        choice = raw.strip()

        console.clear()

        if choice == "1":
            if nim_client is None:
                console.print(
                    f"[{C_WARN}]AI is off.[/{C_WARN}]  Go to Settings to turn it on first."
                )
            else:
                _cmd_generate(store, nim_client)
        elif choice == "2":
            _cmd_manual_write(store)
        elif choice == "3":
            _cmd_view_drafts(store)
        elif choice == "4":
            _cmd_weekly_planner(store)
        elif choice == "5":
            _cmd_stats(store, nim_client)
        elif choice == "6":
            _cmd_templates(store)
        elif choice == "7":
            _cmd_settings(store, nim_client)
            nim_client = _load_nim_client(store)
        elif choice == "8":
            console.print(f"\n[{C_SUCC}]Bye! Happy posting.[/{C_SUCC}]\n")
            break
        elif choice == "9":
            _cmd_help()
        elif choice == "10":
            if nim_client is None:
                console.print(
                    f"[{C_WARN}]AI is off.[/{C_WARN}]  Go to Settings to turn it on first."
                )
            else:
                _cmd_analyze_profile(store, nim_client)
        else:
            console.print(f"[{C_WARN}]Unknown option '{choice}'.[/{C_WARN}]")

        console.print()
        Prompt.ask("Press Enter to continue", console=console, default="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Bootstrap storage.
    store = Store()

    # First-run onboarding.
    if not store.onboarded:
        _run_onboarding(store)

    # NIM client (only if AI is enabled in prefs and config is valid).
    nim_client = _load_nim_client(store)

    # --- parse CLI args (power-user mode) --------------------------------
    parser = _build_arg_parser()
    known, _ = parser.parse_known_args()

    if known.command is None:
        # Interactive menu.
        _interactive_menu(store, nim_client)
        return

    # Non-interactive / sub-command dispatch.
    cmd = known.command.lower()

    if cmd in ("help", "guide"):
        _cmd_help()
        return

    if cmd == "generate":
        _cmd_generate(store, nim_client, topic=known.topic, style=known.style)
        return

    if cmd == "write":
        _cmd_manual_write(store)
        return

    if cmd == "list":
        posts = _load_posts()
        if not posts:
            console.print(f"\n[{C_DIM}]No posts yet.[/{C_DIM}]")
            return
        tbl = Table(
            title="[bold cyan]All Posts[/bold cyan]",
            box=box.ROUNDED,
            expand=True,
            padding=(0, 1),
        )
        tbl.add_column("ID", justify="right", style="bold", width=5)
        tbl.add_column("Title", width=34)
        tbl.add_column("Date", width=12)
        tbl.add_column("Status", justify="center", width=8)
        tbl.add_column("Src", justify="center", width=6)
        for p in sorted(posts, key=lambda x: x.get("created_at", ""), reverse=True):
            st = "✅ pub" if p.get("published_at") else "📝 drft"
            sc = { "ai_generated": "🤖", "ai_variation": "🔄", "manual": "✏️" }.get(
                p.get("source", ""), "?"
            )
            tbl.add_row(str(p.get("id", "?")),
                        Text(p.get("title", "?"), overflow="ellipsis", no_wrap=True),
                        p.get("created_at", "—"), st, sc)
        console.print()
        console.print(tbl)
        return

    if cmd == "view":
        pid = known.id
        if pid is None:
            console.print(f"[{C_ERR}]--id required.[/{C_ERR}]")
            return
        post = next((p for p in _load_posts() if int(p.get("id", -1)) == pid), None)
        if not post:
            console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
            return
        _display_post_panel(post)
        return

    if cmd == "publish":
        pid = known.id
        if pid is None:
            console.print(f"[{C_ERR}]--id required.[/{C_ERR}]")
            return
        post = next((p for p in _load_posts() if int(p.get("id", -1)) == pid), None)
        if not post:
            console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
            return
        _cmd_publish(post, store)
        return

    if cmd == "templates":
        _cmd_templates(store)
        return

    if cmd == "week":
        _cmd_weekly_planner(store, interactive=not known.non_interactive)
        return

    if cmd == "stats":
        _cmd_stats(store, nim_client)
        return

    if cmd == "improve":
        if nim_client is None:
            console.print(f"[{C_ERR}]AI is off.[/{C_ERR}]  Go to Settings or set API Key to enable.")
            return
        pid = known.id
        if pid is None:
            console.print(f"[{C_ERR}]--id required.[/{C_ERR}]")
            return
        post = next((p for p in _load_posts() if int(p.get("id", -1)) == pid), None)
        if not post:
            console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
            return
        _cmd_improve_post(post, store, nim_client)
        return

    if cmd == "variations":
        if nim_client is None:
            console.print(f"[{C_ERR}]AI is off.[/{C_ERR}]  Go to Settings or set API Key to enable.")
            return
        pid = known.id
        if pid is None:
            console.print(f"[{C_ERR}]--id required.[/{C_ERR}]")
            return
        post = next((p for p in _load_posts() if int(p.get("id", -1)) == pid), None)
        if not post:
            console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
            return
        _cmd_generate_variations(post, store, nim_client)
        return

    if cmd == "image":
        if nim_client is None:
            console.print(f"[{C_ERR}]AI is off.[/{C_ERR}]  Go to Settings or set API Key to enable.")
            return
        pid = known.id
        if pid is None:
            console.print(f"[{C_ERR}]--id required.[/{C_ERR}]")
            return
        post = next((p for p in _load_posts() if int(p.get("id", -1)) == pid), None)
        if not post:
            console.print(f"[{C_ERR}]Post #{pid} not found.[/{C_ERR}]")
            return
        _cmd_image_options(post, store, nim_client)
        return

    if cmd in ("analyze-profile", "analyze"):
        if nim_client is None:
            console.print(f"[{C_ERR}]AI is off.[/{C_ERR}]  Go to Settings or set API Key to enable.")
            return
        _cmd_analyze_profile(store, nim_client, default_image_path=known.image or None)
        return

    if cmd in ("settings", "setup"):
        _cmd_settings(store, nim_client)
        return

    console.print(f"[{C_ERR}]Unknown command: {cmd}[/{C_ERR}]\n")
    parser.print_help()


if __name__ == "__main__":
    main()
