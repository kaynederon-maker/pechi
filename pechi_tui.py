#!/usr/bin/env python3
"""pechi: the terminal home of Pechi. Run it from any folder.

pechi                  chat in the current folder
pechi "tidy my notes"  start on a task straight away
pechi --root D:\\work   use another folder
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from random import choice, randint
from typing import ClassVar
from urllib.parse import urlsplit

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from pechi_agent import (
    PECHI_HOME,
    STOPPED,
    Brain,
    Cancelled,
    describe_call,
    run_task,
    split_reply,
    system_prompt,
)
from pechi_sprites import Pose, render
from pechi_themes import THEME_NAMES, THEME_NOTES, THEMES

CONFIG = PECHI_HOME / "config.json"
FRAME_SECONDS = 0.125
STREAM_SECONDS = 0.08  # how often streamed text repaints
FADE_SECONDS = 0.45  # slightly longer than the slowest transition in pechi.tcss
FADE_FRAME_SECONDS = 1 / 30
SLEEP_FRAMES = 720  # a minute and a half of quiet
MAX_HISTORY = 12  # messages carried into the next task
HISTORY_CHARS = 6000  # of each old message, so an old answer cannot fill num_ctx
STEP_ARG_CHARS = 100  # of a tool call shown on its step line
PREVIEW_LINES = 6  # of a tool's output shown under its step line
PREVIEW_CHARS = 500
ASK_POLL_SECONDS = 0.2
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
COMMANDS = (
    "/auto",
    "/boop",
    "/brain",
    "/clear",
    "/help",
    "/pet",
    "/quit",
    "/theme",
    "/treat",
)
HINTS = "click pechi to pet   ctrl+t brain   ctrl+g theme   /auto continue all   pgup/pgdn scroll   esc stop   /help"
# A bare "continue" tells a model nothing, and the 35B answered it by copying its work.
RESUME_NOTE = (
    "Your last task stopped before it finished, and your work so far is above. If this "
    "message means carry on, do not repeat any of it: continue from exactly where it "
    "stopped and finish everything in Final Answer."
)
HELP = """**What Pechi can do**

Read and write files in this folder, run commands and Python, search the web, and read or scrape pages.

**Commands**

- `/brain` or `ctrl+t` switches between 9B and 35B, even in the middle of a task
- `/theme` or `ctrl+g` changes the coat
- `/auto` turns continue all on or off
- `/clear` or `ctrl+l` starts a fresh chat
- `pgup` and `pgdn` scroll the chat
- `esc` stops the current task, and "continue" picks it up again
- `/quit` or `ctrl+q` leaves

Pechi asks before writing files, running anything, or opening a new website. Press `c` on that card, or type `/auto`, to continue all: Pechi stops asking and stays that way, except for high-risk calls, which always ask.

**Pechi loves attention**

Click Pechi to pet: the ears and head get scritches, the nose gets a boop, and the belly gets a rub. Without a mouse, use `/pet` or `ctrl+p`, and `/boop`. `/treat` is always welcome. Keep going and see what happens."""
SAYINGS = {
    "idle": ("wan!", "hff hff", "...?", "ready!", "good day!", "pet me?"),
    "think": ("sniff sniff...", "hmm hmm...", "on it!", "thinking..."),
    "talk": ("wan wan!", "found it!", "here!"),
    "happy": ("yatta!", "all done!", "wan wan wan!"),
    "cry": ("uu...", "oh no...", "sorry..."),
    "sleep": ("zzz...",),
    "love": ("hff hff...", "more pets!", "mmmm..."),
    "boop": ("boop!", "!?"),
    "chew": ("nom nom nom",),
    "zoom": ("ZOOMIES!!",),
    "pout": ("hmph.",),
}
ZONE_MOODS = {
    "ears": "love",
    "head": "love",
    "nose": "boop",
    "belly": "love",
    "treat": "chew",
}
ZONE_WORDS = {
    "ears": ("ear scritches!!", "right there!", "mmm, ears..."),
    "head": ("hff hff...", "good pets!", "more pets!", "mmmm..."),
    "nose": ("boop!", "!?", "achoo!", "hey, my nose!"),
    "belly": ("belly rubs!!", "my leg is kicking!", "wan wan wan!"),
    "treat": ("nom nom nom", "yum!", "more? more?"),
}
MILESTONES = {
    10: "10 pets! we're friends!",
    50: "50 pets! best friends!",
    100: "100 pets!! i love you!",
    500: "500 pets!!! yours forever!",
}
STREAK_SECONDS = 4.0  # a touch within this long of the last one builds the streak
ZOOM_STREAK = 7  # enough attention to set off the zoomies
TIRED_STREAK = 14  # too much attention, even for a very good dog
ZOOM_PATH = ((-2, 0), (0, -2), (2, 0), (0, -1), (-2, 0), (2, -1), (-1, 0), (0, 0))
ZOOM_STEP_SECONDS = 0.14
TOUCH_HEARTS = 2
CELEBRATE_HEARTS = 4
MARKERS = ("Thought:", "Action:", "Final Answer:")
TOOL_WORDS = {
    "list_dir": "peeking in the folder",
    "read_file": "reading a file",
    "write_file": "writing it down",
    "run_shell": "running a command",
    "run_python": "working it out",
    "web_search": "sniffing around the web",
    "web_fetch": "fetching a page",
}
MOOD_SECONDS = {
    "happy": 3.0,
    "cry": 4.0,
    "love": 3.0,
    "boop": 1.6,
    "chew": 2.5,
    "zoom": 2.5,
    "pout": 3.5,
}
HEARTS_TOP = 18  # the first sidebar row below the bubble
SWAP_EXIT_SECONDS = 0.35  # matches the swap card's exit in pechi.tcss
SWAP_MIN_SECONDS = 0.9
MIN_TIMER_SECONDS = 0.01  # Textual divides by the interval, so it cannot be zero


def clean(text: str) -> str:
    """Model and web text can carry terminal control codes, so drop them."""
    return CONTROL_RE.sub("", text)


def site_of(args: dict) -> str | None:
    """The host a web_fetch call names. The model wrote these, so trust nothing."""
    try:
        return urlsplit(args["url"]).hostname
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def brief(args: dict) -> str:
    """A tool call's main argument, always its first, on one line."""
    return " ".join(str(next(iter(args.values()), "")).split())[:STEP_ARG_CHARS]


def preview(text: str) -> str:
    """The top of a tool's output, enough to follow what Pechi found."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:PREVIEW_LINES])[:PREVIEW_CHARS]


def trimmed(text: str) -> str:
    """The newest part of a long message, which is what a follow-up needs most."""
    if len(text) <= HISTORY_CHARS:
        return text
    return f"[start elided]\n{text[-HISTORY_CHARS:]}"


def settled(answer: str) -> str:
    """A streaming answer without a last line that could be a marker still arriving,
    which would otherwise flash on screen until the reply is cut there."""
    head, _, last = answer.rpartition("\n")
    word = last.strip("*_# ")
    unfinished = word and any(marker.startswith(word) for marker in MARKERS)
    return head.rstrip() if unfinished else answer


def load_config() -> dict:
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def save_config(config: dict) -> None:
    CONFIG.parent.mkdir(exist_ok=True)
    temporary = CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps(config), encoding="utf-8")
    temporary.replace(CONFIG)  # atomic, so a crash cannot leave half a file


def pose_for(mood: str, frame: int) -> Pose:
    """The sprite for a mood at a tick of the animation clock."""
    if mood == "think":
        eyes = "look" if frame // 6 % 2 else "open"
        return Pose(eyes=eyes, tail=frame % 3, bob=frame // 3 % 2)
    if mood == "talk":
        return Pose(mouth="open" if frame // 2 % 2 else "smile", tail=frame // 2 % 3)
    if mood == "happy":
        return Pose(eyes="happy", mouth="tongue", tail=frame % 3, bob=frame // 2 % 2)
    if mood == "cry":
        return Pose(eyes="cry", bob=frame // 6 % 2)
    if mood == "sleep":
        return Pose(eyes="blink", bob=frame // 10 % 2)
    if mood == "love":
        return Pose(eyes="love", mouth="tongue", tail=frame % 3, bob=frame // 2 % 2)
    if mood == "boop":
        return Pose(eyes="wide", mouth="wow", tail=frame // 3 % 3)
    if mood == "chew":
        mouth = "open" if frame % 2 else "smile"
        return Pose(eyes="happy", mouth=mouth, tail=frame % 3, bob=frame % 2)
    if mood == "zoom":
        return Pose(eyes="happy", mouth="tongue", tail=frame % 3, bob=frame % 2)
    if mood == "pout":
        return Pose(eyes="cross", mouth="frown", bob=frame // 8 % 2)
    eyes = "blink" if frame % 44 < 2 else "open"
    return Pose(eyes=eyes, tail=(0, 0, 1, 0)[frame // 4 % 4], bob=frame // 10 % 2)


def zone_of(x: int, y: int) -> str:
    """The part of Pechi under a click, in the sprite's 26 by 14 terminal cells."""
    if y <= 4:
        return "ears"
    if y >= 11:
        return "belly"
    if y >= 8 and 8 <= x <= 17:
        return "nose"
    return "head"


def reaction(zone: str, streak: int, asleep: bool) -> tuple[str, str]:
    """The mood and words for a touch, which change as the attention builds up."""
    if asleep:
        return "boop", "!? oh, it's you!"
    if streak >= TIRED_STREAK:
        return "pout", "okay okay, enough!"
    if streak >= ZOOM_STREAK:
        return "zoom", "ZOOMIES!!"
    return ZONE_MOODS[zone], choice(ZONE_WORDS[zone])


class Mascot(Static):
    """Pechi, breathing and blinking on a steady clock, and glad of the attention."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mood = "idle"
        self.frame = 0

    def on_mount(self) -> None:
        self.paint()
        self.set_interval(FRAME_SECONDS, self.tick)

    def tick(self) -> None:
        self.frame += 1
        self.paint()

    def paint(self) -> None:
        self.update(
            render(pose_for(self.mood, self.frame), self.app.current_theme.dark)
        )

    def set_mood(self, mood: str) -> None:
        self.mood, self.frame = mood, 0

    def hop(self) -> None:
        self.styles.offset = (0, -2)
        self.set_timer(0.18, lambda: setattr(self.styles, "offset", (0, 0)))

    def zoom(self) -> None:
        """Tear around the sidebar, ending back where Pechi started."""
        path = iter(ZOOM_PATH)
        self.set_interval(
            ZOOM_STEP_SECONDS,
            lambda: setattr(self.styles, "offset", next(path)),
            repeat=len(ZOOM_PATH),
        )

    def on_click(self, event: Click) -> None:
        if self.id == "mascot":  # the intro has a Pechi too, but no one to react
            self.app.touch(zone_of(event.x, event.y))


class Turn(Vertical):
    """One exchange: the reasoning and tool work of every step, then the answer."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.thought: Static | None = None  # the current step's, once it has any
        self.thought_text = ""

    def compose(self) -> ComposeResult:
        yield Static("pechi", classes="name", markup=False)
        yield Markdown("", classes="answer", open_links=False)

    def next_step(self) -> None:
        """Leave the last step's reasoning on screen and start a new one."""
        self.thought, self.thought_text = None, ""

    async def stream(self, raw: str) -> None:
        thought, answer = split_reply(clean(raw))
        if thought and thought != self.thought_text:
            self.show_thought(thought)
        if settled(answer):
            await self.show_answer(settled(answer))

    def show_thought(self, text: str) -> None:
        self.thought_text = text
        if self.thought is None:
            self.thought = Static(text, classes="thought", markup=False)
            self.mount(self.thought, before=self.query_one(Markdown))
        else:
            self.thought.update(text)

    def drop_echo(self, answer: str) -> None:
        """A reply with no markers streams as a thought, but it is the answer."""
        if self.thought is not None and self.thought_text == clean(answer):
            self.thought.remove()
            self.next_step()

    async def show_answer(self, answer: str) -> None:
        widget = self.query_one(Markdown)
        widget.add_class("on")
        await widget.update(clean(answer))

    def add_step(self, text: str) -> Static:
        step = Static(clean(text), classes="step", markup=False)
        self.mount(step, before=self.query_one(Markdown))
        return step


class Intro(Screen[None]):
    """Pechi bounces in, the name spells itself, and the screen fades away."""

    NAME = "p e c h i"

    def compose(self) -> ComposeResult:
        yield Mascot(id="intro-pup")
        yield Static("", id="intro-name", markup=False)
        yield Static("your little butler", id="intro-tag", markup=False)

    def on_mount(self) -> None:
        self.letters = 0
        self.call_after_refresh(self.query_one("#intro-pup").add_class, "in")
        self.set_timer(0.55, self.spell)
        self.set_timer(2.0, self.leave)

    def spell(self) -> None:
        self.set_interval(0.06, self.next_letter, repeat=len(self.NAME))

    def next_letter(self) -> None:
        self.letters += 1
        self.query_one("#intro-name", Static).update(self.NAME[: self.letters])
        if self.letters == len(self.NAME):
            self.query_one("#intro-tag").add_class("in")

    def leave(self) -> None:
        if self.is_current:
            self.add_class("leaving")
            self.app.redraw_through_fade(self, 0.3)
            self.set_timer(0.3, self.finish)

    def finish(self) -> None:
        if self.is_current:
            self.dismiss(None)

    def on_key(self) -> None:
        self.finish()


class Card(ModalScreen):
    """A soft card that rises from the bottom of the screen."""

    def on_mount(self) -> None:
        self.call_after_refresh(self.query_one(".card").add_class, "shown")
        self.app.redraw_through_fade(self)


class Approval(Card):
    """Pechi asks before doing anything with side effects."""

    ARM_SECONDS = 0.6  # so a keystroke meant for the chat cannot approve anything
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "answer('once')"),
        Binding("a", "answer('always')"),
        Binding("c", "answer('all')"),
        Binding("n,escape", "answer('no')"),
    ]

    def __init__(
        self, name: str, args: dict, flagged: bool, domain: str | None
    ) -> None:
        super().__init__()
        self.tool_name, self.args, self.flagged, self.domain = (
            name,
            args,
            flagged,
            domain,
        )

    def compose(self) -> ComposeResult:
        title = "this one looks risky. read it all." if self.flagged else "may i?"
        keys = "y  yes    n  no"
        if not self.flagged:
            always = f"a  always for {self.domain}    " if self.domain else ""
            keys = f"y  yes    {always}c  continue all    n  no"
        details = describe_call(self.args)
        words = TOOL_WORDS.get(self.tool_name, self.tool_name)
        with Vertical(classes="card"):
            yield Static(f"pechi asks: {title}", classes="card-title", markup=False)
            with VerticalScroll(classes="card-body"):
                yield Static(f"{words}\n{details}", markup=False)
            yield Static(
                f"{len(details)} characters, up/down to scroll\n{keys}",
                classes="card-keys",
                markup=False,
            )

    def on_mount(self) -> None:
        super().on_mount()
        self.opened = time.monotonic()

    def action_answer(self, answer: str) -> None:
        if answer != "no" and time.monotonic() - self.opened < self.ARM_SECONDS:
            return
        if self.flagged and answer in ("always", "all"):  # a risky call stays one-off
            return
        if answer == "always" and not self.domain:
            return
        self.dismiss(answer)


class ThemePicker(Card):
    """Each highlighted theme previews live. Escape puts the old one back."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="card"):
            yield Static("pick a coat for pechi", classes="card-title", markup=False)
            yield OptionList(
                *(
                    Option(f"{name:<10}{THEME_NOTES[name]}", id=name)
                    for name in THEME_NAMES
                ),
                id="themes",
            )
            yield Static(
                "up/down  try on    enter  keep    esc  back",
                classes="card-keys",
                markup=False,
            )

    def on_mount(self) -> None:
        super().on_mount()
        self.original = self.app.theme
        self.query_one(OptionList).highlighted = THEME_NAMES.index(self.original)

    @on(OptionList.OptionHighlighted)
    def preview(self, event: OptionList.OptionHighlighted) -> None:
        self.app.theme = event.option.id

    @on(OptionList.OptionSelected)
    def choose(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.app.theme = self.original
        self.dismiss(None)


class PechiApp(App[None]):
    CSS_PATH = "pechi.tcss"
    TITLE = "pechi"
    ENABLE_COMMAND_PALETTE = False
    HORIZONTAL_BREAKPOINTS: ClassVar[list] = [(0, "-narrow"), (88, "-wide")]
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+t", "toggle_brain", priority=True),
        Binding("ctrl+g", "pick_theme", priority=True),
        Binding("ctrl+l", "clear", priority=True),
        Binding("escape", "stop"),
        Binding("ctrl+q", "quit", priority=True),
        Binding("ctrl+p", "touch('head')"),
        Binding("pageup", "chat_page(-1)"),
        Binding("pagedown", "chat_page(1)"),
    ]

    def __init__(
        self, root: Path, auto: bool = False, first_task: str = "", intro: bool = True
    ) -> None:
        super().__init__()
        self.config = load_config()
        self.workspace, self.first_task = root, first_task
        self.auto = auto or self.config.get("auto") is True
        self.play_intro = intro
        saved_brain = self.config.get("brain")
        self.brain = Brain(index=saved_brain if type(saved_brain) is int else 0)
        self.stop_flag = threading.Event()
        self.busy = False
        self.history: tuple = ()
        self.resuming = False  # the last task stopped and its work is in the history
        self.trusted_sites: set[str] = set()
        saved_pets = self.config.get("pets")
        self.pets = saved_pets if type(saved_pets) is int else 0
        self.streak = 0
        self.last_touch = 0.0
        self.mood_timer = None
        self.quiet_frames = 0
        self.swap_frame = 0
        self.swap_timer = None
        self.swap_open = False
        self.swap_started = 0.0
        for item in THEMES:
            self.register_theme(item)
        saved = self.config.get("theme")
        self.theme = saved if saved in THEME_NAMES else THEME_NAMES[0]

    def compose(self) -> ComposeResult:
        yield Static("", id="top", markup=False)
        with Horizontal(id="main"):
            with Vertical(id="side"):
                yield Mascot(id="mascot")
                yield Static("wan!", id="bubble", markup=False)
            yield VerticalScroll(id="chat")
        yield Input(
            placeholder="tell pechi what to fetch...",
            suggester=SuggestFromList(COMMANDS),
            id="prompt",
        )
        yield Static(HINTS, id="hints", markup=False)
        yield Static("", id="swap", markup=False)

    def on_mount(self) -> None:
        # Held directly: a query only searches the active screen, so a timer firing
        # under an approval card would fail to find these.
        self.mascot = self.query_one("#mascot", Mascot)
        self.bubble = self.query_one("#bubble", Static)
        self.swap = self.query_one("#swap", Static)
        self.top_bar = self.query_one("#top", Static)
        self.side = self.query_one("#side")
        self.chat = self.query_one("#chat")
        self.chat.anchor()  # follows new text until the owner scrolls up to read
        self.theme_changed_signal.subscribe(self, self.theme_changed)
        self.refresh_labels()
        self.say_mood("idle")
        self.query_one("#prompt").focus()
        self.set_interval(FRAME_SECONDS, self.settle)
        self.redraw_through_fade()
        self.greet()
        self.load_brain()
        if self.play_intro and self.animation_level != "none":
            self.push_screen(Intro(), self.start_first_task)
        else:
            self.start_first_task()

    def start_first_task(self, _result: None = None) -> None:
        if self.first_task:
            self.begin(self.first_task)

    def redraw_through_fade(self, widget=None, seconds: float = FADE_SECONDS) -> None:
        """Textual fades a widget but keeps the cached text inside it, so that text
        would stay in its old colour. Redrawing through the fade keeps them in step."""
        targets = [widget] if widget else self.screen_stack

        def redraw() -> None:
            for target in targets:
                for node in target.walk_children(with_self=True):
                    node.refresh()

        repeats = round(seconds / FADE_FRAME_SECONDS)
        self.set_interval(FADE_FRAME_SECONDS, redraw, repeat=repeats)

    def theme_changed(self, _theme) -> None:
        self.redraw_through_fade()

    def refresh_labels(self) -> None:
        model = self.brain.current
        bar = Text(no_wrap=True)
        bar.append("pechi", style="bold")
        bar.append(f"   {model.label} {model.nickname}   {self.workspace}")
        if self.auto:
            bar.append("   continue all on", style="bold")
        self.top_bar.update(bar)

    def save(self) -> None:
        # From the config, not self.auto: the --auto flag lasts one session only.
        save_config(
            {
                "theme": self.theme,
                "brain": self.brain.index,
                "auto": self.config.get("auto") is True,
                "pets": self.pets,
            }
        )

    def set_auto(self, on: bool) -> None:
        self.auto = on
        self.config = {**self.config, "auto": on}
        self.save()
        self.refresh_labels()
        self.say_mood("happy", "i'll keep going!" if on else "i'll ask first!")

    def say_mood(self, mood: str, words: str = "") -> None:
        self.mascot.set_mood(mood)
        bubble = self.bubble
        bubble.update(words or choice(SAYINGS[mood]))
        bubble.styles.opacity = 0.0
        bubble.styles.animate("opacity", 1.0, duration=0.25, easing="out_cubic")
        if self.mood_timer:  # or an old mood's timer would cut this one short
            self.mood_timer.stop()
            self.mood_timer = None
        if mood in MOOD_SECONDS:
            self.mood_timer = self.set_timer(MOOD_SECONDS[mood], self.calm_down)

    def calm_down(self) -> None:
        self.say_mood("think" if self.busy else "idle")

    def action_touch(self, zone: str) -> None:
        self.touch(zone)

    def touch(self, zone: str) -> None:
        """Pechi reacts to a pet, a boop or a treat. It never touches a running task."""
        now = time.monotonic()
        self.streak = self.streak + 1 if now - self.last_touch < STREAK_SECONDS else 1
        self.last_touch, self.quiet_frames = now, 0
        self.pets += 1
        mood, words = reaction(zone, self.streak, self.mascot.mood == "sleep")
        words = MILESTONES.get(self.pets, words)
        self.say_mood(mood, words)
        if mood == "zoom":
            self.mascot.zoom()
            self.float_hearts(TOUCH_HEARTS)
        elif mood != "pout":
            self.mascot.hop()
            self.float_hearts(TOUCH_HEARTS)
        if self.pets in MILESTONES:
            self.save()
        if self.streak == 1 and not self.side.display:
            self.add_note(f"*{words}*")  # no sidebar here, so say it once in the chat

    def settle(self) -> None:
        """After a long quiet spell, Pechi curls up for a nap."""
        mascot = self.mascot
        self.quiet_frames = 0 if mascot.mood != "idle" else self.quiet_frames + 1
        if self.quiet_frames >= SLEEP_FRAMES:
            self.quiet_frames = 0
            self.say_mood("sleep")

    def greet(self) -> None:
        self.add_note(
            f"wan! i'm pechi. i'm in **{self.workspace.name or self.workspace}**. what shall we do?"
        )

    def add_note(self, text: str) -> None:
        turn = Turn(classes="row")
        self.add_row(turn)
        self.call_after_refresh(self.finish_note, turn, text)

    async def finish_note(self, turn: Turn, text: str) -> None:
        await turn.show_answer(text)

    def add_row(self, row: Static | Vertical | Horizontal) -> None:
        chat = self.chat
        chat.mount(row)
        self.call_after_refresh(row.add_class, "shown")
        self.redraw_through_fade(row)
        self.call_after_refresh(chat.scroll_end, animate=True, duration=0.25)

    def begin(self, task: str) -> None:
        you = Horizontal(
            Static(clean(task), classes="you", markup=False), classes="row you-row"
        )
        self.add_row(you)
        turn = Turn(classes="row")
        self.add_row(turn)
        self.busy = True
        self.stop_flag.clear()
        self.say_mood("think")
        self.run_turn(task, turn)

    @on(Input.Submitted, "#prompt")
    def submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.run_command(text)
        elif self.busy:
            self.say_mood("think", "one thing at a time!")
        else:
            self.begin(text)

    def run_command(self, text: str) -> None:
        name = text.split()[0].lower()
        if name == "/auto":
            self.set_auto(not self.auto)
        elif name in ("/pet", "/boop", "/treat"):
            self.touch({"/pet": "head", "/boop": "nose", "/treat": "treat"}[name])
        elif name == "/brain":
            self.action_toggle_brain()
        elif name == "/theme":
            self.action_pick_theme()
        elif name == "/clear":
            self.action_clear()
        elif name == "/help":
            self.add_note(HELP)
        elif name in ("/quit", "/exit"):
            self.exit()
        else:
            self.say_mood("cry", f"i don't know {name}")

    def action_toggle_brain(self) -> None:
        model = self.brain.toggle()
        self.save()
        self.refresh_labels()
        self.show_swap(model.label, model.nickname)
        self.load_brain()

    def show_swap(self, label: str, nickname: str) -> None:
        self.swap_title = f"fetching {label} {nickname}"
        self.swap_frame = 0
        self.paint_swap()
        self.swap_open = True
        self.swap_started = time.monotonic()
        self.swap.add_class("on")  # laid out first, so that the slide has a start
        self.call_after_refresh(self.slide_swap_in)
        if self.swap_timer:
            self.swap_timer.stop()
        self.swap_timer = self.set_interval(0.09, self.paint_swap)

    def paint_swap(self) -> None:
        """A tiny pup trots back and forth along a dotted path."""
        self.swap_frame += 1
        width = 26
        spot = abs(self.swap_frame % (width * 2 - 2) - (width - 1))
        face = "(•ᴥ•)" if self.swap_frame % 4 < 2 else "(^ᴥ^)"
        track = "·" * spot + face + "·" * (width - spot - 1)
        self.swap.update(f"{self.swap_title}\n{track}")

    def slide_swap_in(self) -> None:
        if self.swap_open:  # a fast swap can finish before this frame arrives
            self.swap.add_class("shown")

    def hide_swap(self) -> None:
        """Stay up long enough to read, even when the model was already loaded."""
        shown_for = time.monotonic() - self.swap_started
        self.set_timer(
            max(MIN_TIMER_SECONDS, SWAP_MIN_SECONDS - shown_for), self.close_swap
        )

    def close_swap(self) -> None:
        self.swap_open = False
        self.swap.remove_class("shown")
        if self.swap_timer:
            self.swap_timer.stop()
        self.set_timer(SWAP_EXIT_SECONDS, self.finish_hiding_swap)

    def finish_hiding_swap(self) -> None:
        """At 0% opacity it would still paint blank cells over the top bar."""
        if not self.swap.has_class("shown"):
            self.swap.remove_class("on")

    @work(thread=True, group="brain")
    def load_brain(self) -> None:
        """Swap models off the UI thread. If Ollama is away, the next task reports it."""
        try:
            self.brain.ready()
        except (urllib.error.URLError, OSError, ValueError):
            self.call_from_thread(self.say_mood, "cry", "is ollama awake?")
        self.call_from_thread(self.hide_swap)

    @on(Markdown.LinkClicked)
    def link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Only web links open. On Windows any other scheme, such as file:, would
        launch whatever it points at."""
        if urlsplit(event.href).scheme in ("http", "https"):
            self.open_url(event.href)
        else:
            self.say_mood("cry", "i won't open that")

    def action_pick_theme(self) -> None:
        if not isinstance(self.screen, (Intro, Card)):
            self.push_screen(ThemePicker(), self.theme_chosen)

    def theme_chosen(self, name: str | None) -> None:
        self.save()
        if name:
            self.say_mood("happy", "new coat!")

    def action_clear(self) -> None:
        if self.busy:  # the running task is still writing into this chat
            self.say_mood("think", "press esc first!")
            return
        self.history = ()
        self.resuming = False
        self.chat.remove_children()
        self.greet()

    def action_stop(self) -> None:
        if self.busy:
            self.stop_flag.set()
            self.say_mood("think", "stopping...")

    def action_chat_page(self, direction: int) -> None:
        self.chat.release_anchor()
        if direction < 0:
            self.chat.scroll_page_up()
        else:
            self.chat.scroll_page_down()

    def on_unmount(self) -> None:
        self.stop_flag.set()  # releases a worker waiting on an approval
        self.save()  # keeps the pets, which are not written on every single pet

    def ask(self, name: str, args: dict, flagged: bool) -> bool:
        """Runs on the worker thread and waits for the owner's answer."""
        if self.auto and not flagged:  # continue all can start partway through a task
            return True
        domain = site_of(args) if name == "web_fetch" else None
        if domain in self.trusted_sites and not flagged:
            return True
        answer: Future = Future()
        self.call_from_thread(
            self.push_screen, Approval(name, args, flagged, domain), answer.set_result
        )
        while not self.stop_flag.is_set():
            try:
                reply = answer.result(timeout=ASK_POLL_SECONDS)
            except FutureTimeout:
                continue
            if reply == "always":
                self.trusted_sites.add(domain)
            elif reply == "all":
                self.call_from_thread(self.set_auto, True)
            return reply in ("once", "always", "all")
        return False

    def show_event(self, turn: Turn, kind: str, data: dict) -> None:
        if kind == "step":
            turn.next_step()
            self.say_mood("think")
        elif kind == "action":
            word = TOOL_WORDS.get(data["tool"], data["tool"])
            self.current_step = turn.add_step(f"· {word}: {brief(data['args'])}")
            self.say_mood("think", TOOL_WORDS.get(data["tool"], "working..."))
        elif kind == "result":
            word = TOOL_WORDS.get(data["tool"], data["tool"])
            mark = "♡" if data["ok"] else "×"
            found = preview(data["text"])
            self.current_step.update(
                clean(f"{mark} {word}: {brief(data['args'])}\n{found}")
            )
            self.current_step.add_class("done" if data["ok"] else "failed")

    def remember(self, task: str, reply: str) -> None:
        self.history = (
            *self.history,
            {"role": "user", "content": task},
            {"role": "assistant", "content": trimmed(reply)},
        )[-MAX_HISTORY:]

    def keep_work(self, task: str, replies: list[str]) -> None:
        """Hold on to a stopped task's working, so "continue" has something to continue."""
        work = "\n\n".join(filter(None, replies))
        if work:
            self.remember(task, work)
            self.resuming = True

    async def finish_turn(
        self, turn: Turn, task: str, answer: str, replies: list[str]
    ) -> None:
        turn.drop_echo(answer)
        await turn.show_answer(answer)
        stopped = answer.startswith(STOPPED)
        if stopped:
            self.keep_work(task, replies)
        else:
            self.remember(task, f"Final Answer: {answer}")
        self.busy = False
        self.say_mood("cry" if stopped else "happy")
        if not stopped:
            self.celebrate()

    def celebrate(self) -> None:
        """Pechi hops, and little hearts float up and fade."""
        self.mascot.hop()
        self.float_hearts(CELEBRATE_HEARTS)

    def float_hearts(self, count: int) -> None:
        side = self.side
        if not side.display:
            return
        # They rise over the plain sidebar: a terminal cell cannot be see-through, so
        # a heart drifting across the sprite would drag a pale box with it.
        bottom = side.size.height - 3
        if bottom <= HEARTS_TOP:
            return
        for _ in range(count):
            x, y = randint(3, 26), bottom - randint(0, 2)
            heart = Static("♡", classes="heart", markup=False)
            heart.styles.offset = (x, y)
            side.mount(heart)
            self.set_timer(0.05, lambda h=heart, x=x, y=y: self.float_heart(h, x, y))
            self.set_timer(1.1, heart.remove)

    def float_heart(self, heart: Static, x: int, y: int) -> None:
        heart.styles.offset = (x, max(y - 5, HEARTS_TOP))
        heart.styles.opacity = 0.0
        self.redraw_through_fade(heart, 0.95)

    def turn_failed(self, turn: Turn, mood: str, words: str, message: str) -> None:
        self.busy = False
        self.say_mood(mood, words)
        turn.mount(Static(message, classes="note", markup=False))

    @work(thread=True, exclusive=True, group="turn")
    def run_turn(self, task: str, turn: Turn) -> None:
        last_paint = 0.0
        replies: list[str] = []  # kept for the next turn in case this one stops
        partial = ""

        def on_token(text: str) -> None:
            nonlocal last_paint, partial
            partial = text
            now = time.monotonic()
            if now - last_paint >= STREAM_SECONDS:
                last_paint = now
                self.call_from_thread(turn.stream, text)

        def llm(messages: list[dict]) -> str:
            nonlocal partial
            reply = self.brain.chat(messages, on_token, self.stop_flag)
            partial = ""
            replies.append(reply)
            self.call_from_thread(turn.stream, reply)  # the throttle may skip the end
            return reply

        def on_event(kind: str, **data) -> None:
            self.call_from_thread(self.show_event, turn, kind, data)

        def failed(mood: str, words: str, message: str) -> None:
            self.keep_work(task, [*replies, partial])
            self.call_from_thread(self.turn_failed, turn, mood, words, message)

        request = f"{task}\n\n{RESUME_NOTE}" if self.resuming else task
        self.resuming = False
        try:
            answer = run_task(
                request,
                llm,
                system_prompt(self.workspace),
                self.workspace,
                history=self.history,
                ask=self.ask,  # reads self.auto live, so /auto works mid-task
                on_event=on_event,
                cancelled=self.stop_flag,
            )
        except Cancelled:
            failed("idle", "okay, stopped", 'stopped. say "continue" to pick it up.')
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            failed("cry", "uu...", friendly_error(err.code, detail))
        except Exception as err:  # noqa: BLE001
            # The worker boundary: anything uncaught here would close the whole app.
            failed("cry", "uu...", friendly_error(0, str(err)))
        else:
            self.call_from_thread(self.finish_turn, turn, task, answer, replies)


def friendly_error(code: int, detail: str) -> str:
    if "out of memory" in detail.lower():
        return "This brain did not fit in the graphics card. Close other GPU apps, or lower num_gpu in pechi_agent.py."
    if code == 404:
        return f"Ollama cannot find this model. Check `ollama list`.\n{detail}"
    if code == 0 and ("refused" in detail.lower() or "urlopen" in detail.lower()):
        return "Ollama seems to be asleep. Start it, then ask again."
    return f"Something went wrong: {clean(detail)}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pechi", description="Pechi, your little AI butler"
    )
    parser.add_argument("task", nargs="*", help="a task to start on straight away")
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="folder Pechi works in"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="continue all for this session: no approval prompts except for high-risk calls",
    )
    parser.add_argument(
        "--no-intro", action="store_true", help="skip the opening animation"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a folder")
    PechiApp(root, args.auto, " ".join(args.task), not args.no_intro).run()
    # A worker may still be loading a model or running a tool, and Python would wait
    # for it and leave the terminal hanging. Nothing is left to save at this point.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
