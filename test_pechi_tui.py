"""The interface, web tools and model swapping. Run with: pytest test_pechi_tui.py

No model, GPU or network is used: Ollama is a local fake, and the web is mocked.
"""

import asyncio
import contextlib
import http.server
import json
import socket
import threading
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
from textual.widgets import Input, Markdown, Static

import pechi_agent as agent
import pechi_sprites as sprites
import pechi_themes as themes
import pechi_tui as tui
import pechi_web as web


class FakeOllama(http.server.BaseHTTPRequestHandler):
    state: ClassVar[dict] = {}

    def log_message(self, *args) -> None:
        pass

    def send_json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.send_json({"models": [{"name": n} for n in self.state["running"]]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":
            self.state["calls"].append((body["model"], body.get("keep_alive")))
            if body.get("keep_alive") == 0:
                self.state["running"].remove(body["model"])
            else:
                self.state["running"].append(body["model"])
            return self.send_json({"done": True})
        self.state["chats"].append(body)
        reply = self.state["replies"].pop(0)
        reason = "stop"
        if isinstance(reply, tuple):  # (text, reason), and "hold" stalls mid-reply
            reply, reason = reply
        for word in body["options"].get("stop", []):  # as the real server does
            reply = reply.partition(word)[0]
        self.send_response(200)
        self.end_headers()
        with contextlib.suppress(ConnectionError):  # a stopped reply hangs up early
            for start in range(0, len(reply), 6):
                piece = reply[start : start + 6]
                chunk = {"message": {"content": piece}, "done": False}
                self.wfile.write(json.dumps(chunk).encode() + b"\n")
                self.wfile.flush()
            if reason == "hold":
                self.state["release"].wait(5)
            end = {"message": {"content": ""}, "done": True, "done_reason": reason}
            self.wfile.write(json.dumps(end).encode() + b"\n")


@pytest.fixture(autouse=True)
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(tui, "CONFIG", tmp_path / "home" / "config.json")
    monkeypatch.setattr(agent, "AUDIT_LOG", tmp_path / "audit.jsonl")


@pytest.fixture
def ollama(monkeypatch):
    FakeOllama.state = {
        "running": [],
        "calls": [],
        "chats": [],
        "replies": [],
        "release": threading.Event(),
    }
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(agent, "OLLAMA", f"http://127.0.0.1:{server.server_port}")
    yield FakeOllama.state
    server.shutdown()


def drive(app: tui.PechiApp, script, size: tuple[int, int] = (120, 30)) -> None:
    async def runner() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await script(pilot)

    asyncio.run(runner())


async def send(pilot, text: str) -> None:
    prompt = pilot.app.query_one("#prompt", Input)
    prompt.value = text
    await pilot.press("enter")
    await pilot.pause(0.05)


async def until_idle(pilot) -> None:
    for _ in range(1000):
        await pilot.pause(0.02)
        if not pilot.app.busy:
            break
    await pilot.pause(0.05)


async def until_asked(pilot) -> None:
    for _ in range(100):
        await pilot.pause(0.02)
        if isinstance(pilot.app.screen, tui.Approval):
            return


def answers(app: tui.PechiApp) -> list[str]:
    return [m.source for m in app.query(Markdown) if m.has_class("on")]


def make_app(tmp_path: Path, **options) -> tui.PechiApp:
    return tui.PechiApp(tmp_path, intro=False, **options)


def test_a_question_streams_to_an_answer(tmp_path, ollama):
    ollama["replies"] = ["Thought: easy\nFinal Answer: **Hello** there"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "say hi")
        await until_idle(pilot)
        assert answers(app)[-1] == "**Hello** there"
        assert app.query_one("#mascot").mood == "happy"
        assert len(app.history) == 2

    drive(app, script)


def on_screen(app: tui.PechiApp, selector: str) -> list[str]:
    return [str(w.render()) for w in app.query(selector) if w.display]


def read_step(path: str = "a.txt", note: str = "look") -> str:
    return f'Thought: {note}\nAction: read_file\nAction Input: {{"path": "{path}"}}'


def write_step(path: str) -> str:
    return f'Thought: save\nAction: write_file\nAction Input: {{"path": "{path}", "content": "hi"}}'


def test_every_steps_working_stays_on_screen(tmp_path, ollama):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    ollama["replies"] = [
        read_step("a.txt", "first look"),
        read_step("b.txt", "second look"),
        "Thought: all read\nFinal Answer: done",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "read them")
        await until_idle(pilot)
        assert on_screen(app, ".thought") == ["first look", "second look", "all read"]
        assert answers(app)[-1] == "done"

    drive(app, script)


def test_working_stays_in_order_with_the_tool_steps(tmp_path, ollama):
    (tmp_path / "a.txt").write_text("the file text")
    ollama["replies"] = [read_step(), "Thought: done\nFinal Answer: ok"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "read it")
        await until_idle(pilot)
        turn = list(app.query(tui.Turn))[-1]
        kinds = [c.classes & {"thought", "step", "answer"} for c in turn.children][1:]
        assert kinds == [{"thought"}, {"step"}, {"thought"}, {"answer"}]

    drive(app, script)


def test_tool_steps_show_what_they_did_and_found(tmp_path, ollama):
    (tmp_path / "a.txt").write_text("the file text")
    ollama["replies"] = [read_step(), "Thought: done\nFinal Answer: ok"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "read it")
        await until_idle(pilot)
        step = on_screen(app, ".step")[0]
        assert "a.txt" in step and "the file text" in step

    drive(app, script)


def test_a_reply_without_markers_shows_once_as_the_answer(tmp_path, ollama):
    ollama["replies"] = ["Step 1: mix.\nStep 2: bake."]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "cake?")
        await until_idle(pilot)
        assert answers(app)[-1] == "Step 1: mix.\nStep 2: bake."
        assert on_screen(app, ".thought") == []
        assert app.query_one("#mascot").mood == "happy"

    drive(app, script)


def test_a_long_answer_keeps_the_newest_text_in_view(tmp_path, ollama):
    steps = "\n\n".join(f"Step {n}: do the thing." for n in range(1, 61))
    ollama["replies"] = [f"Thought: ok\nFinal Answer: {steps}"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "long")
        await until_idle(pilot)
        await pilot.pause(0.3)
        assert answers(app)[-1].endswith("Step 60: do the thing.")
        assert app.chat.max_scroll_y > 0
        assert app.chat.scroll_y >= app.chat.max_scroll_y

    drive(app, script)


def test_page_keys_scroll_back_and_release_the_view(tmp_path, ollama):
    steps = "\n\n".join(f"Step {n}" for n in range(1, 61))
    ollama["replies"] = [f"Thought: ok\nFinal Answer: {steps}"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "long")
        await until_idle(pilot)
        await pilot.pause(0.3)
        bottom = app.chat.scroll_y
        await pilot.press("pageup")
        await pilot.pause(0.3)
        assert app.chat.scroll_y < bottom

    drive(app, script)


def test_a_reply_cut_off_by_length_continues_by_itself(tmp_path, ollama):
    ollama["replies"] = [
        ("Thought: ok\nFinal Answer: Step 1: mix. Step 2: ", "length"),
        ("bake. Step 3: eat.", "stop"),
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "cake")
        await until_idle(pilot)
        assert answers(app)[-1] == "Step 1: mix. Step 2: bake. Step 3: eat."
        resumed = ollama["chats"][1]["messages"]
        assert resumed[-2]["role"] == "assistant"
        assert resumed[-2]["content"].endswith("Step 2: ")
        assert resumed[-1]["role"] == "user"

    drive(app, script)


def test_a_huge_cut_reply_sends_back_only_its_tail(ollama):
    ollama["replies"] = [("x" * 50_000, "length"), ("y", "stop")]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert reply == "x" * 50_000 + "y"
    sent = ollama["chats"][1]["messages"][-2]["content"]
    assert len(sent) == agent.CONTINUE_TAIL_CHARS


def test_continuing_cannot_run_for_ever(ollama):
    ollama["replies"] = [("x", "length")] * (agent.MAX_CONTINUES + 5)
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert len(ollama["chats"]) == agent.MAX_CONTINUES + 1
    assert reply == "x" * (agent.MAX_CONTINUES + 1)


def test_a_reply_stops_being_read_once_its_tool_call_is_complete(ollama):
    ollama["replies"] = [read_step() + "\nThought: x\nFinal Answer: invented"]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert reply == read_step()


def test_a_reply_stops_being_read_once_it_has_answered(ollama):
    ollama["replies"] = ["Thought: a\nFinal Answer: done\nThought: b\nFinal Answer: c"]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert reply == "Thought: a\nFinal Answer: done"


def test_an_answer_imagined_before_the_tool_ran_never_reaches_the_screen(
    tmp_path, ollama
):
    (tmp_path / "a.txt").write_text("x")
    invented = "Thought: read it\nFinal Answer: INVENTED " + "word " * 60
    ollama["replies"] = [
        f"{read_step()}\n{invented}",
        "Thought: done\nFinal Answer: the real answer",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "read it")
        shown = []
        for _ in range(300):
            await pilot.pause(0.01)
            shown.extend(m.source for m in app.query(Markdown) if m.has_class("on"))
            if not app.busy:
                break
        await until_idle(pilot)
        assert "INVENTED" not in " ".join(shown)
        assert answers(app)[-1] == "the real answer"
        assert len(ollama["chats"]) == 2

    drive(app, script)


def test_the_word_observation_cannot_cut_an_answer_short(ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: Observation: it rises. Done."]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert reply.endswith("it rises. Done.")


def test_a_model_inventing_a_tool_result_is_still_cut_off(ollama):
    ollama["replies"] = [read_step() + "\nObservation: invented\nFinal Answer: lies"]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert "invented" not in reply and "lies" not in reply
    assert agent.parse_reply(reply)[0] == "action"


def test_the_word_observation_in_a_thought_cannot_lose_the_action(ollama):
    step = 'Action: read_file\nAction Input: {"path": "a.txt"}'
    ollama["replies"] = [f"Thought: my Observation: it needs reading\n{step}"]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert agent.parse_reply(reply) == ("action", "read_file", {"path": "a.txt"})


def test_the_word_observation_in_a_file_cannot_cut_the_call(ollama):
    args = {"path": "a.txt", "content": "Observation: it rises"}
    ollama["replies"] = [f"Action: write_file\nAction Input: {json.dumps(args)}"]
    reply = agent.Brain().chat([{"role": "user", "content": "go"}])
    assert agent.parse_reply(reply) == ("action", "write_file", args)


def test_turning_continue_all_off_applies_to_the_next_step(tmp_path, ollama):
    ollama["replies"] = [
        ("Thought: wait for it", "hold"),
        write_step("a.txt"),
        "Thought: ok\nFinal Answer: left it",
    ]
    app = make_app(tmp_path, auto=True)

    async def script(pilot):
        await send(pilot, "write it")
        for _ in range(200):
            await pilot.pause(0.02)
            if ollama["chats"]:
                break
        await send(pilot, "/auto")
        ollama["release"].set()
        await until_asked(pilot)
        assert isinstance(app.screen, tui.Approval)
        await pilot.press("n")
        await until_idle(pilot)
        assert not (tmp_path / "a.txt").exists()

    drive(app, script)


def test_an_answer_starting_with_stopped_is_still_an_answer(tmp_path, ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: Stopped by the shop, all good."]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "errands?")
        await until_idle(pilot)
        assert app.history[-1]["content"].startswith("Final Answer:")
        assert app.query_one("#mascot").mood == "happy"

    drive(app, script)


def test_continue_all_stops_the_asking_and_is_remembered(tmp_path, ollama):
    ollama["replies"] = [
        write_step("a.txt"),
        write_step("b.txt"),
        "Thought: done\nFinal Answer: saved",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "write both")
        await until_asked(pilot)
        await pilot.pause(tui.Approval.ARM_SECONDS)
        await pilot.press("c")
        await until_idle(pilot)
        assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
        assert not isinstance(app.screen, tui.Approval)
        assert "continue all" in str(app.query_one("#top", Static).render())

    drive(app, script)
    assert make_app(tmp_path).auto is True


def test_continue_all_cannot_be_chosen_on_a_risky_call(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        app.push_screen(tui.Approval("run_shell", {"command": "rm -rf /"}, True, None))
        await pilot.pause(tui.Approval.ARM_SECONDS + 0.1)
        await pilot.press("c")
        await pilot.pause(0.05)
        assert isinstance(app.screen, tui.Approval)
        assert app.auto is False

    drive(app, script)


def test_risky_calls_still_ask_with_continue_all_on(tmp_path, ollama):
    command = "rm -rf ."
    ollama["replies"] = [
        f'Thought: x\nAction: run_shell\nAction Input: {{"command": "{command}"}}',
        "Thought: ok\nFinal Answer: left it",
    ]
    app = make_app(tmp_path, auto=True)

    async def script(pilot):
        await send(pilot, "wipe")
        await until_asked(pilot)
        assert isinstance(app.screen, tui.Approval)
        await pilot.press("n")
        await until_idle(pilot)

    drive(app, script)


def test_the_auto_command_turns_continue_all_on_and_off(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "/auto")
        assert app.auto is True
        await send(pilot, "/auto")
        assert app.auto is False

    drive(app, script)
    assert make_app(tmp_path).auto is False


def test_the_auto_flag_lasts_one_session_only(tmp_path, ollama):
    app = make_app(tmp_path, auto=True)

    async def script(pilot):
        await pilot.press("ctrl+t")
        await app.workers.wait_for_complete()

    drive(app, script)
    assert make_app(tmp_path).auto is False


def test_stopping_keeps_the_work_so_continue_can_pick_it_up(tmp_path, ollama):
    (tmp_path / "a.txt").write_text("x")
    ollama["replies"] = [
        read_step("a.txt", "step one of the plan"),
        ("Thought: halfway through step two", "hold"),
        "Thought: resumed\nFinal Answer: finished",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "big job")
        for _ in range(200):
            await pilot.pause(0.02)
            if len(ollama["chats"]) == 2:
                break
        await pilot.pause(0.3)
        await pilot.press("escape")
        ollama["release"].set()
        await until_idle(pilot)
        kept = app.history[-1]["content"]
        assert "step one of the plan" in kept and "halfway through step two" in kept
        await send(pilot, "continue")
        await until_idle(pilot)
        sent = ollama["chats"][-1]["messages"]
        carried = " ".join(m["content"] for m in sent)
        assert "step one of the plan" in carried and "big job" in carried
        assert sent[-1]["content"] == f"continue\n\n{tui.RESUME_NOTE}"
        assert "not finished" not in carried.lower()  # a label a model will parrot
        assert app.history[-2]["content"] == "continue"

    drive(app, script)


def test_the_resume_note_goes_out_once_and_only_after_a_stop(tmp_path, ollama):
    ollama["replies"] = [
        "Thought: x\nFinal Answer: one",
        ("Thought: the plan", "hold"),
        "Thought: x\nFinal Answer: two",
        "Thought: x\nFinal Answer: three",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "first")
        await until_idle(pilot)
        await send(pilot, "second")
        for _ in range(200):
            await pilot.pause(0.02)
            if len(ollama["chats"]) == 2:
                break
        await pilot.pause(0.3)
        await pilot.press("escape")
        ollama["release"].set()
        await until_idle(pilot)
        await send(pilot, "continue")
        await until_idle(pilot)
        await send(pilot, "fourth")
        await until_idle(pilot)
        last = [chat["messages"][-1]["content"] for chat in ollama["chats"]]
        assert [tui.RESUME_NOTE in text for text in last] == [False, False, True, False]

    drive(app, script)


def test_clearing_the_chat_forgets_a_stopped_task(tmp_path, ollama):
    ollama["replies"] = ["Thought: the plan"] * agent.MAX_FAILURES
    ollama["replies"] = [*ollama["replies"], "Thought: x\nFinal Answer: ok"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        await pilot.press("ctrl+l")
        await send(pilot, "new")
        await until_idle(pilot)
        assert ollama["chats"][-1]["messages"][-1]["content"] == "new"

    drive(app, script)


def test_a_breaker_stop_keeps_the_work_too(tmp_path, ollama):
    ollama["replies"] = ["Thought: the plan so far"] * agent.MAX_FAILURES
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        assert "the plan so far" in app.history[-1]["content"]

    drive(app, script)


def test_old_answers_are_trimmed_to_their_newest_part(tmp_path, ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: " + "a" * 20_000 + "THE END"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        kept = app.history[-1]["content"]
        assert kept.endswith("THE END")
        assert len(kept) < tui.HISTORY_CHARS + 100

    drive(app, script)


def test_a_tool_step_runs_after_approval(tmp_path, ollama):
    ollama["replies"] = [
        'Thought: note it\nAction: write_file\nAction Input: {"path": "a.txt", "content": "hi"}',
        "Thought: done\nFinal Answer: saved",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "write it")
        await until_asked(pilot)
        card = app.screen
        assert isinstance(card, tui.Approval)
        card.ARM_SECONDS = 60  # still arming: a stray keystroke must not approve
        await pilot.press("y")
        await pilot.pause(0.05)
        assert app.screen is card
        card.ARM_SECONDS = 0
        await pilot.press("y")
        await until_idle(pilot)
        assert (tmp_path / "a.txt").read_text() == "hi"
        assert answers(app)[-1] == "saved"

    drive(app, script)


def test_denying_stops_the_write(tmp_path, ollama):
    ollama["replies"] = [
        'Thought: x\nAction: write_file\nAction Input: {"path": "a.txt", "content": "hi"}',
        "Thought: ok\nFinal Answer: left it alone",
    ]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "write it")
        await until_asked(pilot)
        await pilot.press("n")
        await until_idle(pilot)
        assert not (tmp_path / "a.txt").exists()

    drive(app, script)


@pytest.mark.parametrize(
    "args, site",
    [
        ({"url": "https://Safe.Example/a"}, "safe.example"),
        ({"url": 5}, None),
        ({"url": None}, None),
        ({}, None),
        ({"url": "http://["}, None),
    ],
)
def test_site_of_survives_whatever_the_model_sends(args, site):
    assert tui.site_of(args) == site


@pytest.mark.parametrize(
    "href, opened",
    [
        ("https://safe.example/", True),
        ("http://safe.example/", True),
        ("file:///C:/Windows/System32/calc.exe", False),
        ("ms-settings:", False),
        ("javascript:alert(1)", False),
    ],
)
def test_only_web_links_open(tmp_path, ollama, monkeypatch, href, opened):
    app = make_app(tmp_path)
    seen = []
    monkeypatch.setattr(app, "open_url", seen.append)

    async def script(pilot):
        await pilot.pause(0.1)
        first = app.query(Markdown).first()
        first.post_message(Markdown.LinkClicked(first, href))
        await pilot.pause(0.1)
        assert bool(seen) is opened

    drive(app, script)


def test_a_surprise_inside_the_worker_is_reported_not_fatal(tmp_path, ollama):
    app = make_app(tmp_path)

    def explode(*args, **kwargs):
        raise TypeError("surprise")

    app.brain.chat = explode

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        assert app.busy is False
        assert "surprise" in str(app.query_one(".note", Static).render())
        assert app.history == ()  # nothing was done, so nothing to continue

    drive(app, script)


def test_clear_waits_for_the_running_task(tmp_path, ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: done"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        app.busy = True  # as if the task were still running
        await pilot.press("ctrl+l")
        await pilot.pause(0.05)
        assert app.query(".you")
        app.busy = False
        await pilot.press("ctrl+l")
        await pilot.pause(0.05)
        assert not app.query(".you")

    drive(app, script)


def test_the_theme_picker_cannot_stack(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.press("ctrl+g")
        await pilot.pause(0.1)
        await pilot.press("ctrl+g")
        await pilot.pause(0.1)
        assert len(app.screen_stack) == 2

    drive(app, script)


def test_a_command_line_task_waits_for_the_intro(tmp_path, ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: done"]
    app = tui.PechiApp(tmp_path, first_task="hi", intro=True)

    async def script(pilot):
        await pilot.pause(0.1)
        assert not ollama["chats"] and isinstance(app.screen, tui.Intro)
        await pilot.press("x")
        await pilot.pause(0.2)
        await until_idle(pilot)
        assert answers(app)[-1] == "done"

    drive(app, script)


def test_a_trusted_site_is_not_asked_about_again(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        app.trusted_sites.add("safe.example")
        assert app.ask("web_fetch", {"url": "https://safe.example/a"}, False)

    drive(app, script)


def test_model_text_is_never_parsed_as_markup(tmp_path, ollama):
    ollama["replies"] = ["[bold red]Thought: [/nope]\nFinal Answer: [link=x]done"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "[/oops] [red]hi")
        await until_idle(pilot)
        assert "[/oops] [red]hi" in str(app.query_one(".you", Static).render())

    drive(app, script)


def test_a_closed_swap_card_takes_no_space(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        assert app.swap.display is False
        await pilot.press("ctrl+t")
        await pilot.pause(0.02)
        assert app.swap.display is True
        await app.workers.wait_for_complete()
        await pilot.pause(tui.SWAP_MIN_SECONDS + tui.SWAP_EXIT_SECONDS + 0.2)
        assert app.swap.display is False

    drive(app, script)


def test_a_timer_under_an_open_card_does_not_crash(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.press("ctrl+g")
        await pilot.pause(0.3)  # several ticks of the mascot and nap timers
        app.settle()
        app.say_mood("happy")
        assert isinstance(app.screen, tui.ThemePicker)

    drive(app, script)


@pytest.mark.parametrize(
    "x, y, zone",
    [
        (6, 2, "ears"),
        (19, 3, "ears"),
        (4, 6, "head"),
        (12, 6, "head"),
        (2, 9, "head"),
        (12, 9, "nose"),
        (12, 12, "belly"),
    ],
)
def test_each_part_of_pechi_is_its_own_zone(x, y, zone):
    assert tui.zone_of(x, y) == zone


def test_clicking_pechi_gets_a_reaction_and_keeps_the_prompt_ready(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        await pilot.click("#mascot", offset=(12, 9))
        await pilot.pause(0.05)
        assert app.mascot.mood == "boop"
        await pilot.click("#mascot", offset=(4, 6))
        await pilot.pause(0.05)
        assert app.mascot.mood == "love"
        assert app.query_one("#prompt").has_focus
        assert app.pets == 2

    drive(app, script)


def test_a_long_petting_session_builds_to_zoomies_then_enough(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        moods = []
        for _ in range(tui.TIRED_STREAK):
            app.touch("head")
            moods.append(app.mascot.mood)
            await pilot.pause()
        assert moods[0] == "love"
        assert moods[tui.ZOOM_STREAK - 1] == "zoom"
        assert moods[-1] == "pout"

    drive(app, script)


def test_the_streak_fades_after_a_pause(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.streak, app.last_touch = tui.ZOOM_STREAK, 0.0
        app.touch("head")
        assert app.mascot.mood == "love" and app.streak == 1

    drive(app, script)


def test_a_touch_wakes_a_sleeping_pechi(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.say_mood("sleep")
        app.touch("head")
        assert app.mascot.mood == "boop"
        assert "it's you" in str(app.bubble.render())

    drive(app, script)


def test_a_reaction_mid_task_goes_back_to_thinking(tmp_path, ollama, monkeypatch):
    monkeypatch.setitem(tui.MOOD_SECONDS, "love", 0.05)
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.busy = True
        app.touch("head")
        await pilot.pause(0.2)
        assert app.mascot.mood == "think"
        app.busy = False

    drive(app, script)


def test_quick_pets_do_not_cut_each_other_short(tmp_path, ollama, monkeypatch):
    monkeypatch.setitem(tui.MOOD_SECONDS, "love", 0.3)
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.touch("head")
        await pilot.pause(0.2)
        app.touch("head")
        await pilot.pause(0.2)  # past the first pet's timer, not the second's
        assert app.mascot.mood == "love"

    drive(app, script)


def test_pets_survive_a_restart_and_milestones_are_celebrated(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.pets = 9
        app.touch("head")
        assert "10 pets" in str(app.bubble.render())

    drive(app, script)
    assert make_app(tmp_path).pets == 10


@pytest.mark.parametrize(
    "command, mood", [("/pet", "love"), ("/boop", "boop"), ("/treat", "chew")]
)
def test_commands_reach_pechi_without_a_mouse(tmp_path, ollama, command, mood):
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, command)
        assert app.mascot.mood == mood

    drive(app, script)


@pytest.mark.parametrize(
    "streaming, shown",
    [
        ("Done.\nThou", "Done."),
        ("Done.\n\n**Fin", "Done."),
        ("Done.\nAct", "Done."),
        ("Done.\nThousands of them", "Done.\nThousands of them"),
        ("Done.", "Done."),
    ],
)
def test_a_half_written_marker_never_flashes_in_the_answer(streaming, shown):
    assert tui.settled(streaming) == shown


def test_a_last_line_held_back_still_lands_in_the_finished_answer(tmp_path, ollama):
    ollama["replies"] = ["Thought: ok\nFinal Answer: Pick one:\nA"]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await until_idle(pilot)
        assert answers(app)[-1] == "Pick one:\nA"

    drive(app, script)


def test_the_pet_key_stays_out_of_an_approval_card(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        app.push_screen(tui.Approval("write_file", {"path": "a"}, False, None))
        await pilot.pause(0.05)
        await pilot.press("ctrl+p")
        await pilot.pause(0.05)
        assert app.pets == 0

    drive(app, script)


def test_a_petting_streak_adds_one_note_on_a_narrow_window(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        before = len(app.query(tui.Turn))
        for _ in range(5):
            app.touch("head")
            await pilot.pause()
        await pilot.pause(0.2)
        assert len(app.query(tui.Turn)) == before + 1

    drive(app, script, size=(70, 24))


def test_the_pet_key_works_from_the_prompt(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        await pilot.press("ctrl+p")
        await pilot.pause(0.05)
        assert app.pets == 1

    drive(app, script)


def test_a_reaction_shows_in_the_chat_when_the_sidebar_is_hidden(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        before = len(answers(app))
        await send(pilot, "/pet")
        await pilot.pause(0.2)
        assert len(answers(app)) == before + 1

    drive(app, script, size=(70, 24))


def test_the_intro_mascot_ignores_clicks(tmp_path, ollama):
    app = tui.PechiApp(tmp_path, intro=True)

    async def script(pilot):
        await pilot.pause(0.1)
        await pilot.click("#intro-pup")
        await pilot.pause(0.05)
        assert app.pets == 0

    drive(app, script)


def test_terminal_control_codes_are_dropped():
    assert tui.clean("a\x1b[2Jb\x07\nc") == "a[2Jb\nc"
    assert tui.clean("a\x9b2Jb") == "a2Jb"  # the 8-bit form of the same escape


def test_toggle_swaps_models_and_unloads_the_other(tmp_path, ollama):
    fast, big = agent.MODELS
    app = make_app(tmp_path)

    async def script(pilot):
        await app.workers.wait_for_complete()
        assert ollama["running"] == [fast.name]
        await pilot.press("ctrl+t")
        await app.workers.wait_for_complete()
        await pilot.pause(0.05)
        assert ollama["running"] == [big.name]
        assert (fast.name, 0) in ollama["calls"]
        assert "35B" in str(app.query_one("#top", Static).render())
        await pilot.press("ctrl+t")
        await app.workers.wait_for_complete()
        assert ollama["running"] == [fast.name]

    drive(app, script)


def test_toggle_mid_task_applies_at_the_next_step(tmp_path, ollama):
    fast, big = agent.MODELS
    (tmp_path / "a.txt").write_text("x")
    ollama["replies"] = [
        'Thought: look\nAction: read_file\nAction Input: {"path": "a.txt"}',
        "Thought: done\nFinal Answer: ok",
    ]
    app = make_app(tmp_path)
    original = app.show_event

    def toggle_on_action(turn, kind, data):
        original(turn, kind, data)
        if kind == "action":
            app.brain.toggle()

    app.show_event = toggle_on_action

    async def script(pilot):
        await send(pilot, "read it")
        await until_idle(pilot)
        assert [chat["model"] for chat in ollama["chats"]] == [fast.name, big.name]
        assert ollama["chats"][1]["options"]["num_gpu"] == big.num_gpu

    drive(app, script)


def test_brain_choice_and_theme_survive_a_restart(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.press("ctrl+t")
        await app.workers.wait_for_complete()
        app.theme = "burgundy"
        app.save()

    drive(app, script)
    again = make_app(tmp_path)
    assert again.theme == "burgundy" and again.brain.index == 1


@pytest.mark.parametrize(
    "text", ["{not json", "[]", '{"brain": "1"}', '{"brain": null, "theme": 5}']
)
def test_a_broken_config_falls_back_to_defaults(tmp_path, ollama, text):
    tui.CONFIG.parent.mkdir()
    tui.CONFIG.write_text(text)
    app = make_app(tmp_path)
    assert app.theme == "beige" and app.brain.index == 0 and app.pets == 0


def test_theme_picker_previews_live_and_escape_restores(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.press("ctrl+g")
        await pilot.pause(0.1)
        await pilot.press("down")
        await pilot.pause(0.1)
        assert app.theme == themes.THEME_NAMES[1]
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert app.theme == "beige"

    drive(app, script)


def test_stop_cancels_a_running_task(tmp_path, ollama):
    ollama["replies"] = ["Thought: long\nFinal Answer: " + "word " * 4000]
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "go")
        await pilot.press("escape")
        await until_idle(pilot)
        assert app.busy is False

    drive(app, script)


def test_slash_commands(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await send(pilot, "/help")
        await pilot.pause(0.1)
        assert "What Pechi can do" in answers(app)[-1]
        await send(pilot, "/clear")
        await pilot.pause(0.1)
        assert len(answers(app)) == 1

    drive(app, script)


def test_the_sidebar_hides_on_a_narrow_window(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.pause(0.05)
        assert app.query_one("#side").display is False

    drive(app, script, size=(70, 24))


def test_friendly_errors():
    assert "graphics card" in tui.friendly_error(500, "CUDA error: out of memory")
    assert "asleep" in tui.friendly_error(0, "<urlopen error [WinError 10061] refused>")
    assert "ollama list" in tui.friendly_error(404, "model not found")


MOODS = ["idle", "think", "talk", "happy", "cry", "sleep"]
MOODS += ["love", "boop", "chew", "zoom", "pout"]


@pytest.mark.parametrize("mood", MOODS)
def test_every_mood_animates_and_stays_in_shape(mood):
    frames = {sprites.pixels(tui.pose_for(mood, frame)) for frame in range(120)}
    assert len(frames) > 1
    for rows in frames:
        assert len(rows) == len(sprites.BASE)
        assert {len(row) for row in rows} == {sprites.WIDTH}


def test_sprite_uses_a_light_outline_on_dark_themes():
    dark = sprites.render(sprites.Pose(), True).spans
    assert any(sprites.OUTLINE_DARK.lower() in str(span.style) for span in dark)


def luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    )
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one: str, two: str) -> float:
    high, low = sorted((luminance(one), luminance(two)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("item", themes.THEMES, ids=themes.THEME_NAMES)
def test_every_theme_is_readable(item):
    for ground in (item.background, item.surface, item.panel):
        assert contrast(item.foreground, ground) >= 7
        assert contrast(item.variables["pechi-muted"], ground) >= 4.5
    for color in (item.accent, item.error, item.success):
        assert contrast(color, item.surface) >= 4.5


def test_every_theme_has_a_note_and_earth_tones_come_first():
    assert set(themes.THEME_NOTES) == set(themes.THEME_NAMES)
    assert themes.THEME_NAMES[0] == "beige" and "burgundy" in themes.THEME_NAMES


@pytest.fixture
def dns(monkeypatch):
    table = {
        "public.example": "93.184.216.34",
        "inside.example": "10.0.0.5",
        "loop.example": "127.0.0.1",
        "mapped.example": "::ffff:192.168.1.1",
        "meta.example": "169.254.169.254",
    }

    def lookup(host, port, *args, **kwargs):
        address = table[host]
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(web.socket, "getaddrinfo", lookup)


@pytest.mark.parametrize("host", ["inside", "loop", "mapped", "meta"])
def test_private_addresses_are_refused(dns, host):
    with pytest.raises(PermissionError):
        web.require_public(f"https://{host}.example/")


@pytest.mark.parametrize(
    "url", ["file:///C:/secret.txt", "ftp://public.example/x", "http://"]
)
def test_other_schemes_are_refused(dns, url):
    with pytest.raises(PermissionError):
        web.require_public(url)


def mock(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_a_redirect_to_another_site_needs_its_own_approval(dns):
    transport = mock(
        lambda r: httpx.Response(302, headers={"location": "http://inside.example/"})
    )
    with pytest.raises(ValueError, match="another site"):
        web.fetch("https://public.example/", transport)


def test_a_redirect_within_a_site_is_followed(dns, monkeypatch):
    def handler(request):
        if request.url.host == "public.example":
            return httpx.Response(
                301, headers={"location": "https://www.public.example/"}
            )
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"hi"
        )

    table = web.socket.getaddrinfo

    def lookup(host, *args, **kwargs):
        return table(host.removeprefix("www."), *args, **kwargs)

    monkeypatch.setattr(web.socket, "getaddrinfo", lookup)
    assert web.fetch("https://public.example/", mock(handler)).text == "hi"


def test_redirect_loops_end(dns):
    transport = mock(lambda r: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(ValueError, match="redirects"):
        web.fetch("https://public.example/", transport)


def test_binary_files_are_refused(dns):
    transport = mock(
        lambda r: httpx.Response(
            200, headers={"content-type": "application/zip"}, content=b"PK"
        )
    )
    with pytest.raises(ValueError, match="not a text page"):
        web.fetch("https://public.example/a.zip", transport)


def test_huge_pages_are_cut_off(dns):
    big = b"x" * (web.MAX_BYTES * 2)
    transport = mock(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=big
        )
    )
    assert len(web.fetch("https://public.example/", transport).text) == web.MAX_BYTES


def test_readable_prefers_the_main_content():
    page = "<html><body><div>Jump to content</div><main><p>The real text</p></main></body></html>"
    text = web.readable(web.Page("https://public.example/", page, True))
    assert "The real text" in text and "Jump to content" not in text


def test_theme_picker_rows_stay_visible(tmp_path, ollama):
    app = make_app(tmp_path)

    async def script(pilot):
        await pilot.press("ctrl+g")
        await pilot.pause(0.5)
        rows = app.screen.query_one("#themes")
        assert "beige" in rows.render_line(0).text

    drive(app, script)


PAGE = """<html><head><title>Dogs</title><style>.x{}</style></head><body>
<nav>menu</nav><h1>Malinois</h1><p>Very good dogs.</p><script>steal()</script>
<ul><li class="d"><a href="/a">Alpha</a></li><li class="d"><a href="/b">Beta</a></li></ul>
</body></html>"""


def test_readable_keeps_the_text_and_drops_the_furniture():
    text = web.readable(web.Page("https://public.example/", PAGE, True))
    assert "Title: Dogs" in text and "Very good dogs." in text
    assert "menu" not in text and "steal" not in text
    assert "Alpha: https://public.example/a" in text


def test_scrape_returns_matches_with_absolute_links():
    text = web.scrape(web.Page("https://public.example/", PAGE, True), "li.d")
    assert "2 matches" in text
    assert "Beta -> https://public.example/b" in text


def test_scrape_says_so_when_nothing_matches():
    text = web.scrape(web.Page("https://x.example/", PAGE, True), "table")
    assert "No elements" in text


def test_the_launcher_points_at_the_app():
    import install_pechi

    text = install_pechi.launcher("C:/py.exe", Path("C:/p/pechi_tui.py"))
    assert '"C:/py.exe"' in text and "pechi_tui.py" in text and "%*" in text
