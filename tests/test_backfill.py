"""Learning from the sessions that ran before the plugin was installed (`te backfill`).

Claude Code keeps every session's log under `~/.claude/projects/<folder>/<session>.jsonl`. The
backfill reads the most recent of them for a repository, oldest first, through the same fold a
finished session goes through, and replays their turns into the loop guard's envelope. The first
session start in a repository runs it once. Folds run inline here (`TE_FOLD_SYNC=1`).
"""
import json
import os
import time

import pytest

from tokeneconomizer.plugin import backfill, cli, denylist

from test_live_hooks import _catalog, call, env, repo, store_for  # noqa: F401  (fixtures)

OLD = time.time() - 3 * 24 * 3600


def _entry(kind, cwd, sid, **kw):
    return dict(kw, type=kind, cwd=cwd, sessionId=sid)


def history_session(projects, repo, sid, turns, age_h=0.0, folder="W--repo"):
    """A saved Claude Code session as Claude Code writes it: per turn a prompt, its tool calls
    (`tool_use` then a `tool_result` carrying the structured `toolUseResult`), and, when the turn
    ended on its own, the closing text and the Stop hook's summary.

    `turns` is a list of (prompt, calls, ending): calls are (name, input, output, is_error);
    ending is "green" (the agent stopped), "interrupted" or "open" (the log stops mid-turn)."""
    d = os.path.join(projects, folder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    lines, n = [], 0
    for prompt, calls, ending in turns:
        lines.append(_entry("user", repo, sid, message={"role": "user", "content": prompt}))
        for name, inp, out, is_error in calls:
            n += 1
            lines.append(_entry("assistant", repo, sid, message={
                "id": f"msg_{sid}_{n}", "model": "claude-opus-5-5", "role": "assistant",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                          "cache_creation_input_tokens": 100, "output_tokens": 50},
                "content": [{"type": "tool_use", "id": f"tu{n}", "name": name, "input": inp}]}))
            text = out if is_error else (out.get("stdout", "") if isinstance(out, dict) else out)
            lines.append(_entry("user", repo, sid, toolUseResult=out, message={
                "role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu{n}",
                                             "content": text, "is_error": is_error}]}))
        if ending == "green":
            n += 1
            lines.append(_entry("assistant", repo, sid, message={
                "id": f"msg_{sid}_{n}", "model": "claude-opus-5-5", "role": "assistant",
                "stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}]}))
            lines.append(_entry("system", repo, sid, subtype="stop_hook_summary"))
        elif ending == "interrupted":
            lines.append(_entry("user", repo, sid, message={
                "role": "user", "content": [{"type": "text",
                                             "text": "[Request interrupted by user]"}]}))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    at = OLD + age_h * 3600
    os.utime(path, (at, at))
    return path


LS = ("Bash", {"command": "ls"}, {"stdout": "a.py", "stderr": ""}, False)
EDIT = ("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"}, "ok", False)
TEST_OK = ("Bash", {"command": "python -m pytest -q"}, {"stdout": "3 passed", "stderr": ""},
           False)


def green(n_calls=2, prompt="go"):
    return (prompt, [EDIT] + [TEST_OK] * (n_calls - 1), "green")


@pytest.fixture
def projects(tmp_path):
    p = tmp_path / "claude" / "projects"
    p.mkdir(parents=True)
    return str(p)


def run(env, monkeypatch, *argv):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return cli.main(["--host", "claude", *argv])


def test_backfill_learns_from_the_repositorys_saved_sessions_oldest_first(
        env, repo, projects, monkeypatch, capsys):
    for i in range(3):
        history_session(projects, repo, f"h{i}", [green(2 + i)], age_h=i)
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    out = capsys.readouterr().out
    assert "3 earlier session(s)" in out
    st = store_for(env, repo)
    assert sorted(st.read("folded.json")) == ["h0", "h1", "h2"]
    ledger = st.read_lines("ledger.jsonl")
    assert [r["session_id"] for r in ledger] == ["h0", "h1", "h2"]        # oldest first
    assert all(r["backfilled"] for r in ledger)
    assert ledger[0]["requests"] >= 2 and ledger[0]["output_tokens"] >= 100
    # Each green turn fed the envelope with its own number of tool calls.
    assert st.read("envelope.json")["resolved_turns"] == [2, 3, 4]
    assert st.read("sessions/h0.json")["backfilled"] is True
    assert st.read("backfill.json")["sessions"] == 3


def test_only_the_most_recent_sessions_up_to_the_limit_are_read(
        env, repo, projects, monkeypatch, capsys):
    for i in range(5):
        history_session(projects, repo, f"h{i}", [green()], age_h=i)
    env = dict(env, TE_BACKFILL_SESSIONS="3")
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    assert sorted(store_for(env, repo).read("folded.json")) == ["h2", "h3", "h4"]
    capsys.readouterr()
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects,
               "--limit", "1") == 0
    assert "nothing new" in capsys.readouterr().out          # the newest is already learned


def test_open_excluded_and_other_folders_sessions_are_left_alone(
        env, repo, projects, tmp_path, monkeypatch):
    history_session(projects, repo, "done", [green()])
    open_ = history_session(projects, repo, "open", [green()])
    os.utime(open_, None)                                     # written just now: still running
    history_session(projects, repo, "mine", [green()])
    other = tmp_path / "other"
    other.mkdir()
    history_session(projects, str(other), "elsewhere", [green()], folder="W--other")
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects,
               "--exclude", "mine") == 0
    assert sorted(store_for(env, repo).read("folded.json")) == ["done"]
    assert store_for(env, str(other)).read("folded.json") is None


def test_a_second_backfill_learns_nothing_twice(env, repo, projects, monkeypatch):
    for i in range(2):
        history_session(projects, repo, f"h{i}", [green()], age_h=i)
    for _ in range(2):
        assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    st = store_for(env, repo)
    assert len(st.read_lines("ledger.jsonl")) == 2
    assert st.read("envelope.json")["resolved_turns"] == [2, 2]


def test_only_turns_that_ended_on_their_own_feed_the_envelope(env, repo, projects, monkeypatch):
    history_session(projects, repo, "h", [
        ("one", [EDIT, TEST_OK, TEST_OK], "interrupted"),
        ("two", [EDIT, ("Bash", {"command": "python -m pytest -q"},
                        "Exit code 1\nFAILED tests/test_a.py::t - AssertionError", True)],
         "green"),                                                     # ended red
        green(4, "three"),
        ("four", [LS, LS], "open")])
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    st = store_for(env, repo)
    assert st.read("envelope.json")["resolved_turns"] == [4]
    turns = st.read("sessions/h.json")["turns"]
    assert [t["tool_calls"] for t in turns] == [3, 2, 4, 2]
    assert [t["resolved_proxy"] for t in turns] == [False, False, True, False]


def test_a_dry_run_reports_and_writes_nothing(env, repo, projects, monkeypatch, capsys):
    for i in range(2):
        history_session(projects, repo, f"h{i}", [green()], age_h=i)
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects,
               "--dry-run") == 0
    out = capsys.readouterr().out
    assert "would learn from 2 earlier session(s)" in out
    assert not os.path.exists(env["TOKENECONOMIZER_HOME"]) or \
        store_for(env, repo).read("folded.json") is None


def test_tools_history_leaves_unused_are_only_suggested(env, repo, projects, tmp_path,
                                                         monkeypatch):
    """Under the default `auto`, tools the history alone shows unused wait for the person:
    nothing is written to the settings file, the next session start says so, and a later live
    fold does not write them either. `toolprune apply` does."""
    _catalog(env, ["Bash", "Read", "Edit", "Workflow", "CronCreate"])
    for i in range(6):
        history_session(projects, repo, f"h{i}", [green(prompt=f"p{i}"), green(prompt="more")],
                        age_h=i)
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    st = store_for(env, repo)
    unt = st.read("untool.json")
    assert sorted(unt["from_history"]) == ["CronCreate", "Workflow"]
    assert not unt.get("auto_pending")
    assert not os.path.exists(denylist.settings_path(repo))
    # A live session after the backfill, that also left those tools unused.
    t = history_session(projects, repo, "live", [green(), green(prompt="more")], age_h=9)
    notice = call(env, {"hook_event_name": "SessionStart", "session_id": "live",
                        "source": "startup", "transcript_path": t, "cwd": repo})["systemMessage"]
    assert "2 tool(s) went unused" in notice
    assert "nothing changes until you run `/tokeneconomizer:toolprune apply`" in notice
    call(env, {"hook_event_name": "SessionEnd", "session_id": "live", "transcript_path": t,
               "cwd": repo, "reason": "other"})
    assert not os.path.exists(denylist.settings_path(repo))         # a live fold waits too
    assert sorted(st.read("untool.json")["from_history"]) == ["CronCreate", "Workflow"]
    assert run(env, monkeypatch, "--cwd", repo, "toolprune", "apply") == 0
    deny = json.load(open(denylist.settings_path(repo)))["permissions"]["deny"]
    assert sorted(deny) == ["CronCreate", "Workflow"]
    assert st.read("untool.json").get("from_history") in (None, [])


def _start(env, sid, transcript, repo):
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid, "source": "startup",
                      "transcript_path": transcript, "cwd": repo})


def test_the_first_session_start_learns_from_the_history_once(env, repo, projects):
    for i in range(2):
        history_session(projects, repo, f"h{i}", [green()], age_h=i)
    live = os.path.join(projects, "W--repo", "live.jsonl")
    open(live, "w").close()
    msg = _start(env, "live", live, repo)["systemMessage"]
    assert "learning from up to 30 earlier session(s)" in msg
    st = store_for(env, repo)
    assert sorted(st.read("folded.json")) == ["h0", "h1"]           # the live one is not
    history_session(projects, repo, "h2", [green()], age_h=5)
    assert "earlier session" not in _start(env, "live2", live, repo).get("systemMessage", "")
    assert "h2" not in st.read("folded.json")                       # once per repository


def test_no_backfill_when_turned_off_or_outside_claude_codes_folder(env, repo, projects,
                                                                    tmp_path):
    history_session(projects, repo, "h0", [green()])
    live = os.path.join(projects, "W--repo", "live.jsonl")
    open(live, "w").close()
    off = dict(env, TE_BACKFILL_SESSIONS="0")
    assert "earlier session" not in _start(off, "a", live, repo).get("systemMessage", "")
    elsewhere = str(tmp_path / "loose.jsonl")
    open(elsewhere, "w").close()
    assert "earlier session" not in _start(env, "b", elsewhere, repo).get("systemMessage", "")
    assert store_for(env, repo).read("folded.json") is None


def test_the_backfill_never_calls_a_model(env, repo, projects, monkeypatch):
    """Even with the proposer on, history is read with the free layer only."""
    from tokeneconomizer.plugin import sidecall
    calls = []
    monkeypatch.setattr(sidecall, "claude_call", lambda *a, **k: calls.append(a) or ("", None))
    env = dict(env, TE_MICROSKILLS="on")
    for i in range(4):
        history_session(projects, repo, f"h{i}", [green()], age_h=i)
    assert run(env, monkeypatch, "--cwd", repo, "backfill", "--projects-dir", projects) == 0
    assert calls == []
    assert len(store_for(env, repo).read("folded.json")) == 4


def test_the_turns_reader_pairs_calls_with_their_results(projects, repo):
    path = history_session(projects, repo, "h", [
        ("go", [LS, ("Bash", {"command": "false"}, "Exit code 1\n", True)], "green")])
    (turn,) = backfill.claude_turns(path)
    assert turn["ended"] is True and turn["interrupted"] is False
    assert [(c["name"], c["is_error"]) for c in turn["calls"]] == [("Bash", False),
                                                                   ("Bash", True)]
    assert turn["calls"][0]["response"] == {"stdout": "a.py", "stderr": ""}
