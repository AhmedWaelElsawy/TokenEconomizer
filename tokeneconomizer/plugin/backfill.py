"""Learning from the sessions that ran before the plugin was installed.

The plugin learns from a session when it ends, so on its own it starts from nothing. Claude Code
keeps every session's log under `<config dir>/projects/<folder>/<session>.jsonl`; the backfill
reads the most recent of them (`backfill_sessions`, 30 by default) for one repository, oldest
first, through what a finished session goes through:

  * `fold.fold_session` -- micro-skills' facts, tool pruning's evidence, the ledger;
  * each prompt-turn replayed through `loopguard.TurnGuard`, a turn that ended on its own
    feeding the envelope, as the Stop hook would have.

Three differences from a live session, all on purpose:

  * No model is called: micro-skills runs its free layer only, and the replayed guard has no
    judge and no envelope to act on (it records, it decides nothing).
  * Tools the history alone shows unused are only proposed (`untool.json`'s `from_history`),
    even under `toolprune=auto`: they wait for `toolprune apply`, as a deny list nobody watched
    form should.
  * A session already learned (folded, or recorded by the hooks), one still being written, and
    the session that started the backfill are left alone, so nothing is counted twice.

The first session start in a repository runs it once in the background (`backfill.json` marks
it); `te backfill` runs it by hand, with `--dry-run` to see what it would read.
"""
import json
import os
import time
from dataclasses import replace

from .. import loopguard
from ..trace.claude import CONVERSATION, _is_prompt
from . import fold
from .paths import repo_identity

MARKER = "backfill.json"
# Read this far into a log for its session id and directory; Claude Code writes both on its
# first conversation entry, after a few bookkeeping lines.
HEAD_LINES = 200
ENDED = ("end_turn", "stop_sequence")


def session_projects_dir(transcript):
    """The `projects` folder a live session's own log is in, or None: the session start reads
    history only from there, never from a guessed location."""
    if not transcript:
        return None
    d = os.path.dirname(os.path.dirname(os.path.abspath(transcript)))
    return d if os.path.basename(d) == "projects" and os.path.isdir(d) else None


def projects_dir(env):
    """Where Claude Code keeps its session logs, for `te backfill`: `CLAUDE_CONFIG_DIR/projects`,
    else `~/.claude/projects`."""
    base = env.get("CLAUDE_CONFIG_DIR") or os.path.join(
        env.get("USERPROFILE") or env.get("HOME") or os.path.expanduser("~"), ".claude")
    return os.path.join(base, "projects")


def _head(path):
    """(session id, directory) of a saved session, from its first conversation entries."""
    sid = cwd = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if n >= HEAD_LINES or (sid and cwd):
                    break
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                sid = sid or (ev.get("sessionId") if isinstance(ev.get("sessionId"), str)
                              else None)
                if ev.get("type") in CONVERSATION and isinstance(ev.get("cwd"), str):
                    cwd = cwd or ev["cwd"]
    except OSError:
        return None, None
    return sid or os.path.splitext(os.path.basename(path))[0], cwd


def history(root):
    """Every saved session under `root`: [{path, session_id, cwd, mtime}]. A subagent's log
    (under `<session>/subagents/`) belongs to its session and is not listed."""
    out = []
    try:
        folders = sorted(os.listdir(root))
    except OSError:
        return out
    for folder in folders:
        d = os.path.join(root, folder)
        try:
            names = sorted(n for n in os.listdir(d) if n.endswith(".jsonl"))
        except OSError:
            continue
        for name in names:
            path = os.path.join(d, name)
            sid, cwd = _head(path)
            if not cwd:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            out.append({"path": path, "session_id": sid, "cwd": cwd, "mtime": mtime})
    return out


def claude_turns(path):
    """The prompt-turns of a saved Claude Code session, in order: [{calls, ended, interrupted}].
    A call is {name, input, response, is_error}, its `response` the structured result the
    PostToolUse hook would have seen (`toolUseResult`), else the text the model saw. A turn
    `ended` when the agent stopped on its own -- the Stop hook's summary, or a final message
    that ended the turn -- with no tool call after it."""
    turns, cur, by_id = [], None, {}
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return turns
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict) or ev.get("isSidechain"):
                continue
            kind = ev.get("type")
            if _is_prompt(ev):
                cur = {"calls": [], "ended": False, "interrupted": False}
                turns.append(cur)
                continue
            if cur is None:
                continue
            if kind == "system" and ev.get("subtype") == "stop_hook_summary":
                cur["ended"] = True
                continue
            if kind not in CONVERSATION:
                continue
            msg = ev.get("message") or {}
            content = msg.get("content")
            if kind == "assistant" and msg.get("stop_reason") in ENDED:
                cur["ended"] = True
            if isinstance(content, str):
                continue
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and kind == "assistant":
                    if b.get("id") in by_id:
                        continue
                    c = {"name": str(b.get("name") or ""),
                         "input": b.get("input") if isinstance(b.get("input"), dict) else {},
                         "response": None, "is_error": False}
                    by_id[b.get("id")] = c
                    cur["calls"].append(c)
                    cur["ended"] = False
                elif b.get("type") == "tool_result":
                    c = by_id.get(b.get("tool_use_id"))
                    if c is None:
                        continue
                    text = b.get("content")
                    if isinstance(text, list):
                        text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
                    text = text if isinstance(text, str) else ""
                    c["is_error"] = bool(b.get("is_error"))
                    structured = ev.get("toolUseResult")
                    c["response"] = text if c["is_error"] or not isinstance(
                        structured, dict) else structured
                elif (b.get("type") == "text" and kind == "user"
                      and str(b.get("text") or "").startswith("[Request interrupted")):
                    cur["interrupted"] = True
    return turns


def _payload(call):
    """The hook payload this call would have arrived as: a failure as PostToolUseFailure's
    `error`, anything else as PostToolUse's `tool_response`."""
    p = {"tool_name": call["name"], "tool_input": call["input"]}
    if call["is_error"]:
        p["error"] = call["response"] or ""
    else:
        p["tool_response"] = call["response"] if call["response"] is not None else ""
    return p


def replay_turns(ctx, transcript):
    """Each turn through the loop guard as its hooks would have folded it; the green ones feed
    the envelope. The guard is given no envelope, backstop or judge: the turns already ran, so
    it only records. Returns the turns' summaries."""
    thr = loopguard.thresholds(loopguard.load_envelope(
        ctx.store, level=ctx.config.loopguard_aggr, overrides=ctx.config.loopguard_thresholds),
        ctx.config.loopguard_thresholds)
    thr = dict(thr, backstop_turns=None)
    out = []
    for turn in claude_turns(transcript):
        g = loopguard.TurnGuard()
        for c in turn["calls"]:
            name, args, text, code = ctx.host.tool_call(_payload(c))
            g.fold(name, args, text, code, envelope=None, thr=thr, judge=None)
        normal = turn["ended"] and not turn["interrupted"]
        if g.cum_turns and normal:
            loopguard.record_turn(ctx.store, g, level=ctx.config.loopguard_aggr,
                                  overrides=ctx.config.loopguard_thresholds)
        out.append(g.summary(normal))
    return out


def _learned(store, sid):
    folded = store.read("folded.json") or {}
    return sid in folded or store.read(os.path.join("sessions", f"{sid}.json")) is not None


def _history_config(cfg):
    """The settings history is read under: the free micro-skills layer, and deny lists only
    proposed."""
    ms = cfg.microskills
    if "distiller" in ms:
        ms = "envfacts" if ms == "envfacts+distiller" else "off"
    tp = "propose" if cfg.toolprune == "auto" else cfg.toolprune
    return replace(cfg, microskills=ms, toolprune=tp)


def learn_session(ctx, item):
    """Learn from one saved session: its turns, then the fold. False if it was learned already."""
    sid, tp = item["session_id"], item["path"]
    if _learned(ctx.store, sid):
        return False
    ctx.config = _history_config(ctx.config)
    turns = replay_turns(ctx, tp)
    ctx.store.write(os.path.join("sessions", f"{sid}.json"), {
        "session_id": sid, "host": ctx.host.name, "cwd": item["cwd"], "transcript": tp,
        "started": None, "ended": item["mtime"], "turns": turns[-200:], "nudges": 0,
        "aborts": 0, "judge_tokens": 0.0, "writer_tokens": 0.0, "backfilled": True})
    fold.fold_session(ctx, sid, tp)
    return True


def candidates(ctx, items, limit, exclude=(), now=None):
    """The repository's saved sessions to learn from: the `limit` most recent that have
    finished, less those excluded, oldest first; and how many of them were learned already."""
    now = now or time.time()
    mine = [i for i in items if i["session_id"] not in exclude
            and now - i["mtime"] >= fold.STALE_S
            and repo_identity(i["cwd"], ctx.env).key == ctx.repo.key]
    recent = sorted(mine, key=lambda i: i["mtime"])[-limit:] if limit > 0 else []
    new = [i for i in recent if not _learned(ctx.store, i["session_id"])]
    return new, len(recent) - len(new)


def _mark_from_history(ctx):
    """Tools the history proposed wait for the person, even under `auto` (`fold._hold`)."""
    with ctx.store.locked("mechanisms"):
        unt = ctx.store.read("untool.json")
        if not isinstance(unt, dict):
            return []
        prop = list(unt.get("proposal") or [])
        unt["from_history"] = sorted(set(unt.get("from_history") or []) | set(prop))
        ctx.store.write("untool.json", unt)
        return unt["from_history"]


def run(ctx, root, limit=None, exclude=(), dry_run=False):
    """Learn from the repository's saved sessions under `root`. Returns a summary dict."""
    from .context import LiveContext
    limit = ctx.config.backfill_sessions if limit is None else limit
    new, known = candidates(ctx, history(root), limit, exclude)
    summary = {"repo": ctx.repo.label, "limit": limit, "found": len(new),
               "already": known, "learned": 0}
    if dry_run:
        return summary
    for item in new:
        sctx = LiveContext.build(ctx.host, item["cwd"], ctx.env, session_id=item["session_id"])
        if sctx.repo.key != ctx.repo.key:
            continue                    # pinned to another repository by an earlier run
        summary["learned"] += bool(learn_session(sctx, item))
    tracker = loopguard.load_envelope(ctx.store, level=ctx.config.loopguard_aggr,
                                      overrides=ctx.config.loopguard_thresholds)
    summary["green_turns"] = len(tracker.resolved_turns)
    summary["proposal"] = _mark_from_history(ctx) if summary["learned"] else []
    prior = ctx.store.read(MARKER) or {}
    ctx.store.write(MARKER, dict(prior, at=time.time(), limit=limit,
                                 sessions=int(prior.get("sessions", 0)) + summary["learned"]))
    return summary


def should_start(ctx, transcript):
    """Whether this session start runs the repository's first backfill: Claude Code, turned on,
    not shadowed, not run before, and the session's own log in Claude Code's projects folder."""
    return (ctx.host.name == "claude" and ctx.config.backfill_sessions > 0
            and not ctx.config.shadow and ctx.store.read(MARKER) is None
            and session_projects_dir(transcript) is not None)


def start(ctx, transcript, session_id):
    """Mark the repository and run its backfill in a detached process (inline when
    TE_FOLD_SYNC=1, for tests)."""
    root = session_projects_dir(transcript)
    ctx.store.write(MARKER, {"started": time.time(), "sessions": 0})
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return run(ctx, root, exclude=(session_id,))
    return ctx.spawn(["backfill", "--projects-dir", root, "--exclude", session_id],
                     "fold.log")
