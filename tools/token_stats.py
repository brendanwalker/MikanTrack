"""Regenerate TOKEN_STATS.md (and the README badges) from local Claude Code transcripts.

Claude Code writes one JSONL transcript per session under the project folder in
~/.claude/projects/. Every assistant message records exact API token usage per
model. This script aggregates the sessions that belong to MikanMediaPipe work,
rewrites the generated block in TOKEN_STATS.md, and refreshes the badge line in
README.md.

Usage:  python tools/token_stats.py
New sessions must be added to SESSIONS (or IGNORED) below; the script lists any
transcript it does not recognize.
"""

import collections
import glob
import json
import os
import re
from datetime import datetime, timezone

TRANSCRIPT_DIR = os.path.expanduser(
    r"~\.claude\projects\D--Github-git-BrendanWalker-MikanXR"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_MD = os.path.join(REPO_ROOT, "TOKEN_STATS.md")
README_MD = os.path.join(REPO_ROOT, "README.md")

# Session transcript id -> short description. Only these count toward the totals.
SESSIONS = {
    "4ac3774b-763d-45c3-a0ef-f1ca5753b894": "Initial build: scaffold to working tracker, fusion, parametric model (Jul 30 - Aug 1)",
    "2dc53dce-c715-4265-82d3-88c5ad5ae604": "Triangulation overhaul, RTMPose A/B, RealSense, IMU, charuco extrinsics (Aug 2 - Aug 10)",
    "7d1bd754-bd18-405a-92cd-2316852369e8": "Image quality metrics, record/replay, hitch diagnosis (Aug 10 - Aug 11)",
    "21c00bc2-e910-4fe0-b136-3cfd02ad904b": "Depth A/B, marker-scale discovery, bone calibration, seeding fixes (Aug 11 - Aug 12)",
    "aa3a90ea-54b1-4179-bc80-4cb1994f004d": "LEARNINGS.md and token stats (Aug 12)",
    "6f113e65-3d35-483b-ad9c-1e5ecbb32974": "3-camera right-hand dropout: SQPnP fix, minCameraConfidence gate (Aug 12)",
    "8f667e50-cf5d-4d36-8cd6-92b98572a026": "Dedicated linux/macos tracking machine consideration (Aug 12)",
    "356d659d-3cde-4c0b-bbc1-bef43a4d2f3a": "BlazePose revival: opt-in per-camera body pose, shoulders + head OSC (Aug 12)",
    "af7ea373-96e6-4a6b-ac6d-5b23d382bfa1": "Hand state estimator + biomechanical priors: joint fit, limits, fitted angle prior (Aug 15 - Aug 16)",
}

# Same project folder, not MikanMediaPipe work (MikanXR, UE plugin, ARKit, misc).
IGNORED = {
    "0a2b4224-a7b1-41fb-8879-f965933a1cc6",
    "27edf9b1-7c34-4dd4-9179-e604afb801c9",
    "2c5fe4d4-b97f-445f-949d-b6764059d234",
    "347026e9-bc02-4aeb-90f6-add26b750de3",
    "3514f9af-f5a4-4a70-bf06-1c513c02c87a",
    "3b5b23f6-496e-4fc5-9597-78831354b5f4",
    "4b740127-3fa0-4ec1-bccf-3b3377b18758",
    "75aaf43f-4e17-479a-a584-7621552b2f0d",
    "8859e189-cf8e-4d87-b0f0-34188008f1d0",
    "b254da4d-1c3c-4146-8029-cf7f17ed016a",
    "bb1669f1-4176-4f99-b70e-38920391cc44",
    "bf1aab0e-ac0b-4510-a604-2f41651ed6a5",
    "ca1eb0cb-51a9-4351-9309-15a698c674db",
    "d4115c0b-6a16-4dda-8f1e-2413e290e2a7",
    "df57a597-bd73-4d99-9a23-a21d699dc7b4",
    "e1be012c-f0ab-44ef-9c9d-bf96b6ceace9",
    "e54cb883-409e-4494-acec-6dbfa3aa717d",
    "e84a4fe6-dbbb-419c-871e-eb02ff808086",
    "ebeb96da-d1d1-4af0-ba3f-cdafedf91a5d",
    "f606064a-fa31-408f-9d21-74e786c6311f",
    "fd1b0429-d2a8-4ba6-98d0-16ceb78db2ce",
}

# Compute weighting per token class, proportional to Anthropic's price ratios
# (price is the closest public proxy for serving compute): cache reads cost 10%
# of input, input costs 20% of output, cache writes cost 25% of input... see
# TOKEN_STATS.md for the discussion.
WEIGHTS = {"output": 1.0, "input": 1 / 5, "cache_write": 1 / 4, "cache_read": 1 / 50}

# Energy per 1000 weighted (output-equivalent) tokens, Wh: low / central / high.
WH_PER_1K_WEIGHTED = (0.5, 2.0, 6.0)
# Water per kWh: onsite datacenter cooling, and total including power generation.
WATER_ONSITE_L_PER_KWH = 1.1
WATER_TOTAL_L_PER_KWH = 3.0


def aggregate_session(top_id):
    """Sum usage across a session's transcript plus its subagent transcripts."""
    files = [os.path.join(TRANSCRIPT_DIR, top_id + ".jsonl")]
    files += glob.glob(os.path.join(TRANSCRIPT_DIR, top_id, "**", "*.jsonl"), recursive=True)
    messages = {}
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                usage = msg.get("usage")
                if not isinstance(usage, dict):
                    continue
                model = msg.get("model", "unknown")
                if model == "<synthetic>":
                    continue
                # Streaming rewrites the same message id with growing usage:
                # keep the record with the largest output count.
                key = (path, msg.get("id") or rec.get("uuid"))
                out = usage.get("output_tokens") or 0
                prev = messages.get(key)
                if prev is None or out >= prev[4]:
                    messages[key] = (
                        model,
                        usage.get("input_tokens") or 0,
                        usage.get("cache_creation_input_tokens") or 0,
                        usage.get("cache_read_input_tokens") or 0,
                        out,
                    )
    per_model = collections.defaultdict(lambda: collections.Counter())
    for model, inp, cw, cr, out in messages.values():
        c = per_model[model]
        c["input"] += inp
        c["cache_write"] += cw
        c["cache_read"] += cr
        c["output"] += out
        c["calls"] += 1
    return per_model


def weighted(counter):
    return sum(counter[k] * w for k, w in WEIGHTS.items())


def fmt(n):
    return f"{int(n):,}"


def fmt_compact(n):
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.2g}{suffix}" if n < 10 * div else f"{n / div:.0f}{suffix}"
    return str(int(n))


def replace_block(path, marker, content):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    begin, end = f"<!-- {marker}:BEGIN -->", f"<!-- {marker}:END -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"marker {marker} not found in {path}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(pattern.sub(begin + "\n" + content + "\n" + end, text))


def main():
    known = set(SESSIONS) | IGNORED
    for path in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.jsonl")):
        sid = os.path.basename(path)[:-6]
        if sid not in known:
            print(f"UNCLASSIFIED session {sid} - add to SESSIONS or IGNORED")

    per_session = {sid: aggregate_session(sid) for sid in SESSIONS}
    totals = collections.defaultdict(lambda: collections.Counter())
    for models in per_session.values():
        for model, c in models.items():
            totals[model].update(c)

    session_rows = ["| Session | Model | API calls | Output | Cache read | Cache write |",
                    "|---|---|---|---|---|---|"]
    for sid, label in SESSIONS.items():
        for model, c in sorted(per_session[sid].items()):
            session_rows.append(
                f"| {label} | {model} | {fmt(c['calls'])} | {fmt(c['output'])} "
                f"| {fmt(c['cache_read'])} | {fmt(c['cache_write'])} |"
            )

    total_rows = ["| Model | API calls | Input | Cache write | Cache read | Output | Weighted |",
                  "|---|---|---|---|---|---|---|"]
    grand = collections.Counter()
    for model, c in sorted(totals.items()):
        grand.update(c)
        total_rows.append(
            f"| {model} | {fmt(c['calls'])} | {fmt(c['input'])} | {fmt(c['cache_write'])} "
            f"| {fmt(c['cache_read'])} | {fmt(c['output'])} | {fmt(weighted(c))} |"
        )
    gw = weighted(grand)
    total_rows.append(
        f"| **total** | {fmt(grand['calls'])} | {fmt(grand['input'])} | {fmt(grand['cache_write'])} "
        f"| {fmt(grand['cache_read'])} | {fmt(grand['output'])} | **{fmt(gw)}** |"
    )

    est_rows = ["| Scenario | Wh per 1k weighted tokens | Energy | Water (onsite cooling) | Water (incl. generation) |",
                "|---|---|---|---|---|"]
    for label, wh in zip(("low", "central", "high"), WH_PER_1K_WEIGHTED):
        kwh = gw / 1000 * wh / 1000
        est_rows.append(
            f"| {label} | {wh} | {kwh:,.0f} kWh | {kwh * WATER_ONSITE_L_PER_KWH:,.0f} L "
            f"| {kwh * WATER_TOTAL_L_PER_KWH:,.0f} L |"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    central_kwh = gw / 1000 * WH_PER_1K_WEIGHTED[1] / 1000
    block = "\n".join(
        [f"Last regenerated {stamp} by `tools/token_stats.py`.", ""]
        + ["## Per-session usage", ""] + session_rows + [""]
        + ["## Totals per model", ""] + total_rows + [""]
        + ["## Energy and water estimate", ""] + est_rows
    )
    replace_block(STATS_MD, "TOKEN_STATS", block)

    def badge(label, value, color):
        esc = lambda s: s.replace("-", "--").replace(" ", "_").replace("/", "%2F")
        return f"![{label}](https://img.shields.io/badge/{esc(label)}-{esc(value)}-{color})"

    badges = " ".join([
        badge("AI tokens", f"{fmt_compact(grand['output'])} out / {fmt_compact(grand['cache_read'])} read", "blueviolet"),
        badge("est. energy", f"~{central_kwh:,.0f} kWh", "yellow"),
        badge("est. water", f"~{central_kwh * WATER_TOTAL_L_PER_KWH:,.0f} L", "blue"),
    ]) + "\n(estimates, see [TOKEN_STATS.md](TOKEN_STATS.md))"
    replace_block(README_MD, "AI_USAGE_BADGES", badges)
    replace_block(STATS_MD, "AI_USAGE_BADGES", badges)

    print(f"total output {fmt(grand['output'])}, cache read {fmt(grand['cache_read'])}, "
          f"weighted {fmt(gw)}, central estimate {central_kwh:,.0f} kWh")
    print(f"updated {STATS_MD} and {README_MD}")


if __name__ == "__main__":
    main()
