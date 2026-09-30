# AI Token Usage

<!-- AI_USAGE_BADGES:BEGIN -->
![AI tokens](https://img.shields.io/badge/AI_tokens-7.4M_out_%2F_2.3B_read-blueviolet) ![est. energy](https://img.shields.io/badge/est._energy-~123_kWh-yellow) ![est. water](https://img.shields.io/badge/est._water-~369_L-blue)
(estimates, see [TOKEN_STATS.md](TOKEN_STATS.md))
<!-- AI_USAGE_BADGES:END -->

This project was developed almost entirely in Claude Code sessions (Claude Fable 5 and Claude Opus 5). This document tracks the exact API token usage of every MikanTrack development session, and derives an order-of-magnitude estimate of the electricity and water that inference consumed. See `LEARNINGS.md` for what all those tokens actually bought.

## How the numbers are collected

Claude Code stores a JSONL transcript per session (including subagent transcripts) under `~/.claude/projects/`, and every assistant message in it records the exact billed token counts per model. `tools/token_stats.py` scans those transcripts, deduplicates streamed message records, sums the sessions listed as MikanTrack work, and rewrites the generated block below plus the badges here and in `README.md`. To update after new sessions: add the new session id to the script's `SESSIONS` table and run

```bash
python tools/token_stats.py
```

The token counts are exact (they are what the API metered). Everything under "Energy and water estimate" is not.

## Estimation methodology

Anthropic publishes no per-token energy figures, so the estimate chains public reference points with stated assumptions. Treat the result as order-of-magnitude.

**Step 1: reduce four token classes to one.** The API meters four classes that cost very different amounts of compute: fresh input, cache writes, cache reads, and output. We fold them into "weighted output-equivalent tokens" using Anthropic's own price ratios as the compute proxy (input = 1/5 of output, cache write = 1/4, cache read = 1/50). Pricing is the closest public signal of relative serving cost. Note the consequence: cache reads dominate this project's weighted total even at 2% weight, because agentic coding re-reads the growing conversation context on every call. That is real work (attention over the cached context) and should not be dropped.

**Step 2: energy per weighted token.** Public anchors: Google reported the median Gemini Apps text prompt at 0.24 Wh (Aug 2025 technical report), Epoch AI estimated ~0.3 Wh for a typical GPT-4o query, and Sam Altman quoted 0.34 Wh for an average ChatGPT query. A median chat prompt is a few hundred output-equivalent tokens, which lands those figures near 1 Wh per 1000 weighted tokens. Fable/Opus-class models are larger than what serves a median consumer prompt, so we take 2 Wh per 1000 weighted tokens as the central assumption, with 0.5 and 6 as low/high bounds. These constants sit at the top of `tools/token_stats.py`; correct them as better data appears.

**Step 3: water per kWh.** Onsite datacenter cooling: ~1.1 L/kWh (consistent with Google's reported fleet WUE and their 0.26 mL per 0.24 Wh prompt figure). Including the water footprint of electricity generation raises it to roughly 3 L/kWh on a typical US grid mix.

**What the estimate excludes:** model training (amortized per query it is generally estimated to be small relative to inference at scale, but it is not zero), datacenter construction, the dev machine running Claude Code locally, and the RTX 3090 that ran the actual hand-tracking workloads this project exists for. It also cannot know Anthropic's actual serving efficiency, batching, or datacenter locations, which is why the range spans an order of magnitude.

**For scale:** the central estimate (~70 kWh) is about two and a half days of average US household electricity, or roughly 140 hours of the RTX 3090 dev machine at full load. The central water estimate (~200 L) is about one bathtub.

<!-- TOKEN_STATS:BEGIN -->
Last regenerated 2026-09-30 06:06 UTC by `tools/token_stats.py`.

## Per-session usage

| Session | Model | API calls | Output | Cache read | Cache write |
|---|---|---|---|---|---|
| Initial build: scaffold to working tracker, fusion, parametric model (Jul 30 - Aug 1) | claude-fable-5 | 745 | 1,109,801 | 245,527,743 | 5,178,551 |
| Initial build: scaffold to working tracker, fusion, parametric model (Jul 30 - Aug 1) | claude-opus-5 | 218 | 239,263 | 56,255,854 | 876,017 |
| Triangulation overhaul, RTMPose A/B, RealSense, IMU, charuco extrinsics (Aug 2 - Aug 10) | claude-fable-5 | 441 | 428,792 | 207,103,745 | 1,080,665 |
| Triangulation overhaul, RTMPose A/B, RealSense, IMU, charuco extrinsics (Aug 2 - Aug 10) | claude-opus-5 | 947 | 1,012,882 | 453,133,941 | 5,483,413 |
| Image quality metrics, record/replay, hitch diagnosis (Aug 10 - Aug 11) | claude-fable-5 | 227 | 355,415 | 73,796,202 | 1,877,293 |
| Image quality metrics, record/replay, hitch diagnosis (Aug 10 - Aug 11) | claude-opus-5 | 199 | 245,359 | 98,648,329 | 1,551,818 |
| Depth A/B, marker-scale discovery, bone calibration, seeding fixes (Aug 11 - Aug 12) | claude-opus-5 | 373 | 406,564 | 143,251,867 | 1,024,437 |
| LEARNINGS.md and token stats (Aug 12) | claude-fable-5 | 21 | 42,320 | 2,036,500 | 116,150 |
| 3-camera right-hand dropout: SQPnP fix, minCameraConfidence gate (Aug 12) | claude-fable-5 | 163 | 150,814 | 31,096,087 | 303,891 |
| Dedicated linux/macos tracking machine consideration (Aug 12) | claude-fable-5 | 14 | 13,324 | 1,013,749 | 70,154 |
| BlazePose revival: opt-in per-camera body pose, shoulders + head OSC (Aug 12) | claude-fable-5 | 151 | 176,162 | 34,416,298 | 379,915 |
| BlazePose revival: opt-in per-camera body pose, shoulders + head OSC (Aug 12) | claude-opus-5 | 785 | 882,945 | 373,642,254 | 3,352,535 |
| Hand state estimator + biomechanical priors: joint fit, limits, fitted angle prior (Aug 15 - Aug 16) | claude-fable-5 | 429 | 443,619 | 245,581,671 | 1,549,360 |
| Hand state estimator + biomechanical priors: joint fit, limits, fitted angle prior (Aug 15 - Aug 16) | claude-opus-5 | 95 | 99,784 | 53,053,665 | 707,108 |
| VMC OSC output mode for VMC4UE and VSeeFace comparison (Aug 16) | claude-opus-5 | 181 | 169,468 | 47,773,144 | 508,165 |
| Left elbow stuck in a non-anatomic pose: recorded-replay diagnosis (Aug 16) | claude-opus-5 | 146 | 113,124 | 33,808,344 | 322,776 |
| OSC forearm position, wrist rotation removal, UE plugin update (Aug 16 - Aug 17) | claude-opus-5 | 92 | 52,147 | 16,392,624 | 217,274 |
| Open-sourcing docs pass: CLAUDE.md, docs/reference, rename to MikanTrack (Aug 17) | claude-fable-5 | 127 | 159,017 | 17,666,658 | 770,851 |
| Open-sourcing docs pass: CLAUDE.md, docs/reference, rename to MikanTrack (Aug 17) | claude-opus-5 | 59 | 53,593 | 15,022,036 | 473,308 |
| Open-sourcing docs pass: CLAUDE.md, docs/reference, rename to MikanTrack (Aug 17) | claude-sonnet-5 | 85 | 72,133 | 8,571,544 | 386,751 |
| Project files, guided setup flow, tracking configurations (Aug 17 - Aug 18) | claude-fable-5 | 273 | 314,321 | 83,780,411 | 1,071,121 |
| Project files, guided setup flow, tracking configurations (Aug 17 - Aug 18) | claude-opus-5 | 113 | 93,677 | 9,050,492 | 601,205 |
| Project files, guided setup flow, tracking configurations (Aug 17 - Aug 18) | claude-sonnet-5 | 360 | 455,086 | 36,320,823 | 938,224 |
| Single-camera removal and SystemAnimatorOnline transfer consideration (Aug 18 - Aug 19) | claude-opus-5 | 56 | 77,596 | 8,637,133 | 342,360 |
| testing.md reference doc (Aug 19) | claude-opus-5 | 38 | 24,631 | 3,748,073 | 111,058 |
| Cleanup pass: flicker tracker, ImuService split, CameraContext, capture timing, MainWindow split, loop stages (Sep 29) | claude-fable-5-1 | 102 | 242,525 | 34,360,308 | 502,477 |

## Totals per model

| Model | API calls | Input | Cache write | Cache read | Output | Weighted |
|---|---|---|---|---|---|---|
| claude-fable-5 | 2,591 | 21,074 | 12,397,951 | 942,019,064 | 3,193,585 | 25,137,668 |
| claude-fable-5-1 | 102 | 2,754 | 502,477 | 34,360,308 | 242,525 | 1,055,901 |
| claude-opus-5 | 3,302 | 45,258 | 15,571,474 | 1,312,417,756 | 3,471,033 | 33,621,308 |
| claude-sonnet-5 | 445 | 13,920 | 1,324,975 | 44,892,367 | 527,219 | 1,759,094 |
| **total** | 6,440 | 83,006 | 29,796,877 | 2,333,689,495 | 7,434,362 | **61,573,972** |

## Energy and water estimate

| Scenario | Wh per 1k weighted tokens | Energy | Water (onsite cooling) | Water (incl. generation) |
|---|---|---|---|---|
| low | 0.5 | 31 kWh | 34 L | 92 L |
| central | 2.0 | 123 kWh | 135 L | 369 L |
| high | 6.0 | 369 kWh | 406 L | 1,108 L |
<!-- TOKEN_STATS:END -->
