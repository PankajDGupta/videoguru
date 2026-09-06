# VideoGuru — Progress Tracker

> **Last Updated:** 2026-09-06
>
> Track the completion status of each spec. Update this file as work progresses.

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ⬜ | Not Started |
| 🟡 | In Progress |
| ✅ | Completed |
| 🔴 | Blocked |
| ⏭️ | Skipped / Deferred |

---

## Phase 0 — Project Scaffolding

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-001 | Initialize Project Structure | ✅ | Antigravity | Scaffolding created, virtual environment & dependencies installed, FFmpeg verified, unit tests passing |
| SPEC-002 | ADK Project Bootstrap | ✅ | Antigravity | Root agent & App configured, InMemorySessionService integrated, main.py CLI & Web UI launcher implemented, unit & integration tests passing |


---

## Phase I — Intent Capture & Media Ingestion

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-003 | Root Greeter Agent | ✅ | Antigravity | RootGreeterAgent implemented, custom record_theme tool stores theme in session.state['theme'] and hands off to Ingestion Agent, unit & integration tests passing |
| SPEC-004 | Local Directory Scanner Tool | ✅ | Antigravity | Custom ADK tool scan_local_directory and scan_local_directory_with_metadata implemented using pathlib, ScannedVideoFile schema created, CLI --scan-dir added, unit & integration tests passing |
| SPEC-005 | Clip Metadata Extraction Tool | ✅ | Antigravity | Custom ADK tool extract_clip_metadata and probe_video_file implemented using ffprobe, ClipManifestEntry schema created with UUID-based clip_id, CLI --extract-metadata added, unit & integration tests passing |
| SPEC-006 | Ingestion Agent | ✅ | Antigravity | IngestionAgent implemented with scan_local_directory and extract_clip_metadata orchestration, build_clip_manifest and ingest_media_directory tools created, Clip Manifest stored in session.state['clip_manifest'], CLI --ingest-dir added, unit & integration tests passing |


---

## Phase II — Multimodal Video Analysis & EDL Generation

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-007 | Pydantic EDL Schema | ✅ | Antigravity | TransitionIntent enum, EDLEntry & EditDecisionList models implemented, validation rules (end > start, clip references), CLI --validate-edl added, 42 unit tests passing |
| SPEC-008 | Gemini Video Analysis Tool | ✅ | Antigravity | analyze_clip tool implemented using Gemini 2.0 Flash multimodal API, EDLEntry structured output enforced with engagement scoring, guaranteed Files API resource cleanup, offline/mock mode supported, CLI --analyze-clip added, 19 unit & integration tests passing (162 total tests passing) |
| SPEC-009 | Curation Agent | ✅ | Antigravity | CurationAgent implemented with assemble_edl_from_manifest and curate_edit_decision_list tools, stores EDL in session.state['edl'], supports .env loading and offline mode, CLI --curate added, 15 unit & integration tests passing (177 total tests passing) |

---

## Phase III — Autonomous AI Review Loop

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-010 | Reviewer Agent (Algorithmic) | ✅ | Antigravity | ReviewerAgent implemented with review_edl_algorithmically tool, evaluating pacing (CPM, cadence variance), hook strength (first 5s), and retention curve, writes feedback & scores to session.state, CLI --review-edl added, 24 unit & integration tests passing (201 total tests passing) |
| SPEC-011 | Critic Agent (Human-Centric) | ⬜ | — | — |
| SPEC-012 | LoopAgent Assembly | ⬜ | — | Depends on SPEC-009, SPEC-010, SPEC-011 |

---

## Phase IV — Human-in-the-Loop Review

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-013 | OpenTimelineIO Converter Tool | ⬜ | — | — |
| SPEC-014 | Review Orchestrator Agent | ⬜ | — | Depends on SPEC-013 |
| SPEC-015 | Gemini Live API Integration | ⬜ | — | Stretch goal |

---

## Phase V — Rendering & Post-Production

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-016 | FFmpeg Command Builder | ⬜ | — | — |
| SPEC-017 | Transition Rendering Tool | ⬜ | — | Depends on SPEC-016 |
| SPEC-018 | Audio Ducking Tool | ⬜ | — | Depends on SPEC-016 |
| SPEC-019 | Whisper Captioning Tool | ⬜ | — | Requires Whisper installed |
| SPEC-020 | Enhancement & Rendering Agent | ⬜ | — | Depends on SPEC-016–019 |

---

## Phase VI — Security & Callbacks

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-021 | before_model_callback — Input Guardrail | ⬜ | — | — |
| SPEC-022 | before_tool_callback — FFmpeg Sandboxing | ⬜ | — | — |
| SPEC-023 | after_tool_callback — Error Recovery | ⬜ | — | — |
| SPEC-024 | after_model_callback — Schema Validation | ⬜ | — | — |

---

## Phase VII — Root Pipeline Assembly & Integration

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-025 | Root Workflow Agent | ⬜ | — | Depends on all prior specs |
| SPEC-026 | End-to-End Integration Test | ⬜ | — | Depends on SPEC-025 |
| SPEC-027 | Configuration & Environment Setup | ⬜ | — | Can be done in parallel |

---

## Phase VIII — Polish & Deployment

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-028 | Logging & Observability | ⬜ | — | — |
| SPEC-029 | Error Handling & Graceful Degradation | ⬜ | — | — |
| SPEC-030 | Documentation & README | ⬜ | — | — |

---

## Summary

| Metric | Count |
|--------|-------|
| **Total Specs** | 30 |
| **Completed** | 10 |
| **In Progress** | 0 |
| **Blocked** | 0 |
| **Not Started** | 20 |

---

## Resolved Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Gemini model version | **`gemini-2.0-flash`** | Best balance of speed, cost, and large context window for iterative video editing workflows. Pro variants are slower and more expensive per token. |
| 2 | Whisper model size | **`medium`** | ~5 GB VRAM but WER drops to ~2.9% (vs ~5.0% for `base`). Quality-focused auto-captions justify the resource cost. |
| 3 | Background music | **User-supplied (MVP)** | User drops an audio track (e.g., `music.mp3`) into the local ingestion folder. The Enhancement & Rendering Agent picks it up and applies FFmpeg `sidechaincompress` for automatic dialogue-aware ducking. |
| 4 | Max video length | **~1 hour combined raw footage** | Gemini 2.0 Flash samples at ~1 fps and its context window supports up to ~1 hour of total video. Keep combined clip duration under this threshold to avoid token overflow. |
| 5 | Deployment target | **Local-only MVP** | Raw clips are downloaded from Google Photos to a local folder. Local ADK gives FFmpeg unthrottled access to media bytes. Cloud deployment would require uploading GBs of footage first, adding severe latency. |

---

## Next Steps

- **SPEC-010 (Reviewer Agent (Algorithmic)) is 100% complete and unit tested.**
- Next Step: Proceed with Phase III: **SPEC-011** (Critic Agent (Human-Centric)).
  - Implement the `CriticAgent` in `agents/critic.py`.
  - System prompt: act as a human viewer, thumbnail strategist, and narrative evaluator.
  - Analyze the EDL for:
    - 30-second hook / Average View Duration (AVD) potential.
    - Click-Through Rate (CTR) and thumbnail frame viability.
    - Emotional resonance and storytelling.
  - Implement loop control actions (`exit_loop` tool to break LoopAgent on pass, or append critical feedback to state on fail).





