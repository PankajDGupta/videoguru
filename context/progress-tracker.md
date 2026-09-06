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
| SPEC-004 | Local Directory Scanner Tool | ⬜ | — | — |
| SPEC-005 | Clip Metadata Extraction Tool | ⬜ | — | Requires FFmpeg/ffprobe |
| SPEC-006 | Ingestion Agent | ⬜ | — | Depends on SPEC-004, SPEC-005 |

---

## Phase II — Multimodal Video Analysis & EDL Generation

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-007 | Pydantic EDL Schema | ⬜ | — | — |
| SPEC-008 | Gemini Video Analysis Tool | ⬜ | — | Requires Gemini API key |
| SPEC-009 | Curation Agent | ⬜ | — | Depends on SPEC-007, SPEC-008 |

---

## Phase III — Autonomous AI Review Loop

| Spec | Title | Status | Assignee | Notes |
|------|-------|--------|----------|-------|
| SPEC-010 | Reviewer Agent (Algorithmic) | ⬜ | — | — |
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
| **Completed** | 3 |
| **In Progress** | 0 |
| **Blocked** | 0 |
| **Not Started** | 27 |

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

- Proceed with Phase I: **SPEC-004** (Local Directory Scanner Tool).
- Build custom ADK tool `scan_local_directory(directory_path: str)` using `pathlib`.
- Recursively discover `.mp4`, `.mov`, `.avi`, and `.mkv` files and extract initial filesystem metadata.
