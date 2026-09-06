# VideoGuru — Project Overview

## What Is VideoGuru?

VideoGuru is an **automated, multi-agent video production pipeline** built on the **Google Agent Development Kit (ADK)**. It takes raw video clips downloaded from Google Photos, analyzes them with Gemini multimodal models, autonomously curates and reviews a draft edit, presents it to a human for approval, and renders a final broadcast-quality video complete with smooth transitions, audio ducking, and burned-in captions — all optimized for YouTube engagement.

---

## Goals

| # | Goal | Why It Matters |
|---|------|----------------|
| 1 | **Zero-touch editing** | Eliminate manual NLE work for routine vlogs. |
| 2 | **Algorithmic optimization** | Maximize AVD, CTR, and recommendation-algorithm reach. |
| 3 | **Human-in-the-loop safety** | No video is rendered without explicit creator approval. |
| 4 | **Professional output quality** | FFmpeg-powered xfade transitions, audio ducking, Whisper captions. |
| 5 | **Security** | ADK Callback architecture sandboxes all system-level commands. |

---

## Core Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Google ADK (SequentialAgent, LoopAgent) |
| LLM / Vision | Gemini 2.0 Flash (multimodal video understanding) |
| Structured Output | Pydantic schemas via Gemini Structured Output |
| Timeline Interchange | OpenTimelineIO (.otio) |
| Video Rendering | FFmpeg (xfade transitions, acrossfade, sidechaincompress) |
| Captioning | OpenAI Whisper → .srt → FFmpeg subtitle burn-in |
| Session / State | ADK InMemorySessionService |
| UI | ADK Web UI / Gemini Live API (bidirectional streaming) |

---

## Pipeline Stages (Six-Phase Workflow)

```
┌─────────────────┐
│ 1. Intent Capture│ ── User provides theme / highlight
└───────┬─────────┘
        ▼
┌─────────────────┐
│ 2. Ingestion     │ ── Scan local dir, build clip manifest
└───────┬─────────┘
        ▼
┌─────────────────────────────────────────┐
│ 3. Autonomous Editing Loop (LoopAgent)  │
│   ├── Curation Agent   → EDL draft     │
│   ├── Reviewer Agent   → algo check    │
│   └── Critic Agent     → human-feel    │
│       (loops until exit_loop invoked)   │
└───────┬─────────────────────────────────┘
        ▼
┌─────────────────┐
│ 4. Human Review  │ ── OTIO preview → approve / revise
└───────┬─────────┘
        ▼
┌─────────────────┐
│ 5. Rendering     │ ── FFmpeg xfade, audio ducking, captions
└───────┬─────────┘
        ▼
┌─────────────────┐
│ 6. Final Output  │ ── Broadcast-quality .mp4
└─────────────────┘
```

---

## Agent Roster

| Agent | Type | Responsibility |
|-------|------|---------------|
| **Root Greeter** | Entry | Captures user's theme / highlight intent. |
| **Ingestion Agent** | Sequential | Scans local directory, extracts metadata, builds clip manifest. |
| **Curation Agent** | Loop member | Generates initial Edit Decision List (EDL) via Gemini multimodal analysis. |
| **Reviewer Agent** | Loop member | Evaluates EDL against YouTube algorithm metrics (pacing, hooks, retention). |
| **Critic Agent** | Loop member | Evaluates EDL from a human-viewer perspective (30-sec hook, CTR, thumbnail). |
| **Review Orchestrator** | Sequential | Converts EDL → OTIO, pauses pipeline, presents draft to human for approval. |
| **Enhancement & Rendering Agent** | Sequential | Executes FFmpeg rendering with transitions, audio ducking, and Whisper captions. |

---

## Key Data Structures

### Clip Manifest (Phase I output)

| Field | Type | Purpose |
|-------|------|---------|
| `clip_id` | `str (UUID)` | Unique clip identifier |
| `absolute_path` | `str` | Local file path for FFmpeg |
| `duration_seconds` | `float` | Timeline offset calculation |
| `frame_rate` | `float` | Normalization for concat |
| `resolution` | `str` | Pre-scaling detection |

### Edit Decision List — EDL (Phase II output)

| Field | Type | Purpose |
|-------|------|---------|
| `file_reference` | `str` | Maps to `clip_id` |
| `start_trim` | `float` | Segment start (seconds) |
| `end_trim` | `float` | Segment end (seconds) |
| `scene_rationale` | `str` | Qualitative selection reason |
| `transition_intent` | `str` | Desired xfade type |

---

## Security Model

The ADK **Callback** architecture provides four interception points:

1. **`before_model_callback`** — Input guardrail; blocks prompt injection.
2. **`before_tool_callback`** — Validates FFmpeg args; restricts file paths.
3. **`after_tool_callback`** — Traps tool errors for self-correction.
4. **`after_model_callback`** — Validates EDL JSON schema before downstream use.

---

## Target Platform

- **Primary output:** YouTube-optimized `.mp4`
- **Supported host OS:** Windows (local Google Photos download directory)
- **Deployment:** Local-only MVP (no cloud deployment — avoids uploading GBs of raw footage)
- **Python version:** 3.11+
- **FFmpeg:** System-installed, accessible on PATH

---

## Key Decisions & Constraints

| Decision | Value | Notes |
|----------|-------|-------|
| **Gemini model** | `gemini-2.0-flash` | Best speed/cost/context-window balance for iterative editing |
| **Whisper model** | `medium` (~5 GB VRAM) | WER ~2.9% — quality-focused captions |
| **Background music** | User-supplied | User drops `music.mp3` into the ingestion folder; auto-ducked via `sidechaincompress` |
| **Max input duration** | ~1 hour combined | Gemini 2.0 Flash samples ~1 fps; context window supports ~1 hr of video |
| **Deployment** | Local-only MVP | FFmpeg needs unthrottled local access to raw media bytes |
