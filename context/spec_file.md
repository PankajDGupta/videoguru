# VideoGuru — Spec File

> Each spec is an individually executable development step.
> Specs are ordered by dependency — complete them top-to-bottom.

---

## Phase 0 — Project Scaffolding

### SPEC-001: Initialize Project Structure

- Create Python project skeleton with `pyproject.toml` / `requirements.txt`.
- Set up directory layout:
  ```
  videoguru/
  ├── agents/          # ADK agent definitions
  ├── tools/           # Custom ADK tools
  ├── schemas/         # Pydantic models
  ├── services/        # Shared services (session, state)
  ├── rendering/       # FFmpeg wrapper modules
  ├── tests/           # Unit & integration tests
  ├── config/          # App configuration
  ├── context/         # Project documentation (this folder)
  └── main.py          # Entry point
  ```
- Install core dependencies: `google-adk`, `google-genai`, `pydantic`, `opentimelineio`, `openai-whisper`.
- Verify FFmpeg is accessible on PATH.

### SPEC-002: ADK Project Bootstrap

- Create the root ADK application entry point (`main.py`).
- Configure `InMemorySessionService` for state management.
- Implement basic ADK web UI launch (`adk web`) for testing.
- Verify a minimal "hello world" agent runs end-to-end.

---

## Phase I — Intent Capture & Media Ingestion

### SPEC-003: Root Greeter Agent

- Implement the `RootGreeterAgent` that prompts the user for the day's theme/highlight.
- Store the user's thematic context in ADK session state under `session.state["theme"]`.
- On receiving input, trigger handoff to the Ingestion Agent.

### SPEC-004: Local Directory Scanner Tool

- Build a custom ADK tool `scan_local_directory(directory_path: str)`.
- Use `pathlib` to recursively find `.mp4`, `.mov`, `.avi`, `.mkv` files.
- Extract basic metadata per file: file name, file size, last modified timestamp.
- Return a list of file path strings for downstream processing.

### SPEC-005: Clip Metadata Extraction Tool

- Build a custom ADK tool `extract_clip_metadata(file_path: str)`.
- Use FFmpeg (`ffprobe`) to extract: duration, frame rate, resolution, codec.
- Generate a UUID-based `clip_id` for each clip.
- Return a structured `ClipManifestEntry` (Pydantic model).

### SPEC-006: Ingestion Agent

- Implement the `IngestionAgent` as a sequential agent.
- Orchestrate `scan_local_directory` → `extract_clip_metadata` (per file).
- Build the full **Clip Manifest** (list of `ClipManifestEntry`).
- Store the manifest in session state: `session.state["clip_manifest"]`.

---

## Phase II — Multimodal Video Analysis & EDL Generation

### SPEC-007: Pydantic EDL Schema

- Define the `EDLEntry` Pydantic model:
  - `file_reference: str` (maps to `clip_id`)
  - `start_trim: float` (≥ 0.0)
  - `end_trim: float` (> `start_trim`)
  - `scene_rationale: str`
  - `transition_intent: str` (enum: `cut`, `fade`, `wipe`, `slide`, `dissolve`)
- Define the `EditDecisionList` model as a list wrapper of `EDLEntry`.
- Add validation rules (end > start, valid clip_id references).

### SPEC-008: Gemini Video Analysis Tool

- Build a custom ADK tool `analyze_clip(clip_path: str, theme: str)`.
- Use Gemini 2.0 Flash multimodal API to analyze the video file.
- Pass structured output schema (`EDLEntry`) to enforce deterministic JSON.
- Return the scored/analyzed clip data.

### SPEC-009: Curation Agent

- Implement the `CurationAgent`.
- Iterate through the clip manifest from session state.
- Call `analyze_clip` for each clip, cross-referencing against the stored theme.
- Assemble the initial **Edit Decision List** (JSON array of `EDLEntry`).
- Store EDL in session state: `session.state["edl"]`.

---

## Phase III — Autonomous AI Review Loop

### SPEC-010: Reviewer Agent (Algorithmic)

- Implement the `ReviewerAgent`.
- System prompt: act as a YouTube algorithm expert.
- Analyze the EDL for:
  - Pacing (cut frequency, energy flow).
  - Hook strength (first 5 seconds).
  - Narrative arc and retention curve.
- Output: feedback string + pass/fail flag.

### SPEC-011: Critic Agent (Human-Centric)

- Implement the `CriticAgent`.
- System prompt: act as a human viewer and thumbnail strategist.
- Analyze the EDL for:
  - 30-second hook / AVD potential.
  - CTR / thumbnail viability.
  - Emotional engagement and storytelling.
- If **pass**: invoke `exit_loop` tool to break the LoopAgent.
- If **fail**: invoke `append_to_state` with specific revision feedback, loop restarts.

### SPEC-012: LoopAgent Assembly

- Wire `CurationAgent`, `ReviewerAgent`, and `CriticAgent` into an ADK `LoopAgent`.
- Configure max iterations (e.g., 5) as a safety circuit-breaker.
- Implement `exit_loop` and `append_to_state` tools.
- Test the loop with mock EDL data to verify convergence.

---

## Phase IV — Human-in-the-Loop Review

### SPEC-013: OpenTimelineIO Converter Tool

- Build a custom ADK tool `edl_to_otio(edl: list, clip_manifest: list)`.
- Use the `opentimelineio` library to:
  - Create a `Timeline` with video and audio `Track` objects.
  - Map each EDL entry to a `Clip` with `TimeRange` from timestamps.
  - Set `media_reference` to the local file path.
- Save the `.otio` file to a staging directory.
- Return the file path.

### SPEC-014: Review Orchestrator Agent

- Implement the `ReviewOrchestratorAgent`.
- Convert the approved EDL to `.otio` using the converter tool.
- Present the draft to the user via ADK Web UI chat.
- Message: summary of the edit + path to the `.otio` file.
- Wait for user response:
  - **"Approve"** → advance to rendering phase.
  - **Feedback text** → append feedback to session state, re-trigger LoopAgent.

### SPEC-015: Gemini Live API Integration (Stretch)

- Implement bidirectional streaming review using `LiveRequestQueue` + `run_live()`.
- Allow voice-based feedback from the user.
- Parse voice input and route to approve/revise flow.
- *(This is a stretch goal; SPEC-014 is the MVP path.)*

---

## Phase V — Rendering & Post-Production

### SPEC-016: FFmpeg Command Builder

- Build a Python module `rendering/ffmpeg_builder.py`.
- Implement functions to construct FFmpeg CLI strings:
  - `build_trim_command(clip_path, start, end, output_path)` — trim individual clips.
  - `build_xfade_chain(clips, transitions, durations)` — chain xfade filters.
  - `build_audio_crossfade(clips)` — parallel `acrossfade` filters.
- All commands use `subprocess.run()` with strict argument validation.

### SPEC-017: Transition Rendering Tool

- Build a custom ADK tool `render_with_transitions(edl, clip_manifest)`.
- Parse the approved EDL for `transition_intent` values.
- Map intents to FFmpeg xfade parameters:
  - `fade` → `xfade=transition=fade`
  - `wipe` → `xfade=transition=wipeleft`
  - `slide` → `xfade=transition=slideleft`
  - `dissolve` → `xfade=transition=dissolve`
- Calculate cumulative offsets and durations dynamically.
- Execute the full FFmpeg filter_complex pipeline.

### SPEC-018: Audio Ducking Tool

- Build a custom ADK tool `apply_audio_ducking(video_path, music_path)`.
- Use FFmpeg `sidechaincompress` filter to duck music under speech.
- Parameters: threshold, ratio, attack, release (configurable).
- Output: video with ducked audio mix.

### SPEC-019: Whisper Captioning Tool

- Build a custom ADK tool `generate_captions(video_path)`.
- Run OpenAI Whisper on the rendered video to produce `.srt` file.
- Auto-detect language.
- Use FFmpeg `subtitles` filter to burn `.srt` into the final video.

### SPEC-020: Enhancement & Rendering Agent

- Implement the `EnhancementRenderingAgent`.
- Orchestrate the rendering pipeline in order:
  1. Trim clips per EDL.
  2. Apply xfade transitions.
  3. Apply audio ducking (if background music provided).
  4. Generate and burn-in Whisper captions.
- Save final output as `.mp4` in the designated output directory.
- Store final path in session state: `session.state["final_video_path"]`.

---

## Phase VI — Security & Callbacks

### SPEC-021: before_model_callback — Input Guardrail

- Implement callback to sanitize user prompts.
- Block prompt injection attempts (e.g., attempts to reference external URLs or override agent instructions).
- Log blocked attempts.

### SPEC-022: before_tool_callback — FFmpeg Sandboxing

- Implement callback to validate all FFmpeg command arguments before execution.
- Whitelist permitted flags and binaries (`ffmpeg`, `ffprobe`).
- Restrict file paths to the designated media and staging directories only.
- Block any command attempting shell injection or path traversal.

### SPEC-023: after_tool_callback — Error Recovery

- Implement callback to intercept FFmpeg/tool errors.
- Parse stderr for common failure patterns.
- Format error into LLM-friendly feedback for self-correction.
- Retry logic with backoff (max 3 retries).

### SPEC-024: after_model_callback — Schema Validation

- Implement callback to validate all model-generated JSON.
- Verify EDL entries conform to `EDLEntry` Pydantic schema.
- Reject malformed responses and request re-generation.

---

## Phase VII — Root Pipeline Assembly & Integration

### SPEC-025: Root Workflow Agent

- Wire all agents into the root `SequentialAgent`:
  1. `RootGreeterAgent`
  2. `IngestionAgent`
  3. `LoopAgent` (Curation → Reviewer → Critic)
  4. `ReviewOrchestratorAgent`
  5. `EnhancementRenderingAgent`
- Register all four security callbacks.
- Configure session state passthrough across agent boundaries.

### SPEC-026: End-to-End Integration Test

- Create a test suite with sample video clips (short, ~5 sec each).
- Run the full pipeline from intent capture → final render.
- Verify:
  - Clip manifest is correctly built.
  - EDL is valid JSON conforming to schema.
  - OTIO file is generated and importable.
  - Final `.mp4` plays correctly with transitions and captions.
  - Security callbacks fire correctly.

### SPEC-027: Configuration & Environment Setup

- Create `config/settings.py` with configurable parameters:
  - `MEDIA_INPUT_DIR` — local Google Photos directory.
  - `STAGING_DIR` — intermediate files.
  - `OUTPUT_DIR` — final rendered videos.
  - `MAX_LOOP_ITERATIONS` — safety limit for LoopAgent.
  - `WHISPER_MODEL` — Whisper model size (tiny/base/small/medium/large).
  - `GEMINI_MODEL` — Gemini model identifier.
- Support `.env` file for API keys (Gemini API key, etc.).

---

## Phase VIII — Polish & Deployment

### SPEC-028: Logging & Observability

- Add structured logging throughout all agents and tools.
- Log agent transitions, tool invocations, EDL iterations, and render progress.
- Use Python `logging` with JSON formatter.

### SPEC-029: Error Handling & Graceful Degradation

- Add try/except wrappers around all external calls (Gemini API, FFmpeg, Whisper).
- Implement graceful fallbacks (e.g., skip captions if Whisper fails).
- Surface user-friendly error messages via ADK Web UI.

### SPEC-030: Documentation & README

- Write comprehensive `README.md` with:
  - Project description.
  - Setup instructions (Python, FFmpeg, API keys).
  - Usage guide (how to run the pipeline).
  - Architecture diagram.
- Add inline docstrings to all agents, tools, and schemas.
