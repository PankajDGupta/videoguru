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

---

## Phase IX — Feature Extensions

### SPEC-031: Overlay Text Agent (YouTube Shorts Engagement)

- Implement `OverlayTextAgent` as a new agent in `agents/overlay_text.py`.
- System prompt: Act as a YouTube Shorts viral content strategist.
- Generate short, punchy overlay text for each EDL cut based on the theme:
  - Cut 1 must be a strong HOOK (larger font, emojis, curiosity-driven).
  - Subsequent cuts get scene-specific engaging text (3-8 words max).
- Implement Pydantic schemas in `schemas/overlay_text.py`:
  - `OverlayTextStyle`: Font family, size, color, border, shadow, position, box styling.
  - `OverlayTextEntry`: Per-cut text content, timing, and visual style.
  - `OverlayTextPlan`: Container for all overlay entries.
- Add `build_drawtext_overlay_command()` to `rendering/ffmpeg_builder.py`:
  - Build FFmpeg `drawtext` filter chains with per-entry enable/disable timing.
  - Support vibrant rotating color palette (`YOUTUBE_SHORTS_COLORS`).
  - Position text in upper-third (safe from YouTube Shorts UI).
- Implement tools in `tools/overlay_text_tools.py`:
  - `generate_overlay_text_plan`: ADK tool using Gemini to generate per-cut text.
  - `generate_overlay_texts_with_gemini`: Direct Gemini API text generation.
  - `burn_overlay_text`: FFmpeg drawtext burn-in execution.
  - `_compute_cut_timeline_offsets`: Calculate absolute timing accounting for xfade overlaps.
  - Offline mock mode with deterministic fallback texts.
- Integrate into `EnhancementRenderingAgent.execute_rendering_pipeline()`:
  - New Step 1b between transitions (Step 1) and audio ducking (Step 2).
  - Graceful degradation: rendering continues without overlay text on failure.
  - Store overlay plan in `session.state["overlay_texts"]`.
- Add unit tests in `tests/test_overlay_text.py`.

### SPEC-032: YouTube Shorts Format & Safe-Zone Text Burning (1080x1920)

- Add YouTube Shorts vertical aspect ratio (9:16) and 1080x1920 dimension support:
  - Prevent YouTube algorithmic rejections of horizontal videos as Shorts.
  - `VIDEO_TYPE` (`shorts` / `landscape`) and `TARGET_RESOLUTION` (`1080x1920` / `1920x1080`) in `config/settings.py`.
- Dynamic video normalization in FFmpeg:
  - `render_with_transitions` in `tools/transition_renderer.py` accepts and resolves `target_resolution`.
  - Normalizes clips via `scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2`.
- YouTube Shorts UI Safe-Zone Text Burning:
  - `build_drawtext_overlay_command` in `rendering/ffmpeg_builder.py`:
    - Shift vertical positions out of mobile UI dead zones (`top=h*0.14`, `upper_third=h*0.18`, `lower_third=h*0.62`, `bottom=h*0.65`).
    - Avoid obscuration by YouTube Shorts channel metadata, sound bar, and action icons.
  - `_generate_mock_overlay_texts` in `tools/overlay_text_tools.py`:
    - Scale font sizes for 1080px canvas (`HOOK_FONT_SIZE_SHORTS=46`, `SCENE_FONT_SIZE_SHORTS=36`).
  - `build_caption_burn_command` and `generate_captions` in `tools/whisper_captioning.py`:
    - Elevate vertical subtitle margin (`MarginV=220`, `Alignment=2`) above YouTube Shorts bottom UI.
    - Scale subtitle font size to 24pt for vertical mobile viewing.
- Pipeline & CLI Integration:
  - Pass `target_resolution` throughout `EnhancementRenderingAgent.execute_rendering_pipeline()`.
  - Auto-name output with `final_shorts_{ts}.mp4` when rendering Shorts.
  - Auto-detect Shorts format in `record_theme` from keywords (`short`, `shorts`, `reel`, `vertical`).
  - Add `--shorts` and `--resolution` flags to CLI in `main.py`.
- Comprehensive unit tests in `tests/test_youtube_shorts.py`.

### SPEC-033: Scene-Aware Text Overlays & Subtitle Controls

- Speech-to-text Subtitles Disabled by Default:
  - Set `ENABLE_CAPTIONS = False` by default in `config/settings.py` and `.env.example`.
  - Add CLI flags `--captions` / `--subtitles` and `--no-captions` in `main.py` for explicit opt-in subtitle transcription.
  - Skip Whisper transcription in `EnhancementRenderingAgent.execute_rendering_pipeline()` when `enable_captions=False`, eliminating unwanted ambient speech subtitles and saving rendering time.
- Scene-Aware Engaging Text Overlays:
  - Maintain `ENABLE_OVERLAY_TEXT = True` by default with `--no-overlay-text` opt-out.
  - In `tools/overlay_text_tools.py`, enhance Gemini prompt and schema pipeline to analyze the theme, cut rationales, and clip metadata descriptions to produce engaging, stylish scene titles and insights.
  - Explicitly instruct Gemini not to transcribe background speech or foreign language chatter.
- Windows Subprocess Encoding Hardening:
  - Add `encoding="utf-8", errors="replace"` and null-safe `stderr`/`stdout` handling in `execute_ffmpeg_command` (`rendering/ffmpeg_builder.py`), `tools/clip_metadata.py`, and `tools/video_analysis.py` to eliminate Windows `cp1252` `UnicodeDecodeError`.
- FFmpeg Filtergraph Comma Escaping:
  - Properly escape commas with `\,` in `build_drawtext_overlay_command` for text content and timeline expressions (`gte(t\,{start})*lte(t\,{end})`) to prevent filtergraph parsing errors.
- Unit and integration tests in `tests/test_scene_text_overlays.py`.

### SPEC-034: Google Photos Integration — OAuth2 Video Download

- Implement `services/google_photos_client.py` — a standalone OAuth2 + REST client:
  - `_TokenCache` class persisting `access_token`, `refresh_token`, and `expires_at` in a JSON file (`token.json`).
  - `GooglePhotosClient.from_credentials_file(credentials_json, token_file)` factory loading a Desktop OAuth2 credentials JSON.
  - `authenticate()` method: reuses cached token, silently refreshes via `refresh_token`, or runs full browser OAuth2 consent flow via `http.server.HTTPServer` on `localhost:8080`.
  - `iter_videos_for_date(target_date)` / `iter_videos_for_date_range(start, end)` — paginated `mediaItems:search` with `dateFilter` + `mediaTypeFilter=VIDEO` via Google Photos Library API v1.
  - `download_video(media_item, dest_dir, skip_if_id_cached, id_cache_file)` — streaming download using `baseUrl=dv`, deduplication via `.gphotos_downloaded_ids.json`.
  - `revoke_token()` — revokes the access token and clears the local cache.
- Implement `tools/google_photos_downloader.py` — ADK tool wrapper:
  - `download_from_google_photos(start_date, end_date, dest_dir, credentials_json, token_file)` — resolves defaults, connects client, iterates items, returns `DownloadResult`.
  - `download_videos_from_google_photos(start_date, end_date)` — ADK tool returning structured dict with `success`, `downloaded_count`, `skipped_count`, `failed_count`, `downloaded_paths`, `summary`, `errors`.
- Configuration:
  - `GOOGLE_PHOTOS_CREDENTIALS` and `GOOGLE_PHOTOS_TOKEN_FILE` settings in `config/settings.py`.
  - `.env.example` section documenting Google Cloud Console credential setup steps.
  - `requests>=2.31.0` added to `requirements.txt`.
- CLI (in `main.py`):
  - `--fetch-photos` — download today's videos from Google Photos to `input_videos/`.
  - `--photos-date YYYY-MM-DD` — download for a specific date.
  - `--photos-start YYYY-MM-DD` / `--photos-end YYYY-MM-DD` — custom date range.
  - Composable: `--fetch-photos --pipeline` downloads then immediately runs the editing pipeline.
- Unit tests in `tests/test_google_photos_downloader.py` covering token cache, client factory, auth flow, pagination, download deduplication, ADK tool schema, and CLI argument parsing.

### SPEC-035: Theme-Related Motion Graphics

- Analyze the rendered video and composite animated motion graphics that relate to the theme:
  - New Step 1c in `EnhancementRenderingAgent.execute_rendering_pipeline()` (after overlay text, before audio ducking); analyzes the transition render so graphic timing matches the final timeline.
  - Graceful degradation: any analysis or FFmpeg failure logs a warning and rendering continues without graphics.
- Implement Pydantic schemas in `schemas/motion_graphics.py`:
  - `MotionGraphicType`: `kinetic_title`, `lower_third`, `stat_callout`, `corner_badge`, `progress_bar`.
  - `MotionGraphicElement`: type, start/end time, text, subtext, accent colour, cut index, rationale; validates timing, requires copy for text graphics, strips emoji/backslashes, normalises hex colours.
  - `MotionGraphicsPlan`: elements + Gemini-chosen theme palette (`primary_color`, `secondary_color`) + `content_summary`; `to_dict_list()` / `from_list()` for session state.
- Add `build_motion_graphics_command()` to `rendering/ffmpeg_builder.py`:
  - Pure-FFmpeg animation: `color` sources + `overlay`, and `drawtext` with `expansion=none`, driven by time-based easing expressions (slide in/out, rise, overshoot pop, fades, growing bar).
  - Per-format zones keep Shorts graphics inside the safe area and clear of the overlay text position.
  - Escapes copy safely (colon, apostrophe -> typographic, backslash stripped, no `%` expansion); rejects invalid colours and undrawable elements; caps at 14 elements.
- Implement `tools/motion_graphics_tools.py`:
  - `analyze_video_for_motion_graphics`: uploads a small silent proxy to Gemini, requests structured JSON (`_GeminiMotionPlan`), always deletes the uploaded file, skips malformed graphics.
  - `build_motion_graphics_prompt`: theme, cut timeline with scene rationales, existing overlay text, per-type copy limits, no-emoji / no-invented-facts / no-speech-transcription rules.
  - `sanitize_motion_graphics_plan`: clamps to the video, enforces min/max lifetimes, trims copy at word boundaries, removes same-type and shared-zone overlaps, caps density (progress bar exempt).
  - Deterministic fact-safe fallback plan for offline mode, missing API key or API errors.
  - `burn_motion_graphics` and ADK tool `generate_motion_graphics_plan` (stores `session.state["motion_graphics"]`).
- Configuration and CLI:
  - `ENABLE_MOTION_GRAPHICS` (default `true`) in `config/settings.py` and `.env.example`; `--no-motion-graphics` in `main.py`; `state["enable_motion_graphics"]` and `enable_motion_graphics=` argument on the rendering agent.
- Unit and integration tests (including real FFmpeg renders) in `tests/test_motion_graphics.py`.


### SPEC-036: MCP Server Interface

- Expose VideoGuru to MCP clients (Claude Code, Claude Desktop) over stdio via `mcp_server.py`:
  - Uses the official `mcp` SDK (`MCPServer` on mcp 2.x, `FastMCP` fallback on 1.x).
  - stdout is the transport: logging goes to stderr (`configure_logging`) and every tool body runs with `sys.stdout` redirected to stderr.
- Implement a background `JobManager` for long-running runs:
  - `start_pipeline` validates input (directory, video files, theme, music file), builds the same `initial_state` as `main.py --pipeline`, and runs `RootWorkflowAgent.run_pipeline_async` as an asyncio task with `auto_approve=True`.
  - `get_job_status` / `list_jobs` report status (`running`/`completed`/`failed`), current stage, elapsed time, and result paths (`final_video_path`, `otio_file_path`, `review_status`).
  - `RootWorkflowAgent.run_pipeline_async` gains an optional `on_stage` callback fired on each agent transition.
- Granular tools: `scan_media_directory`, `get_clip_metadata`, `build_clip_manifest`, `apply_audio_ducking`, `generate_captions`, `download_from_google_photos` (returns an "authorization required" error when no OAuth token exists instead of blocking on a browser flow), and `list_outputs`.
- Errors are returned as `{success: false, error}` instead of raised.
- Add `mcp` to `requirements.txt` and `pyproject.toml`; document registration in `README.md` and `.mcp.json.example`.
- Unit and integration tests in `tests/test_mcp_server.py` (tool registration, job lifecycle, stdout cleanliness, error paths, real offline pipeline job on synthetic FFmpeg clips).
