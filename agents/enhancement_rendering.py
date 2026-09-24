"""Enhancement & Rendering Agent for VideoGuru (SPEC-020).

Orchestrates the Phase V rendering pipeline in sequence:
1. Render cuts with visual transitions and audio crossfades (SPEC-017).
2. Apply automatic audio ducking if background music is provided (SPEC-018).
3. Transcribe speech and burn in subtitles using OpenAI Whisper (SPEC-019).
4. Save the final broadcast-ready video to settings.OUTPUT_DIR.
5. Record the final path in session state under `session.state["final_video_path"]`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
from typing import Any, AsyncGenerator, Optional, Sequence
import uuid

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from config import settings
from schemas.edl import EditDecisionList
from tools.audio_ducking import apply_audio_ducking
from tools.curation_tools import get_edl_from_state
from tools.ingestion_tools import get_clip_manifest_from_state
from tools.overlay_text_tools import burn_overlay_text, generate_overlay_texts_with_gemini, _generate_mock_overlay_texts
from tools.transition_renderer import render_with_transitions
from tools.whisper_captioning import generate_captions
from schemas.overlay_text import OverlayTextPlan

logger = logging.getLogger(__name__)

ENHANCEMENT_RENDERING_INSTRUCTION = (
    "You are the Enhancement & Rendering Agent for VideoGuru, an automated AI video production system. "
    "Your responsibility is Phase V High-Fidelity Rendering & Post-Production:\n"
    "1. Render the approved Edit Decision List (EDL) using the `render_with_transitions` tool to apply smooth "
    "   video xfade transitions and audio crossfades.\n"
    "2. If background music is provided, invoke `apply_audio_ducking` using sidechain compression to automatically "
    "   duck background music levels whenever dialogue/vocals occur.\n"
    "3. Invoke `generate_captions` to transcribe dialogue using Whisper and burn clean, high-retention subtitles "
    "   directly into the video frames.\n"
    "4. Ensure the final broadcast-quality .mp4 is stored in the output directory, and save its path into "
    "   session.state['final_video_path'].\n"
    "5. Provide the user with a concise summary of the rendered video, including resolution, duration, and file path."
)


class SimpleToolContext:
    """Lightweight ToolContext adapter for offline tool invocation."""

    def __init__(self, state_dict: dict[str, Any]) -> None:
        self.state = state_dict


class EnhancementRenderingAgent(Agent):
    """Enhancement & Rendering Agent executing Phase V High-Fidelity Rendering."""

    offline: bool = False

    def __init__(
        self,
        name: str = "enhancement_rendering_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Executes post-production rendering for VideoGuru: applies xfade video transitions, "
            "dialogue-aware audio ducking with background music, and Whisper-powered burned-in subtitles."
        ),
        instruction: str = ENHANCEMENT_RENDERING_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [render_with_transitions, apply_audio_ducking, generate_captions]

        super().__init__(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            tools=tools,
            sub_agents=sub_agents or [],
            offline=is_offline,
            **kwargs,
        )

    def execute_rendering_pipeline(
        self,
        state: dict[str, Any],
        music_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        transition_duration: float = 1.0,
    ) -> dict[str, Any]:
        """Execute the end-to-end rendering pipeline deterministically.

        Args:
            state: Session state dictionary containing 'edl' and 'clip_manifest'.
            music_path: Optional path to background music track.
            output_path: Optional explicit final destination path.
            transition_duration: Overlap seconds for xfade transitions.

        Returns:
            Dictionary containing final paths and render metadata:
            {
                "final_video_path": str,
                "transition_video_path": str,
                "ducked_video_path": Optional[str],
                "srt_path": Optional[str],
                "captioned_video_path": str,
                "has_ducking": bool,
                "has_captions": bool,
            }
        """
        tool_ctx = SimpleToolContext(state)

        # 1. Resolve EDL and Clip Manifest
        edl = get_edl_from_state(state)
        manifest = get_clip_manifest_from_state(state)

        if not edl:
            raise ValueError(
                "EnhancementRenderingAgent: Missing Edit Decision List in session state ('edl')."
            )
        if not manifest:
            raise ValueError(
                "EnhancementRenderingAgent: Missing Clip Manifest in session state ('clip_manifest')."
            )

        logger.info(
            "EnhancementRenderingAgent: Starting Phase V render pipeline with %d cuts.",
            len(edl),
        )

        # Sanitize EDL cuts if any cut duration <= transition_duration to safeguard rendering
        manifest_map = {c.clip_id: c for c in manifest}
        sanitized_cuts = []
        edl_modified = False
        for idx, cut in enumerate(edl.entries):
            if cut.duration <= transition_duration:
                clip = manifest_map.get(cut.file_reference)
                clip_dur = clip.duration_seconds if clip else cut.end_trim
                if clip_dur > transition_duration:
                    target_len = min(clip_dur, max(2.5, transition_duration + 1.0))
                    new_start = cut.start_trim
                    new_end = min(clip_dur, new_start + target_len)
                    if new_end - new_start <= transition_duration:
                        new_start = max(0.0, new_end - target_len)
                    logger.warning(
                        "EnhancementRenderingAgent: Cut %d (%s) duration (%.3fs) was <= transition_duration (%.3fs). Auto-extended to [%.2fs, %.2fs] (duration: %.2fs).",
                        idx,
                        cut.file_reference,
                        cut.duration,
                        transition_duration,
                        new_start,
                        new_end,
                        new_end - new_start,
                    )
                    cut = cut.model_copy(update={"start_trim": new_start, "end_trim": new_end})
                    edl_modified = True
            sanitized_cuts.append(cut)

        if edl_modified:
            edl = EditDecisionList(entries=sanitized_cuts)
            if tool_ctx is not None and hasattr(tool_ctx, "state") and tool_ctx.state is not None:
                tool_ctx.state["edl"] = edl

        # 2. Step 1: Render Transitions
        staging_dir = settings.STAGING_DIR / f"render_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        transition_output = staging_dir / "step1_transitions.mp4"

        transition_path = render_with_transitions(
            edl=edl,
            clip_manifest=manifest,
            output_path=transition_output,
            transition_duration=transition_duration,
            tool_context=tool_ctx,
            staging_dir=staging_dir,
        )
        logger.info("EnhancementRenderingAgent: Step 1 (Transitions) completed -> %s", transition_path)

        from services.observability import log_render_progress
        log_render_progress("transitions", progress_percent=33.0, output_path=str(transition_path))

        # 1b. Step 1b: Overlay Text Generation & Burn-in
        overlay_input = transition_path
        has_overlay_text = False

        try:
            # Check for pre-generated overlay texts in state, or generate new ones
            overlay_texts_raw = state.get("overlay_texts")
            if overlay_texts_raw and isinstance(overlay_texts_raw, list) and len(overlay_texts_raw) > 0:
                overlay_plan = OverlayTextPlan.from_list(overlay_texts_raw, theme=state.get("theme", ""))
            else:
                # Generate overlay texts from theme and EDL
                theme = state.get("theme", "")
                if theme and edl:
                    if self.offline:
                        overlay_plan = _generate_mock_overlay_texts(edl, theme, transition_duration)
                    else:
                        overlay_plan = generate_overlay_texts_with_gemini(edl, theme, transition_duration)
                    state["overlay_texts"] = overlay_plan.to_dict_list()
                else:
                    overlay_plan = None

            if overlay_plan and len(overlay_plan) > 0:
                overlay_output = staging_dir / "step1b_overlayed.mp4"
                overlay_result = burn_overlay_text(
                    video_path=transition_path,
                    overlay_plan=overlay_plan,
                    output_path=overlay_output,
                )
                overlay_input = overlay_result
                has_overlay_text = True
                logger.info("EnhancementRenderingAgent: Step 1b (Overlay Text) completed -> %s", overlay_result)
            else:
                logger.info("EnhancementRenderingAgent: No overlay text plan. Skipping overlay step.")
        except Exception as exc:
            logger.warning(
                "EnhancementRenderingAgent: Overlay text burn-in failed (%s). Continuing without overlay text.",
                exc,
            )
            overlay_input = transition_path

        log_render_progress("overlay_text", progress_percent=45.0, output_path=str(overlay_input))

        # 3. Step 2: Audio Ducking (if background music exists)
        resolved_music: Optional[Path] = None
        if music_path is not None:
            resolved_music = Path(music_path).resolve()
        elif state.get("music_path"):
            resolved_music = Path(state["music_path"]).resolve()
        else:
            # Check for standard background music in media_dir or MEDIA_INPUT_DIR
            search_dirs: list[Path] = []
            if state.get("media_dir"):
                search_dirs.append(Path(state["media_dir"]))
            search_dirs.append(settings.MEDIA_INPUT_DIR)

            for sdir in search_dirs:
                for candidate_name in ["music.mp3", "background.mp3", "bgm.mp3", "music.wav"]:
                    candidate = sdir / candidate_name
                    if candidate.is_file():
                        resolved_music = candidate.resolve()
                        break
                if resolved_music:
                    break

        ducked_path = overlay_input
        has_ducking = False

        if resolved_music and resolved_music.is_file():
            logger.info("EnhancementRenderingAgent: Found background music track: %s", resolved_music)
            ducked_output = staging_dir / "step2_ducked.mp4"
            try:
                ducked_path = apply_audio_ducking(
                    video_path=overlay_input,
                    music_path=str(resolved_music),
                    output_path=str(ducked_output),
                    tool_context=tool_ctx,
                )
                has_ducking = True
                logger.info("EnhancementRenderingAgent: Step 2 (Ducking) completed -> %s", ducked_path)
            except Exception as exc:
                logger.warning(
                    "EnhancementRenderingAgent: Audio ducking failed (%s). Continuing with un-ducked audio.",
                    exc,
                )
                ducked_path = overlay_input
        else:
            logger.info("EnhancementRenderingAgent: No background music track provided. Skipping ducking.")
            tool_ctx.state["ducked_video_path"] = overlay_input

        log_render_progress("ducking", progress_percent=66.0, output_path=str(ducked_path), has_ducking=has_ducking)

        # 4. Step 3: Whisper Subtitle Generation & Burn-in
        captioned_path = ducked_path
        srt_path: Optional[str] = None
        has_captions = False

        caption_output = staging_dir / "step3_captioned.mp4"
        srt_output = staging_dir / "subtitles.srt"

        try:
            logger.info("EnhancementRenderingAgent: Transcribing and burning subtitles...")
            cap_result = generate_captions(
                video_path=ducked_path,
                output_srt_path=srt_output,
                output_video_path=caption_output,
                tool_context=tool_ctx,
                offline=self.offline,
            )
            captioned_path = cap_result.get("captioned_video_path", ducked_path)
            srt_path = cap_result.get("srt_path")
            has_captions = True
            logger.info("EnhancementRenderingAgent: Step 3 (Captions) completed -> %s", captioned_path)
        except Exception as exc:
            logger.warning(
                "EnhancementRenderingAgent: Captioning failed or skipped (%s). Continuing with uncaptioned video.",
                exc,
            )
            captioned_path = ducked_path

        log_render_progress("captions", progress_percent=90.0, output_path=str(captioned_path), has_captions=has_captions)

        # 5. Final Output Placement
        if output_path is not None:
            final_target = Path(output_path).resolve()
        else:
            settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            final_target = (settings.OUTPUT_DIR / f"final_vlog_{ts}.mp4").resolve()

        final_target.parent.mkdir(parents=True, exist_ok=True)

        if Path(captioned_path).resolve() != final_target:
            shutil.copy2(captioned_path, final_target)

        # 6. Record state
        state["final_video_path"] = str(final_target)
        state["rendering_complete"] = True

        logger.info("EnhancementRenderingAgent: Final broadcast video saved to %s", final_target)
        log_render_progress("completed", progress_percent=100.0, output_path=str(final_target))

        return {
            "final_video_path": str(final_target),
            "transition_video_path": str(transition_path),
            "overlay_text_video_path": str(overlay_input) if has_overlay_text else None,
            "has_overlay_text": has_overlay_text,
            "ducked_video_path": str(ducked_path) if has_ducking else None,
            "srt_path": str(srt_path) if srt_path else None,
            "captioned_video_path": str(captioned_path),
            "has_ducking": has_ducking,
            "has_captions": has_captions,
        }

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
        """Execute the rendering agent turn across live ADK LLM or offline modes."""
        state = ctx.session.state
        try:
            result = self.execute_rendering_pipeline(state)
            final_path = result["final_video_path"]
            ducking_str = "Applied" if result["has_ducking"] else "None (skipped)"
            captions_str = "Burned-in via Whisper" if result["has_captions"] else "None (skipped)"

            overlay_str = "Applied" if result.get("has_overlay_text", False) else "None (skipped)"
            report = (
                "🎬 [Enhancement & Rendering Agent] Rendering Complete!\n\n"
                f"- Final Video Path: `{final_path}`\n"
                f"- Transitions: Applied xfade / acrossfade\n"
                f"- Overlay Text: {overlay_str}\n"
                f"- Background Audio Ducking: {ducking_str}\n"
                f"- Captions: {captions_str}\n"
                f"- Status: Ready for broadcast / YouTube upload!"
            )

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(text=report)]),
                actions=EventActions(
                    state_delta={
                        "final_video_path": final_path,
                        "rendering_complete": True,
                        "transition_rendered_path": result["transition_video_path"],
                        "ducked_video_path": result["ducked_video_path"],
                        "srt_path": result["srt_path"],
                    }
                ),
            )
        except Exception as exc:
            logger.error("EnhancementRenderingAgent execution failed: %s", exc)
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[
                        types.Part.from_text(
                            text=f"❌ [Enhancement & Rendering Agent] Rendering failed: {exc}"
                        )
                    ]
                ),
                actions=EventActions(
                    state_delta={
                        "rendering_error": str(exc),
                        "rendering_complete": False,
                    }
                ),
            )


def create_enhancement_rendering_agent(
    name: str = "enhancement_rendering_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> EnhancementRenderingAgent:
    """Factory function creating a configured EnhancementRenderingAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return EnhancementRenderingAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
enhancement_rendering_agent = create_enhancement_rendering_agent()
