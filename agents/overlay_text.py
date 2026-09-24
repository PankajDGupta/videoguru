"""Overlay Text Agent for VideoGuru (SPEC-031).

Generates engaging, theme-aware overlay text for each cut in the EDL
using Gemini multimodal analysis and YouTube Shorts engagement strategies.
Burns the generated text into the rendered video using FFmpeg drawtext filters.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Optional

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from config import settings
from schemas.edl import EditDecisionList
from schemas.overlay_text import OverlayTextPlan
from tools.curation_tools import get_edl_from_state
from tools.overlay_text_tools import (
    generate_overlay_text_plan,
    generate_overlay_texts_with_gemini,
    _generate_mock_overlay_texts,
)

logger = logging.getLogger(__name__)

OVERLAY_TEXT_INSTRUCTION = (
    "You are the Overlay Text Agent for VideoGuru, an automated AI video production system. "
    "Your responsibility is generating engaging YouTube Shorts-style overlay text for each scene cut:\n"
    "1. Read the finalized EDL (Edit Decision List) and the video theme from session state.\n"
    "2. Generate short, punchy overlay text for each cut using the `generate_overlay_text_plan` tool.\n"
    "   - The first cut MUST have a strong HOOK text to stop viewers from scrolling.\n"
    "   - Each text should be 3-8 words with relevant emojis.\n"
    "   - Use vibrant colors that rotate across cuts for visual variety.\n"
    "3. Store the overlay text plan in session.state['overlay_texts'] for the rendering pipeline.\n"
    "4. Provide a summary of the generated overlay texts."
)


class OverlayTextAgent(Agent):
    """Overlay Text Agent generating per-cut engaging text for YouTube Shorts."""

    offline: bool = False

    def __init__(
        self,
        name: str = "overlay_text_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Generates engaging, theme-aware overlay text for each video cut "
            "using Gemini AI, optimized for YouTube Shorts CTR and retention."
        ),
        instruction: str = OVERLAY_TEXT_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [generate_overlay_text_plan]

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

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
        """Execute overlay text generation in offline or live mode."""
        state = ctx.session.state

        try:
            # Resolve EDL from session state
            edl = get_edl_from_state(state)
            if not edl or not edl.entries:
                logger.warning("OverlayTextAgent: No EDL in session state. Skipping overlay text generation.")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part.from_text(
                            text="⚠️ [Overlay Text Agent] No EDL found. Skipping overlay text generation."
                        )]
                    ),
                )
                return

            theme = state.get("theme", "Engaging YouTube Shorts highlights")

            if self.offline:
                # Offline deterministic mode
                plan = _generate_mock_overlay_texts(edl, theme)
            else:
                plan = generate_overlay_texts_with_gemini(edl, theme)

            # Build summary report
            lines = [f"🌟 Generated {len(plan)} overlay texts for: \"{theme[:60]}\""]
            for entry in plan.entries:
                hook_marker = " 🎯 [HOOK]" if entry.is_hook else ""
                lines.append(
                    f"  Cut {entry.cut_index + 1}: \"{entry.text}\" "
                    f"({entry.start_time:.1f}s-{entry.end_time:.1f}s){hook_marker}"
                )

            report = "\n".join(lines)

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(text=report)]),
                actions=EventActions(
                    state_delta={
                        "overlay_texts": plan.to_dict_list(),
                    }
                ),
            )

        except Exception as exc:
            logger.error("OverlayTextAgent execution failed: %s", exc)
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[
                        types.Part.from_text(
                            text=f"⚠️ [Overlay Text Agent] Text generation failed ({exc}). "
                            f"Rendering will continue without overlay text."
                        )
                    ]
                ),
                actions=EventActions(
                    state_delta={
                        "overlay_text_error": str(exc),
                    }
                ),
            )


def create_overlay_text_agent(
    name: str = "overlay_text_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> OverlayTextAgent:
    """Factory function creating a configured OverlayTextAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return OverlayTextAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
overlay_text_agent = create_overlay_text_agent()
