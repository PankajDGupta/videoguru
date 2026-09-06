"""Curation and Narrative Agent for VideoGuru (SPEC-009).

Responsible for Phase II Multimodal Video Analysis and Initial Scene Selection:
- Iterates through the Clip Manifest stored in session state (`session.state["clip_manifest"]`).
- Evaluates raw footage against the creator's theme (`session.state["theme"]`) using multimodal analysis.
- Assembles the initial Edit Decision List (EDL) of type-safe EDLEntry cuts.
- Stores the approved EDL in session state under `session.state["edl"]`.
- Supports editorial revision feedback when invoked within Phase III's autonomous LoopAgent.
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
from tools.curation_tools import (
    assemble_edl_from_manifest,
    curate_edit_decision_list,
    get_edl_from_state,
)
from tools.ingestion_tools import get_clip_manifest_from_state
from tools.intent_tools import get_theme_from_state
from tools.video_analysis import analyze_clip

logger = logging.getLogger(__name__)

CURATION_INSTRUCTION = (
    "You are the Curation and Narrative Agent for VideoGuru, an automated AI video production system. "
    "Your primary responsibility is Multimodal Video Analysis and Initial Scene Selection (Phase II):\n"
    "1. Retrieve the Clip Manifest from session state ('clip_manifest') and the creator's theme ('theme').\n"
    "2. For each raw video clip in the manifest, invoke the `analyze_clip` tool to analyze the video frames and audio "
    "against the creator's stated theme, or call `curate_edit_decision_list` to assemble the entire cut list.\n"
    "3. Select the most visually engaging, continuous moments that highlight the theme, eliminating dead air and low-energy footage.\n"
    "4. Assemble the initial Edit Decision List (EDL) conforming strictly to the EDLEntry schema, setting start_trim, "
    "end_trim, scene_rationale, transition_intent, and engagement_score.\n"
    "5. Save the assembled EDL in session state under 'edl' and summarize the curated timeline (clip count, total duration, "
    "average engagement score) for downstream algorithmic review (Phase III)."
)


class CurationAgent(Agent):
    """Curation and Narrative Agent executing Phase II video curation and EDL assembly."""

    offline: bool = False

    def __init__(
        self,
        name: str = "curation_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Analyzes raw video footage against the creator's theme via Gemini multimodal vision, "
            "selects high-engagement segments, assembles the Edit Decision List (EDL), "
            "and stores it in session state under 'edl'."
        ),
        instruction: str = CURATION_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [curate_edit_decision_list, analyze_clip]
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

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        """Execute the curation turn, supporting both live Gemini LLM and deterministic offline mode."""
        if self.offline:
            theme = get_theme_from_state(ctx.session.state) or "General highlights and engaging moments"
            manifest = get_clip_manifest_from_state(ctx.session.state)
            feedback = ctx.session.state.get("critic_feedback") or ctx.session.state.get("revision_feedback")

            if not manifest:
                logger.warning("CurationAgent: No clip manifest available in session state.")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[CurationAgent] Cannot curate video: No clip manifest found in session state. "
                                    "Please run Phase I Media Ingestion (Ingestion Agent) before curating cuts."
                                )
                            )
                        ]
                    ),
                )
                return

            logger.info(
                "OfflineCurationAgent: Curating %d clip(s) for theme: '%s' (feedback: %s)",
                len(manifest),
                theme,
                feedback or "none",
            )

            edl = assemble_edl_from_manifest(
                manifest=manifest,
                theme=theme,
                offline=True,
                feedback=str(feedback) if feedback else None,
            )

            serialized_edl = edl.to_dict_list()
            minutes = int(edl.total_duration // 60)
            seconds = int(edl.total_duration % 60)
            formatted_duration = f"{minutes}m {seconds}s ({edl.total_duration:.2f}s)"

            scores = [e.engagement_score for e in edl if e.engagement_score is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            summary_text = (
                f"[CurationAgent] Successfully assembled initial Edit Decision List (EDL) with {len(edl)} cut(s).\n"
                f"Theme: {theme}\n"
                f"Total Duration: {formatted_duration} | Average Engagement: {avg_score:.1f}/10.0\n"
                f"EDL locked into session state under key 'edl' for autonomous review (Phase III)."
            )

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(text=summary_text)]),
                actions=EventActions(
                    state_delta={
                        "edl": serialized_edl,
                        "edl_total_duration": edl.total_duration,
                    }
                ),
            )
        else:
            # Live LLM execution via Google ADK
            async for event in super()._run_async_impl(ctx):
                yield event


def create_curation_agent(
    name: str = "curation_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> CurationAgent:
    """Factory function creating a configured CurationAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return CurationAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
curation_agent = create_curation_agent()
