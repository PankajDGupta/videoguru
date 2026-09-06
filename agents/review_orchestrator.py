"""Review Orchestrator Agent for VideoGuru (SPEC-014).

Responsible for Phase IV Human-in-the-Loop Review:
- Converts the approved Edit Decision List (EDL) to OpenTimelineIO (.otio) format.
- Generates a human-readable summary of the edit.
- Presents the draft to the user and waits for feedback.
- Processes user feedback to either advance to rendering or trigger a revision loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Optional, Sequence

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from config import settings
from schemas.edl import EditDecisionList
from tools.intent_tools import get_theme_from_state
from tools.otio_converter import edl_to_otio
from tools.review_orchestrator_tools import (
    prepare_review_summary,
    process_user_review_response,
)
from tools.curation_tools import get_edl_from_state
from tools.ingestion_tools import get_clip_manifest_from_state

logger = logging.getLogger(__name__)

REVIEW_ORCHESTRATOR_INSTRUCTION = (
    "You are the Review Orchestrator Agent for VideoGuru, an automated AI video production system. "
    "Your responsibility is Phase IV Human-in-the-Loop Review:\n"
    "1. You must convert the approved Edit Decision List (EDL) from session state into OpenTimelineIO (.otio) format "
    "using the `edl_to_otio` tool.\n"
    "2. Generate a comprehensive summary of the edit using the `prepare_review_summary` tool, incorporating the "
    "theme and OTIO file path.\n"
    "3. Present this summary to the user and ask for their feedback.\n"
    "4. When the user responds, use the `process_user_review_response` tool to evaluate their input:\n"
    "   - If they approve ('approve', 'yes', 'looks good'), the tool will set the state to 'approved' so we can advance to rendering.\n"
    "   - If they provide revision feedback, the tool will append it to the session state so the editing loop can restart.\n"
    "5. Ensure the user is fully aware of their options."
)


class ReviewOrchestratorAgent(Agent):
    """Review Orchestrator Agent executing Phase IV Human-in-the-Loop Review."""

    offline: bool = False

    def __init__(
        self,
        name: str = "review_orchestrator_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Presents the approved Edit Decision List to the human user for review. "
            "Converts the timeline to .otio format and processes user approval or revision feedback."
        ),
        instruction: str = REVIEW_ORCHESTRATOR_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [edl_to_otio, prepare_review_summary, process_user_review_response]
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
        """Execute review orchestration turn, supporting live LLM and offline handling."""
        if self.offline:
            # First, determine if we are responding to a user review or presenting the summary
            user_input = ""
            if hasattr(ctx, "new_message") and ctx.new_message and ctx.new_message.parts:
                user_input = ctx.new_message.parts[0].text
            elif hasattr(ctx.invocation_context, "new_message") and ctx.invocation_context.new_message and ctx.invocation_context.new_message.parts:
                 user_input = ctx.invocation_context.new_message.parts[0].text
                 
            state = ctx.session.state
            status = state.get("review_status", "")
            
            # If review is already pending, treat this as user feedback
            if status == "pending" and user_input:
                # Wrap tool context logic
                class SimpleToolContext:
                    def __init__(self, state_dict):
                        self.state = state_dict

                tool_ctx = SimpleToolContext(state)
                response_msg = process_user_review_response(user_input, tool_ctx)
                
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(text=response_msg)]),
                    actions=EventActions(
                        state_delta={
                            "review_approved": state.get("review_approved"),
                            "review_status": state.get("review_status"),
                            "revision_feedback": state.get("revision_feedback", []),
                        }
                    ),
                )
                return

            # Otherwise, present the summary
            theme = get_theme_from_state(state) or "General Highlights"
            raw_edl = state.get("edl")
            manifest = get_clip_manifest_from_state(state)

            if not raw_edl or not manifest:
                logger.warning("ReviewOrchestratorAgent: Missing EDL or clip manifest in state.")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[ReviewOrchestratorAgent] Cannot present review draft: "
                                    "Missing Edit Decision List or Clip Manifest in session state."
                                )
                            )
                        ]
                    ),
                )
                return

            if isinstance(raw_edl, EditDecisionList):
                edl = raw_edl
            elif isinstance(raw_edl, Sequence):
                edl = EditDecisionList.from_dict_list(raw_edl)
            else:
                logger.error("ReviewOrchestratorAgent: Invalid EDL format in state.")
                return

            # Create OTIO file via tool helper in offline mode
            # Mock ToolContext for the tools
            class MockToolContext:
                def __init__(self, s):
                    self.state = s
                    
            tctx = MockToolContext(state)
            try:
                otio_path = edl_to_otio(edl=edl, clip_manifest=manifest, tool_context=tctx)
            except Exception as e:
                logger.error(f"ReviewOrchestratorAgent: Failed to convert EDL to OTIO: {e}")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part.from_text(text=f"[ReviewOrchestratorAgent] OTIO conversion failed: {e}")]
                    ),
                )
                return

            summary_text = prepare_review_summary(edl, theme, otio_path)

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(text=summary_text)]),
                actions=EventActions(
                    state_delta={
                        "otio_file_path": otio_path,
                        "review_status": "pending",
                    }
                ),
            )
        else:
            # Live LLM execution via Google ADK
            async for event in super()._run_async_impl(ctx):
                yield event


def create_review_orchestrator_agent(
    name: str = "review_orchestrator_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> ReviewOrchestratorAgent:
    """Factory function creating a configured ReviewOrchestratorAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return ReviewOrchestratorAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
review_orchestrator_agent = create_review_orchestrator_agent()
