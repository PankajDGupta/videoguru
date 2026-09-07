"""Critic Agent (Human-Centric) for VideoGuru (SPEC-011).

Responsible for Phase III Human-Centric Review & Loop Control within the autonomous editing loop:
- Evaluates the proposed Edit Decision List (EDL) from a human viewer perspective.
- Analyzes 30-second hook & Average View Duration (AVD) potential.
- Evaluates Click-Through Rate (CTR) viability and identifies the best candidate thumbnail frame.
- Evaluates narrative arc progression, emotional peaks, and theme alignment.
- Autonomous Loop Control:
  - If pass: invokes exit_loop tool to break the LoopAgent and advance to Phase IV Human Review.
  - If fail: invokes append_to_state with actionable revision feedback for the Curation Agent.
- Synchronizes feedback and scores into session state ('critic_passed', 'critic_score',
  'critic_feedback', 'critic_review', 'candidate_thumbnail').
- Supports both live Gemini 2.0 Flash analysis and deterministic offline heuristic evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Optional, Sequence, Union

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from config import settings
from schemas.critic import CriticReviewResult
from schemas.edl import EditDecisionList
from tools.critic_tools import (
    evaluate_critic_heuristically,
    get_critic_review_from_state,
    review_edl_as_critic,
)
from tools.intent_tools import get_theme_from_state
from tools.loop_tools import append_to_state, exit_loop

logger = logging.getLogger(__name__)

CRITIC_INSTRUCTION = (
    "You are the Critic Agent (Human-Centric) for VideoGuru, an automated AI video production system. "
    "Your responsibility is Human-Centric Quality Assurance and Loop Control (Phase III) within the autonomous editing loop:\n"
    "1. Retrieve the Edit Decision List (EDL) from session state ('edl') and creator theme ('theme').\n"
    "2. Scrutinize the timeline using the `review_edl_as_critic` tool for:\n"
    "   - 30-Second Hook & AVD: Ensure the opening 30 seconds delivers immediate narrative payoff without dead air or slow buildup.\n"
    "   - Thumbnail CTR Viability: Select the most compelling, high-contrast, emotionally engaging visual frame to serve as a thumbnail.\n"
    "   - Emotional Storytelling: Evaluate narrative arc progression, emotional peaks, and human resonance.\n"
    "3. Autonomous Loop Control:\n"
    "   - If pass (score >= 7.0, engaging 30s hook, viable thumbnail): invoke `exit_loop` to break the LoopAgent cycle.\n"
    "   - If fail: invoke `append_to_state` with specific, actionable revision feedback so the Curation Agent can refine cuts.\n"
    "4. Record evaluation in session state under 'critic_passed', 'critic_score', 'critic_feedback', 'critic_review', and 'candidate_thumbnail'."
)


class CriticAgent(Agent):
    """Critic Agent executing Phase III human viewer evaluation, thumbnail analysis, and loop control."""

    offline: bool = False

    def __init__(
        self,
        name: str = "critic_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Evaluates Edit Decision Lists (EDLs) from a human viewer perspective: "
            "30-second hook AVD, thumbnail CTR viability, and emotional storytelling. "
            "Breaks the editing loop via exit_loop on pass or requests revisions via append_to_state."
        ),
        instruction: str = CRITIC_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [review_edl_as_critic, exit_loop, append_to_state]
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
        """Execute human-centric critic turn, supporting both live Gemini and offline heuristics."""
        if self.offline:
            theme = get_theme_from_state(ctx.session.state) or "General Highlights"
            raw_edl = ctx.session.state.get("edl")

            if not raw_edl:
                logger.warning("CriticAgent: No EDL found in session state.")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[CriticAgent] Cannot perform critic review: No Edit Decision List "
                                    "found in session state. Please run Phase II Curation (Curation Agent) first."
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
                logger.error("CriticAgent: Invalid EDL format in state: %s", type(raw_edl))
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=f"[CriticAgent] Invalid EDL format in session state: {type(raw_edl).__name__}"
                            )
                        ]
                    ),
                )
                return

            logger.info(
                "OfflineCriticAgent: Reviewing EDL with %d cut(s) for theme '%s'",
                len(edl),
                theme,
            )

            result = evaluate_critic_heuristically(edl=edl, theme=theme)

            state_delta: dict[str, Any] = {
                "critic_passed": result.passed,
                "critic_score": result.score,
                "critic_feedback": result.feedback,
                "critic_review": result.to_dict(),
                "candidate_thumbnail": result.candidate_thumbnail.model_dump(),
            }

            if result.passed:
                state_delta["loop_status"] = "exited"
                state_delta["exit_loop_invoked"] = True
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(text=result.feedback)]),
                    actions=EventActions(
                        state_delta=state_delta,
                        escalate=True,  # Signal ADK LoopAgent to exit
                    ),
                )
            else:
                # Append to revision_feedback list
                current_feedback = list(ctx.session.state.get("revision_feedback", []))
                current_feedback.append(result.feedback)
                state_delta["revision_feedback"] = current_feedback
                state_delta["loop_status"] = "iterating"

                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(text=result.feedback)]),
                    actions=EventActions(
                        state_delta=state_delta,
                        escalate=False,  # Signal ADK LoopAgent to continue loop
                    ),
                )
        else:
            # Pre-check: ensure EDL exists before delegating to LLM
            raw_edl = ctx.session.state.get("edl")
            if not raw_edl:
                logger.warning("CriticAgent: No EDL found in session state (live mode).")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[CriticAgent] Cannot perform critic review: No Edit Decision List "
                                    "found in session state. The Curation Agent may not have stored the EDL. "
                                    "Please ensure curate_edit_decision_list was called."
                                )
                            )
                        ]
                    ),
                )
                return

            # Live LLM execution via Google ADK
            async for event in super()._run_async_impl(ctx):
                yield event

    def run_review(
        self,
        session: Any,
        edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]]]] = None,
        theme: Optional[str] = None,
        offline: Optional[bool] = None,
    ) -> CriticReviewResult:
        """Programmatic helper to evaluate an EDL and synchronize results to session state.

        Args:
            session: ADK Session instance holding state dictionary.
            edl: Optional explicit EDL. Defaults to session.state['edl'].
            theme: Optional creator theme. Defaults to session.state['theme'].
            offline: Optional offline override flag.

        Returns:
            CriticReviewResult containing evaluation report and thumbnail recommendation.
        """
        # Resolve EDL
        resolved_edl: Optional[EditDecisionList] = None
        if edl is not None:
            if isinstance(edl, EditDecisionList):
                resolved_edl = edl
            elif isinstance(edl, Sequence):
                resolved_edl = EditDecisionList.from_dict_list(edl)
        elif "edl" in session.state:
            raw = session.state["edl"]
            if isinstance(raw, EditDecisionList):
                resolved_edl = raw
            elif isinstance(raw, Sequence):
                resolved_edl = EditDecisionList.from_dict_list(raw)

        if not resolved_edl:
            raise ValueError(
                "CriticAgent.run_review requires an EDL either in arguments or in session.state['edl']."
            )

        resolved_theme = theme or session.state.get("theme", "")
        is_offline = offline if offline is not None else self.offline

        # Create lightweight ToolContext-like container for state synchronization
        class SimpleToolContext:
            def __init__(self, state_dict: dict[str, Any]) -> None:
                self.state = state_dict
                self.actions = type("Actions", (), {"escalate": False, "skip_summarization": False})()

        ctx = SimpleToolContext(session.state)

        result = review_edl_as_critic(
            tool_context=ctx,
            edl=resolved_edl,
            theme=resolved_theme,
            offline=is_offline,
        )

        return result


def create_critic_agent(
    name: str = "critic_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> CriticAgent:
    """Factory function creating a configured CriticAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return CriticAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
critic_agent = create_critic_agent()
