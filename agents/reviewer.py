"""Reviewer Agent (Algorithmic) for VideoGuru (SPEC-010).

Responsible for Phase III Algorithmic Optimization within the autonomous review loop:
- Evaluates the proposed Edit Decision List (EDL) against YouTube recommendation algorithm metrics.
- Analyzes pacing (cut frequency, CPM, cadence variance), hook strength (first 5 seconds),
  and viewer retention curve / drop-off risks.
- Emits pass/fail status, composite algorithmic score (0.0 - 10.0), and structured markdown critique.
- Synchronizes feedback into session state ('reviewer_passed', 'reviewer_score', 'reviewer_feedback', 'algorithmic_review').
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
from schemas.edl import EditDecisionList
from schemas.review import AlgorithmicReviewResult
from tools.intent_tools import get_theme_from_state
from tools.review_tools import (
    evaluate_edl_heuristically,
    get_review_from_state,
    review_edl_algorithmically,
)

logger = logging.getLogger(__name__)

REVIEWER_INSTRUCTION = (
    "You are the Reviewer Agent (Algorithmic) for VideoGuru, an automated AI video production system. "
    "Your sole responsibility is Algorithmic Optimization (Phase III) to maximize YouTube recommendation reach:\n"
    "1. Retrieve the Edit Decision List (EDL) from session state ('edl') and creator theme ('theme').\n"
    "2. Analyze the flow of the clips using the `review_edl_algorithmically` tool for:\n"
    "   - Pacing: Check cut frequency (CPM), cadence rhythm, and attention-reset intervals.\n"
    "   - Hook Strength: Scrutinize the critical first 5 seconds. Demand immediate energy and thematic value.\n"
    "   - Retention Curve: Identify long cuts (>10.0s) or slow passages that risk viewer drop-off.\n"
    "3. Issue a definitive pass/fail determination along with a composite algorithmic score (0.0 to 10.0).\n"
    "4. Record your evaluation in session state under 'reviewer_passed', 'reviewer_score', 'reviewer_feedback', "
    "and 'algorithmic_review'.\n"
    "5. If revisions are required, provide specific, actionable trim and cut recommendations so the editing loop "
    "can adjust timestamps and pacing."
)


class ReviewerAgent(Agent):
    """Reviewer Agent executing Phase III YouTube algorithmic optimization and retention analysis."""

    offline: bool = False

    def __init__(
        self,
        name: str = "reviewer_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Analyzes Edit Decision Lists (EDLs) against YouTube recommendation algorithm metrics: "
            "pacing, hook strength, and retention curves. Emits pass/fail status and algorithmic feedback."
        ),
        instruction: str = REVIEWER_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key
        if tools is None:
            tools = [review_edl_algorithmically]
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
        """Execute algorithmic review turn, supporting both live Gemini and offline heuristics."""
        if self.offline:
            theme = get_theme_from_state(ctx.session.state) or "General Highlights"
            raw_edl = ctx.session.state.get("edl")

            if not raw_edl:
                logger.warning("ReviewerAgent: No EDL found in session state.")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[ReviewerAgent] Cannot perform algorithmic review: No Edit Decision List "
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
                logger.error("ReviewerAgent: Invalid EDL format in state: %s", type(raw_edl))
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=f"[ReviewerAgent] Invalid EDL format in session state: {type(raw_edl).__name__}"
                            )
                        ]
                    ),
                )
                return

            logger.info(
                "OfflineReviewerAgent: Reviewing EDL with %d cut(s) for theme '%s'",
                len(edl),
                theme,
            )

            result = evaluate_edl_heuristically(edl=edl, theme=theme)

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part.from_text(text=result.feedback)]),
                actions=EventActions(
                    state_delta={
                        "reviewer_passed": result.passed,
                        "reviewer_score": result.score,
                        "reviewer_feedback": result.feedback,
                        "algorithmic_review": result.to_dict(),
                    }
                ),
            )
        else:
            # Pre-check: ensure EDL exists before delegating to LLM
            raw_edl = ctx.session.state.get("edl")
            if not raw_edl:
                logger.warning("ReviewerAgent: No EDL found in session state (live mode).")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "[ReviewerAgent] Cannot perform algorithmic review: No Edit Decision List "
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
    ) -> AlgorithmicReviewResult:
        """Programmatic helper to evaluate an EDL and synchronize results to session state.

        Args:
            session: ADK Session instance holding state dictionary.
            edl: Optional explicit EDL. Defaults to session.state['edl'].
            theme: Optional creator theme. Defaults to session.state['theme'].
            offline: Optional offline override flag.

        Returns:
            AlgorithmicReviewResult containing review report.
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
                "ReviewerAgent.run_review requires an EDL either in arguments or in session.state['edl']."
            )

        resolved_theme = theme or session.state.get("theme", "")
        is_offline = offline if offline is not None else self.offline

        result = review_edl_algorithmically(
            edl=resolved_edl,
            theme=resolved_theme,
            offline=is_offline,
        )

        session.state["reviewer_passed"] = result.passed
        session.state["reviewer_score"] = result.score
        session.state["reviewer_feedback"] = result.feedback
        session.state["algorithmic_review"] = result.to_dict()

        return result


def create_reviewer_agent(
    name: str = "reviewer_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> ReviewerAgent:
    """Factory function creating a configured ReviewerAgent instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return ReviewerAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
reviewer_agent = create_reviewer_agent()
