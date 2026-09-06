"""Root Workflow Agent for VideoGuru (SPEC-025).

Assembles the full 5-stage autonomous video production pipeline into an ADK SequentialAgent:
1. RootGreeterAgent: Intent capture and theme definition (Phase I, SPEC-003)
2. IngestionAgent: Local media discovery and clip manifest generation (Phase I, SPEC-006)
3. LoopAgent: Autonomous editing loop (Curation -> Reviewer -> Critic) (Phase III, SPEC-012)
4. ReviewOrchestratorAgent: OpenTimelineIO export and draft review (Phase IV, SPEC-014)
5. EnhancementRenderingAgent: Transitions, audio ducking, Whisper captions (Phase V, SPEC-020)

Registers all four Phase VI security and validation callbacks across all stages.
Manages session state passthrough across agent boundaries:
theme -> clip_manifest -> edl -> review_status / otio_file_path -> final_video_path.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import uuid
from typing import Any, Optional, Sequence

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.enhancement_rendering import (
    EnhancementRenderingAgent,
    create_enhancement_rendering_agent,
)
from agents.ingestion import (
    IngestionAgent,
    create_ingestion_agent,
)
from agents.loop_agent import (
    create_loop_agent,
)
from agents.review_orchestrator import (
    ReviewOrchestratorAgent,
    create_review_orchestrator_agent,
)
from agents.root_greeter import (
    RootGreeterAgent,
    create_root_greeter_agent,
)
from callbacks import register_security_callbacks
from config import settings
from services import get_session_service

logger = logging.getLogger(__name__)

ROOT_PIPELINE_DESCRIPTION = (
    "VideoGuru Root Workflow Pipeline: End-to-end autonomous video production digital assembly line. "
    "Orchestrates Root Greeter (theme capture) -> Ingestion Agent (clip scanning & metadata extraction) -> "
    "Loop Agent (curation, algorithmic review, and critic iteration) -> Review Orchestrator Agent "
    "(OpenTimelineIO export and draft review) -> Enhancement & Rendering Agent (xfade transitions, "
    "audio ducking, Whisper captions, and broadcast-ready output rendering)."
)


class RootWorkflowAgent(SequentialAgent):
    """Root Workflow Agent orchestrating the 5-stage VideoGuru video production pipeline.

    Subclasses ADK SequentialAgent to enforce strict sequential execution across:
    1. RootGreeterAgent
    2. IngestionAgent
    3. LoopAgent (Curation -> Reviewer -> Critic)
    4. ReviewOrchestratorAgent
    5. EnhancementRenderingAgent
    """

    offline: bool = False

    def __init__(
        self,
        name: str = "videoguru_root_pipeline",
        sub_agents: Optional[Sequence[BaseAgent]] = None,
        description: str = ROOT_PIPELINE_DESCRIPTION,
        offline: bool = False,
        enable_security_callbacks: bool = True,
        **kwargs: Any,
    ) -> None:
        has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        is_offline = offline if offline else not has_api_key

        if sub_agents is None:
            greeter = create_root_greeter_agent(offline=is_offline)
            ingestion = create_ingestion_agent(offline=is_offline)
            loop = create_loop_agent(offline=is_offline)
            review = create_review_orchestrator_agent(offline=is_offline)
            enhancement = create_enhancement_rendering_agent(offline=is_offline)
            sub_agents = [greeter, ingestion, loop, review, enhancement]

        if enable_security_callbacks:
            for sub in sub_agents:
                register_security_callbacks(sub)

        super().__init__(
            name=name,
            sub_agents=list(sub_agents),
            description=description,
            **kwargs,
        )
        self.offline = is_offline

        if enable_security_callbacks:
            register_security_callbacks(self)

    async def run_pipeline_async(
        self,
        user_prompt: str = "Create a broadcast-quality travel vlog",
        user_id: str = settings.DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        initial_state: Optional[dict[str, Any]] = None,
        session_service: Optional[InMemorySessionService] = None,
        auto_approve: bool = True,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously execute the full 5-stage root workflow pipeline using ADK Runner.

        Args:
            user_prompt: Initial creator prompt / theme description.
            user_id: User identifier for session isolation.
            session_id: Optional session ID (auto-generated if None).
            initial_state: Optional pre-seeded session state dict.
            session_service: Optional InMemorySessionService instance.
            auto_approve: If True, flags the review orchestrator stage to auto-approve.
            raise_on_error: If True, re-raises any caught exception.

        Returns:
            Dictionary containing pipeline execution results, state, and event logs:
            {
                "success": bool,
                "session_id": str,
                "user_id": str,
                "theme": Optional[str],
                "clip_manifest": Optional[list],
                "edl": Optional[list],
                "otio_file_path": Optional[str],
                "review_status": Optional[str],
                "final_video_path": Optional[str],
                "rendering_complete": bool,
                "events": list[dict[str, Any]],
                "session_state": dict[str, Any],
                "error": Optional[str],
            }
        """
        sid = session_id or f"root_pipeline_{uuid.uuid4().hex[:8]}"
        svc = session_service or get_session_service()

        state_to_set: dict[str, Any] = dict(initial_state or {})
        if auto_approve and "auto_approve" not in state_to_set:
            state_to_set["auto_approve"] = True

        # Ensure session exists with initial state
        session = await svc.get_session(app_name=settings.APP_NAME, user_id=user_id, session_id=sid)
        if session is None:
            session = await svc.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=sid,
                state=state_to_set,
            )
        else:
            for k, v in state_to_set.items():
                session.state[k] = v

        runner = Runner(
            app_name=settings.APP_NAME,
            agent=self,
            session_service=svc,
        )

        user_content = types.Content(parts=[types.Part.from_text(text=user_prompt)])
        collected_events: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        error_msg: Optional[str] = None

        logger.info(
            "RootWorkflowAgent: Starting pipeline execution for session '%s' (user: '%s', offline: %s)",
            sid,
            user_id,
            self.offline,
        )

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=sid,
                new_message=user_content,
            ):
                event_data: dict[str, Any] = {
                    "author": event.author,
                    "text": None,
                }
                if event.content and event.content.parts:
                    texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                    if texts:
                        event_data["text"] = " ".join(texts)
                collected_events.append(event_data)

            # Retrieve final session state
            refreshed_session = await svc.get_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=sid,
            )
            if refreshed_session:
                final_state = dict(refreshed_session.state)

            success = True
            logger.info("RootWorkflowAgent: Pipeline execution finished successfully for session '%s'", sid)

        except Exception as exc:
            logger.error("RootWorkflowAgent: Pipeline execution failed for session '%s': %s", sid, exc)
            error_msg = str(exc)
            success = False

            # Capture whatever partial state exists
            refreshed_session = await svc.get_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=sid,
            )
            if refreshed_session:
                final_state = dict(refreshed_session.state)

            if raise_on_error:
                raise

        return {
            "success": success,
            "session_id": sid,
            "user_id": user_id,
            "theme": final_state.get("theme"),
            "clip_manifest": final_state.get("clip_manifest"),
            "edl": final_state.get("edl"),
            "otio_file_path": final_state.get("otio_file_path"),
            "review_status": final_state.get("review_status"),
            "final_video_path": final_state.get("final_video_path"),
            "rendering_complete": bool(final_state.get("rendering_complete", False)),
            "events": collected_events,
            "session_state": final_state,
            "error": error_msg,
        }

    def execute_pipeline(
        self,
        user_prompt: str = "Create a broadcast-quality travel vlog",
        user_id: str = settings.DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        initial_state: Optional[dict[str, Any]] = None,
        session_service: Optional[InMemorySessionService] = None,
        auto_approve: bool = True,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Synchronously execute the full 5-stage root workflow pipeline.

        Handles both standard synchronous contexts and execution within an active
        event loop (e.g., interactive shells or notebook runners).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.run_pipeline_async(
            user_prompt=user_prompt,
            user_id=user_id,
            session_id=session_id,
            initial_state=initial_state,
            session_service=session_service,
            auto_approve=auto_approve,
            raise_on_error=raise_on_error,
        )

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)


def create_root_workflow_agent(
    name: str = "videoguru_root_pipeline",
    offline: bool = False,
    enable_security_callbacks: bool = True,
    sub_agents: Optional[Sequence[BaseAgent]] = None,
    **kwargs: Any,
) -> RootWorkflowAgent:
    """Factory function creating a configured RootWorkflowAgent instance.

    Args:
        name: Name identifier for the sequential agent.
        offline: If True, forces all constituent sub-agents to operate in offline heuristic mode.
        enable_security_callbacks: If True, attaches all four Phase VI security callbacks.
        sub_agents: Optional explicit list of sub-agents to override default 5-stage setup.
        **kwargs: Additional parameters forwarded to RootWorkflowAgent / SequentialAgent.

    Returns:
        An initialized RootWorkflowAgent instance.
    """
    return RootWorkflowAgent(
        name=name,
        offline=offline,
        enable_security_callbacks=enable_security_callbacks,
        sub_agents=sub_agents,
        **kwargs,
    )


def execute_pipeline(
    agent: Optional[RootWorkflowAgent] = None,
    user_prompt: str = "Create a broadcast-quality travel vlog",
    user_id: str = settings.DEFAULT_USER_ID,
    session_id: Optional[str] = None,
    initial_state: Optional[dict[str, Any]] = None,
    session_service: Optional[InMemorySessionService] = None,
    auto_approve: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Helper function to execute the root workflow pipeline synchronously.

    Args:
        agent: RootWorkflowAgent instance (defaults to singleton root_workflow_agent).
        user_prompt: Initial creator prompt / theme description.
        user_id: User identifier.
        session_id: Optional unique session ID.
        initial_state: Optional initial session state dict.
        session_service: Optional InMemorySessionService instance.
        auto_approve: Whether to auto-approve review draft for end-to-end completion.
        raise_on_error: Whether to re-raise any exception.

    Returns:
        Pipeline execution result dictionary.
    """
    target = agent or root_workflow_agent
    return target.execute_pipeline(
        user_prompt=user_prompt,
        user_id=user_id,
        session_id=session_id,
        initial_state=initial_state,
        session_service=session_service,
        auto_approve=auto_approve,
        raise_on_error=raise_on_error,
    )


async def run_pipeline_async(
    agent: Optional[RootWorkflowAgent] = None,
    user_prompt: str = "Create a broadcast-quality travel vlog",
    user_id: str = settings.DEFAULT_USER_ID,
    session_id: Optional[str] = None,
    initial_state: Optional[dict[str, Any]] = None,
    session_service: Optional[InMemorySessionService] = None,
    auto_approve: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Helper function to execute the root workflow pipeline asynchronously."""
    target = agent or root_workflow_agent
    return await target.run_pipeline_async(
        user_prompt=user_prompt,
        user_id=user_id,
        session_id=session_id,
        initial_state=initial_state,
        session_service=session_service,
        auto_approve=auto_approve,
        raise_on_error=raise_on_error,
    )


# Default singleton instance
root_workflow_agent = create_root_workflow_agent()
