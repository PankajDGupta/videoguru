"""Unit tests for IngestionAgent safety fallback (SPEC-006 / SPEC-025)."""
import asyncio
from pathlib import Path
import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.ingestion import create_ingestion_agent, IngestionAgent
from config import settings


def test_ingestion_agent_fallback_populates_manifest():
    """Verify that IngestionAgent fallback triggers if clip_manifest is not set."""
    async def _test():
        agent = create_ingestion_agent(offline=True)
        svc = InMemorySessionService()
        session_id = "test_fallback_session"
        await svc.create_session(
            app_name=settings.APP_NAME,
            user_id="test_user",
            session_id=session_id,
            state={"media_dir": "input_videos", "theme": "Workout"},
        )
        session = await svc.get_session(app_name=settings.APP_NAME, user_id="test_user", session_id=session_id)
        class MockCtx:
            def __init__(self, s):
                self.session = s
                self.user_content = types.Content(parts=[types.Part.from_text(text="Workout")])
        ctx = MockCtx(session)
        events = []
        async for evt in agent._run_async_impl(ctx):
            events.append(evt)
            if evt.actions and evt.actions.state_delta:
                for k, v in evt.actions.state_delta.items():
                    session.state[k] = v
        assert "clip_manifest" in session.state
        assert len(session.state["clip_manifest"]) >= 1

    asyncio.run(_test())
