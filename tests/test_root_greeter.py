"""Unit and Integration tests for SPEC-003: Root Greeter Agent (Intent Capture)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from unittest.mock import MagicMock
import pytest

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from agents import (
    ROOT_GREETER_INSTRUCTION,
    RootGreeterAgent,
    create_root_greeter_agent,
    get_root_agent,
    ingestion_agent,
    root_agent,
    root_greeter_agent,
)
from config import settings
from schemas import ThemeIntent
from services import (
    SessionManager,
    get_session_manager,
    get_session_service,
    reset_session_service,
)
from tools import get_theme_from_state, record_theme


class TestThemeSchema:
    """Validate ThemeIntent Pydantic model for theme capture."""

    def test_valid_theme_intent(self):
        intent = ThemeIntent(theme="Surfing in Hawaii", notes="Use tropical music")
        assert intent.theme == "Surfing in Hawaii"
        assert intent.notes == "Use tropical music"

    def test_theme_strips_whitespace(self):
        intent = ThemeIntent(theme="   Mountain Climbing in Alps   ")
        assert intent.theme == "Mountain Climbing in Alps"

    def test_empty_theme_rejected(self):
        with pytest.raises(ValueError, match="Theme cannot be empty"):
            ThemeIntent(theme="   ")

        with pytest.raises(ValueError):
            ThemeIntent(theme="")


class TestRecordThemeTool:
    """Validate the custom record_theme tool and state/action mutations."""

    def test_record_theme_success(self):
        actions = EventActions()
        mock_invocation = MagicMock()
        mock_invocation.session.state = {}

        tool_context = ToolContext(invocation_context=mock_invocation, event_actions=actions)

        response = record_theme("Scuba diving in Great Barrier Reef", tool_context)

        assert "Scuba diving in Great Barrier Reef" in response
        assert "ingestion_agent" in response
        assert tool_context.state["theme"] == "Scuba diving in Great Barrier Reef"
        assert actions.state_delta.get("theme") == "Scuba diving in Great Barrier Reef"
        assert actions.transfer_to_agent == "ingestion_agent"

    def test_record_theme_empty_string(self):
        actions = EventActions()
        mock_invocation = MagicMock()
        mock_invocation.session.state = {}

        tool_context = ToolContext(invocation_context=mock_invocation, event_actions=actions)

        response = record_theme("    ", tool_context)

        assert "Theme cannot be empty" in response
        assert "theme" not in tool_context.state
        assert actions.transfer_to_agent is None

    def test_get_theme_from_state(self):
        state = {"theme": "Tokyo Night Walk", "clip_count": 4}
        assert get_theme_from_state(state) == "Tokyo Night Walk"
        assert get_theme_from_state({}) is None


class TestRootGreeterAgentDefinition:
    """Validate agent configuration, tools, sub-agents, and inheritance."""

    def test_agent_attributes(self):
        agent = create_root_greeter_agent()
        assert isinstance(agent, Agent)
        assert agent.name == "root_greeter"
        assert agent.model == settings.GEMINI_MODEL
        assert agent.instruction == ROOT_GREETER_INSTRUCTION
        assert record_theme in agent.tools
        assert len(agent.sub_agents) == 1
        assert agent.sub_agents[0].name == "ingestion_agent"

    def test_root_agent_wiring(self):
        assert root_agent is not None
        assert isinstance(root_agent, RootGreeterAgent)
        assert root_agent.name == "videoguru"
        assert get_root_agent() is root_agent
        assert record_theme in root_agent.tools
        assert any(sa.name == "ingestion_agent" for sa in root_agent.sub_agents)


class TestIntentCaptureTurnWithRunner:
    """Validate full multi-turn conversational intent capture and handoff with Runner."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        reset_session_service()
        yield
        reset_session_service()

    def test_multi_turn_greeting_then_theme(self):
        async def _test():
            service = InMemorySessionService()
            user_id = "creator_01"
            session_id = "sess_intent_01"

            await service.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

            agent = create_root_greeter_agent(name="root_greeter", offline=True)
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=service,
            )

            # Turn 1: Creator greets agent without theme
            turn1_events: list[Event] = []
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Hello VideoGuru!")]),
            ):
                turn1_events.append(ev)

            assert len(turn1_events) >= 1
            turn1_text = turn1_events[0].content.parts[0].text
            assert "Welcome to VideoGuru" in turn1_text
            assert "theme or main highlight" in turn1_text

            sess_turn1 = await service.get_session(app_name=settings.APP_NAME, user_id=user_id, session_id=session_id)
            assert "theme" not in sess_turn1.state

            # Turn 2: Creator provides theme
            turn2_events: list[Event] = []
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Road trip across Arizona desert")]),
            ):
                turn2_events.append(ev)

            # Expect events from root_greeter confirming theme AND handoff to ingestion_agent
            response_texts = [
                e.content.parts[0].text
                for e in turn2_events
                if e.content and e.content.parts and getattr(e.content.parts[0], "text", None)
            ]

            assert any("Road trip across Arizona desert" in t for t in response_texts)
            assert any("[IngestionAgent] Ready!" in t for t in response_texts)

            # Verify session state has theme locked in
            sess_turn2 = await service.get_session(app_name=settings.APP_NAME, user_id=user_id, session_id=session_id)
            assert sess_turn2.state["theme"] == "Road trip across Arizona desert"

        asyncio.run(_test())

    def test_single_turn_with_theme_upfront(self):
        async def _test():
            service = InMemorySessionService()
            user_id = "creator_02"
            session_id = "sess_intent_02"

            await service.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

            agent = create_root_greeter_agent(name="root_greeter", offline=True)
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=service,
            )

            # Single turn with theme provided right away
            events: list[Event] = []
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Family reunion barbecue in Texas")]),
            ):
                events.append(ev)

            response_texts = [
                e.content.parts[0].text
                for e in events
                if e.content and e.content.parts and getattr(e.content.parts[0], "text", None)
            ]

            assert any("Family reunion barbecue in Texas" in t for t in response_texts)
            assert any("[IngestionAgent]" in t for t in response_texts)

            session = await service.get_session(app_name=settings.APP_NAME, user_id=user_id, session_id=session_id)
            assert session.state["theme"] == "Family reunion barbecue in Texas"

        asyncio.run(_test())


class TestMainCLIThemeFlag:
    """Validate main.py CLI invocations with --theme."""

    def test_main_theme_flag(self):
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--theme",
                "Photography walk in Central Park",
                "--offline",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(settings.BASE_DIR),
        )
        assert result.returncode == 0
        assert "Photography walk in Central Park" in result.stdout
        assert "Session State Theme: 'Photography walk in Central Park'" in result.stdout
        assert "Turn completed successfully." in result.stdout

    def test_main_run_greeting_flag(self):
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--run",
                "Hello VideoGuru",
                "--offline",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(settings.BASE_DIR),
        )
        assert result.returncode == 0
        assert "Welcome to VideoGuru" in result.stdout
        assert "Session State Theme: None (Awaiting creator input)" in result.stdout
        assert "Turn completed successfully." in result.stdout
