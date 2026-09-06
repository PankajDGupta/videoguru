"""Unit and Integration tests for SPEC-002: ADK Project Bootstrap."""

import asyncio
import subprocess
import sys
from pathlib import Path
import pytest

from google.adk.agents import Agent, BaseAgent
from google.adk.apps import App
from google.adk.cli.cli import AgentLoader
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents import app, get_app, get_root_agent, root_agent
from config import settings
from main import OfflineBootstrapAgent, run_agent_turn
from services import (
    SessionManager,
    get_session_manager,
    get_session_service,
    reset_session_service,
)


class TestSessionService:
    """Validate InMemorySessionService integration and SessionManager functionality."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        reset_session_service()
        yield
        reset_session_service()

    def test_singleton_instances(self):
        service1 = get_session_service()
        service2 = get_session_service()
        assert service1 is service2
        assert isinstance(service1, InMemorySessionService)

        manager1 = get_session_manager()
        manager2 = get_session_manager()
        assert manager1 is manager2
        assert isinstance(manager1, SessionManager)

    def test_create_and_get_session_sync(self):
        mgr = SessionManager()
        user_id = "test_user_sync"
        session_id = "sess_sync_001"
        initial_state = {"theme": "mountain hike", "clip_count": 5}

        session = mgr.create_session_sync(
            user_id=user_id,
            session_id=session_id,
            state=initial_state,
        )
        assert session is not None
        assert session.id == session_id
        assert session.state["theme"] == "mountain hike"
        assert session.state["clip_count"] == 5

        fetched = mgr.get_session_sync(user_id=user_id, session_id=session_id)
        assert fetched is not None
        assert fetched.id == session_id
        assert fetched.state["theme"] == "mountain hike"

    def test_create_and_get_session_async(self):
        async def _test():
            mgr = SessionManager()
            user_id = "test_user_async"
            session_id = "sess_async_001"

            session = await mgr.create_session(
                user_id=user_id,
                session_id=session_id,
                state={"status": "initialized"},
            )
            assert session.id == session_id

            fetched_state = await mgr.get_state(user_id=user_id, session_id=session_id)
            assert fetched_state.get("status") == "initialized"

        asyncio.run(_test())

    def test_state_mutation_sync(self):
        mgr = SessionManager()
        user_id = "test_user_mutate"
        session_id = "sess_mutate_001"

        mgr.create_session_sync(user_id=user_id, session_id=session_id, state={"step": 1})
        mgr.update_state_sync(
            user_id=user_id,
            session_id=session_id,
            state_delta={"step": 2, "approved": True},
        )

        state = mgr.get_state_sync(user_id=user_id, session_id=session_id)
        assert state["step"] == 2
        assert state["approved"] is True

    def test_state_mutation_async(self):
        async def _test():
            mgr = SessionManager()
            user_id = "test_user_async_mutate"
            session_id = "sess_async_mutate_001"

            await mgr.create_session(user_id=user_id, session_id=session_id)
            await mgr.update_state(
                user_id=user_id,
                session_id=session_id,
                state_delta={"edl_count": 12},
            )

            state = await mgr.get_state(user_id=user_id, session_id=session_id)
            assert state.get("edl_count") == 12

        asyncio.run(_test())

    def test_list_and_delete_sessions(self):
        mgr = SessionManager()
        user_id = "test_user_list"

        mgr.create_session_sync(user_id=user_id, session_id="s1")
        mgr.create_session_sync(user_id=user_id, session_id="s2")

        sessions = mgr.list_sessions_sync(user_id=user_id)
        session_ids = [s.id for s in sessions]
        assert "s1" in session_ids
        assert "s2" in session_ids

        mgr.delete_session_sync(user_id=user_id, session_id="s1")
        remaining = mgr.list_sessions_sync(user_id=user_id)
        remaining_ids = [s.id for s in remaining]
        assert "s1" not in remaining_ids
        assert "s2" in remaining_ids


class TestRootAgent:
    """Validate root agent configuration, App packaging, and ADK discovery."""

    def test_root_agent_definition(self):
        assert root_agent is not None
        assert isinstance(root_agent, Agent)
        assert root_agent.name == "videoguru"
        assert root_agent.model == settings.GEMINI_MODEL
        assert "VideoGuru" in root_agent.instruction
        assert get_root_agent() is root_agent

    def test_app_definition(self):
        assert app is not None
        assert isinstance(app, App)
        assert app.name == settings.APP_NAME
        assert app.root_agent is root_agent
        assert get_app() is app

    def test_adk_agent_loader_discovery(self):
        """Verify that ADK's native AgentLoader discovers and loads videoguru agent."""
        loader = AgentLoader("agents")
        assert loader._is_single_agent is True
        loaded_agent = loader.load_agent("agents")
        # AgentLoader can return App or BaseAgent
        if isinstance(loaded_agent, App):
            assert loaded_agent.root_agent.name == "videoguru"
        else:
            assert loaded_agent.name == "videoguru"


class TestAgentRunner:
    """Validate end-to-end agent execution with Runner and InMemorySessionService."""

    def test_offline_bootstrap_agent_with_runner(self):
        async def _test():
            session_service = InMemorySessionService()
            user_id = "runner_test_user"
            session_id = "runner_test_session"

            await session_service.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

            agent = OfflineBootstrapAgent(name="videoguru_test")
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=session_service,
            )

            user_content = types.Content(parts=[types.Part.from_text(text="Hello VideoGuru")])
            received_events: list[Event] = []

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
            ):
                received_events.append(event)

            assert len(received_events) >= 1
            response_text = received_events[0].content.parts[0].text
            assert "Hello from VideoGuru" in response_text
            assert "InMemorySessionService and ADK Runner are operational" in response_text

        asyncio.run(_test())

    def test_run_agent_turn_helper(self):
        async def _test():
            responses = await run_agent_turn(
                user_message="Test greeting",
                force_offline=True,
            )
            assert len(responses) >= 1
            assert any("VideoGuru" in r for r in responses)

        asyncio.run(_test())


class TestFastApiApp:
    """Validate ADK Web UI FastAPI app creation."""

    def test_fast_api_app_initialization(self):
        agents_dir = str(settings.BASE_DIR / "agents")
        web_app = get_fast_api_app(
            agents_dir=agents_dir,
            session_service_uri="memory://",
            web=True,
            host=settings.WEB_HOST,
            port=settings.WEB_PORT,
        )
        assert web_app is not None
        assert hasattr(web_app, "routes")
        assert len(web_app.routes) > 0


class TestMainCLI:
    """Validate main.py CLI invocations."""

    def test_main_info_flag(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--info"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(settings.BASE_DIR),
        )
        assert result.returncode == 0
        assert "VideoGuru" in result.stdout
        assert "InMemorySessionService" in result.stdout
        assert any(m in result.stdout for m in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.6-flash"))

    def test_main_run_flag(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--run", "Hello CLI Test", "--offline"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(settings.BASE_DIR),
        )
        assert result.returncode == 0
        assert "VideoGuru: Executing Bootstrap Agent Turn" in result.stdout
        assert "Hello from VideoGuru" in result.stdout
        assert "Turn completed successfully." in result.stdout
