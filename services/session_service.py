"""Session Service implementation for VideoGuru using Google ADK InMemorySessionService."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from google.adk.sessions import InMemorySessionService, Session

from config import settings

_SESSION_SERVICE_INSTANCE: Optional[InMemorySessionService] = None
_SESSION_MANAGER_INSTANCE: Optional[SessionManager] = None


class SessionManager:
    """Manages session lifecycle and state for VideoGuru agents."""

    def __init__(
        self,
        session_service: Optional[InMemorySessionService] = None,
        app_name: str = settings.APP_NAME,
    ) -> None:
        self.session_service = session_service or InMemorySessionService()
        self.app_name = app_name

    async def create_session(
        self,
        user_id: str,
        session_id: str,
        state: Optional[dict[str, Any]] = None,
    ) -> Session:
        """Asynchronously create a new session."""
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            state=state or {},
        )
        return session

    def create_session_sync(
        self,
        user_id: str,
        session_id: str,
        state: Optional[dict[str, Any]] = None,
    ) -> Session:
        """Synchronously create a new session."""
        return self.session_service.create_session_sync(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            state=state or {},
        )

    async def get_session(self, user_id: str, session_id: str) -> Optional[Session]:
        """Asynchronously retrieve a session by user_id and session_id."""
        return await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    def get_session_sync(self, user_id: str, session_id: str) -> Optional[Session]:
        """Synchronously retrieve a session by user_id and session_id."""
        return self.session_service.get_session_sync(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def get_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Retrieve state dictionary for a given session."""
        session = await self.get_session(user_id, session_id)
        if session is None:
            return {}
        return dict(session.state)

    def get_state_sync(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Synchronously retrieve state dictionary for a given session."""
        session = self.get_session_sync(user_id, session_id)
        if session is None:
            return {}
        return dict(session.state)

    async def update_state(
        self,
        user_id: str,
        session_id: str,
        state_delta: dict[str, Any],
    ) -> None:
        """Update state in a session with delta key-values."""
        # Update canonical session state in the session service storage
        if (
            self.app_name in self.session_service.sessions
            and user_id in self.session_service.sessions[self.app_name]
            and session_id in self.session_service.sessions[self.app_name][user_id]
        ):
            self.session_service.sessions[self.app_name][user_id][session_id].state.update(state_delta)
        else:
            raise ValueError(
                f"Session '{session_id}' not found for user '{user_id}' in app '{self.app_name}'."
            )

    def update_state_sync(
        self,
        user_id: str,
        session_id: str,
        state_delta: dict[str, Any],
    ) -> None:
        """Synchronously update state in a session with delta key-values."""
        if (
            self.app_name in self.session_service.sessions
            and user_id in self.session_service.sessions[self.app_name]
            and session_id in self.session_service.sessions[self.app_name][user_id]
        ):
            self.session_service.sessions[self.app_name][user_id][session_id].state.update(state_delta)
        else:
            raise ValueError(
                f"Session '{session_id}' not found for user '{user_id}' in app '{self.app_name}'."
            )

    async def list_sessions(self, user_id: str) -> list[Session]:
        """List all sessions for a specific user."""
        res = await self.session_service.list_sessions(
            app_name=self.app_name,
            user_id=user_id,
        )
        if hasattr(res, "sessions"):
            return list(res.sessions)
        return list(res)

    def list_sessions_sync(self, user_id: str) -> list[Session]:
        """Synchronously list all sessions for a specific user."""
        res = self.session_service.list_sessions_sync(
            app_name=self.app_name,
            user_id=user_id,
        )
        if hasattr(res, "sessions"):
            return list(res.sessions)
        return list(res)

    async def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete an existing session."""
        await self.session_service.delete_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    def delete_session_sync(self, user_id: str, session_id: str) -> None:
        """Synchronously delete an existing session."""
        self.session_service.delete_session_sync(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )


def get_session_service() -> InMemorySessionService:
    """Return the shared in-memory session service singleton."""
    global _SESSION_SERVICE_INSTANCE
    if _SESSION_SERVICE_INSTANCE is None:
        _SESSION_SERVICE_INSTANCE = InMemorySessionService()
    return _SESSION_SERVICE_INSTANCE


def get_session_manager(app_name: str = settings.APP_NAME) -> SessionManager:
    """Return the shared SessionManager singleton."""
    global _SESSION_MANAGER_INSTANCE
    if _SESSION_MANAGER_INSTANCE is None or _SESSION_MANAGER_INSTANCE.app_name != app_name:
        _SESSION_MANAGER_INSTANCE = SessionManager(
            session_service=get_session_service(),
            app_name=app_name,
        )
    return _SESSION_MANAGER_INSTANCE


def reset_session_service() -> None:
    """Reset the session service and manager singletons (useful for test isolation)."""
    global _SESSION_SERVICE_INSTANCE, _SESSION_MANAGER_INSTANCE
    _SESSION_SERVICE_INSTANCE = None
    _SESSION_MANAGER_INSTANCE = None
