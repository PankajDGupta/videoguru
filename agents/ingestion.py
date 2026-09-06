"""Ingestion Agent stub for VideoGuru.

This stub provides the target agent node for handoff from the RootGreeterAgent.
It will be fully implemented in SPEC-006 with local directory scanning and
metadata extraction.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types

logger = logging.getLogger(__name__)


class StubIngestionAgent(BaseAgent):
    """Placeholder Ingestion Agent that receives handoff from RootGreeterAgent."""

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        theme = ctx.session.state.get("theme", "Not specified")
        logger.info("IngestionAgent active. Current session theme: '%s'", theme)
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            f"[IngestionAgent] Ready! Received handoff for theme: '{theme}'. "
                            "Awaiting media scan directory (Phase I, SPEC-004 to SPEC-006)."
                        )
                    )
                ]
            ),
        )


def create_ingestion_agent() -> StubIngestionAgent:
    """Factory function creating a new instance of StubIngestionAgent."""
    return StubIngestionAgent(
        name="ingestion_agent",
        description="Ingestion and Contextualization Agent responsible for scanning media files and building clip manifest.",
    )


# Default ingestion agent instance for module import
ingestion_agent = create_ingestion_agent()
