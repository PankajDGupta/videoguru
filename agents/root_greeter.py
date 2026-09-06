"""Root Greeter Agent for VideoGuru (SPEC-003).

Responsible for Phase I Intent Capture:
- Welcomes the creator to VideoGuru.
- Prompts for the day's theme or main highlight.
- Stores user's thematic context in session state under `session.state["theme"]`.
- Initiates handoff to the Ingestion Agent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncGenerator, Optional

from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from agents.ingestion import create_ingestion_agent
from config import settings
from tools.intent_tools import record_theme

logger = logging.getLogger(__name__)

ROOT_GREETER_INSTRUCTION = (
    "You are the Root Greeter Agent for VideoGuru, an automated AI video production system. "
    "Your primary responsibility is Intent Capture:\n"
    "1. Welcome the creator warmly to VideoGuru.\n"
    "2. Prompt the creator to describe the day's theme, main highlight, or story focus for their video "
    "(e.g., 'Hiking in Yosemite', 'Cooking Italian pasta with family', 'Tokyo travel vlog').\n"
    "3. Once the creator provides their theme or highlight, immediately call the `record_theme` tool "
    "with their theme string. This stores the theme in session state under 'theme' and initiates handoff "
    "to the Ingestion Agent.\n"
    "4. Acknowledge their theme enthusiastically, confirm it has been locked in, and inform them "
    "that VideoGuru is now transitioning to scanning and ingesting their raw video footage."
)

_GREETING_STOP_WORDS = {
    "hello",
    "hi",
    "hey",
    "videoguru",
    "start",
    "session",
    "greetings",
    "video",
    "guru",
    "test",
    "cli",
    "greeting",
    "good",
    "morning",
    "afternoon",
    "evening",
}


def is_greeting_prompt(text: str) -> bool:
    """Determine whether the user text is a greeting without a specific video theme."""
    if not text:
        return True
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().strip()
    words = [w for w in cleaned.split() if w not in _GREETING_STOP_WORDS]
    return len(words) == 0


class RootGreeterAgent(Agent):
    """Root Greeter Agent executing Phase I intent capture and routing."""

    offline: bool = False

    def __init__(
        self,
        name: str = "root_greeter",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Prompts the creator for the day's theme or main highlight, saves theme "
            "intent into session state under 'theme', and initiates handoff to the Ingestion Agent."
        ),
        instruction: str = ROOT_GREETER_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        if tools is None:
            tools = [record_theme]
        if sub_agents is None:
            sub_agents = [create_ingestion_agent()]

        super().__init__(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            tools=tools,
            sub_agents=sub_agents,
            offline=offline,
            **kwargs,
        )

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        """Execute the agent turn, supporting both live Gemini LLM and deterministic offline mode."""
        if self.offline:
            # Extract user message text from context
            user_text = ""
            if ctx.user_content and ctx.user_content.parts:
                user_text = " ".join(
                    p.text for p in ctx.user_content.parts if getattr(p, "text", None)
                ).strip()

            existing_theme = ctx.session.state.get("theme")
            is_greeting = is_greeting_prompt(user_text)

            if is_greeting and not existing_theme:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "Hello from VideoGuru! Welcome to VideoGuru, your AI video production assistant. "
                                    "What is the day's theme or main highlight for your video?"
                                )
                            )
                        ]
                    ),
                )
            else:
                # Theme provided or explicitly present
                theme_to_set = existing_theme if (is_greeting and existing_theme) else user_text
                logger.info("OfflineRootGreeter: Capturing theme '%s' and transferring to ingestion_agent.", theme_to_set)

                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[
                            types.Part.from_text(
                                text=(
                                    f"Got it! Thematic context locked in: '{theme_to_set}'. "
                                    "Handing off to Ingestion Agent to scan raw clips."
                                )
                            )
                        ]
                    ),
                    actions=EventActions(
                        state_delta={"theme": theme_to_set},
                        transfer_to_agent="ingestion_agent",
                    ),
                )
        else:
            # Live LLM execution via Google ADK
            async for event in super()._run_async_impl(ctx):
                yield event


def create_root_greeter_agent(
    name: str = "root_greeter",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> RootGreeterAgent:
    """Factory function to instantiate a configured RootGreeterAgent."""
    return RootGreeterAgent(name=name, model=model, offline=offline, **kwargs)


# Default singleton instance
root_greeter_agent = create_root_greeter_agent()
