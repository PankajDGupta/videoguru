"""Root Agent and Application definition for VideoGuru."""

from google.adk.agents import Agent
from google.adk.apps import App

from config import settings

ROOT_INSTRUCTION = (
    "You are VideoGuru, an automated AI video production assistant built on the Google Agent "
    "Development Kit (ADK). Your goal is to guide the creator through producing a broadcast-quality "
    "vlog or video from raw clips. Welcome the creator and ask for the day's theme or main highlight "
    "to begin the autonomous curation and editing pipeline."
)

# Root agent instance for VideoGuru
root_agent = Agent(
    name="videoguru",
    model=settings.GEMINI_MODEL,
    description="VideoGuru Root Multi-Agent Orchestrator",
    instruction=ROOT_INSTRUCTION,
)

# ADK App wrapping the root agent
app = App(
    name=settings.APP_NAME,
    root_agent=root_agent,
)


def get_root_agent() -> Agent:
    """Return the configured root agent instance."""
    return root_agent


def get_app() -> App:
    """Return the configured ADK App instance."""
    return app
