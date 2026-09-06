"""Ingestion and Contextualization Agent for VideoGuru (SPEC-006).

Responsible for Phase I Media Ingestion:
- Scans the creator's local directory containing raw video footage (downloaded from Google Photos).
- Extracts deep stream and container metadata (duration, frame rate, resolution, codecs, clip ID).
- Assembles the comprehensive Clip Manifest (list of ClipManifestEntry).
- Stores the manifest in session state under `session.state["clip_manifest"]`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Optional


from google.adk.agents import Agent
from google.adk.events import Event, EventActions
from google.genai import types

from config import settings
from tools.clip_metadata import extract_clip_metadata
from tools.directory_scanner import scan_local_directory
from tools.ingestion_tools import build_clip_manifest, ingest_media_directory

logger = logging.getLogger(__name__)

INGESTION_INSTRUCTION = (
    "You are the Ingestion and Contextualization Agent for VideoGuru, an automated AI video production system. "
    "Your primary responsibility is Media Ingestion and Manifest Construction (Phase I):\n"
    "1. Ingest raw video files from the creator's local media directory (downloaded from Google Photos).\n"
    "2. Use the `scan_local_directory` and `extract_clip_metadata` tools, or the composite `ingest_media_directory` tool, "
    "to discover video files (.mp4, .mov, .avi, .mkv) and extract deep stream metadata (duration, frame rate, resolution, codecs, clip ID).\n"
    "3. Build the full Clip Manifest (list of ClipManifestEntry) and ensure it is saved in session state under 'clip_manifest'.\n"
    "4. Provide a clear, structured summary of the ingested media to the creator, including total clip count, combined duration, "
    "and resolution breakdown."
)


class IngestionAgent(Agent):
    """Ingestion and Contextualization Agent executing Phase I media ingestion."""

    offline: bool = False

    def __init__(
        self,
        name: str = "ingestion_agent",
        model: str = settings.GEMINI_MODEL,
        description: str = (
            "Scans the creator's local media directory, extracts clip metadata via ffprobe, "
            "assembles the Clip Manifest, and stores it in session state under 'clip_manifest'."
        ),
        instruction: str = INGESTION_INSTRUCTION,
        tools: Optional[list[Any]] = None,
        sub_agents: Optional[list[Any]] = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        is_offline = offline if offline else not bool(os.getenv("GEMINI_API_KEY"))
        if tools is None:
            tools = [scan_local_directory, extract_clip_metadata, ingest_media_directory]
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
        """Execute the ingestion turn, supporting both live Gemini LLM and deterministic offline mode."""
        if self.offline:
            theme = ctx.session.state.get("theme", "Not specified")
            media_dir = ctx.session.state.get("media_dir")

            # Extract user message text to check if a directory path was provided
            user_text = ""
            if ctx.user_content and ctx.user_content.parts:
                user_text = " ".join(
                    p.text for p in ctx.user_content.parts if getattr(p, "text", None)
                ).strip()

            target_dir: Optional[Path] = None
            if user_text:
                candidate = Path(user_text).resolve()
                if candidate.exists() and candidate.is_dir():
                    target_dir = candidate

            if target_dir is None and media_dir:
                candidate = Path(media_dir).resolve()
                if candidate.exists() and candidate.is_dir():
                    target_dir = candidate

            if target_dir is None and settings.MEDIA_INPUT_DIR:
                candidate = Path(settings.MEDIA_INPUT_DIR).resolve()
                if candidate.exists() and candidate.is_dir():
                    target_dir = candidate

            if target_dir is not None:
                logger.info(
                    "OfflineIngestionAgent: Building manifest for directory '%s' (theme: '%s')",
                    target_dir,
                    theme,
                )
                manifest = build_clip_manifest(target_dir)
                serialized = [e.model_dump() for e in manifest]
                total_duration = sum(e.duration_seconds for e in manifest)

                summary_text = (
                    f"[IngestionAgent] Ready! Received handoff for theme: '{theme}'.\n"
                    f"Successfully ingested {len(manifest)} clip(s) from '{target_dir}'. "
                    f"Total duration: {total_duration:.2f}s. "
                    f"Clip manifest locked in session state under 'clip_manifest'."
                )

                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part.from_text(text=summary_text)]),
                    actions=EventActions(
                        state_delta={
                            "clip_manifest": serialized,
                            "media_dir": str(target_dir),
                        }
                    ),
                )
            else:
                # Awaiting directory path
                logger.info(
                    "OfflineIngestionAgent: Active with theme '%s', awaiting media directory.",
                    theme,
                )
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
        else:
            # Live LLM execution via Google ADK
            async for event in super()._run_async_impl(ctx):
                yield event


# Backwards-compatibility alias
StubIngestionAgent = IngestionAgent


def create_ingestion_agent(
    name: str = "ingestion_agent",
    model: str = settings.GEMINI_MODEL,
    offline: bool = False,
    **kwargs: Any,
) -> IngestionAgent:
    """Factory function creating a configured IngestionAgent instance."""
    is_offline = offline if offline else not bool(os.getenv("GEMINI_API_KEY"))
    return IngestionAgent(name=name, model=model, offline=is_offline, **kwargs)


# Default singleton instance
ingestion_agent = create_ingestion_agent()

