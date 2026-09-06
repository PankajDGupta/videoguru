"""Tests for VideoGuru Review Orchestrator Agent (SPEC-014).

Covers:
- `prepare_review_summary` tool.
- `process_user_review_response` tool.
- `ReviewOrchestratorAgent` offline mode yielding summary and OTIO path.
- `ReviewOrchestratorAgent` processing user response.
- Error handling for missing EDL/manifest.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.adk.events import EventActions
from google.adk.sessions import Session
from google.genai import types

from agents.review_orchestrator import create_review_orchestrator_agent
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.review_orchestrator_tools import (
    prepare_review_summary,
    process_user_review_response,
)


def _build_sample_edl() -> EditDecisionList:
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_001",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Intro",
                transition_intent=TransitionIntent.CUT,
                engagement_score=8.5,
            ),
            EDLEntry(
                file_reference="clip_002",
                start_trim=1.0,
                end_trim=6.0,
                scene_rationale="Action",
                transition_intent=TransitionIntent.FADE,
                engagement_score=9.0,
            ),
        ]
    )


def _build_sample_manifest() -> list[ClipManifestEntry]:
    return [
        ClipManifestEntry(
            clip_id="clip_001",
            absolute_path="/media/clip_001.mp4",
            file_name="clip_001.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            has_audio=True,
            file_size_bytes=1000,
        ),
        ClipManifestEntry(
            clip_id="clip_002",
            absolute_path="/media/clip_002.mp4",
            file_name="clip_002.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            has_audio=True,
            file_size_bytes=1000,
        ),
    ]


class TestReviewOrchestratorTools:
    
    def test_prepare_review_summary(self):
        edl = _build_sample_edl()
        theme = "Action Highlights"
        otio_path = "/staging/timeline.otio"
        
        summary = prepare_review_summary(edl, theme, otio_path)
        
        assert "VideoGuru Edit Draft Ready for Review" in summary
        assert "Theme:** Action Highlights" in summary
        assert "Total Cuts:** 2" in summary
        assert "Total Duration:** 10.00 seconds" in summary
        assert "Unique Clips Used:** 2" in summary
        assert "1 cut, 1 fade" in summary or "1 fade, 1 cut" in summary
        assert "8.8/10.0" in summary
        assert "/staging/timeline.otio" in summary

    def test_process_user_review_response_approve(self):
        class MockToolContext:
            def __init__(self):
                self.state = {}
                
        ctx = MockToolContext()
        response = process_user_review_response("Looks good to me!", tool_context=ctx)
        
        assert "Edit approved" in response
        assert ctx.state["review_approved"] is True
        assert ctx.state["review_status"] == "approved"
        
    def test_process_user_review_response_revise(self):
        class MockToolContext:
            def __init__(self):
                self.state = {"revision_feedback": []}
                
        ctx = MockToolContext()
        response = process_user_review_response("Make the intro faster", tool_context=ctx)
        
        assert "Revision requested" in response
        assert ctx.state["review_approved"] is False
        assert ctx.state["review_status"] == "revision_requested"
        assert len(ctx.state["revision_feedback"]) == 1
        assert ctx.state["revision_feedback"][0] == "Make the intro faster"


class TestReviewOrchestratorAgent:

    @patch("agents.review_orchestrator.edl_to_otio")
    def test_agent_present_summary(self, mock_edl_to_otio):
        mock_edl_to_otio.return_value = "/mock/path/timeline.otio"
        
        agent = create_review_orchestrator_agent(offline=True)
        edl = _build_sample_edl()
        manifest = _build_sample_manifest()
        
        mock_session = Session(
            id="session_test",
            state={"edl": edl, "theme": "Test Theme", "clip_manifest": manifest},
            app_name="videoguru",
            user_id="test_user",
        )
        mock_ctx = MagicMock()
        mock_ctx.session = mock_session
        mock_ctx.new_message = None
        mock_ctx.invocation_context.new_message = None

        events = []
        async def run():
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)
                
        asyncio.run(run())
        
        assert len(events) == 1
        event = events[0]
        assert "VideoGuru Edit Draft Ready for Review" in event.content.parts[0].text
        assert "/mock/path/timeline.otio" in event.content.parts[0].text
        assert event.actions.state_delta["otio_file_path"] == "/mock/path/timeline.otio"
        assert event.actions.state_delta["review_status"] == "pending"

    def test_agent_process_feedback(self):
        agent = create_review_orchestrator_agent(offline=True)
        
        mock_session = Session(
            id="session_test2",
            state={"review_status": "pending", "revision_feedback": []},
            app_name="videoguru",
            user_id="test_user",
        )
        mock_ctx = MagicMock()
        mock_ctx.session = mock_session
        mock_ctx.new_message = types.Content(parts=[types.Part.from_text(text="I approve!")])
        
        events = []
        async def run():
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)
                
        asyncio.run(run())
        
        assert len(events) == 1
        event = events[0]
        assert "Edit approved" in event.content.parts[0].text
        assert event.actions.state_delta["review_status"] == "approved"
        assert event.actions.state_delta["review_approved"] is True

    def test_agent_missing_edl(self):
        agent = create_review_orchestrator_agent(offline=True)
        
        mock_session = Session(
            id="session_test3",
            state={"theme": "Test Theme"},
            app_name="videoguru",
            user_id="test_user",
        )
        mock_ctx = MagicMock()
        mock_ctx.session = mock_session
        mock_ctx.new_message = None
        mock_ctx.invocation_context.new_message = None
        
        events = []
        async def run():
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)
                
        asyncio.run(run())
        
        assert len(events) == 1
        assert "Missing Edit Decision List" in events[0].content.parts[0].text
