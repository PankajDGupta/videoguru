"""Tests for Gemini Live API Review Integration for VideoGuru (SPEC-015).

Covers:
- LiveReviewSession initialization and LiveRequestQueue wiring.
- Sending text and real-time audio chunk requests to the live request queue.
- Voice transcription helper fallback / mock handling.
- Routing transcribed voice review input into approve vs revise flow.
- Session state synchronization ('review_approved', 'review_status', 'revision_feedback').
- Live stream bidirectional event generation with ADK Runner in offline mode.
- CLI argument parsing and flags for --live-review.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from google.adk.events import Event
from google.adk.runners import LiveRequestQueue, Runner
from google.genai import types

from agents.review_orchestrator import create_review_orchestrator_agent
from config import settings
from services.live_review import (
    LiveReviewSession,
    create_live_review_session,
    create_mock_wav_bytes,
    transcribe_voice_chunk,
)
from services.session_service import get_session_service


class TestLiveReviewToolsAndMockHelpers:

    def test_create_mock_wav_bytes(self):
        wav_data = create_mock_wav_bytes(duration_seconds=0.5, sample_rate=16000)
        assert len(wav_data) > 44  # WAV header is 44 bytes
        assert wav_data.startswith(b"RIFF")

    def test_transcribe_voice_chunk_empty_input(self):
        result = transcribe_voice_chunk(b"")
        assert result == ""

    @patch("services.live_review.transcribe_voice_chunk")
    def test_voice_review_approval_routing(self, mock_transcribe):
        mock_transcribe.return_value = "Everything looks solid, approve this draft."

        state = {"review_status": "pending", "review_approved": False}
        runner = MagicMock(spec=Runner)
        session = create_live_review_session(
            runner=runner,
            session_id="test_live_session",
            offline=True,
        )

        audio_bytes = create_mock_wav_bytes(0.5)
        result = asyncio.run(
            session.process_voice_review(
                audio_data=audio_bytes,
                session_state=state,
            )
        )

        assert result["review_approved"] is True
        assert result["review_status"] == "approved"
        assert state["review_approved"] is True
        assert state["review_status"] == "approved"
        assert "approve" in result["transcription"].lower()

    @patch("services.live_review.transcribe_voice_chunk")
    def test_voice_review_revision_routing(self, mock_transcribe):
        mock_transcribe.return_value = "The intro is sluggish, make the first cut faster."

        state = {"review_status": "pending", "review_approved": False}
        runner = MagicMock(spec=Runner)
        session = create_live_review_session(
            runner=runner,
            session_id="test_live_session_rev",
            offline=True,
        )

        audio_bytes = create_mock_wav_bytes(0.5)
        result = asyncio.run(
            session.process_voice_review(
                audio_data=audio_bytes,
                session_state=state,
            )
        )

        assert result["review_approved"] is False
        assert result["review_status"] == "revision_requested"
        assert state["review_approved"] is False
        assert len(state.get("revision_feedback", [])) == 1
        assert "sluggish" in state["revision_feedback"][0]


class TestLiveReviewSessionStreaming:

    def test_live_queue_send_text_and_audio(self):
        runner = MagicMock(spec=Runner)
        queue = LiveRequestQueue()
        session = LiveReviewSession(
            runner=runner,
            session_id="test_stream_session",
            live_request_queue=queue,
            offline=True,
        )

        async def run_queue_operations():
            await session.send_text_message("Hello from user voice stream")
            await session.send_audio_chunk(create_mock_wav_bytes(0.2))

            req1 = await queue.get()
            assert req1 is not None
            assert req1.content.parts[0].text == "Hello from user voice stream"

            req2 = await queue.get()
            assert req2 is not None
            assert req2.blob is not None
            assert req2.blob.mime_type == "audio/wav"

        asyncio.run(run_queue_operations())
        session.close()

    def test_offline_run_live_stream_generator(self):
        runner = MagicMock(spec=Runner)
        session = create_live_review_session(
            runner=runner,
            session_id="test_gen_session",
            offline=True,
        )

        async def run_stream():
            await session.send_text_message("Streaming review turn 1")
            events = []
            async for event in session.run_live_stream(max_events=1):
                events.append(event)
            return events

        events = asyncio.run(run_stream())
        assert len(events) == 1
        assert events[0].author == "review_orchestrator_live"
        assert "Streaming review turn 1" in events[0].content.parts[0].text


class TestLiveReviewCliArguments:

    def test_main_cli_arguments_support_live_review(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--live-review", "work_dir/edl.json", "--voice-file", "voice.wav"]):
            args = parse_arguments()
            assert args.live_review == "work_dir/edl.json"
            assert args.voice_file == "voice.wav"
