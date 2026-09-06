"""Gemini Live API Review Integration for VideoGuru (SPEC-015).

Implements bidirectional streaming review using Google ADK's LiveRequestQueue
and Runner.run_live(), supporting:
- Voice & audio feedback input ingestion (WAV/PCM byte chunks or audio files).
- Audio transcription and text feedback parsing.
- Routing transcribed voice input into the approve/revise review flow.
- Session state synchronization ('review_approved', 'review_status', 'revision_feedback').
- Deterministic offline mock runner for testing and environments without live WebSocket credentials.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from typing import Any, AsyncGenerator, Callable, Optional, Sequence, Union

from google.adk.events import Event, EventActions
from google.adk.runners import LiveRequestQueue, Runner
from google.genai import types

from config import settings
from schemas.edl import EditDecisionList
from tools.review_orchestrator_tools import process_user_review_response

logger = logging.getLogger(__name__)


def create_mock_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid PCM WAV bytes for testing audio input streams."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        num_frames = int(sample_rate * max(0.1, duration_seconds))
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


def transcribe_voice_chunk(
    audio_data: bytes,
    mime_type: str = "audio/wav",
    client: Optional[Any] = None,
    whisper_model: Optional[str] = None,
) -> str:
    """Transcribe an audio chunk to text using available Gemini, Whisper, or fallback.

    Args:
        audio_data: Raw audio byte stream.
        mime_type: MIME type of the audio data (e.g., 'audio/wav', 'audio/pcm').
        client: Optional Gemini client.
        whisper_model: Whisper model name if local whisper fallback is preferred.

    Returns:
        Transcribed text string.
    """
    if not audio_data:
        return ""

    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if api_key and client is None:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.debug("Failed to initialize genai client for transcription: %s", exc)

    if client is not None:
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=audio_data, mime_type=mime_type),
                    "Transcribe the spoken audio review feedback precisely as plain text.",
                ],
            )
            if response.text:
                return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini voice transcription failed: %s; falling back", exc)

    # Local Whisper fallback if available
    try:
        import whisper
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(audio_data)
            tmp_path = tmp_wav.name

        try:
            model = whisper.load_model(whisper_model or "base")
            result = model.transcribe(tmp_path)
            return result.get("text", "").strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as exc:
        logger.debug("Local Whisper transcription unavailable: %s", exc)

    return ""


class LiveReviewSession:
    """Manages a bidirectional streaming review session using ADK LiveRequestQueue and run_live()."""

    def __init__(
        self,
        runner: Runner,
        session_id: str,
        user_id: str = settings.DEFAULT_USER_ID,
        live_request_queue: Optional[LiveRequestQueue] = None,
        offline: bool = False,
    ) -> None:
        self.runner = runner
        self.session_id = session_id
        self.user_id = user_id
        self.queue = live_request_queue or LiveRequestQueue()
        self.offline = offline
        self.is_active = False
        self._received_events: list[Event] = []

    async def send_text_message(self, text: str) -> None:
        """Send a text turn into the live request queue."""
        content = types.Content(parts=[types.Part.from_text(text=text)])
        self.queue.send_content(content=content)

    async def send_audio_chunk(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/wav",
        end_of_turn: bool = True,
    ) -> None:
        """Send a real-time raw audio chunk into the live request queue."""
        blob = types.Blob(data=audio_bytes, mime_type=mime_type)
        self.queue.send_realtime(blob=blob)
        if end_of_turn:
            self.queue.send_audio_stream_end()

    async def process_voice_review(
        self,
        audio_data: bytes,
        session_state: dict[str, Any],
        mime_type: str = "audio/wav",
        transcription_override: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process voice review feedback: transcribe audio and route through review flow.

        Updates session_state with:
        - 'voice_transcription': string of transcribed text
        - 'review_approved': bool (True if approve, False if revision)
        - 'review_status': 'approved' | 'revision_requested'
        - 'revision_feedback': list of feedback items if revised

        Args:
            audio_data: Audio bytes containing creator voice feedback.
            session_state: Mutable session state dictionary.
            mime_type: MIME type of the audio stream.
            transcription_override: Explicit text transcription (e.g., in unit tests).

        Returns:
            Dictionary containing transcription and review status summary.
        """
        if transcription_override is not None:
            transcribed_text = transcription_override.strip()
        else:
            transcribed_text = transcribe_voice_chunk(audio_data, mime_type=mime_type)

        if not transcribed_text:
            transcribed_text = "No audible voice feedback detected."

        class SimpleToolContext:
            def __init__(self, st: dict[str, Any]) -> None:
                self.state = st

        tool_ctx = SimpleToolContext(session_state)
        result_message = process_user_review_response(transcribed_text, tool_ctx)
        session_state["voice_transcription"] = transcribed_text

        return {
            "transcription": transcribed_text,
            "result_message": result_message,
            "review_approved": session_state.get("review_approved", False),
            "review_status": session_state.get("review_status", "pending"),
            "revision_feedback": session_state.get("revision_feedback", []),
        }

    async def run_live_stream(
        self,
        max_events: Optional[int] = None,
    ) -> AsyncGenerator[Event, None]:
        """Execute the bidirectional streaming loop via Runner.run_live()."""
        self.is_active = True
        event_count = 0

        try:
            if self.offline:
                # In offline mode, consume queue items and simulate response events
                while self.is_active:
                    try:
                        req = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        break

                    if req is None:
                        break

                    # Synthesize an event based on request
                    text_parts = []
                    if hasattr(req, "content") and req.content and req.content.parts:
                        for part in req.content.parts:
                            if getattr(part, "text", None):
                                text_parts.append(part.text)

                    response_text = " ".join(text_parts) if text_parts else "Live audio feedback received."
                    event = Event(
                        author="review_orchestrator_live",
                        content=types.Content(parts=[types.Part.from_text(text=response_text)]),
                    )
                    self._received_events.append(event)
                    yield event
                    event_count += 1
                    if max_events is not None and event_count >= max_events:
                        break
            else:
                async for event in self.runner.run_live(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    live_request_queue=self.queue,
                ):
                    self._received_events.append(event)
                    yield event
                    event_count += 1
                    if max_events is not None and event_count >= max_events:
                        break
        finally:
            self.is_active = False
            self.queue.close()

    def close(self) -> None:
        """Close the live request queue and terminate session."""
        self.is_active = False
        self.queue.close()


def create_live_review_session(
    runner: Runner,
    session_id: str,
    user_id: str = settings.DEFAULT_USER_ID,
    offline: bool = False,
) -> LiveReviewSession:
    """Factory creating a LiveReviewSession instance."""
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    is_offline = offline if offline else not has_api_key
    return LiveReviewSession(
        runner=runner,
        session_id=session_id,
        user_id=user_id,
        offline=is_offline,
    )
