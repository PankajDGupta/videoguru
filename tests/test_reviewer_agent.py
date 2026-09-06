"""Tests for VideoGuru Reviewer Agent and Algorithmic Review Tools (SPEC-010).

Covers:
- Algorithmic review schemas (PacingMetrics, HookMetrics, RetentionMetrics, AlgorithmicReviewResult)
- Pacing, hook, and retention calculation heuristics
- Passing vs failing EDL evaluations
- Custom ADK tool review_edl_algorithmically with session state synchronization
- ReviewerAgent initialization, instructions, and tools
- ReviewerAgent execution (both programmatic run_review and async turn with state delta)
- Multi-agent pipeline integration: Curation Agent -> Reviewer Agent
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from google.adk.events import EventActions
from google.adk.sessions import Session
from google.adk.tools import ToolContext

from agents import (
    CurationAgent,
    ReviewerAgent,
    create_curation_agent,
    create_reviewer_agent,
    reviewer_agent,
)
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from schemas.review import (
    AlgorithmicReviewResult,
    HookMetrics,
    PacingMetrics,
    RetentionMetrics,
)
from tools.clip_metadata import find_ffprobe_executable
from tools.curation_tools import assemble_edl_from_manifest
from tools.review_tools import (
    calculate_hook_metrics,
    calculate_pacing_metrics,
    calculate_retention_metrics,
    evaluate_edl_heuristically,
    get_review_from_state,
    review_edl_algorithmically,
)


@pytest.fixture(scope="module")
def sample_review_media(tmp_path_factory) -> dict[str, Any]:
    """Generate two small synthetic MP4 video clips for pipeline testing."""
    ffprobe_bin = find_ffprobe_executable()
    ffmpeg_bin = str(Path(ffprobe_bin).parent / "ffmpeg.exe")
    if not Path(ffmpeg_bin).exists():
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    temp_dir = tmp_path_factory.mktemp("review_media_dir")
    clip_1 = temp_dir / "clip_action.mp4"
    clip_2 = temp_dir / "clip_scenic.mp4"

    # Clip 1: 2.0s 1280x720 video with audio
    cmd_1 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2.0:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=800:duration=2.0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(clip_1),
    ]
    subprocess.run(cmd_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Clip 2: 1.5s 1920x1080 video with audio
    cmd_2 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.5:size=1920x1080:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=400:duration=1.5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(clip_2),
    ]
    subprocess.run(cmd_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    manifest_entries = [
        ClipManifestEntry(
            clip_id="vid_action_001",
            absolute_path=str(clip_1),
            file_name="clip_action.mp4",
            duration_seconds=2.0,
            frame_rate=30.0,
            resolution="1280x720",
            width=1280,
            height=720,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=clip_1.stat().st_size,
        ),
        ClipManifestEntry(
            clip_id="vid_scenic_002",
            absolute_path=str(clip_2),
            file_name="clip_scenic.mp4",
            duration_seconds=1.5,
            frame_rate=24.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=clip_2.stat().st_size,
        ),
    ]

    return {
        "dir": temp_dir,
        "clip_1": clip_1,
        "clip_2": clip_2,
        "manifest": manifest_entries,
    }


def _create_mock_tool_context(initial_state: dict[str, Any] | None = None) -> ToolContext:
    actions = EventActions()
    mock_invocation = MagicMock()
    mock_invocation.session.state = initial_state if initial_state is not None else {}
    return ToolContext(invocation_context=mock_invocation, event_actions=actions)


def _build_sample_passing_edl() -> EditDecisionList:
    """Helper creating a fast-paced, high-engagement EDL that meets all algorithmic benchmarks."""
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_001",
                start_trim=0.0,
                end_trim=2.5,
                scene_rationale="Fast, punchy drone opening establishing setting",
                transition_intent=TransitionIntent.CUT,
                engagement_score=8.8,
            ),
            EDLEntry(
                file_reference="clip_002",
                start_trim=1.0,
                end_trim=4.5,
                scene_rationale="High energy action shot maintaining momentum",
                transition_intent=TransitionIntent.WIPE,
                engagement_score=8.2,
            ),
            EDLEntry(
                file_reference="clip_003",
                start_trim=0.5,
                end_trim=4.0,
                scene_rationale="Dynamic scene with steady movement",
                transition_intent=TransitionIntent.DISSOLVE,
                engagement_score=7.9,
            ),
            EDLEntry(
                file_reference="clip_004",
                start_trim=2.0,
                end_trim=5.0,
                scene_rationale="Memorable closing climax clip",
                transition_intent=TransitionIntent.FADE,
                engagement_score=8.5,
            ),
        ]
    )


def _build_sample_failing_edl() -> EditDecisionList:
    """Helper creating an EDL with sluggish hook and excessively long cut that fails benchmarks."""
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_slow_hook",
                start_trim=0.0,
                end_trim=8.0,  # 8.0s hook > 5.0s limit
                scene_rationale="Slow static intro landscape",
                transition_intent=TransitionIntent.CUT,
                engagement_score=5.5,
            ),
            EDLEntry(
                file_reference="clip_too_long",
                start_trim=0.0,
                end_trim=14.0,  # 14.0s cut > 10.0s threshold
                scene_rationale="Monotonous static interview without b-roll",
                transition_intent=TransitionIntent.CUT,
                engagement_score=4.8,
            ),
        ]
    )


class TestAlgorithmicReviewSchemas:
    """Test serialization, validation, and deserialization of review models."""

    def test_pacing_metrics_model(self):
        metrics = PacingMetrics(
            avg_cut_duration=3.5,
            cut_count=4,
            cuts_per_minute=17.14,
            max_cut_duration=4.5,
            min_cut_duration=2.5,
            pacing_variance=0.82,
            excessive_cuts_count=0,
        )
        assert metrics.avg_cut_duration == 3.5
        assert metrics.cuts_per_minute == 17.14
        assert metrics.excessive_cuts_count == 0

    def test_hook_metrics_model(self):
        hook = HookMetrics(
            hook_duration=2.5,
            first_cut_engagement=8.5,
            hook_score=9.0,
            is_punchy=True,
        )
        assert hook.hook_duration == 2.5
        assert hook.is_punchy is True
        assert hook.hook_score == 9.0

    def test_algorithmic_review_result_serialization(self):
        edl = _build_sample_passing_edl()
        result = evaluate_edl_heuristically(edl, theme="Travel Adventure")

        as_dict = result.to_dict()
        assert as_dict["passed"] is True
        assert as_dict["score"] >= 7.0
        assert "PacingMetrics" not in as_dict  # flattened dict

        as_json = result.to_json()
        restored = AlgorithmicReviewResult.from_json(as_json)
        assert restored.passed == result.passed
        assert restored.score == result.score
        assert restored.hook_metrics.hook_duration == result.hook_metrics.hook_duration


class TestAlgorithmicCalculations:
    """Test pacing, hook, and retention calculation functions."""

    def test_pacing_metrics_calculation(self):
        edl = _build_sample_passing_edl()
        metrics = calculate_pacing_metrics(edl)

        # Durations: [2.5, 3.5, 3.5, 3.0], total = 12.5s, count = 4
        assert metrics.cut_count == 4
        assert metrics.min_cut_duration == 2.5
        assert metrics.max_cut_duration == 3.5
        assert metrics.avg_cut_duration == round(12.5 / 4, 3)
        assert metrics.excessive_cuts_count == 0
        assert metrics.cuts_per_minute > 0.0

    def test_pacing_metrics_with_excessive_cut(self):
        edl = _build_sample_failing_edl()
        metrics = calculate_pacing_metrics(edl)
        assert metrics.excessive_cuts_count == 1  # 14.0s cut
        assert metrics.max_cut_duration == 14.0

    def test_pacing_metrics_empty_edl_raises(self):
        empty_edl = EditDecisionList(entries=[])
        with pytest.raises(ValueError, match="empty EditDecisionList"):
            calculate_pacing_metrics(empty_edl)

    def test_hook_metrics_punchy_opening(self):
        edl = _build_sample_passing_edl()
        hook = calculate_hook_metrics(edl)

        assert hook.hook_duration == 2.5
        assert hook.is_punchy is True
        assert hook.hook_score >= 8.0

    def test_hook_metrics_sluggish_opening(self):
        edl = _build_sample_failing_edl()
        hook = calculate_hook_metrics(edl)

        assert hook.hook_duration == 8.0
        assert hook.is_punchy is False
        assert hook.hook_score < 7.0

    def test_hook_metrics_empty_edl_raises(self):
        empty_edl = EditDecisionList(entries=[])
        with pytest.raises(ValueError, match="empty EditDecisionList"):
            calculate_hook_metrics(empty_edl)

    def test_retention_metrics_calculation(self):
        edl = _build_sample_passing_edl()
        pacing = calculate_pacing_metrics(edl)
        retention = calculate_retention_metrics(edl, pacing)

        assert retention.total_duration == 12.5
        assert retention.retention_score >= 7.0
        assert retention.transition_diversity_score > 0.0


class TestHeuristicReviewEvaluation:
    """Test deterministic algorithmic evaluation for passing and failing edits."""

    def test_passing_edl_evaluation(self):
        edl = _build_sample_passing_edl()
        result = evaluate_edl_heuristically(edl, theme="High Energy Vlog")

        assert result.passed is True
        assert result.score >= 7.0
        assert result.hook_score >= 7.0
        assert result.pacing_score >= 7.0
        assert "[PASS]" in result.feedback
        assert len(result.recommendations) == 0

    def test_failing_edl_evaluation(self):
        edl = _build_sample_failing_edl()
        result = evaluate_edl_heuristically(edl, theme="Static Vlog")

        assert result.passed is False
        assert result.score < 7.0 or result.hook_metrics.is_punchy is False
        assert "[REVISE]" in result.feedback
        assert len(result.recommendations) > 0

        # Should recommend trimming opening cut and the 14s cut
        combined_recs = " ".join(result.recommendations)
        assert "5-second hook" in combined_recs or "opening cut" in combined_recs.lower()
        assert "14.0s" in combined_recs or "too long" in combined_recs.lower()


class TestReviewToolWithContext:
    """Test custom ADK tool review_edl_algorithmically with session state interaction."""

    def test_tool_with_explicit_edl(self):
        edl = _build_sample_passing_edl()
        result = review_edl_algorithmically(edl=edl, theme="Action Montage", offline=True)

        assert result.passed is True
        assert result.score >= 7.0

    def test_tool_with_tool_context(self):
        ctx = _create_mock_tool_context(
            {
                "theme": "Drone Showcase",
                "edl": _build_sample_passing_edl().to_dict_list(),
            }
        )
        result = review_edl_algorithmically(tool_context=ctx, offline=True)

        assert result.passed is True
        assert ctx.state["reviewer_passed"] is True
        assert ctx.state["reviewer_score"] == result.score
        assert "reviewer_feedback" in ctx.state
        assert "algorithmic_review" in ctx.state

        # Check helper reconstructs review
        restored = get_review_from_state(ctx.state)
        assert restored is not None
        assert restored.passed == result.passed
        assert restored.score == result.score

    def test_tool_missing_edl_raises(self):
        ctx = _create_mock_tool_context({})
        with pytest.raises(ValueError, match="No Edit Decision List"):
            review_edl_algorithmically(tool_context=ctx, offline=True)


class TestReviewerAgentCreation:
    """Test ReviewerAgent initialization and configuration."""

    def test_agent_initialization(self):
        agent = create_reviewer_agent()
        assert agent.name == "reviewer_agent"
        assert len(agent.tools) == 1
        assert agent.tools[0] == review_edl_algorithmically

    def test_agent_instruction_contains_key_elements(self):
        agent = create_reviewer_agent()
        instruction = agent.instruction
        assert "YouTube" in instruction
        assert "Pacing" in instruction
        assert "Hook Strength" in instruction
        assert "Retention Curve" in instruction
        assert "reviewer_passed" in instruction

    def test_singleton_export(self):
        assert isinstance(reviewer_agent, ReviewerAgent)
        assert reviewer_agent.name == "reviewer_agent"


class TestReviewerAgentExecution:
    """Test ReviewerAgent session execution and async turn behavior."""

    def test_run_review_passing(self):
        agent = create_reviewer_agent(offline=True)
        session = Session(id="test_sess_001", app_name="videoguru", user_id="creator_1", state={})
        session.state["theme"] = "Viral Travel Highlights"
        session.state["edl"] = _build_sample_passing_edl().to_dict_list()

        result = agent.run_review(session)
        assert result.passed is True
        assert session.state["reviewer_passed"] is True
        assert session.state["reviewer_score"] >= 7.0
        assert "reviewer_feedback" in session.state

    def test_run_review_failing(self):
        agent = create_reviewer_agent(offline=True)
        session = Session(id="test_sess_002", app_name="videoguru", user_id="creator_1", state={})
        session.state["theme"] = "Raw Archive"
        session.state["edl"] = _build_sample_failing_edl().to_dict_list()

        result = agent.run_review(session)
        assert result.passed is False
        assert session.state["reviewer_passed"] is False
        assert len(result.recommendations) > 0

    def test_run_review_missing_edl_raises(self):
        agent = create_reviewer_agent(offline=True)
        session = Session(id="test_sess_003", app_name="videoguru", user_id="creator_1", state={})
        with pytest.raises(ValueError, match="requires an EDL"):
            agent.run_review(session)

    def test_async_turn_offline_with_state_delta(self):
        async def _test():
            agent = create_reviewer_agent(offline=True)
            session = Session(id="test_sess_async", app_name="videoguru", user_id="creator_1", state={})
            session.state["theme"] = "Rapid vlog"
            session.state["edl"] = _build_sample_passing_edl().to_dict_list()

            class MockContext:
                def __init__(self, sess):
                    self.session = sess

            ctx = MockContext(session)
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

            assert len(events) == 1
            ev = events[0]
            assert ev.author == "reviewer_agent"
            assert "Algorithmic Review Report" in ev.content.parts[0].text
            assert ev.actions is not None
            assert ev.actions.state_delta["reviewer_passed"] is True
            assert ev.actions.state_delta["reviewer_score"] >= 7.0

        asyncio.run(_test())

    def test_async_turn_missing_edl_emits_warning(self):
        async def _test():
            agent = create_reviewer_agent(offline=True)
            session = Session(id="test_sess_empty", app_name="videoguru", user_id="creator_1", state={})

            class MockContext:
                def __init__(self, sess):
                    self.session = sess

            ctx = MockContext(session)
            events = []
            async for event in agent._run_async_impl(ctx):
                events.append(event)

            assert len(events) == 1
            ev = events[0]
            assert "Cannot perform algorithmic review" in ev.content.parts[0].text

        asyncio.run(_test())


class TestCurationToReviewerPipeline:
    """Integration test verifying output from CurationAgent feeds cleanly into ReviewerAgent."""

    def test_curation_to_reviewer_flow(self, sample_review_media):
        manifest = sample_review_media["manifest"]

        # 1. Curation Agent assembles EDL
        theme = "Extreme Mountain Biking Adventure"
        curated_edl = assemble_edl_from_manifest(manifest=manifest, theme=theme, offline=True)
        assert len(curated_edl) == 2

        # 2. Setup session with curated EDL
        session = Session(id="test_pipeline_session", app_name="videoguru", user_id="creator_1", state={})
        session.state["theme"] = theme
        session.state["clip_manifest"] = [c.model_dump() for c in manifest]
        session.state["edl"] = curated_edl.to_dict_list()

        # 3. Reviewer Agent evaluates the curated EDL
        reviewer = create_reviewer_agent(offline=True)
        review_result = reviewer.run_review(session)

        # 4. Verify review results
        assert review_result.pacing_metrics.cut_count == 2
        assert review_result.hook_metrics.hook_duration == curated_edl[0].duration
        assert "reviewer_passed" in session.state
        assert "reviewer_score" in session.state
        assert "reviewer_feedback" in session.state
        assert "algorithmic_review" in session.state
