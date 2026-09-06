"""Tests for VideoGuru Critic Agent and Human-Centric Review Tools (SPEC-011).

Covers:
- Critic schemas (Hook30sMetrics, ThumbnailMetrics, StorytellingMetrics, CriticReviewResult)
- 30-second hook AVD, thumbnail CTR, and narrative calculation heuristics
- Passing vs failing EDL evaluations from human viewer perspective
- Loop control tools: exit_loop (escalate=True) and append_to_state (feedback accumulation)
- Custom ADK tool review_edl_as_critic with session state synchronization
- CriticAgent initialization, instructions, and tools
- CriticAgent execution (both programmatic run_review and async turn with state delta and escalate flag)
- Multi-agent pipeline integration: Curation Agent -> Reviewer Agent -> Critic Agent
- CLI interface: main.py --critic
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from google.adk.events import EventActions
from google.adk.sessions import Session
from google.adk.tools import ToolContext

from agents import (
    CriticAgent,
    CurationAgent,
    ReviewerAgent,
    create_critic_agent,
    create_curation_agent,
    create_reviewer_agent,
    critic_agent,
)
from schemas.critic import (
    CriticReviewResult,
    Hook30sMetrics,
    StorytellingMetrics,
    ThumbnailMetrics,
)
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.critic_tools import (
    calculate_30s_hook_metrics,
    calculate_storytelling_metrics,
    calculate_thumbnail_metrics,
    evaluate_critic_heuristically,
    evaluate_critic_with_gemini,
    get_critic_review_from_state,
    review_edl_as_critic,
)
from tools.curation_tools import assemble_edl_from_manifest
from tools.loop_tools import append_to_state, exit_loop
from tools.review_tools import review_edl_algorithmically


def _create_mock_tool_context(initial_state: dict[str, Any] | None = None) -> ToolContext:
    actions = EventActions()
    mock_invocation = MagicMock()
    mock_invocation.session.state = initial_state if initial_state is not None else {}
    return ToolContext(invocation_context=mock_invocation, event_actions=actions)


def _build_sample_passing_edl() -> EditDecisionList:
    """Helper creating an engaging, fast-paced EDL with compelling shots and punchy hook."""
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_drone_001",
                start_trim=0.0,
                end_trim=2.5,
                scene_rationale="Fast drone sweep revealing stunning mountain sunset horizon",
                transition_intent=TransitionIntent.CUT,
                engagement_score=9.0,
            ),
            EDLEntry(
                file_reference="clip_action_002",
                start_trim=1.0,
                end_trim=4.0,
                scene_rationale="Dynamic action shot of athlete jump with intense facial expression",
                transition_intent=TransitionIntent.WIPE,
                engagement_score=9.2,
            ),
            EDLEntry(
                file_reference="clip_reaction_003",
                start_trim=0.5,
                end_trim=3.5,
                scene_rationale="Vibrant reaction close-up smile and celebration climax",
                transition_intent=TransitionIntent.DISSOLVE,
                engagement_score=8.7,
            ),
            EDLEntry(
                file_reference="clip_closing_004",
                start_trim=2.0,
                end_trim=5.0,
                scene_rationale="Scenic sunset resolution tying back to the travel theme",
                transition_intent=TransitionIntent.FADE,
                engagement_score=8.5,
            ),
        ]
    )


def _build_sample_failing_edl() -> EditDecisionList:
    """Helper creating an EDL with dead air in intro (>8s cut) and sluggish pacing."""
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_slow_001",
                start_trim=0.0,
                end_trim=12.0,  # 12s cut causes dead air in first 30s
                scene_rationale="Static unmoving shot of empty room without subject",
                transition_intent=TransitionIntent.CUT,
                engagement_score=4.5,
            ),
            EDLEntry(
                file_reference="clip_flat_002",
                start_trim=0.0,
                end_trim=9.5,
                scene_rationale="Low energy hallway pan without clear focus",
                transition_intent=TransitionIntent.CUT,
                engagement_score=5.0,
            ),
        ]
    )


# ==============================================================================
# 1. Schema Tests
# ==============================================================================

class TestCriticSchemas:
    """Validate Pydantic models for human-centric critic evaluation."""

    def test_hook_30s_metrics(self):
        metrics = Hook30sMetrics(
            duration_analyzed=15.0,
            cut_count_in_30s=4,
            hook_avd_score=8.5,
            theme_setup_score=8.0,
            has_dead_air=False,
            is_engaging=True,
        )
        assert metrics.duration_analyzed == 15.0
        assert metrics.cut_count_in_30s == 4
        assert metrics.hook_avd_score == 8.5
        assert metrics.is_engaging is True
        assert not metrics.has_dead_air

    def test_thumbnail_metrics(self):
        thumb = ThumbnailMetrics(
            candidate_clip_reference="clip_action_002",
            candidate_timestamp=2.5,
            visual_interest_score=9.1,
            ctr_score=9.3,
            thumbnail_rationale="High contrast athletic jump with facial expression",
            is_viable=True,
        )
        assert thumb.candidate_clip_reference == "clip_action_002"
        assert thumb.ctr_score == 9.3
        assert thumb.is_viable is True

    def test_storytelling_metrics(self):
        story = StorytellingMetrics(
            narrative_arc_score=8.5,
            emotional_peak_score=9.0,
            human_resonance_score=8.6,
            theme_alignment_score=8.8,
        )
        assert story.narrative_arc_score == 8.5
        assert story.human_resonance_score == 8.6

    def test_critic_review_result_serialization(self):
        hook = Hook30sMetrics(
            duration_analyzed=14.0,
            cut_count_in_30s=4,
            hook_avd_score=8.8,
            theme_setup_score=8.5,
            has_dead_air=False,
            is_engaging=True,
        )
        thumb = ThumbnailMetrics(
            candidate_clip_reference="clip_action_002",
            candidate_timestamp=2.5,
            visual_interest_score=9.0,
            ctr_score=9.2,
            thumbnail_rationale="Dynamic action shot",
            is_viable=True,
        )
        story = StorytellingMetrics(
            narrative_arc_score=8.5,
            emotional_peak_score=8.8,
            human_resonance_score=8.6,
            theme_alignment_score=8.5,
        )
        result = CriticReviewResult(
            passed=True,
            score=8.8,
            hook_30s_score=8.8,
            thumbnail_score=9.2,
            storytelling_score=8.6,
            feedback="### Critic Report: [PASS]",
            recommendations=[],
            candidate_thumbnail=thumb,
            hook_30s_metrics=hook,
            storytelling_metrics=story,
            loop_action="exit_loop",
        )

        d = result.to_dict()
        assert d["passed"] is True
        assert d["loop_action"] == "exit_loop"
        assert d["candidate_thumbnail"]["candidate_clip_reference"] == "clip_action_002"

        json_str = result.to_json()
        assert "exit_loop" in json_str

        reconstructed = CriticReviewResult.from_json(json_str)
        assert reconstructed.score == 8.8
        assert reconstructed.passed is True
        assert reconstructed.candidate_thumbnail.candidate_clip_reference == "clip_action_002"


# ==============================================================================
# 2. Heuristic Calculation Tests
# ==============================================================================

class TestCriticHeuristics:
    """Validate 30s hook, thumbnail selection, and storytelling heuristics."""

    def test_calculate_30s_hook_engaging(self):
        edl = _build_sample_passing_edl()
        metrics = calculate_30s_hook_metrics(edl, theme="Travel sunset adventure")
        assert metrics.cut_count_in_30s == 4
        assert metrics.duration_analyzed == 11.5  # 2.5 + 3.0 + 3.0 + 3.0 = 11.5
        assert not metrics.has_dead_air
        assert metrics.is_engaging is True
        assert metrics.hook_avd_score >= 7.0
        assert metrics.theme_setup_score >= 7.0

    def test_calculate_30s_hook_dead_air(self):
        edl = _build_sample_failing_edl()
        metrics = calculate_30s_hook_metrics(edl, theme="Action highlights")
        assert metrics.has_dead_air is True  # First cut is 12s (>8s threshold)
        assert metrics.is_engaging is False
        assert metrics.hook_avd_score < 7.0

    def test_calculate_thumbnail_selection(self):
        edl = _build_sample_passing_edl()
        thumb = calculate_thumbnail_metrics(edl, theme="Travel adventure")
        assert thumb.candidate_clip_reference in {"clip_action_002", "clip_drone_001", "clip_reaction_003"}
        assert thumb.is_viable is True
        assert thumb.ctr_score >= 7.0
        assert thumb.candidate_timestamp > 0.0

    def test_calculate_storytelling_metrics(self):
        edl = _build_sample_passing_edl()
        story = calculate_storytelling_metrics(edl, theme="sunset adventure")
        assert story.narrative_arc_score >= 8.0
        assert story.emotional_peak_score >= 8.0
        assert story.human_resonance_score >= 8.0

    def test_evaluate_critic_heuristically_pass(self):
        edl = _build_sample_passing_edl()
        result = evaluate_critic_heuristically(edl, theme="sunset adventure")
        assert result.passed is True
        assert result.score >= 7.0
        assert result.loop_action == "exit_loop"
        assert "[PASS]" in result.feedback
        assert "exit_loop" in result.feedback

    def test_evaluate_critic_heuristically_fail(self):
        edl = _build_sample_failing_edl()
        result = evaluate_critic_heuristically(edl, theme="fast action sports")
        assert result.passed is False
        assert result.score < 7.0
        assert result.loop_action == "append_to_state"
        assert "[REVISE]" in result.feedback
        assert len(result.recommendations) > 0

    def test_evaluate_critic_heuristically_empty_error(self):
        empty_edl = EditDecisionList(entries=[])
        with pytest.raises(ValueError, match="Cannot review an empty EditDecisionList as critic"):
            evaluate_critic_heuristically(empty_edl)


# ==============================================================================
# 3. Loop Control Tools Tests
# ==============================================================================

class TestLoopTools:
    """Validate loop control tools: exit_loop and append_to_state."""

    def test_exit_loop_tool(self):
        ctx = _create_mock_tool_context(initial_state={"loop_status": "iterating"})
        msg = exit_loop(tool_context=ctx)
        assert "exited successfully" in msg
        assert ctx.actions.escalate is True
        assert ctx.actions.skip_summarization is True
        assert ctx.state.get("loop_status") == "exited"
        assert ctx.state.get("exit_loop_invoked") is True

    def test_exit_loop_without_context(self):
        msg = exit_loop(tool_context=None)
        assert "exited successfully" in msg

    def test_append_to_state_tool(self):
        ctx = _create_mock_tool_context(initial_state={})
        msg1 = append_to_state(key="revision_feedback", value="First revision critique", tool_context=ctx)
        assert "entry #1" in msg1
        assert ctx.state.get("revision_feedback") == ["First revision critique"]
        assert ctx.state.get("loop_status") == "iterating"

        msg2 = append_to_state(key="revision_feedback", value="Second revision critique", tool_context=ctx)
        assert "entry #2" in msg2
        assert len(ctx.state.get("revision_feedback")) == 2
        assert ctx.state.get("revision_feedback")[1] == "Second revision critique"

    def test_append_to_state_without_context(self):
        msg = append_to_state(key="test_key", value="val", tool_context=None)
        assert "Warning" in msg


# ==============================================================================
# 4. ADK Critic Tool & State Sync Tests
# ==============================================================================

class TestCriticTools:
    """Validate review_edl_as_critic tool and session state synchronization."""

    def test_review_edl_as_critic_pass_triggers_exit_loop(self):
        edl = _build_sample_passing_edl()
        ctx = _create_mock_tool_context(initial_state={"edl": edl, "theme": "Mountain trip"})

        result = review_edl_as_critic(
            tool_context=ctx,
            edl=edl,
            theme="Mountain trip",
            offline=True,
        )

        assert result.passed is True
        assert result.loop_action == "exit_loop"
        assert ctx.actions.escalate is True  # exit_loop invoked!
        assert ctx.state.get("critic_passed") is True
        assert ctx.state.get("critic_score") == result.score
        assert ctx.state.get("critic_feedback") == result.feedback
        assert ctx.state.get("loop_status") == "exited"
        assert "candidate_thumbnail" in ctx.state

    def test_review_edl_as_critic_fail_triggers_append_to_state(self):
        edl = _build_sample_failing_edl()
        ctx = _create_mock_tool_context(initial_state={"edl": edl, "theme": "Action"})

        result = review_edl_as_critic(
            tool_context=ctx,
            edl=edl,
            theme="Action",
            offline=True,
        )

        assert result.passed is False
        assert result.loop_action == "append_to_state"
        assert not ctx.actions.escalate  # Loop continues!
        assert ctx.state.get("critic_passed") is False
        assert ctx.state.get("loop_status") == "iterating"
        assert "revision_feedback" in ctx.state
        assert len(ctx.state.get("revision_feedback")) == 1

    def test_review_edl_as_critic_reads_from_state(self):
        edl = _build_sample_passing_edl()
        ctx = _create_mock_tool_context(initial_state={"edl": edl, "theme": "Sunset travel"})

        result = review_edl_as_critic(tool_context=ctx, offline=True)
        assert result.passed is True
        assert ctx.state.get("critic_passed") is True

    def test_review_edl_as_critic_missing_edl_error(self):
        ctx = _create_mock_tool_context(initial_state={})
        with pytest.raises(ValueError, match="No Edit Decision List provided or found"):
            review_edl_as_critic(tool_context=ctx, offline=True)

    def test_get_critic_review_from_state(self):
        edl = _build_sample_passing_edl()
        ctx = _create_mock_tool_context(initial_state={"edl": edl, "theme": "Sunset travel"})
        result = review_edl_as_critic(tool_context=ctx, offline=True)

        extracted = get_critic_review_from_state(ctx.state)
        assert extracted is not None
        assert extracted.passed is True
        assert extracted.score == result.score
        assert extracted.candidate_thumbnail.candidate_clip_reference == result.candidate_thumbnail.candidate_clip_reference


# ==============================================================================
# 5. Critic Agent Initialization & Execution Tests
# ==============================================================================

class TestCriticAgent:
    """Validate CriticAgent attributes, offline run, state delta, and escalation."""

    def test_critic_agent_initialization(self):
        agent = create_critic_agent(offline=True)
        assert agent.name == "critic_agent"
        assert agent.offline is True
        assert "Human-Centric" in agent.instruction
        assert "exit_loop" in agent.instruction
        assert "append_to_state" in agent.instruction

        tool_names = [getattr(t, "__name__", str(t)) for t in agent.tools]
        assert "review_edl_as_critic" in tool_names
        assert "exit_loop" in tool_names
        assert "append_to_state" in tool_names

    def test_critic_agent_singleton(self):
        assert critic_agent is not None
        assert isinstance(critic_agent, CriticAgent)

    def test_critic_agent_run_async_passing_edl(self):
        async def _test():
            agent = create_critic_agent(offline=True)
            edl = _build_sample_passing_edl()

            mock_session = Session(
                id="session_critic_test",
                state={"edl": edl, "theme": "Epic travel highlight"},
                app_name="videoguru",
                user_id="test_user",
            )
            mock_ctx = MagicMock()
            mock_ctx.session = mock_session

            events = []
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)

            assert len(events) == 1
            event = events[0]
            assert event.author == "critic_agent"
            assert "[PASS]" in event.content.parts[0].text
            assert event.actions.escalate is True
            assert event.actions.state_delta["critic_passed"] is True
            assert event.actions.state_delta["loop_status"] == "exited"
            assert "candidate_thumbnail" in event.actions.state_delta

        asyncio.run(_test())

    def test_critic_agent_run_async_failing_edl(self):
        async def _test():
            agent = create_critic_agent(offline=True)
            edl = _build_sample_failing_edl()

            mock_session = Session(
                id="session_critic_fail",
                state={"edl": edl, "theme": "Action sports"},
                app_name="videoguru",
                user_id="test_user",
            )
            mock_ctx = MagicMock()
            mock_ctx.session = mock_session

            events = []
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)

            assert len(events) == 1
            event = events[0]
            assert event.author == "critic_agent"
            assert "[REVISE]" in event.content.parts[0].text
            assert event.actions.escalate is False
            assert event.actions.state_delta["critic_passed"] is False
            assert event.actions.state_delta["loop_status"] == "iterating"
            assert "revision_feedback" in event.actions.state_delta

        asyncio.run(_test())

    def test_critic_agent_run_async_missing_edl(self):
        async def _test():
            agent = create_critic_agent(offline=True)
            mock_session = Session(
                id="session_critic_no_edl",
                state={"theme": "Any theme"},
                app_name="videoguru",
                user_id="test_user",
            )
            mock_ctx = MagicMock()
            mock_ctx.session = mock_session

            events = []
            async for event in agent._run_async_impl(mock_ctx):
                events.append(event)

            assert len(events) == 1
            assert "No Edit Decision List found" in events[0].content.parts[0].text

        asyncio.run(_test())

    def test_critic_agent_programmatic_run_review(self):
        agent = create_critic_agent(offline=True)
        edl = _build_sample_passing_edl()
        session = Session(
            id="session_prog",
            state={"theme": "Mountain sunset"},
            app_name="videoguru",
            user_id="test_user",
        )

        result = agent.run_review(session=session, edl=edl, theme="Mountain sunset")
        assert result.passed is True
        assert session.state["critic_passed"] is True
        assert session.state["critic_score"] == result.score
        assert "candidate_thumbnail" in session.state

    def test_evaluate_critic_with_gemini_fallback(self):
        edl = _build_sample_passing_edl()
        result = evaluate_critic_with_gemini(edl, theme="Test theme", client=None)
        assert result.score >= 7.0
        assert result.passed is True


# ==============================================================================
# 6. Multi-Agent Pipeline Integration Tests (Curation -> Reviewer -> Critic)
# ==============================================================================

class TestMultiAgentPipelineIntegration:
    """Validate seamless interaction across Curation, Reviewer, and Critic agents."""

    @patch("tools.curation_tools.analyze_clip")
    def test_curation_reviewer_critic_pipeline_handoff(self, mock_analyze):
        mock_analyze.side_effect = [
            EDLEntry(
                file_reference="clip_01",
                start_trim=0.0,
                end_trim=2.5,
                scene_rationale="Fast action shot of athlete mountain biking",
                transition_intent=TransitionIntent.CUT,
                engagement_score=9.0,
            ),
            EDLEntry(
                file_reference="clip_02",
                start_trim=1.0,
                end_trim=4.0,
                scene_rationale="Dynamic downhill speed run",
                transition_intent=TransitionIntent.WIPE,
                engagement_score=8.5,
            ),
            EDLEntry(
                file_reference="clip_03",
                start_trim=0.5,
                end_trim=3.5,
                scene_rationale="Vibrant reaction celebration at finish line",
                transition_intent=TransitionIntent.DISSOLVE,
                engagement_score=8.8,
            ),
        ]

        # 1. Synthesize media manifest
        manifest = [
            ClipManifestEntry(
                clip_id="clip_01",
                absolute_path="C:/media/clip_01.mp4",
                file_name="clip_01.mp4",
                duration_seconds=10.0,
                frame_rate=30.0,
                resolution="1920x1080",
                width=1920,
                height=1080,
                video_codec="h264",
                audio_codec="aac",
                has_audio=True,
                file_size_bytes=1000000,
            ),
            ClipManifestEntry(
                clip_id="clip_02",
                absolute_path="C:/media/clip_02.mp4",
                file_name="clip_02.mp4",
                duration_seconds=12.0,
                frame_rate=30.0,
                resolution="1920x1080",
                width=1920,
                height=1080,
                video_codec="h264",
                audio_codec="aac",
                has_audio=True,
                file_size_bytes=1200000,
            ),
            ClipManifestEntry(
                clip_id="clip_03",
                absolute_path="C:/media/clip_03.mp4",
                file_name="clip_03.mp4",
                duration_seconds=8.0,
                frame_rate=30.0,
                resolution="1920x1080",
                width=1920,
                height=1080,
                video_codec="h264",
                audio_codec="aac",
                has_audio=True,
                file_size_bytes=800000,
            ),
        ]

        session = Session(
            id="session_multi_agent",
            state={"theme": "High energy action sports vlog", "clip_manifest": manifest},
            app_name="videoguru",
            user_id="test_user",
        )

        # 2. Run Phase II Curation (assemble_edl_from_manifest)
        edl = assemble_edl_from_manifest(
            manifest=manifest,
            theme=session.state["theme"],
            offline=True,
        )
        assert len(edl) == 3
        session.state["edl"] = edl

        # 3. Run Phase III Reviewer Agent (Algorithmic)
        reviewer = create_reviewer_agent(offline=True)
        review_res = reviewer.run_review(session=session, offline=True)
        assert session.state["reviewer_passed"] is not None
        assert session.state["reviewer_score"] is not None

        # 4. Run Phase III Critic Agent (Human-Centric)
        critic = create_critic_agent(offline=True)
        critic_res = critic.run_review(session=session, offline=True)
        assert session.state["critic_passed"] is not None
        assert session.state["critic_score"] is not None
        assert "candidate_thumbnail" in session.state

        # Both agents evaluated the exact same curated EDL
        assert critic_res.candidate_thumbnail.candidate_clip_reference in edl.clip_references


# ==============================================================================
# 7. CLI Tests
# ==============================================================================

class TestCriticCli:
    """Validate the --critic-review / --critic CLI options in main.py."""

    def test_critic_cli_execution(self, tmp_path):
        edl = _build_sample_passing_edl()
        edl_file = tmp_path / "test_edl.json"
        edl.save_json(edl_file)

        cmd = [
            sys.executable,
            "main.py",
            "--critic",
            str(edl_file),
            "--theme",
            "Epic sunset adventure",
            "--offline",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        output = proc.stdout
        assert "VideoGuru: Critic Agent (Human-Centric Review - SPEC-011)" in output
        assert "Human-Centric Verdict: [PASS]" in output
        assert "30s Hook AVD:" in output
        assert "Thumbnail CTR:" in output
        assert "Storytelling:" in output
        assert "Loop Action:           exit_loop" in output

    def test_critic_cli_failing_edl(self, tmp_path):
        edl = _build_sample_failing_edl()
        edl_file = tmp_path / "failing_edl.json"
        edl.save_json(edl_file)

        cmd = [
            sys.executable,
            "main.py",
            "--critic",
            str(edl_file),
            "--theme",
            "Fast action",
            "--offline",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        output = proc.stdout
        assert "Human-Centric Verdict: [REVISE]" in output
        assert "Loop Action:           append_to_state" in output
        assert "Dead Air: Yes" in output

    def test_critic_cli_file_not_found(self):
        cmd = [
            sys.executable,
            "main.py",
            "--critic",
            "non_existent_file.json",
            "--offline",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.returncode != 0
        assert "not found" in proc.stderr
