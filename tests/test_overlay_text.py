"""Unit tests for SPEC-031: Overlay Text Agent.

Covers:
- OverlayTextStyle, OverlayTextEntry, OverlayTextPlan schema validation
- Timeline offset computation for cuts with xfade overlaps
- Mock overlay text generation (offline mode)
- FFmpeg drawtext command building
- burn_overlay_text path validation
- OverlayTextAgent initialization and offline mode
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from schemas.edl import EDLEntry, EditDecisionList
from schemas.overlay_text import (
    YOUTUBE_SHORTS_COLORS,
    OverlayTextEntry,
    OverlayTextPlan,
    OverlayTextStyle,
    TextPosition,
)
from tools.overlay_text_tools import (
    DEFAULT_OVERLAY_DURATION,
    HOOK_FONT_SIZE,
    SCENE_FONT_SIZE,
    _compute_cut_timeline_offsets,
    _generate_mock_overlay_texts,
)
from rendering.ffmpeg_builder import build_drawtext_overlay_command


# ---------------------
# Test Fixtures
# ---------------------

def _make_edl(count: int = 3, duration: float = 5.0) -> EditDecisionList:
    """Create a test EDL with `count` entries, each lasting `duration` seconds."""
    entries = [
        EDLEntry(
            file_reference=f"vid_{i:04d}",
            start_trim=0.0,
            end_trim=duration,
            scene_rationale=f"Test scene {i}",
            transition_intent="cut",
        )
        for i in range(count)
    ]
    return EditDecisionList(entries=entries)


# ---------------------
# Test TextPosition Enum
# ---------------------

class TestTextPosition:
    """Test suite for TextPosition enum."""

    def test_all_values(self):
        assert TextPosition.TOP.value == "top"
        assert TextPosition.UPPER_THIRD.value == "upper_third"
        assert TextPosition.CENTER.value == "center"
        assert TextPosition.LOWER_THIRD.value == "lower_third"
        assert TextPosition.BOTTOM.value == "bottom"


# ---------------------
# Test OverlayTextStyle
# ---------------------

class TestOverlayTextStyle:
    """Test suite for OverlayTextStyle model."""

    def test_default_values(self):
        style = OverlayTextStyle()
        assert style.font_family == "Impact"
        assert style.font_size == 48
        assert style.font_color == "#FFD700"
        assert style.border_color == "#000000"
        assert style.border_width == 3
        assert style.position == TextPosition.UPPER_THIRD
        assert style.box_enabled is True
        assert style.box_opacity == 0.5

    def test_custom_values(self):
        style = OverlayTextStyle(
            font_family="Arial Black",
            font_size=60,
            font_color="#FF1493",
            border_width=5,
            position=TextPosition.CENTER,
        )
        assert style.font_family == "Arial Black"
        assert style.font_size == 60
        assert style.font_color == "#FF1493"
        assert style.position == TextPosition.CENTER

    def test_font_size_bounds(self):
        with pytest.raises(ValidationError):
            OverlayTextStyle(font_size=5)
        with pytest.raises(ValidationError):
            OverlayTextStyle(font_size=200)

    def test_box_opacity_bounds(self):
        with pytest.raises(ValidationError):
            OverlayTextStyle(box_opacity=-0.1)
        with pytest.raises(ValidationError):
            OverlayTextStyle(box_opacity=1.5)

    def test_position_normalization(self):
        style = OverlayTextStyle(position="CENTER")
        assert style.position == TextPosition.CENTER

    def test_position_invalid_fallback(self):
        style = OverlayTextStyle(position="nonexistent")
        assert style.position == TextPosition.UPPER_THIRD

    def test_serialization_roundtrip(self):
        style = OverlayTextStyle(font_size=52, font_color="#00FFFF")
        data = style.model_dump()
        restored = OverlayTextStyle.model_validate(data)
        assert restored.font_size == 52
        assert restored.font_color == "#00FFFF"


# ---------------------
# Test OverlayTextEntry
# ---------------------

class TestOverlayTextEntry:
    """Test suite for OverlayTextEntry model."""

    def test_valid_entry(self):
        entry = OverlayTextEntry(
            cut_index=0,
            text="Back to the grind",
            start_time=0.3,
            end_time=3.8,
            is_hook=True,
        )
        assert entry.cut_index == 0
        assert entry.text == "Back to the grind"
        assert entry.is_hook is True
        assert entry.style.font_family == "Impact"  # default

    def test_text_strips_whitespace(self):
        entry = OverlayTextEntry(
            cut_index=1,
            text="  Trim me  ",
            start_time=1.0,
            end_time=4.0,
        )
        assert entry.text == "Trim me"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            OverlayTextEntry(cut_index=0, text="", start_time=0.0, end_time=3.0)

    def test_whitespace_only_text_rejected(self):
        with pytest.raises(ValidationError):
            OverlayTextEntry(cut_index=0, text="   ", start_time=0.0, end_time=3.0)

    def test_negative_cut_index_rejected(self):
        with pytest.raises(ValidationError):
            OverlayTextEntry(cut_index=-1, text="Test", start_time=0.0, end_time=3.0)

    def test_negative_start_time_rejected(self):
        with pytest.raises(ValidationError):
            OverlayTextEntry(cut_index=0, text="Test", start_time=-1.0, end_time=3.0)

    def test_custom_style(self):
        style = OverlayTextStyle(font_size=56, font_color="#FF1493")
        entry = OverlayTextEntry(
            cut_index=0,
            text="Hook text",
            start_time=0.0,
            end_time=3.0,
            style=style,
        )
        assert entry.style.font_size == 56
        assert entry.style.font_color == "#FF1493"


# ---------------------
# Test OverlayTextPlan
# ---------------------

class TestOverlayTextPlan:
    """Test suite for OverlayTextPlan container model."""

    def test_empty_plan(self):
        plan = OverlayTextPlan()
        assert len(plan) == 0
        assert not plan

    def test_plan_with_entries(self):
        entries = [
            OverlayTextEntry(cut_index=i, text=f"Text {i}", start_time=i * 5.0, end_time=i * 5.0 + 3.0)
            for i in range(3)
        ]
        plan = OverlayTextPlan(entries=entries, theme="Test theme")
        assert len(plan) == 3
        assert plan.theme == "Test theme"
        assert plan[0].text == "Text 0"

    def test_iteration(self):
        entries = [
            OverlayTextEntry(cut_index=i, text=f"Text {i}", start_time=0.0, end_time=3.0)
            for i in range(2)
        ]
        plan = OverlayTextPlan(entries=entries)
        texts = [e.text for e in plan]
        assert texts == ["Text 0", "Text 1"]

    def test_to_dict_list(self):
        entries = [
            OverlayTextEntry(cut_index=0, text="Hook!", start_time=0.0, end_time=3.0, is_hook=True),
        ]
        plan = OverlayTextPlan(entries=entries, theme="Gym day")
        dicts = plan.to_dict_list()
        assert len(dicts) == 1
        assert dicts[0]["text"] == "Hook!"
        assert dicts[0]["is_hook"] is True

    def test_from_list(self):
        raw = [
            {"cut_index": 0, "text": "Hello", "start_time": 0.0, "end_time": 3.0},
            {"cut_index": 1, "text": "World", "start_time": 5.0, "end_time": 8.0},
        ]
        plan = OverlayTextPlan.from_list(raw, theme="Test")
        assert len(plan) == 2
        assert plan.theme == "Test"
        assert plan[1].text == "World"


# ---------------------
# Test YOUTUBE_SHORTS_COLORS
# ---------------------

class TestYouTubeShortsColors:
    """Test the color palette constant."""

    def test_has_enough_colors(self):
        assert len(YOUTUBE_SHORTS_COLORS) >= 6

    def test_all_hex_format(self):
        for color in YOUTUBE_SHORTS_COLORS:
            assert color.startswith("#")
            assert len(color) == 7


# ---------------------
# Test Timeline Offset Computation
# ---------------------

class TestComputeCutTimelineOffsets:
    """Test _compute_cut_timeline_offsets function."""

    def test_single_cut(self):
        edl = _make_edl(count=1, duration=5.0)
        offsets = _compute_cut_timeline_offsets(edl, transition_duration=1.0)
        assert len(offsets) == 1
        assert offsets[0] == (0.0, 5.0)

    def test_two_cuts_with_overlap(self):
        edl = _make_edl(count=2, duration=5.0)
        offsets = _compute_cut_timeline_offsets(edl, transition_duration=1.0)
        assert len(offsets) == 2
        # First cut: 0.0 to 5.0
        assert offsets[0] == (0.0, 5.0)
        # Second cut starts at 5.0 - 1.0 overlap = 4.0
        assert offsets[1][0] == pytest.approx(4.0, abs=0.01)
        assert offsets[1][1] == pytest.approx(9.0, abs=0.01)

    def test_three_cuts(self):
        edl = _make_edl(count=3, duration=4.0)
        offsets = _compute_cut_timeline_offsets(edl, transition_duration=1.0)
        assert len(offsets) == 3
        # Verify each cut starts after previous minus overlap
        for i in range(1, len(offsets)):
            assert offsets[i][0] < offsets[i][1]  # start < end

    def test_zero_transition_duration(self):
        edl = _make_edl(count=2, duration=5.0)
        offsets = _compute_cut_timeline_offsets(edl, transition_duration=0.0)
        assert offsets[0] == (0.0, 5.0)
        assert offsets[1] == (5.0, 10.0)


# ---------------------
# Test Mock Overlay Text Generation
# ---------------------

class TestGenerateMockOverlayTexts:
    """Test _generate_mock_overlay_texts function."""

    def test_generates_entries_per_cut(self):
        edl = _make_edl(count=5, duration=5.0)
        plan = _generate_mock_overlay_texts(edl, "Gym workout motivation")
        assert len(plan) == 5

    def test_first_entry_is_hook(self):
        edl = _make_edl(count=3, duration=5.0)
        plan = _generate_mock_overlay_texts(edl, "Test theme")
        assert plan[0].is_hook is True
        assert plan[0].style.font_size == HOOK_FONT_SIZE

    def test_subsequent_entries_are_not_hooks(self):
        edl = _make_edl(count=3, duration=5.0)
        plan = _generate_mock_overlay_texts(edl, "Test theme")
        for entry in plan.entries[1:]:
            assert entry.is_hook is False
            assert entry.style.font_size == SCENE_FONT_SIZE

    def test_rotating_colors(self):
        edl = _make_edl(count=4, duration=5.0)
        plan = _generate_mock_overlay_texts(edl, "Test")
        colors = [e.style.font_color for e in plan.entries]
        # Each cut should get the next color in the palette
        for i, color in enumerate(colors):
            assert color == YOUTUBE_SHORTS_COLORS[i % len(YOUTUBE_SHORTS_COLORS)]

    def test_text_timing_within_cut_bounds(self):
        edl = _make_edl(count=3, duration=5.0)
        offsets = _compute_cut_timeline_offsets(edl, transition_duration=1.0)
        plan = _generate_mock_overlay_texts(edl, "Test", transition_duration=1.0)
        for i, entry in enumerate(plan.entries):
            cut_start, cut_end = offsets[i]
            assert entry.start_time >= cut_start
            assert entry.end_time <= cut_end + 0.5  # Allow small margin

    def test_overlay_duration_capped(self):
        edl = _make_edl(count=1, duration=2.0)
        plan = _generate_mock_overlay_texts(edl, "Test")
        entry = plan[0]
        overlay_duration = entry.end_time - entry.start_time
        assert overlay_duration <= DEFAULT_OVERLAY_DURATION + 0.01

    def test_theme_stored_in_plan(self):
        edl = _make_edl(count=1)
        plan = _generate_mock_overlay_texts(edl, "My cool theme")
        assert plan.theme == "My cool theme"


# ---------------------
# Test FFmpeg Drawtext Command Building
# ---------------------

class TestBuildDrawtextOverlayCommand:
    """Test build_drawtext_overlay_command function."""

    def test_basic_command_structure(self):
        entries = [
            {
                "text": "Hook text!",
                "start_time": 0.3,
                "end_time": 3.8,
                "style": {
                    "font_family": "Impact",
                    "font_size": 56,
                    "font_color": "#FFD700",
                    "border_color": "#000000",
                    "border_width": 3,
                    "shadow_x": 2,
                    "shadow_y": 2,
                    "position": "upper_third",
                    "box_enabled": True,
                    "box_color": "black",
                    "box_opacity": 0.5,
                    "box_border_width": 15,
                },
            }
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command(
                video_path="input.mp4",
                overlay_entries=entries,
                output_path="output.mp4",
            )
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-vf" in cmd
        assert "drawtext" in cmd[cmd.index("-vf") + 1]
        assert "output.mp4" == cmd[-1]

    def test_multiple_entries_chained(self):
        entries = [
            {"text": "First", "start_time": 0.0, "end_time": 3.0, "style": {}},
            {"text": "Second", "start_time": 5.0, "end_time": 8.0, "style": {}},
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")
        vf_str = cmd[cmd.index("-vf") + 1]
        # Multiple drawtext filters should be comma-separated
        assert vf_str.count("drawtext") == 2

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="overlay_entries list cannot be empty"):
            build_drawtext_overlay_command("in.mp4", [], "out.mp4")

    def test_empty_video_path_raises(self):
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            build_drawtext_overlay_command("", [{"text": "x"}], "out.mp4")

    def test_empty_output_path_raises(self):
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            build_drawtext_overlay_command("in.mp4", [{"text": "x"}], "")

    def test_text_with_special_chars_escaped(self):
        entries = [
            {"text": "It's awesome: 100%", "start_time": 0.0, "end_time": 3.0, "style": {}},
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")
        vf_str = cmd[cmd.index("-vf") + 1]
        # Colon should be escaped for FFmpeg
        assert "\\:" in vf_str
        # Percent should be doubled
        assert "%%" in vf_str

    def test_enable_timing(self):
        entries = [
            {"text": "Timed", "start_time": 2.5, "end_time": 6.0, "style": {}},
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")
        vf_str = cmd[cmd.index("-vf") + 1]
        assert "gte(t,2.500)*lte(t,6.000)" in vf_str


# ---------------------
# Test OverlayTextAgent
# ---------------------

class TestOverlayTextAgent:
    """Test OverlayTextAgent initialization and configuration."""

    def test_agent_creation(self):
        from agents.overlay_text import create_overlay_text_agent
        agent = create_overlay_text_agent(offline=True)
        assert agent.name == "overlay_text_agent"
        assert agent.offline is True

    def test_agent_custom_name(self):
        from agents.overlay_text import create_overlay_text_agent
        agent = create_overlay_text_agent(name="custom_overlay", offline=True)
        assert agent.name == "custom_overlay"

    def test_agent_has_tools(self):
        from agents.overlay_text import OverlayTextAgent
        agent = OverlayTextAgent(offline=True)
        assert len(agent.tools) > 0

    def test_agent_instruction(self):
        from agents.overlay_text import OVERLAY_TEXT_INSTRUCTION
        assert "overlay" in OVERLAY_TEXT_INSTRUCTION.lower()
        assert "hook" in OVERLAY_TEXT_INSTRUCTION.lower()
        assert "youtube" in OVERLAY_TEXT_INSTRUCTION.lower()
