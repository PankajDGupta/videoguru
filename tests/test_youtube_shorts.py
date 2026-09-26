"""Unit and integration tests for SPEC-032: YouTube Shorts Format & Safe-Zone Text Burning (1080x1920).

Tests:
1. Configuration & Settings: VIDEO_TYPE, TARGET_RESOLUTION, RESOLUTION_MAP.
2. Intent Capture: Auto-detection of YouTube Shorts from theme keywords.
3. FFmpeg Command Builder: Normalization to 1080x1920, Shorts-safe drawtext positions, and elevated subtitle margins.
4. Overlay Text Tools: Shorts-optimized font sizes (46px / 36px) and safe-zone burn-in.
5. Transition Renderer: Normalizing cuts to 1080x1920 based on session state or parameter.
6. Whisper Captioning: Elevated MarginV (220) and FontSize (24pt) for Shorts readability.
7. Enhancement & Rendering Pipeline: End-to-end execution producing 1080x1920 video with final_shorts naming.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import settings
from rendering.ffmpeg_builder import (
    build_caption_burn_command,
    build_drawtext_overlay_command,
    build_trim_command,
)
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from schemas.overlay_text import OverlayTextEntry, OverlayTextPlan, OverlayTextStyle, TextPosition
from tools.intent_tools import record_theme
from tools.overlay_text_tools import (
    HOOK_FONT_SIZE,
    HOOK_FONT_SIZE_SHORTS,
    SCENE_FONT_SIZE,
    SCENE_FONT_SIZE_SHORTS,
    _generate_mock_overlay_texts,
    burn_overlay_text,
)
from tools.transition_renderer import render_with_transitions
from tools.whisper_captioning import burn_subtitles_to_video, generate_captions
from agents.enhancement_rendering import EnhancementRenderingAgent


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_manifest() -> list[ClipManifestEntry]:
    return [
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
        ),
    ]


@pytest.fixture
def sample_edl() -> EditDecisionList:
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_01",
                start_trim=1.0,
                end_trim=5.0,
                scene_rationale="Shorts hook clip",
                transition_intent=TransitionIntent.CUT,
            ),
            EDLEntry(
                file_reference="clip_02",
                start_trim=2.0,
                end_trim=7.0,
                scene_rationale="Shorts action clip",
                transition_intent=TransitionIntent.WIPE,
            ),
        ]
    )


# ---------------------------------------------------------
# 1. Configuration & Settings Tests
# ---------------------------------------------------------

class TestShortsConfiguration:
    def test_settings_supported_video_types(self):
        assert "shorts" in settings.SUPPORTED_VIDEO_TYPES
        assert "vertical" in settings.SUPPORTED_VIDEO_TYPES
        assert "landscape" in settings.SUPPORTED_VIDEO_TYPES

    def test_settings_resolution_map(self):
        assert settings.RESOLUTION_MAP["shorts"] == "1080x1920"
        assert settings.RESOLUTION_MAP["vertical"] == "1080x1920"
        assert settings.RESOLUTION_MAP["landscape"] == "1920x1080"

    def test_validate_settings_valid_video_type(self):
        assert settings.validate_settings() is True


# ---------------------------------------------------------
# 2. Intent Capture Tests (Auto-detecting Shorts)
# ---------------------------------------------------------

class TestShortsIntentDetection:
    def test_record_theme_detects_shorts(self):
        mock_ctx = MagicMock()
        mock_ctx.state = {}
        record_theme("Morning workout session for youtube shorts", mock_ctx)

        assert mock_ctx.state["theme"] == "Morning workout session for youtube shorts"
        assert mock_ctx.state["video_type"] == "shorts"
        assert mock_ctx.state["target_resolution"] == "1080x1920"

    def test_record_theme_detects_vertical_reel(self):
        mock_ctx = MagicMock()
        mock_ctx.state = {}
        record_theme("My travel reels highlights", mock_ctx)

        assert mock_ctx.state["video_type"] == "shorts"
        assert mock_ctx.state["target_resolution"] == "1080x1920"

    def test_record_theme_landscape_no_shorts(self):
        mock_ctx = MagicMock()
        mock_ctx.state = {}
        record_theme("Full 20-minute landscape documentary vlog", mock_ctx)

        assert "video_type" not in mock_ctx.state
        assert "target_resolution" not in mock_ctx.state


# ---------------------------------------------------------
# 3. FFmpeg Command Builder Tests
# ---------------------------------------------------------

class TestShortsFFmpegCommands:
    def test_build_trim_command_1080x1920(self):
        cmd = build_trim_command(
            clip_path="input.mp4",
            start=0.0,
            end=5.0,
            output_path="trimmed.mp4",
            target_resolution="1080x1920",
            target_fps=30.0,
        )
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale=1080:1920:force_original_aspect_ratio=decrease" in vf
        assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2" in vf
        assert "setsar=1" in vf
        assert "fps=30.0" in vf

    def test_build_caption_burn_command_shorts_mode(self):
        cmd = build_caption_burn_command(
            video_path="input.mp4",
            srt_path="subtitles.srt",
            output_path="captioned.mp4",
            is_shorts=True,
        )
        vf = cmd[cmd.index("-vf") + 1]
        # Shorts mode defaults: FontSize=24, MarginV=220, Alignment=2
        assert "FontSize=24" in vf
        assert "MarginV=220" in vf
        assert "Alignment=2" in vf

    def test_build_caption_burn_command_resolution_1080x1920(self):
        cmd = build_caption_burn_command(
            video_path="input.mp4",
            srt_path="subtitles.srt",
            output_path="captioned.mp4",
            target_resolution="1080x1920",
        )
        vf = cmd[cmd.index("-vf") + 1]
        assert "FontSize=24" in vf
        assert "MarginV=220" in vf
        assert "Alignment=2" in vf

    def test_build_caption_burn_command_landscape_default(self):
        cmd = build_caption_burn_command(
            video_path="input.mp4",
            srt_path="subtitles.srt",
            output_path="captioned.mp4",
            target_resolution="1920x1080",
        )
        vf = cmd[cmd.index("-vf") + 1]
        assert "FontSize=16" in vf
        assert "MarginV" not in vf

    def test_build_drawtext_overlay_command_shorts_safe_zones(self):
        entries = [
            {
                "text": "EPIC SHORT HOOK",
                "start_time": 0.0,
                "end_time": 3.0,
                "style": {"position": "upper_third"},
            },
            {
                "text": "LOWER TEXT",
                "start_time": 3.0,
                "end_time": 6.0,
                "style": {"position": "lower_third"},
            },
            {
                "text": "BOTTOM TEXT",
                "start_time": 6.0,
                "end_time": 9.0,
                "style": {"position": "bottom"},
            },
        ]
        cmd = build_drawtext_overlay_command(
            video_path="input.mp4",
            overlay_entries=entries,
            output_path="output.mp4",
            is_shorts=True,
        )
        vf = cmd[cmd.index("-vf") + 1]
        # In shorts mode, upper_third is h*0.18, lower_third is h*0.62, bottom is h*0.65 (clamped)
        assert "y=h*0.18" in vf
        assert "y=h*0.62" in vf
        assert "y=h*0.65" in vf

    def test_build_drawtext_overlay_command_landscape_zones(self):
        entries = [
            {
                "text": "LANDSCAPE HOOK",
                "start_time": 0.0,
                "end_time": 3.0,
                "style": {"position": "upper_third"},
            },
            {
                "text": "LANDSCAPE BOTTOM",
                "start_time": 3.0,
                "end_time": 6.0,
                "style": {"position": "bottom"},
            },
        ]
        cmd = build_drawtext_overlay_command(
            video_path="input.mp4",
            overlay_entries=entries,
            output_path="output.mp4",
            is_shorts=False,
            target_resolution="1920x1080",
        )
        vf = cmd[cmd.index("-vf") + 1]
        assert "y=h*0.15" in vf
        assert "y=h*0.88" in vf


# ---------------------------------------------------------
# 4. Overlay Text Tools Tests
# ---------------------------------------------------------

class TestShortsOverlayTextTools:
    def test_mock_overlay_texts_font_scaling(self, sample_edl):
        # Landscape
        landscape_plan = _generate_mock_overlay_texts(sample_edl, "Test", is_shorts=False)
        assert landscape_plan.entries[0].style.font_size == HOOK_FONT_SIZE  # 56
        assert landscape_plan.entries[1].style.font_size == SCENE_FONT_SIZE  # 42

        # Shorts (1080x1920)
        shorts_plan = _generate_mock_overlay_texts(sample_edl, "Test", is_shorts=True)
        assert shorts_plan.entries[0].style.font_size == HOOK_FONT_SIZE_SHORTS  # 46
        assert shorts_plan.entries[1].style.font_size == SCENE_FONT_SIZE_SHORTS  # 36

    @patch("tools.overlay_text_tools.execute_ffmpeg_command")
    @patch("tools.overlay_text_tools.build_drawtext_overlay_command")
    def test_burn_overlay_text_propagates_shorts(self, mock_build, mock_exec, tmp_path):
        dummy_in = tmp_path / "in.mp4"
        dummy_in.write_text("fake video")
        dummy_out = tmp_path / "out.mp4"

        plan = [
            {
                "text": "Shorts text",
                "start_time": 0.0,
                "end_time": 2.0,
                "style": {"position": "upper_third"},
            }
        ]

        mock_build.return_value = ["ffmpeg", "-y", "-i", str(dummy_in), str(dummy_out)]

        res = burn_overlay_text(
            video_path=dummy_in,
            overlay_plan=plan,
            output_path=dummy_out,
            target_resolution="1080x1920",
            is_shorts=True,
        )

        assert res == str(dummy_out)
        mock_build.assert_called_once_with(
            video_path=dummy_in.resolve(),
            overlay_entries=plan,
            output_path=dummy_out.resolve(),
            target_resolution="1080x1920",
            is_shorts=True,
        )


# ---------------------------------------------------------
# 5. Transition Renderer Tests
# ---------------------------------------------------------

class TestShortsTransitionRenderer:
    @patch("tools.transition_renderer.execute_ffmpeg_command")
    @patch("tools.transition_renderer.build_trim_command")
    def test_render_with_transitions_single_cut_shorts(self, mock_trim, mock_exec, sample_manifest, tmp_path):
        single_cut_edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=0.0,
                    end_trim=5.0,
                    scene_rationale="Single clip",
                    transition_intent=TransitionIntent.CUT,
                )
            ]
        )
        dummy_out = tmp_path / "out_shorts.mp4"
        mock_trim.return_value = ["ffmpeg", "-y", "trim"]

        render_with_transitions(
            edl=single_cut_edl,
            clip_manifest=sample_manifest,
            output_path=dummy_out,
            target_resolution="1080x1920",
        )

        mock_trim.assert_called_once()
        assert mock_trim.call_args.kwargs["target_resolution"] == "1080x1920"

    @patch("tools.transition_renderer.execute_ffmpeg_command")
    @patch("tools.transition_renderer.build_trim_command")
    def test_render_with_transitions_state_shorts(self, mock_trim, mock_exec, sample_edl, sample_manifest, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.state = {
            "video_type": "shorts",
            "target_resolution": "1080x1920",
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
        }
        dummy_out = tmp_path / "multi_shorts.mp4"
        mock_trim.return_value = ["ffmpeg", "-y", "trim"]

        render_with_transitions(
            output_path=dummy_out,
            tool_context=mock_ctx,
        )

        for c in mock_trim.call_args_list:
            assert c.kwargs["target_resolution"] == "1080x1920"


# ---------------------------------------------------------
# 6. Whisper Captioning Tests
# ---------------------------------------------------------

class TestShortsWhisperCaptioning:
    @patch("tools.whisper_captioning.execute_ffmpeg_command")
    @patch("tools.whisper_captioning.build_caption_burn_command")
    def test_burn_subtitles_to_video_shorts(self, mock_build, mock_exec, tmp_path):
        vid_p = tmp_path / "test.mp4"
        vid_p.write_text("vid")
        srt_p = tmp_path / "test.srt"
        srt_p.write_text("srt")
        out_p = tmp_path / "out.mp4"

        mock_build.return_value = ["ffmpeg", "burn"]

        burn_subtitles_to_video(
            video_path=vid_p,
            srt_path=srt_p,
            output_path=out_p,
            target_resolution="1080x1920",
            is_shorts=True,
        )

        mock_build.assert_called_once_with(
            video_path=vid_p.resolve(),
            srt_path=srt_p.resolve(),
            output_path=out_p.resolve(),
            font_size=16,
            target_resolution="1080x1920",
            is_shorts=True,
        )

    @patch("tools.whisper_captioning.burn_subtitles_to_video")
    @patch("tools.whisper_captioning.write_srt_file")
    @patch("tools.whisper_captioning.transcribe_audio_whisper")
    def test_generate_captions_state_shorts(self, mock_transcribe, mock_write, mock_burn, tmp_path):
        dummy_vid = tmp_path / "v.mp4"
        dummy_vid.write_text("video")
        mock_transcribe.return_value = {"segments": []}
        mock_write.return_value = tmp_path / "v.srt"
        mock_burn.return_value = str(tmp_path / "v_captioned.mp4")

        mock_ctx = MagicMock()
        mock_ctx.state = {"video_type": "shorts", "target_resolution": "1080x1920"}

        generate_captions(
            video_path=dummy_vid,
            tool_context=mock_ctx,
            offline=True,
        )

        mock_burn.assert_called_once()
        assert mock_burn.call_args.kwargs["is_shorts"] is True
        assert mock_burn.call_args.kwargs.get("target_resolution") == "1080x1920"


# ---------------------------------------------------------
# 7. Enhancement & Rendering Agent Pipeline Tests
# ---------------------------------------------------------

class TestShortsEnhancementRenderingPipeline:
    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.burn_overlay_text")
    @patch("agents.enhancement_rendering.generate_captions")
    def test_execute_rendering_pipeline_shorts_mode(
        self,
        mock_caps,
        mock_overlay,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        step1_out = tmp_path / "step1.mp4"
        step1_out.write_text("step1")
        step1b_out = tmp_path / "step1b.mp4"
        step1b_out.write_text("step1b")
        step3_out = tmp_path / "step3.mp4"
        step3_out.write_text("step3")

        mock_transitions.return_value = str(step1_out)
        mock_overlay.return_value = str(step1b_out)
        mock_caps.return_value = {
            "captioned_video_path": str(step3_out),
            "srt_path": str(tmp_path / "subs.srt"),
        }

        agent = EnhancementRenderingAgent(offline=True)

        state = {
            "theme": "Fitness Shorts",
            "video_type": "shorts",
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
            "enable_captions": True,
        }

        res = agent.execute_rendering_pipeline(state=state)

        # Check resolution passed to step 1
        assert mock_transitions.call_args.kwargs["target_resolution"] == "1080x1920"

        # Check resolution passed to step 1b
        assert mock_overlay.call_args.kwargs["target_resolution"] == "1080x1920"
        assert mock_overlay.call_args.kwargs["is_shorts"] is True

        # Check resolution passed to step 3
        assert mock_caps.call_args.kwargs["target_resolution"] == "1080x1920"
        assert mock_caps.call_args.kwargs["is_shorts"] is True

        # Check naming contains final_shorts
        assert "final_shorts" in res["final_video_path"]
        assert state["target_resolution"] == "1080x1920"
        assert state["video_type"] == "shorts"
