"""Unit tests for SPEC-017: Transition Rendering Tool (tools/transition_renderer.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest

from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.transition_renderer import (
    _extract_transitions,
    _resolve_clip_manifest,
    _resolve_edl,
    render_with_transitions,
)


@pytest.fixture
def sample_clip_manifest() -> list[ClipManifestEntry]:
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
        ClipManifestEntry(
            clip_id="clip_03",
            absolute_path="C:/media/clip_03.mp4",
            file_name="clip_03.mp4",
            duration_seconds=15.0,
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
                end_trim=6.0,
                scene_rationale="Intro shot",
                transition_intent=TransitionIntent.CUT,
            ),
            EDLEntry(
                file_reference="clip_02",
                start_trim=2.0,
                end_trim=7.0,
                scene_rationale="Action shot",
                transition_intent=TransitionIntent.WIPE,
            ),
            EDLEntry(
                file_reference="clip_03",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Outro shot",
                transition_intent=TransitionIntent.FADE,
            ),
        ]
    )


class TestResolveEDL:
    def test_resolve_from_instance(self, sample_edl):
        res = _resolve_edl(sample_edl, {})
        assert res == sample_edl
        assert len(res) == 3

    def test_resolve_from_dict(self, sample_edl):
        res = _resolve_edl(sample_edl.model_dump(), {})
        assert len(res) == 3
        assert res[0].file_reference == "clip_01"

    def test_resolve_from_list_of_entries(self, sample_edl):
        res = _resolve_edl(sample_edl.entries, {})
        assert len(res) == 3

    def test_resolve_from_json_string(self, sample_edl):
        res = _resolve_edl(sample_edl.model_dump_json(), {})
        assert len(res) == 3

    def test_resolve_from_state(self, sample_edl):
        state = {"edl": sample_edl}
        res = _resolve_edl(None, state)
        assert res == sample_edl

    def test_resolve_from_file(self, tmp_path, sample_edl):
        p = tmp_path / "edl.json"
        p.write_text(sample_edl.model_dump_json(), encoding="utf-8")
        res = _resolve_edl(str(p), {})
        assert len(res) == 3

    def test_resolve_missing_raises_error(self):
        with pytest.raises(ValueError, match="No Edit Decision List provided"):
            _resolve_edl(None, {})

    def test_resolve_empty_edl_raises_error(self):
        empty_edl = EditDecisionList(entries=[])
        with pytest.raises(ValueError, match="Edit Decision List contains no cut entries"):
            _resolve_edl(empty_edl, {})


class TestResolveClipManifest:
    def test_resolve_from_list(self, sample_clip_manifest):
        res = _resolve_clip_manifest(sample_clip_manifest, {})
        assert res == sample_clip_manifest

    def test_resolve_from_dict_list(self, sample_clip_manifest):
        raw_list = [c.model_dump() for c in sample_clip_manifest]
        res = _resolve_clip_manifest(raw_list, {})
        assert len(res) == 3
        assert res[0].clip_id == "clip_01"

    def test_resolve_from_state(self, sample_clip_manifest):
        state = {"clip_manifest": sample_clip_manifest}
        res = _resolve_clip_manifest(None, state)
        assert res == sample_clip_manifest

    def test_resolve_from_json_file(self, tmp_path, sample_clip_manifest):
        p = tmp_path / "manifest.json"
        p.write_text(
            json.dumps([c.model_dump() for c in sample_clip_manifest]),
            encoding="utf-8",
        )
        res = _resolve_clip_manifest(str(p), {})
        assert len(res) == 3

    def test_resolve_missing_manifest_raises_error(self):
        with pytest.raises(ValueError, match="No clip manifest provided"):
            _resolve_clip_manifest(None, {})

    def test_resolve_empty_manifest_raises_error(self):
        with pytest.raises(ValueError, match="Clip manifest is empty"):
            _resolve_clip_manifest([], {})


class TestExtractTransitions:
    def test_extract_transitions(self, sample_edl):
        transitions = _extract_transitions(sample_edl)
        assert len(transitions) == 2
        assert transitions[0] == "wipeleft"
        assert transitions[1] == "fade"

    def test_extract_transitions_all_cut(self):
        edl = EditDecisionList(
            entries=[
                EDLEntry(file_reference="c1", start_trim=0.0, end_trim=2.0, scene_rationale="r1"),
                EDLEntry(file_reference="c2", start_trim=0.0, end_trim=2.0, scene_rationale="r2"),
            ]
        )
        transitions = _extract_transitions(edl)
        assert len(transitions) == 1
        assert transitions[0] == "fade"


class TestRenderWithTransitions:
    @patch("tools.transition_renderer.execute_ffmpeg_command")
    def test_render_single_cut(self, mock_exec, sample_clip_manifest, tmp_path):
        single_edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=1.0,
                    end_trim=5.0,
                    scene_rationale="Single clip",
                )
            ]
        )
        out_file = tmp_path / "out_single.mp4"

        class MockToolContext:
            def __init__(self):
                self.state = {}

        ctx = MockToolContext()
        res_path = render_with_transitions(
            edl=single_edl,
            clip_manifest=sample_clip_manifest,
            output_path=out_file,
            tool_context=ctx,
        )

        assert res_path == str(out_file.resolve())
        assert ctx.state["transition_rendered_path"] == str(out_file.resolve())
        assert mock_exec.call_count == 1
        cmd = mock_exec.call_args[0][0]
        assert "-ss" in cmd
        assert "1.000" in cmd
        assert "-to" in cmd
        assert "5.000" in cmd

    @patch("tools.transition_renderer.execute_ffmpeg_command")
    def test_render_multi_cut_with_transitions(
        self, mock_exec, sample_edl, sample_clip_manifest, tmp_path
    ):
        out_file = tmp_path / "rendered_output.mp4"
        staging_dir = tmp_path / "custom_staging"

        class MockToolContext:
            def __init__(self):
                self.state = {}

        ctx = MockToolContext()
        res = render_with_transitions(
            edl=sample_edl,
            clip_manifest=sample_clip_manifest,
            output_path=out_file,
            transition_duration=1.0,
            tool_context=ctx,
            staging_dir=staging_dir,
        )

        assert res == str(out_file.resolve())
        assert ctx.state["transition_rendered_path"] == str(out_file.resolve())
        assert mock_exec.call_count == 4

        final_cmd = mock_exec.call_args[0][0]
        assert "-filter_complex" in final_cmd
        fc_idx = final_cmd.index("-filter_complex")
        fc_arg = final_cmd[fc_idx + 1]
        assert "xfade=transition=wipeleft" in fc_arg
        assert "xfade=transition=fade" in fc_arg
        assert "acrossfade=d=1.0" in fc_arg

    @patch("tools.transition_renderer.execute_ffmpeg_command")
    def test_render_with_transitions_from_state(
        self, mock_exec, sample_edl, sample_clip_manifest, tmp_path
    ):
        out_file = tmp_path / "out_from_state.mp4"

        class MockToolContext:
            def __init__(self):
                self.state = {
                    "edl": sample_edl,
                    "clip_manifest": sample_clip_manifest,
                }

        ctx = MockToolContext()
        res = render_with_transitions(
            output_path=out_file,
            tool_context=ctx,
        )
        assert res == str(out_file.resolve())
        assert mock_exec.call_count == 4

    def test_render_missing_clip_in_manifest_raises_error(self, sample_clip_manifest):
        bad_edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="nonexistent_clip",
                    start_trim=0.0,
                    end_trim=3.0,
                    scene_rationale="Missing",
                )
            ]
        )
        with pytest.raises(ValueError, match="not found in manifest"):
            render_with_transitions(edl=bad_edl, clip_manifest=sample_clip_manifest)

    def test_render_cut_shorter_than_transition_raises_error(self, sample_clip_manifest):
        short_edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=0.0,
                    end_trim=0.8,
                    scene_rationale="Too short",
                ),
                EDLEntry(
                    file_reference="clip_02",
                    start_trim=0.0,
                    end_trim=5.0,
                    scene_rationale="Normal",
                ),
            ]
        )
        with pytest.raises(ValueError, match="must be strictly greater than transition_duration"):
            render_with_transitions(
                edl=short_edl,
                clip_manifest=sample_clip_manifest,
                transition_duration=1.0,
            )

    def test_render_invalid_transition_duration_raises_error(
        self, sample_edl, sample_clip_manifest
    ):
        with pytest.raises(ValueError, match="transition_duration must be positive"):
            render_with_transitions(
                edl=sample_edl,
                clip_manifest=sample_clip_manifest,
                transition_duration=-0.5,
            )

    @patch("tools.transition_renderer.execute_ffmpeg_command", side_effect=RuntimeError("FFmpeg failed"))
    def test_render_ffmpeg_failure_propagates(self, mock_exec, sample_clip_manifest, tmp_path):
        single_edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=0.0,
                    end_trim=2.0,
                    scene_rationale="Single",
                )
            ]
        )
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            render_with_transitions(
                edl=single_edl,
                clip_manifest=sample_clip_manifest,
                output_path=tmp_path / "out.mp4",
            )

    @patch("tools.transition_renderer.execute_ffmpeg_command")
    def test_render_single_cut_fast_forward(self, mock_exec, sample_clip_manifest, tmp_path):
        out_file = tmp_path / "single_2x.mp4"
        edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=0.0,
                    end_trim=10.0,
                    scene_rationale="2x single cut",
                    playback_speed=2.0,
                )
            ]
        )
        res = render_with_transitions(
            edl=edl,
            clip_manifest=sample_clip_manifest,
            output_path=out_file,
        )
        assert res == str(out_file.resolve())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        vf_idx = cmd.index("-vf")
        assert "setpts=0.500000*PTS" in cmd[vf_idx + 1]
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert "atempo=2.0000" in cmd[af_idx + 1]

    @patch("tools.transition_renderer.execute_ffmpeg_command")
    def test_render_multi_cut_fast_forward(self, mock_exec, sample_clip_manifest, tmp_path):
        out_file = tmp_path / "multi_2x.mp4"
        edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference="clip_01",
                    start_trim=0.0,
                    end_trim=10.0,
                    scene_rationale="Fast cut 1",
                    playback_speed=2.0,
                ),
                EDLEntry(
                    file_reference="clip_02",
                    start_trim=0.0,
                    end_trim=6.0,
                    scene_rationale="Normal cut 2",
                    playback_speed=1.0,
                ),
            ]
        )
        res = render_with_transitions(
            edl=edl,
            clip_manifest=sample_clip_manifest,
            output_path=out_file,
        )
        assert res == str(out_file.resolve())
        assert mock_exec.call_count == 3
        # Pre-trim for clip_01 (index 0)
        trim_cmd_0 = mock_exec.call_args_list[0][0][0]
        vf_idx = trim_cmd_0.index("-vf")
        assert "setpts=0.500000*PTS" in trim_cmd_0[vf_idx + 1]
        # Pre-trim for clip_02 (index 1)
        trim_cmd_1 = mock_exec.call_args_list[1][0][0]
        vf_idx_1 = trim_cmd_1.index("-vf")
        assert "setpts=" not in trim_cmd_1[vf_idx_1 + 1]
        # xfade command (index 2): clip_01 is 10/2=5s, transition=1s -> offset=4.0
        xfade_cmd = mock_exec.call_args_list[2][0][0]
        fc_idx = xfade_cmd.index("-filter_complex")
        assert "offset=4.0" in xfade_cmd[fc_idx + 1]

    def test_render_fast_forward_real_ffmpeg(self, tmp_path):
        """End-to-end integration test executing real FFmpeg binary to verify 2x speed rendering."""
        import subprocess
        from rendering.ffmpeg_builder import find_ffmpeg_executable
        from tools.clip_metadata import extract_clip_metadata

        ffmpeg_bin = find_ffmpeg_executable()
        synth_clip = tmp_path / "synth_4s.mp4"
        out_rendered = tmp_path / "rendered_2x.mp4"

        # Generate a 4.0-second synthetic test video with audio
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=4",
                "-c:v", "libx264", "-c:a", "aac",
                str(synth_clip),
            ],
            check=True,
            capture_output=True,
        )

        manifest_entry = extract_clip_metadata(str(synth_clip))
        assert abs(manifest_entry.duration_seconds - 4.0) < 0.2

        edl = EditDecisionList(
            entries=[
                EDLEntry(
                    file_reference=manifest_entry.clip_id,
                    start_trim=0.0,
                    end_trim=4.0,
                    scene_rationale="2x fast forward test",
                    playback_speed=2.0,
                )
            ]
        )

        rendered_path = render_with_transitions(
            edl=edl,
            clip_manifest=[manifest_entry],
            output_path=out_rendered,
        )

        assert Path(rendered_path).exists()
        out_meta = extract_clip_metadata(str(out_rendered))
        # 4s clip played at 2x speed must be ~2.0s duration
        assert abs(out_meta.duration_seconds - 2.0) < 0.3
        assert out_meta.has_audio
