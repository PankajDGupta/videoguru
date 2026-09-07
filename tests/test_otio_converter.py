"""Tests for OpenTimelineIO Converter Tool (SPEC-013)."""

from __future__ import annotations

import os
from pathlib import Path

import opentimelineio as otio
import pytest

from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.otio_converter import _create_otio_timeline, edl_to_otio, load_otio_timeline


@pytest.fixture
def mock_clip_manifest():
    return [
        ClipManifestEntry(
            clip_id="vid_1",
            absolute_path="/path/to/clip1.mp4",
            file_name="clip1.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=1000,
        ),
        ClipManifestEntry(
            clip_id="vid_2",
            absolute_path="/path/to/clip2.mp4",
            file_name="clip2.mp4",
            duration_seconds=5.0,
            frame_rate=24.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec=None,
            has_audio=False,
            file_size_bytes=500,
        ),
    ]


@pytest.fixture
def mock_edl():
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="vid_1",
                start_trim=1.0,
                end_trim=4.0,
                scene_rationale="action",
                transition_intent=TransitionIntent.CUT,
            ),
            EDLEntry(
                file_reference="vid_2",
                start_trim=0.0,
                end_trim=2.0,
                scene_rationale="b-roll",
                transition_intent=TransitionIntent.FADE,
            ),
        ]
    )


def test_create_otio_timeline(mock_edl, mock_clip_manifest):
    timeline = _create_otio_timeline(mock_edl, mock_clip_manifest)

    assert isinstance(timeline, otio.schema.Timeline)
    assert timeline.name == "VideoGuru Timeline"
    assert len(timeline.tracks) == 2

    video_track = timeline.tracks[0]
    audio_track = timeline.tracks[1]

    assert video_track.name == "V1"
    assert audio_track.name == "A1"
    assert len(video_track) == 2
    assert len(audio_track) == 2

    # Check first clip
    vc1 = video_track[0]
    assert vc1.name == "clip1.mp4"
    assert isinstance(vc1.media_reference, otio.schema.ExternalReference)
    assert vc1.media_reference.target_url == "file:///path/to/clip1.mp4"
    assert vc1.source_range.start_time.to_seconds() == pytest.approx(1.0)
    assert vc1.source_range.duration.to_seconds() == pytest.approx(3.0)

    ac1 = audio_track[0]
    assert isinstance(ac1, otio.schema.Clip)
    assert ac1.name == "clip1.mp4"

    # Check second clip
    vc2 = video_track[1]
    assert vc2.name == "clip2.mp4"
    assert vc2.media_reference.target_url == "file:///path/to/clip2.mp4"
    assert vc2.source_range.start_time.to_seconds() == pytest.approx(0.0)
    assert vc2.source_range.duration.to_seconds() == pytest.approx(2.0)

    ac2 = audio_track[1]
    assert isinstance(ac2, otio.schema.Gap)  # Because vid_2 has no audio
    assert ac2.source_range.duration.to_seconds() == pytest.approx(2.0)


def test_create_otio_timeline_missing_clip(mock_edl):
    with pytest.raises(ValueError, match="Clip reference 'vid_1' not found in manifest."):
        _create_otio_timeline(mock_edl, [])


def test_edl_to_otio_and_load(tmp_path, mock_edl, mock_clip_manifest):
    output_dir = tmp_path / "staging"
    output_dir.mkdir()

    class MockToolContext:
        def __init__(self):
            self.state = {}

    tool_context = MockToolContext()

    otio_path = edl_to_otio(
        edl=mock_edl,
        clip_manifest=mock_clip_manifest,
        output_dir=str(output_dir),
        tool_context=tool_context,
    )

    assert Path(otio_path).exists()
    assert tool_context.state.get("otio_file_path") == otio_path
    
    # Load it back
    loaded_timeline = load_otio_timeline(otio_path)
    assert isinstance(loaded_timeline, otio.schema.Timeline)
    assert loaded_timeline.name == "VideoGuru Timeline"
    assert len(loaded_timeline.tracks) == 2


def test_edl_to_otio_from_state(tmp_path, mock_edl, mock_clip_manifest):
    output_dir = tmp_path / "staging"
    output_dir.mkdir()

    class MockToolContext:
        def __init__(self):
            self.state = {
                "edl": mock_edl.to_dict_list(),
                "clip_manifest": [m.model_dump() for m in mock_clip_manifest],
            }

    tool_context = MockToolContext()

    otio_path = edl_to_otio(
        output_dir=str(output_dir),
        tool_context=tool_context,
    )

    assert Path(otio_path).exists()


def test_edl_to_otio_empty_state_errors():
    class MockToolContext:
        def __init__(self):
            self.state = {}
            
    with pytest.raises(ValueError, match="No Edit Decision List provided"):
        edl_to_otio(tool_context=MockToolContext())


def test_create_otio_timeline_with_fast_forward(mock_clip_manifest):
    fast_edl = EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="vid_1",
                start_trim=0.0,
                end_trim=6.0,
                scene_rationale="Fast forward run",
                playback_speed=2.0,
            ),
            EDLEntry(
                file_reference="vid_2",
                start_trim=1.0,
                end_trim=5.0,
                scene_rationale="Normal cut",
                playback_speed=1.0,
            ),
        ]
    )
    timeline = _create_otio_timeline(fast_edl, mock_clip_manifest)
    video_track = timeline.tracks[0]
    audio_track = timeline.tracks[1]

    # Clip 1 (fast forward 2x)
    vc1 = video_track[0]
    ac1 = audio_track[0]
    assert len(vc1.effects) == 1
    assert isinstance(vc1.effects[0], otio.schema.LinearTimeWarp)
    assert vc1.effects[0].time_scalar == 2.0
    assert len(ac1.effects) == 1
    assert isinstance(ac1.effects[0], otio.schema.LinearTimeWarp)
    assert ac1.effects[0].time_scalar == 2.0

    # Clip 2 (normal speed 1x)
    vc2 = video_track[1]
    assert len(vc2.effects) == 0
