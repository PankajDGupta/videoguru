"""Tests for Theme-Related Motion Graphics (SPEC-035).

Verifies:
1. Motion graphics schemas validate timing, copy, colours and palette resolution.
2. Plans returned by Gemini are sanitized (clamped, de-overlapped, capped, copy-trimmed).
3. The Gemini analysis prompt carries the theme, cut timeline and safety rules.
4. The deterministic offline plan never invents facts.
5. The FFmpeg builder emits correct, safely escaped, safe-zone-aware animated filtergraphs.
6. Real FFmpeg rendering produces a valid video that differs from its source.
7. The Gemini analysis path uploads a proxy, parses structured output, cleans up and degrades gracefully.
8. EnhancementRenderingAgent Step 1c runs by default, honours the opt-out and never fails the render.
9. Settings and CLI flags route correctly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from agents.enhancement_rendering import EnhancementRenderingAgent
from config import settings
from rendering.ffmpeg_builder import (
    MAX_MOTION_GRAPHIC_ELEMENTS,
    build_motion_graphics_command,
    execute_ffmpeg_command,
    find_ffmpeg_executable,
)
from schemas.edl import EditDecisionList, EDLEntry, TransitionIntent
from schemas.media import ClipManifestEntry
from schemas.motion_graphics import (
    DEFAULT_PRIMARY_COLOR,
    MotionGraphicElement,
    MotionGraphicsPlan,
    MotionGraphicType,
    clean_graphic_text,
    normalize_hex_color,
)
from tools import motion_graphics_tools as mgt
from tools.clip_metadata import find_ffprobe_executable
from tools.motion_graphics_tools import (
    _GeminiGraphic,
    _GeminiMotionPlan,
    analyze_video_for_motion_graphics,
    build_motion_graphics_prompt,
    burn_motion_graphics,
    generate_motion_graphics_plan,
    sanitize_motion_graphics_plan,
)


def _el(g_type, start, end, text="Hello", **kw) -> MotionGraphicElement:
    if g_type == MotionGraphicType.PROGRESS_BAR:
        text = None
    return MotionGraphicElement(graphic_type=g_type, start_time=start, end_time=end, text=text, **kw)


def _make_video(path: Path, size: str = "1080x1920", seconds: int = 6) -> Path:
    subprocess.run(
        [
            find_ffmpeg_executable(), "-y", "-f", "lavfi", "-i", f"color=c=0x334455:s={size}:r=30:d={seconds}",
            "-f", "lavfi", "-i", f"sine=d={seconds}", "-shortest",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def _frame_bytes(video: Path, t: float, out: Path) -> bytes:
    subprocess.run(
        [find_ffmpeg_executable(), "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-update", "1", str(out)],
        check=True, capture_output=True,
    )
    return out.read_bytes()


@pytest.fixture(scope="module")
def vertical_video(tmp_path_factory) -> Path:
    """One shared read-only 1080x1920 8s source clip (encoding it per test dominated the runtime)."""
    return _make_video(tmp_path_factory.mktemp("mg_src") / "vertical.mp4", seconds=8)


@pytest.fixture
def sample_edl() -> EditDecisionList:
    return EditDecisionList(
        entries=[
            EDLEntry(file_reference="clip_01", start_trim=0.0, end_trim=4.0,
                     scene_rationale="Morning run along the river", transition_intent=TransitionIntent.CUT),
            EDLEntry(file_reference="clip_02", start_trim=5.0, end_trim=9.0,
                     scene_rationale="Stretching at the trailhead", transition_intent=TransitionIntent.FADE),
        ]
    )


@pytest.fixture
def sample_manifest() -> list[ClipManifestEntry]:
    return [
        ClipManifestEntry(
            clip_id=cid, absolute_path=f"C:/media/{cid}.mp4", file_name=f"{cid}.mp4", duration_seconds=12.0,
            frame_rate=30.0, resolution="1920x1080", width=1920, height=1080, video_codec="h264",
            audio_codec="aac", has_audio=True, file_size_bytes=1000,
        )
        for cid in ("clip_01", "clip_02")
    ]


class TestSchemas:
    def test_valid_element(self):
        e = _el(MotionGraphicType.LOWER_THIRD, 1.0, 4.0, text="Riverside", subtext="Start of the run")
        assert e.duration == 3.0

    def test_end_must_follow_start(self):
        with pytest.raises(ValidationError):
            _el(MotionGraphicType.KINETIC_TITLE, 3.0, 3.0)

    def test_text_graphics_require_text(self):
        with pytest.raises(ValidationError):
            MotionGraphicElement(graphic_type=MotionGraphicType.CORNER_BADGE, start_time=0, end_time=3)

    def test_progress_bar_needs_no_text(self):
        assert _el(MotionGraphicType.PROGRESS_BAR, 0, 10).text is None

    def test_emoji_and_backslashes_stripped(self):
        assert clean_graphic_text("Big run \U0001F525 \\ now") == "Big run now"
        assert _el(MotionGraphicType.KINETIC_TITLE, 0, 3, text="GO \U0001F680 GO").text == "GO GO"

    def test_text_that_is_only_emoji_is_rejected(self):
        with pytest.raises(ValidationError):
            _el(MotionGraphicType.KINETIC_TITLE, 0, 3, text="\U0001F525")

    def test_color_normalisation(self):
        assert normalize_hex_color("ff6b00", "#000000") == "#FF6B00"
        assert normalize_hex_color("not a colour", "#123456") == "#123456"
        assert _el(MotionGraphicType.CORNER_BADGE, 0, 3, accent_color="00e5ff").accent_color == "#00E5FF"
        assert _el(MotionGraphicType.CORNER_BADGE, 0, 3, accent_color="bogus").accent_color is None

    def test_plan_palette_falls_back_to_defaults(self):
        plan = MotionGraphicsPlan(primary_color="nonsense", secondary_color="")
        assert plan.primary_color == DEFAULT_PRIMARY_COLOR

    def test_to_dict_list_resolves_palette(self):
        plan = MotionGraphicsPlan(
            elements=[_el(MotionGraphicType.CORNER_BADGE, 0, 3), _el(MotionGraphicType.CORNER_BADGE, 4, 7, accent_color="#00E5FF")],
            primary_color="#112233", secondary_color="#445566",
        )
        rows = plan.to_dict_list()
        assert rows[0]["accent_color"] == "#112233"
        assert rows[1]["accent_color"] == "#00E5FF"
        assert all(r["secondary_color"] == "#445566" for r in rows)
        json.dumps(rows)  # session-state serialisable

    def test_round_trip_through_state(self):
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.STAT_CALLOUT, 1, 4, text="5 KM", subtext="Distance")],
                                  primary_color="#112233", secondary_color="#445566")
        restored = MotionGraphicsPlan.from_list(plan.to_dict_list(), theme="t")
        assert len(restored) == 1 and restored[0].text == "5 KM"
        assert restored.secondary_color == "#445566"

    def test_container_protocol(self):
        assert not MotionGraphicsPlan()
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 0, 3)])
        assert plan and len(plan) == 1 and list(plan)[0] is plan[0]


class TestSanitizer:
    def test_clamps_to_video_and_drops_too_short(self):
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.LOWER_THIRD, 8.0, 14.0),   # clamped to 8-10 (2.0s)
            _el(MotionGraphicType.CORNER_BADGE, 9.5, 10.5),  # only 0.5s left -> dropped
        ])
        out = sanitize_motion_graphics_plan(plan, 10.0)
        assert [e.graphic_type for e in out.elements] == [MotionGraphicType.LOWER_THIRD]
        assert out.elements[0].end_time == 10.0

    def test_caps_text_graphic_lifetime_but_not_progress_bar(self):
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.LOWER_THIRD, 0.0, 9.0),
            _el(MotionGraphicType.PROGRESS_BAR, 0.0, 20.0),
        ])
        out = sanitize_motion_graphics_plan(plan, 20.0)
        by_type = {e.graphic_type: e for e in out.elements}
        assert by_type[MotionGraphicType.LOWER_THIRD].duration == mgt.MAX_GRAPHIC_DURATION
        assert by_type[MotionGraphicType.PROGRESS_BAR].duration == 20.0

    def test_trims_copy_to_limits(self):
        plan = MotionGraphicsPlan(elements=[
            MotionGraphicElement(graphic_type=MotionGraphicType.CORNER_BADGE, start_time=0, end_time=3,
                                 text="A" * 40, subtext="ignored for badges"),
            MotionGraphicElement(graphic_type=MotionGraphicType.LOWER_THIRD, start_time=5, end_time=8,
                                 text="Title", subtext="S" * 70),
        ])
        out = sanitize_motion_graphics_plan(plan, 30.0)
        badge, lower = out.elements
        assert len(badge.text) == mgt.MAX_TEXT_CHARS[MotionGraphicType.CORNER_BADGE]
        assert badge.subtext is None
        assert len(lower.subtext) == mgt.MAX_SUBTEXT_CHARS

    def test_progress_bar_copy_removed(self):
        el = MotionGraphicElement(graphic_type=MotionGraphicType.PROGRESS_BAR, start_time=0, end_time=5,
                                  text="stray", subtext="stray")
        out = sanitize_motion_graphics_plan(MotionGraphicsPlan(elements=[el]), 10.0)
        assert out.elements[0].text is None and out.elements[0].subtext is None

    def test_same_type_overlap_removed(self):
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.LOWER_THIRD, 0.0, 3.0, text="first"),
            _el(MotionGraphicType.LOWER_THIRD, 2.0, 5.0, text="second"),
            _el(MotionGraphicType.LOWER_THIRD, 3.0, 6.0, text="third"),
        ])
        out = sanitize_motion_graphics_plan(plan, 30.0)
        assert [e.text for e in out.elements] == ["first", "third"]

    def test_shared_zone_types_never_overlap(self):
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.KINETIC_TITLE, 0.0, 3.0, text="HOOK"),
            _el(MotionGraphicType.STAT_CALLOUT, 1.0, 4.0, text="5 KM"),
            _el(MotionGraphicType.STAT_CALLOUT, 3.0, 6.0, text="3 KM"),
        ])
        out = sanitize_motion_graphics_plan(plan, 30.0)
        assert [e.text for e in out.elements] == ["HOOK", "3 KM"]

    def test_different_zone_types_may_overlap(self):
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.KINETIC_TITLE, 0.0, 3.0),
            _el(MotionGraphicType.CORNER_BADGE, 0.5, 3.5),
            _el(MotionGraphicType.PROGRESS_BAR, 0.0, 10.0),
        ])
        assert len(sanitize_motion_graphics_plan(plan, 30.0)) == 3

    def test_density_cap_keeps_progress_bar(self):
        elements = [_el(MotionGraphicType.PROGRESS_BAR, 0, 20)] + [
            _el(MotionGraphicType.CORNER_BADGE, i * 2.0, i * 2.0 + 1.8, text=f"B{i}") for i in range(9)
        ]
        out = sanitize_motion_graphics_plan(MotionGraphicsPlan(elements=elements), 20.0)
        assert mgt._max_elements(20.0) == 4
        assert len(out) == 5  # four capped graphics + the progress bar, which is exempt from the cap
        assert [e.text for e in out.elements if e.text] == ["B0", "B1", "B2", "B3"]
        assert any(e.graphic_type == MotionGraphicType.PROGRESS_BAR for e in out.elements)

    def test_preserves_palette_and_summary(self):
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 0, 3)], primary_color="#112233",
                                  content_summary="A run")
        out = sanitize_motion_graphics_plan(plan, 10.0)
        assert out.primary_color == "#112233" and out.content_summary == "A run"


class TestPrompt:
    def test_prompt_contains_context_and_rules(self, sample_edl):
        prompt = build_motion_graphics_prompt(
            theme="Morning run vlog", video_duration=7.0, is_shorts=True, resolution="1080x1920", edl=sample_edl,
            overlay_entries=[{"text": "RUN CLUB", "start_time": 0.3, "end_time": 3.5}],
        )
        assert "Morning run vlog" in prompt
        assert "Morning run along the river" in prompt and "Cut 1:" in prompt
        assert "RUN CLUB" in prompt
        assert "vertical 9:16" in prompt and "1080x1920" in prompt
        for graphic in MotionGraphicType:
            assert graphic.value in prompt
        assert "NO emojis" in prompt
        assert "Do NOT invent" in prompt
        assert "primary_color" in prompt

    def test_prompt_without_optional_context(self):
        prompt = build_motion_graphics_prompt("", 30.0, False, "1920x1080")
        assert "landscape 16:9" in prompt
        assert "CUT TIMELINE" not in prompt and "ALREADY IN THE VIDEO" not in prompt
        assert "infer it from the footage" in prompt


class TestMockPlan:
    def test_deterministic_and_fact_safe(self, sample_edl):
        a = mgt._generate_mock_motion_graphics(sample_edl, "Morning Run Vlog", 7.0)
        b = mgt._generate_mock_motion_graphics(sample_edl, "Morning Run Vlog", 7.0)
        assert a.to_dict_list() == b.to_dict_list()
        types = {e.graphic_type for e in a.elements}
        assert MotionGraphicType.PROGRESS_BAR in types and MotionGraphicType.CORNER_BADGE in types
        lower = next(e for e in a.elements if e.graphic_type == MotionGraphicType.LOWER_THIRD)
        assert lower.subtext == "2 moments"  # only a fact derived from the EDL
        assert not any(e.graphic_type == MotionGraphicType.STAT_CALLOUT for e in a.elements)
        assert all(0.0 <= e.start_time < e.end_time <= 7.0 for e in a.elements)

    def test_copy_is_cut_at_word_boundaries(self):
        assert mgt._truncate_copy("Morning workout vlog", 14) == "Morning"
        assert mgt._truncate_copy("Morning workout", 15) == "Morning workout"
        assert mgt._truncate_copy("Morning run today", 11) == "Morning run"  # limit lands exactly on a space
        assert mgt._truncate_copy("Supercalifragilistic", 8) == "Supercal"  # no space: hard cut
        plan = mgt._generate_mock_motion_graphics(None, "Morning workout vlog", 10.0)
        badge = next(e for e in plan.elements if e.graphic_type == MotionGraphicType.CORNER_BADGE)
        assert badge.text == "Morning"

    def test_without_edl_or_theme(self):
        plan = mgt._generate_mock_motion_graphics(None, "", 10.0)
        assert plan.elements and all(e.end_time <= 10.0 for e in plan.elements)


class TestBuilder:
    ELEMENTS = [
        {"graphic_type": "progress_bar", "start_time": 0, "end_time": 8, "accent_color": "#FF6B00"},
        {"graphic_type": "kinetic_title", "start_time": 0.5, "end_time": 3.5, "text": "Go, now: 100%", "accent_color": "#FF6B00"},
        {"graphic_type": "corner_badge", "start_time": 1, "end_time": 4, "text": "Day 1", "accent_color": "#FFD700"},
        {"graphic_type": "stat_callout", "start_time": 4, "end_time": 7, "text": "5 KM", "subtext": "Distance", "accent_color": "#00E5FF"},
        {"graphic_type": "lower_third", "start_time": 4, "end_time": 7.5, "text": "Riverside", "subtext": "It's the start", "accent_color": "#FF6B00", "secondary_color": "#FFD700"},
    ]

    def _graph(self, cmd) -> str:
        return cmd[cmd.index("-filter_complex") + 1]

    def test_command_structure(self):
        cmd = build_motion_graphics_command("in.mp4", self.ELEMENTS, "out.mp4", target_resolution="1080x1920",
                                            is_shorts=True, use_gpu=False)
        assert Path(cmd[0]).name.lower().startswith("ffmpeg")
        assert cmd[cmd.index("-i") + 1] == "in.mp4"
        assert "0:a?" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
        assert cmd[-1] == "out.mp4"
        assert cmd[cmd.index("-c:v") + 1] == "libx264"
        graph = self._graph(cmd)
        assert cmd[cmd.index("-map") + 1] == f"[{graph.split('[')[-1].rstrip(']')}]"
        assert "overlay=" in graph and "drawtext=" in graph and "color=c=" in graph
        assert graph.count("drawtext=") >= 5

    def test_gpu_encoder_selected(self):
        cmd = build_motion_graphics_command("in.mp4", self.ELEMENTS, "out.mp4", use_gpu=True)
        assert "h264_nvenc" in cmd

    def test_text_is_escaped_literally(self):
        graph = self._graph(build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4", is_shorts=True))
        assert "expansion=none" in graph
        assert r"GO, NOW\: 100%" in graph            # colon escaped, comma and % left literal
        assert "IT\u2019S" not in graph and "It\u2019s the start" in graph  # apostrophe made typographic
        assert "'s the start" not in graph

    def test_backslashes_and_quotes_cannot_break_out(self):
        els = [{"graphic_type": "corner_badge", "start_time": 0, "end_time": 3, "text": "a\\'b];rm", "accent_color": "#FFFFFF"}]
        graph = self._graph(build_motion_graphics_command("in.mp4", els, "o.mp4"))
        assert "\\'" not in graph and "a\\" not in graph

    def test_shorts_zones_inside_safe_area(self):
        graph = self._graph(build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4", target_resolution="1080x1920"))
        # Lower third starts at 56.5% of 1920 = 1084.8 (above the bottom 32% UI band starting at 1305)
        assert "y='1084.8'" in graph
        assert "y='259.2'" in graph  # progress bar at 13.5%, below the top channel bar

    def test_landscape_zones_differ(self):
        shorts = self._graph(build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4", target_resolution="1080x1920"))
        wide = self._graph(build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4", target_resolution="1920x1080"))
        assert shorts != wide
        assert "y='756.0'" in wide  # lower third at 70% of 1080

    def test_colours_converted(self):
        graph = self._graph(build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4"))
        assert "0xFF6B00" in graph and "0x00E5FF" in graph and "#" not in graph

    def test_invalid_colour_falls_back(self):
        els = [{"graphic_type": "corner_badge", "start_time": 0, "end_time": 3, "text": "X", "accent_color": "red;drop"}]
        graph = self._graph(build_motion_graphics_command("in.mp4", els, "o.mp4"))
        assert "red;drop" not in graph

    def test_skips_undrawable_elements(self):
        els = [
            {"graphic_type": "confetti", "start_time": 0, "end_time": 3, "text": "x"},
            {"graphic_type": "corner_badge", "start_time": 0, "end_time": 0.3, "text": "too short"},
            {"graphic_type": "corner_badge", "start_time": 0, "end_time": 3, "text": ""},
            {"graphic_type": "corner_badge", "start_time": 0, "end_time": 3, "text": "ok"},
        ]
        graph = self._graph(build_motion_graphics_command("in.mp4", els, "o.mp4"))
        assert graph.count("drawtext=") == 1

    def test_only_undrawable_raises(self):
        with pytest.raises(ValueError, match="No valid motion graphic"):
            build_motion_graphics_command("in.mp4", [{"graphic_type": "confetti", "start_time": 0, "end_time": 3}], "o.mp4")

    @pytest.mark.parametrize("video,out,els", [("", "o.mp4", [{}]), ("i.mp4", "", [{}]), ("i.mp4", "o.mp4", [])])
    def test_argument_validation(self, video, out, els):
        with pytest.raises(ValueError):
            build_motion_graphics_command(video, els, out)

    def test_element_cap(self):
        els = [{"graphic_type": "corner_badge", "start_time": i, "end_time": i + 2, "text": f"B{i}"} for i in range(30)]
        graph = self._graph(build_motion_graphics_command("in.mp4", els, "o.mp4"))
        assert graph.count("drawtext=") == MAX_MOTION_GRAPHIC_ELEMENTS

    def test_command_passes_executor_binary_check(self):
        cmd = build_motion_graphics_command("in.mp4", self.ELEMENTS, "o.mp4")
        assert all(isinstance(a, str) for a in cmd)
        with pytest.raises(FileNotFoundError):  # binary check passes; ffmpeg itself never runs on a fake path
            burn_motion_graphics("does_not_exist.mp4", MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 0, 3)]), "o.mp4")


class TestRealFfmpegRendering:
    def test_burn_produces_valid_changed_video(self, tmp_path, vertical_video):
        src = vertical_video
        plan = MotionGraphicsPlan(elements=[
            _el(MotionGraphicType.PROGRESS_BAR, 0, 8),
            _el(MotionGraphicType.KINETIC_TITLE, 0.5, 3.5, text="Morning Run, 5K: Go!"),
            _el(MotionGraphicType.CORNER_BADGE, 1, 4, text="Day 1"),
            _el(MotionGraphicType.STAT_CALLOUT, 3.6, 6, text="50% DONE", subtext="Halfway there"),
            _el(MotionGraphicType.LOWER_THIRD, 2, 5.5, text="Riverside Trail", subtext="It's the start"),
        ])
        out = burn_motion_graphics(src, plan, tmp_path / "out.mp4", target_resolution="1080x1920", is_shorts=True)
        assert Path(out).is_file()

        probe = json.loads(subprocess.run(
            [find_ffprobe_executable(), "-v", "error", "-show_streams", "-show_format", "-of", "json", out],
            check=True, capture_output=True, text=True).stdout)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (1080, 1920)
        assert any(s["codec_type"] == "audio" for s in probe["streams"])
        assert abs(float(probe["format"]["duration"]) - 8.0) < 0.5

        # Graphics visible mid-animation, and gone from a frame with nothing scheduled
        assert _frame_bytes(Path(out), 2.7, tmp_path / "a.png") != _frame_bytes(src, 2.7, tmp_path / "b.png")

    def test_graphics_absent_outside_their_window(self, tmp_path, vertical_video):
        src = vertical_video
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 1, 3, text="Day 1")])
        out = burn_motion_graphics(src, plan, tmp_path / "out.mp4", target_resolution="1080x1920")
        assert _frame_bytes(Path(out), 5.0, tmp_path / "late.png") == _frame_bytes(src, 5.0, tmp_path / "base.png")
        assert _frame_bytes(Path(out), 2.0, tmp_path / "mid.png") != _frame_bytes(src, 2.0, tmp_path / "base2.png")

    def test_landscape_render(self, tmp_path):
        src = _make_video(tmp_path / "wide.mp4", size="1920x1080", seconds=3)
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.LOWER_THIRD, 0.5, 2.8, text="Harbour", subtext="Evening")])
        out = burn_motion_graphics(src, plan, tmp_path / "wide_out.mp4", target_resolution="1920x1080")
        assert Path(out).stat().st_size > 0

    def test_burn_validation(self, tmp_path, vertical_video):
        plan = MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 0, 3)])
        src = vertical_video
        with pytest.raises(ValueError):
            burn_motion_graphics("", plan, tmp_path / "o.mp4")
        with pytest.raises(ValueError):
            burn_motion_graphics(src, plan, "")
        with pytest.raises(FileNotFoundError):
            burn_motion_graphics(tmp_path / "missing.mp4", plan, tmp_path / "o.mp4")
        with pytest.raises(ValueError, match="No motion graphics"):
            burn_motion_graphics(src, MotionGraphicsPlan(), tmp_path / "o.mp4")
        with pytest.raises(TypeError):
            burn_motion_graphics(src, "nope", tmp_path / "o.mp4")


class TestGeminiAnalysis:
    @pytest.fixture
    def video(self, vertical_video):
        return vertical_video

    def _client(self, generate):
        client = MagicMock()
        client.files.upload.return_value = SimpleNamespace(name="files/abc", state=SimpleNamespace(name="ACTIVE"))
        client.models.generate_content.side_effect = generate
        return client

    def _run(self, video, client, tmp_path, **kw):
        with patch.object(settings, "GEMINI_API_KEY", "fake-key"), \
             patch.object(settings, "STAGING_DIR", tmp_path / "stage"), \
             patch("google.genai.Client", return_value=client):
            return analyze_video_for_motion_graphics(video, "Morning run", is_shorts=True, **kw)

    def test_structured_response_becomes_sanitized_plan(self, video, tmp_path, sample_edl):
        payload = _GeminiMotionPlan(
            content_summary="A runner by a river",
            primary_color="#00e5ff", secondary_color="#ffd700",
            graphics=[
                _GeminiGraphic(graphic_type=MotionGraphicType.KINETIC_TITLE, start_time=0.5, end_time=3.0, text="RIVER RUN"),
                _GeminiGraphic(graphic_type=MotionGraphicType.LOWER_THIRD, start_time=5.0, end_time=20.0, text="Riverside", subtext="Trailhead"),
                _GeminiGraphic(graphic_type=MotionGraphicType.LOWER_THIRD, start_time=6.0, end_time=7.0, text="Overlaps"),
            ],
        )
        client = self._client(lambda **kw: SimpleNamespace(parsed=payload, text=""))
        plan = self._run(video, client, tmp_path, edl=sample_edl, overlay_entries=[{"text": "RUN CLUB", "start_time": 0, "end_time": 3}])

        assert plan.content_summary == "A runner by a river"
        assert (plan.primary_color, plan.secondary_color) == ("#00E5FF", "#FFD700")
        assert [e.text for e in plan.elements] == ["RIVER RUN", "Riverside"]
        assert plan.elements[1].end_time == 8.0  # clamped to the video duration
        client.files.delete.assert_called_once_with(name="files/abc")

        sent = client.models.generate_content.call_args.kwargs
        assert sent["config"].response_schema is _GeminiMotionPlan
        assert "RUN CLUB" in sent["contents"][1] and "Morning run along the river" in sent["contents"][1]

    def test_json_text_response_parsed_and_bad_graphics_skipped(self, video, tmp_path):
        body = json.dumps({
            "content_summary": "x", "primary_color": "#112233", "secondary_color": "#445566",
            "graphics": [
                {"graphic_type": "corner_badge", "start_time": 1, "end_time": 4, "text": "DAY 1"},
                {"graphic_type": "lower_third", "start_time": 2, "end_time": 5},  # missing text -> skipped
            ],
        })
        client = self._client(lambda **kw: SimpleNamespace(parsed=None, text="```json\n" + body + "\n```"))
        plan = self._run(video, client, tmp_path)
        assert [e.text for e in plan.elements] == ["DAY 1"]

    def test_api_failure_falls_back_and_still_cleans_up(self, video, tmp_path):
        def boom(**kw):
            raise RuntimeError("quota exceeded")
        client = self._client(boom)
        plan = self._run(video, client, tmp_path)
        assert plan.elements  # deterministic fallback
        assert any(e.graphic_type == MotionGraphicType.PROGRESS_BAR for e in plan.elements)
        client.files.delete.assert_called_once()

    def test_empty_gemini_plan_falls_back(self, video, tmp_path):
        client = self._client(lambda **kw: SimpleNamespace(parsed=_GeminiMotionPlan(), text=""))
        assert self._run(video, client, tmp_path).elements

    def test_offline_never_calls_gemini(self, video):
        with patch("google.genai.Client") as client_cls, patch.object(settings, "GEMINI_API_KEY", "fake-key"):
            plan = analyze_video_for_motion_graphics(video, "Morning run", offline=True)
        client_cls.assert_not_called()
        assert plan.elements

    def test_missing_api_key_uses_fallback(self, video, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch.object(settings, "GEMINI_API_KEY", None), patch("google.genai.Client") as client_cls:
            plan = analyze_video_for_motion_graphics(video, "Morning run")
        client_cls.assert_not_called()
        assert plan.elements

    def test_missing_video_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_video_for_motion_graphics(tmp_path / "nope.mp4", "t")

    def test_analysis_proxy_is_small_and_silent(self, video, tmp_path):
        proxy = mgt._create_analysis_proxy(video, tmp_path / "proxy")
        probe = json.loads(subprocess.run(
            [find_ffprobe_executable(), "-v", "error", "-show_streams", "-of", "json", str(proxy)],
            check=True, capture_output=True, text=True).stdout)
        assert [s["codec_type"] for s in probe["streams"]] == ["video"]
        assert max(probe["streams"][0]["width"], probe["streams"][0]["height"]) == 480


class TestAdkTool:
    def test_generates_and_stores_plan(self, vertical_video):
        video = vertical_video
        ctx = SimpleNamespace(state={"theme": "Morning run", "transition_rendered_path": str(video),
                                     "video_type": "shorts", "target_resolution": "1080x1920"})
        summary = generate_motion_graphics_plan(ctx, offline=True)
        assert summary.startswith("Designed ")
        assert ctx.state["motion_graphics"] and len(ctx.state["motion_graphics_palette"]) == 2
        MotionGraphicsPlan.from_list(ctx.state["motion_graphics"])  # stored form is reloadable

    def test_no_video_message(self):
        assert "No rendered video" in generate_motion_graphics_plan(SimpleNamespace(state={}), offline=True)


class TestRenderingAgentStep:
    @pytest.fixture
    def rendered(self, vertical_video):
        return vertical_video

    def _state(self, sample_edl, sample_manifest, **extra):
        return {"edl": sample_edl, "clip_manifest": sample_manifest, "theme": "Morning run vlog", **extra}

    def test_runs_by_default_and_changes_the_video(self, rendered, sample_edl, sample_manifest, tmp_path):
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)):
            state = self._state(sample_edl, sample_manifest)
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                state, output_path=tmp_path / "final.mp4", target_resolution="1080x1920",
                enable_overlay_text=False, enable_captions=False,
            )
        assert result["has_motion_graphics"] is True
        assert Path(result["motion_graphics_video_path"]).name == "step1c_motion_graphics.mp4"
        assert state["motion_graphics"]
        assert Path(result["final_video_path"]).is_file()
        assert _frame_bytes(Path(result["final_video_path"]), 1.0, tmp_path / "f.png") != _frame_bytes(rendered, 1.0, tmp_path / "s.png")

    def test_opt_out_argument(self, rendered, sample_edl, sample_manifest, tmp_path):
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.burn_motion_graphics") as burn:
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest), output_path=tmp_path / "final.mp4",
                enable_overlay_text=False, enable_motion_graphics=False,
            )
        burn.assert_not_called()
        assert result["has_motion_graphics"] is False and result["motion_graphics_video_path"] is None

    def test_opt_out_via_state(self, rendered, sample_edl, sample_manifest, tmp_path):
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.burn_motion_graphics") as burn:
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest, enable_motion_graphics=False),
                output_path=tmp_path / "final.mp4", enable_overlay_text=False,
            )
        burn.assert_not_called()
        assert result["has_motion_graphics"] is False

    def test_render_survives_motion_graphics_failure(self, rendered, sample_edl, sample_manifest, tmp_path):
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.burn_motion_graphics", side_effect=RuntimeError("ffmpeg exploded")):
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest), output_path=tmp_path / "final.mp4", enable_overlay_text=False,
            )
        assert result["has_motion_graphics"] is False
        assert Path(result["final_video_path"]).is_file()

    def test_render_survives_analysis_failure(self, rendered, sample_edl, sample_manifest, tmp_path):
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.analyze_video_for_motion_graphics", side_effect=RuntimeError("no ffprobe")):
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest), output_path=tmp_path / "final.mp4", enable_overlay_text=False,
            )
        assert result["has_motion_graphics"] is False and Path(result["final_video_path"]).is_file()

    def test_preplanned_graphics_skip_analysis(self, rendered, sample_edl, sample_manifest, tmp_path):
        planned = MotionGraphicsPlan(elements=[_el(MotionGraphicType.CORNER_BADGE, 0.5, 3.5, text="Day 1")]).to_dict_list()
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.analyze_video_for_motion_graphics") as analyze:
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest, motion_graphics=planned),
                output_path=tmp_path / "final.mp4", enable_overlay_text=False,
            )
        analyze.assert_not_called()
        assert result["has_motion_graphics"] is True

    def test_downstream_steps_consume_motion_graphics_output(self, rendered, sample_edl, sample_manifest, tmp_path):
        music = tmp_path / "music.mp3"
        music.write_bytes(b"x")
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch("agents.enhancement_rendering.apply_audio_ducking", side_effect=lambda **kw: kw["video_path"]) as duck:
            result = EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                self._state(sample_edl, sample_manifest), output_path=tmp_path / "final.mp4",
                music_path=music, enable_overlay_text=False,
            )
        assert Path(duck.call_args.kwargs["video_path"]).name == "step1c_motion_graphics.mp4"
        assert result["has_ducking"] is True

    def test_agent_instruction_mentions_motion_graphics(self):
        from agents.enhancement_rendering import ENHANCEMENT_RENDERING_INSTRUCTION
        assert "motion graphics" in ENHANCEMENT_RENDERING_INSTRUCTION


class TestSettingsAndCli:
    def test_enabled_by_default(self):
        assert settings.ENABLE_MOTION_GRAPHICS is True

    @pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("true", True), ("ON", True)])
    def test_env_toggle_and_reload(self, monkeypatch, value, expected):
        monkeypatch.setenv("ENABLE_MOTION_GRAPHICS", value)
        try:
            settings.reload_settings()
            assert settings.ENABLE_MOTION_GRAPHICS is expected
        finally:
            monkeypatch.delenv("ENABLE_MOTION_GRAPHICS", raising=False)
            settings.reload_settings()
        assert settings.ENABLE_MOTION_GRAPHICS is True

    def test_setting_drives_default_behaviour(self, sample_edl, sample_manifest, tmp_path, vertical_video):
        rendered = vertical_video
        with patch("agents.enhancement_rendering.render_with_transitions", return_value=str(rendered)), \
             patch.object(settings, "ENABLE_MOTION_GRAPHICS", False), \
             patch("agents.enhancement_rendering.burn_motion_graphics") as burn:
            EnhancementRenderingAgent(offline=True).execute_rendering_pipeline(
                {"edl": sample_edl, "clip_manifest": sample_manifest, "theme": "t"},
                output_path=tmp_path / "f.mp4", enable_overlay_text=False,
            )
        burn.assert_not_called()

    def test_cli_flag(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--no-motion-graphics"]):
            assert parse_arguments().no_motion_graphics is True
        with patch("sys.argv", ["main.py"]):
            assert parse_arguments().no_motion_graphics is False

    def test_tools_exported(self):
        import tools
        for name in ("analyze_video_for_motion_graphics", "burn_motion_graphics", "generate_motion_graphics_plan"):
            assert name in tools.__all__ and hasattr(tools, name)
