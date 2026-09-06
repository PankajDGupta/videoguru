"""Unit and Integration Tests for SPEC-023 Tool Error Recovery & Self-Correction Callback."""

from unittest.mock import MagicMock
import pytest

from callbacks.error_recovery import (
    after_tool_error_recovery_callback,
    execute_with_retry,
    on_tool_error_recovery_callback,
    parse_tool_error,
)


class TestErrorPatternParsing:
    """Test suite for parse_tool_error classification and guidance."""

    @pytest.mark.parametrize(
        "stderr_snippet, expected_cat, expected_retryable",
        [
            ("Invalid data found when processing input 'corrupt.mp4'", "CORRUPT_MEDIA", False),
            ("[mov,mp4,m4a,3gp,3g2,mj2 @ 000001] moov atom not found", "CORRUPT_MEDIA", False),
            ("No such filter: 'xfade_nonexistent'", "FILTER_GRAPH_ERROR", False),
            ("Cannot find a matching stream for unlabeled input pad 1 on filter 'Parsed_xfade_0'", "FILTER_GRAPH_ERROR", False),
            ("width not divisible by 2 (1921x1080)", "DIMENSION_FORMAT_MISMATCH", False),
            ("Incompatible pixel format 'yuv422p' for xfade", "DIMENSION_FORMAT_MISMATCH", False),
            ("Unknown encoder 'libx265_unsupported'", "CODEC_ERROR", False),
            ("Could not find codec parameters for stream 0", "CODEC_ERROR", False),
            ("FFmpeg command timed out after 300.0 seconds", "TIMEOUT_ERROR", True),
            ("No such file or directory: 'missing_clip.mp4'", "FILE_IO_ERROR", True),
            ("Permission denied: 'C:\\media\\locked.mp4'", "FILE_IO_ERROR", True),
            ("Some totally unexpected segmentation fault crash", "GENERAL_ERROR", False),
        ],
    )
    def test_parse_tool_error_categories(self, stderr_snippet: str, expected_cat: str, expected_retryable: bool):
        parsed = parse_tool_error(stderr_snippet)
        assert parsed.category == expected_cat
        assert parsed.is_retryable is expected_retryable
        assert len(parsed.recommendation) > 10
        assert len(parsed.root_cause) > 5

    def test_to_llm_feedback_formatting(self):
        parsed = parse_tool_error("width not divisible by 2")
        feedback = parsed.to_llm_feedback()
        assert "[TOOL ERROR RECOVERY: DIMENSION_FORMAT_MISMATCH]" in feedback
        assert "Root Cause:" in feedback
        assert "Recommendation:" in feedback
        assert "1920x1080" in feedback


class TestExecuteWithRetry:
    """Test suite for execute_with_retry retry mechanism and exponential backoff."""

    def test_retry_succeeds_immediately(self):
        mock_fn = MagicMock(return_value="success")
        result = execute_with_retry(mock_fn, "arg1", max_retries=3)
        assert result == "success"
        assert mock_fn.call_count == 1

    def test_retry_succeeds_on_second_attempt(self):
        # First call fails with retryable error (file locked), second succeeds
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("Permission denied: file locked")
            return "recovered"

        result = execute_with_retry(flaky, max_retries=3, initial_delay=0.01)
        assert result == "recovered"
        assert len(calls) == 2

    def test_retry_exhaustion_raises_last_error(self):
        def always_fails():
            raise TimeoutError("timed out after 10 seconds")

        with pytest.raises(TimeoutError, match="timed out"):
            execute_with_retry(always_fails, max_retries=3, initial_delay=0.01)

    def test_non_retryable_error_does_not_retry(self):
        mock_fn = MagicMock(side_effect=RuntimeError("No such filter: 'invalid'"))
        with pytest.raises(RuntimeError):
            execute_with_retry(mock_fn, max_retries=3, initial_delay=0.01)
        assert mock_fn.call_count == 1  # No repeated attempts for syntax error


class TestCallbacksIntegration:
    """Test suite for after_tool_callback and on_tool_error_callback hooks."""

    class MockTool:
        name = "render_with_transitions"

    def test_after_tool_callback_leaves_success_unchanged(self):
        success_response = {"status": "success", "output_path": "/path/video.mp4"}
        result = after_tool_error_recovery_callback(
            tool=self.MockTool(),
            args={},
            tool_response=success_response,
        )
        assert result == success_response
        assert "self_correction_guidance" not in result

    def test_after_tool_callback_enriches_error_response(self):
        error_response = {
            "status": "error",
            "stderr": "Cannot find a matching stream for unlabeled input pad 1",
            "returncode": 1,
        }
        result = after_tool_error_recovery_callback(
            tool=self.MockTool(),
            args={},
            tool_response=error_response,
        )
        assert result["status"] == "error"
        assert "parsed_error" in result
        assert result["parsed_error"]["category"] == "FILTER_GRAPH_ERROR"
        assert "self_correction_guidance" in result
        assert "[TOOL ERROR RECOVERY: FILTER_GRAPH_ERROR]" in result["self_correction_guidance"]

    def test_on_tool_error_callback_traps_exception(self):
        err = RuntimeError("Invalid data found when processing input 'damaged.mp4'")
        result = on_tool_error_recovery_callback(
            tool=self.MockTool(),
            args={"clip_path": "damaged.mp4"},
            error=err,
        )
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["tool_name"] == "render_with_transitions"
        assert result["parsed_error"]["category"] == "CORRUPT_MEDIA"
        assert "[TOOL ERROR RECOVERY: CORRUPT_MEDIA]" in result["self_correction_guidance"]
