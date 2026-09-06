"""Unit Tests for VideoGuru Logging & Observability (SPEC-028)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from config import settings
from main import parse_arguments
from services.observability import (
    JsonFormatter,
    configure_logging,
    log_agent_transition,
    log_edl_iteration,
    log_render_progress,
    log_tool_invocation,
)


class TestJsonFormatter:
    """Tests for JsonFormatter structured serialization."""

    def test_json_formatter_basic_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Test message %s",
            args=("param1",),
            exc_info=None,
        )
        record.funcName = "test_func"
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "Test message param1"
        assert data["line"] == 42
        assert data["function"] == "test_func"
        assert "timestamp" in data
        assert "module" in data

    def test_json_formatter_extra_fields(self):
        formatter = JsonFormatter(extra_fields={"app": "videoguru", "version": "1.0"})
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname=__file__,
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.event = "custom_event"
        record.custom_metric = 99.5

        output = formatter.format(record)
        data = json.loads(output)

        assert data["app"] == "videoguru"
        assert data["version"] == "1.0"
        assert data["event"] == "custom_event"
        assert data["custom_metric"] == 99.5

    def test_json_formatter_exception_traceback(self):
        formatter = JsonFormatter(include_traceback=True)
        try:
            raise ValueError("Test error for traceback")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=50,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error for traceback"
        assert isinstance(data["exception"]["traceback"], list)
        assert len(data["exception"]["traceback"]) > 0

    def test_json_formatter_non_serializable_fallback(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Non-serializable extra",
            args=(),
            exc_info=None,
        )
        record.complex_obj = object()
        output = formatter.format(record)
        data = json.loads(output)

        assert "complex_obj" in data
        assert "object at" in data["complex_obj"]


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_json(self):
        test_logger = configure_logging(
            log_level="DEBUG",
            json_format=True,
            logger_name="test_videoguru_json",
        )
        assert test_logger.level == logging.DEBUG
        assert len(test_logger.handlers) >= 1
        assert isinstance(test_logger.handlers[0].formatter, JsonFormatter)

    def test_configure_logging_text(self):
        test_logger = configure_logging(
            log_level="WARNING",
            json_format=False,
            logger_name="test_videoguru_text",
        )
        assert test_logger.level == logging.WARNING
        assert len(test_logger.handlers) >= 1
        assert not isinstance(test_logger.handlers[0].formatter, JsonFormatter)

    def test_configure_logging_file_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            test_logger = configure_logging(
                log_level="INFO",
                json_format=True,
                log_file=log_file,
                logger_name="test_videoguru_file",
            )
            try:
                test_logger.info("Test log line to file", extra={"event": "unit_test"})

                # Flush handlers
                for h in test_logger.handlers:
                    h.flush()

                assert log_file.exists()
                content = log_file.read_text(encoding="utf-8").strip()
                data = json.loads(content)
                assert data["message"] == "Test log line to file"
                assert data["event"] == "unit_test"
            finally:
                for h in list(test_logger.handlers):
                    h.close()
                    test_logger.removeHandler(h)


class TestEventLoggingHelpers:
    """Tests for structured event logging helper functions."""

    def test_log_agent_transition(self, caplog):
        logger = logging.getLogger("test_transitions")
        logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            log_agent_transition(
                from_agent="RootGreeterAgent",
                to_agent="IngestionAgent",
                session_id="session_123",
                theme="Hiking",
                logger=logger,
            )

        assert "Agent transition: 'RootGreeterAgent' -> 'IngestionAgent'" in caplog.text

    def test_log_tool_invocation(self, caplog):
        logger = logging.getLogger("test_tools")
        logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            log_tool_invocation(
                tool_name="render_with_transitions",
                session_id="session_123",
                agent_name="EnhancementRenderingAgent",
                duration_seconds=2.45,
                status="success",
                logger=logger,
            )

        assert "Tool 'render_with_transitions' completed with status 'success'" in caplog.text

    def test_log_edl_iteration(self, caplog):
        logger = logging.getLogger("test_edl")
        logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            log_edl_iteration(
                iteration=2,
                cut_count=5,
                total_duration=28.5,
                score=8.2,
                passed=True,
                feedback="Pacing looks great.",
                session_id="session_123",
                logger=logger,
            )

        assert "EDL Iteration #2: 5 cuts, 28.50s, status=PASSED, score=8.2" in caplog.text

    def test_log_render_progress(self, caplog):
        logger = logging.getLogger("test_render")
        logger.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            log_render_progress(
                stage="xfade_transitions",
                progress_percent=50.0,
                output_path="staging/transitions.mp4",
                session_id="session_123",
                logger=logger,
            )

        assert "Render milestone: stage='xfade_transitions' (50%) -> staging/transitions.mp4" in caplog.text


class TestObservabilitySettingsAndCli:
    """Tests for settings validation and CLI argument parsing for logging."""

    def test_settings_validation_log_level(self):
        with patch.object(settings, "LOG_LEVEL", "INVALID_LEVEL"):
            assert not settings.validate_settings()

        with patch.object(settings, "LOG_LEVEL", "DEBUG"):
            assert settings.validate_settings()

    def test_cli_logging_arguments(self):
        with patch("sys.argv", ["main.py", "--log-level", "DEBUG", "--log-json", "--log-file", "test.log"]):
            args = parse_arguments()
            assert args.log_level == "DEBUG"
            assert args.log_json is True
            assert args.log_file == "test.log"

        with patch("sys.argv", ["main.py", "--no-log-json"]):
            args = parse_arguments()
            assert args.log_json is False
