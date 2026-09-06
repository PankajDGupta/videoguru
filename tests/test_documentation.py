"""Unit Tests for Documentation & README (SPEC-030)."""

from __future__ import annotations

import importlib
from pathlib import Path
import pytest

from config import settings

REQUIRED_README_SECTIONS = [
    "Architecture Overview",
    "Agent Roster",
    "Security & Sandboxing",
    "Prerequisites & Setup",
    "Usage Guide",
    "Testing & Verification",
    "Project Structure",
]

CRITICAL_MODULES = [
    "agents.root_pipeline",
    "agents.root_greeter",
    "agents.ingestion",
    "agents.curation",
    "agents.reviewer",
    "agents.critic",
    "agents.loop_agent",
    "agents.review_orchestrator",
    "agents.enhancement_rendering",
    "callbacks.input_guardrail",
    "callbacks.tool_sandbox",
    "callbacks.error_recovery",
    "callbacks.schema_validation",
    "rendering.ffmpeg_builder",
    "schemas.edl",
    "schemas.media",
    "schemas.review",
    "schemas.critic",
    "services.exceptions",
    "services.resilience",
    "services.observability",
    "services.session_service",
    "tools.directory_scanner",
    "tools.clip_metadata",
    "tools.video_analysis",
    "tools.curation_tools",
    "tools.review_tools",
    "tools.critic_tools",
    "tools.otio_converter",
    "tools.transition_renderer",
    "tools.audio_ducking",
    "tools.whisper_captioning",
    "config.settings",
]


class TestReadmeDocumentation:
    """Tests to verify the presence and completeness of README.md."""

    def test_readme_file_exists(self):
        readme_path = settings.BASE_DIR / "README.md"
        assert readme_path.exists(), "README.md must exist in project root."
        assert readme_path.stat().st_size > 1000, "README.md should be comprehensive."

    @pytest.mark.parametrize("section", REQUIRED_README_SECTIONS)
    def test_readme_contains_required_sections(self, section: str):
        readme_path = settings.BASE_DIR / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        assert section.lower() in content.lower(), f"README.md missing section: '{section}'"

    def test_readme_contains_cli_commands(self):
        readme_path = settings.BASE_DIR / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        assert "--pipeline" in content
        assert "--scan-dir" in content
        assert "--extract-metadata" in content
        assert "--ingest-dir" in content
        assert "--curate" in content
        assert "--review-edl" in content
        assert "--critic" in content
        assert "--loop" in content
        assert "--web" in content
        assert "--offline" in content


class TestModuleDocstrings:
    """Tests to verify that critical modules, classes, and tools contain docstrings."""

    @pytest.mark.parametrize("module_name", CRITICAL_MODULES)
    def test_module_has_docstring(self, module_name: str):
        mod = importlib.import_module(module_name)
        assert mod.__doc__ is not None and len(mod.__doc__.strip()) > 0, (
            f"Module '{module_name}' must have a module-level docstring."
        )
