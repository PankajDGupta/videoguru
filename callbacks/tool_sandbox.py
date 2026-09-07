"""Tool Sandboxing & FFmpeg Validation Callback for VideoGuru (SPEC-022).

Provides before_tool_callback implementation to sandbox external tools and FFmpeg executions:
- Whitelists permitted binaries (`ffmpeg`, `ffprobe`) and registered VideoGuru tools.
- Whitelists permitted FFmpeg flags, blocking hazardous options.
- Restricts all file paths to designated media, staging, output, or temporary directories.
- Strictly blocks shell injection metacharacters and path traversal attempts.
- Raises SecuritySandboxingError or returns a security violation response.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Optional, Sequence, Union

from config import settings

logger = logging.getLogger("videoguru.security.sandbox")

# Custom exception for security violations in tool invocation
class SecuritySandboxingError(PermissionError):
    """Raised when a tool argument or FFmpeg command violates security policies."""
    pass


# Whitelisted binary names (lower-cased without extension)
PERMITTED_BINARIES: frozenset[str] = frozenset({
    "ffmpeg",
    "ffmpeg.exe",
    "ffprobe",
    "ffprobe.exe",
})

# Whitelisted FFmpeg CLI option flags
PERMITTED_FFMPEG_FLAGS: frozenset[str] = frozenset({
    "-y",
    "-n",
    "-i",
    "-filter_complex",
    "-vf",
    "-af",
    "-c:v",
    "-c:a",
    "-c",
    "-b:v",
    "-b:a",
    "-preset",
    "-crf",
    "-ss",
    "-to",
    "-t",
    "-map",
    "-v",
    "-loglevel",
    "-threads",
    "-r",
    "-s",
    "-aspect",
    "-pix_fmt",
    "-f",
    "-an",
    "-vn",
    # ffprobe flags
    "-show_entries",
    "-select_streams",
    "-of",
    "-print_format",
    "-show_format",
    "-show_streams",
    "-hide_banner",
})

# Dangerous flags strictly forbidden
FORBIDDEN_FLAGS: frozenset[str] = frozenset({
    "-protocol_whitelist",
    "-protocol_blacklist",
    "-filter_script",
    "-dump_attachment",
    "-re",
})

# Dangerous protocols forbidden in URLs/filter graphs
FORBIDDEN_PROTOCOLS: Sequence[str] = [
    "http://", "https://", "ftp://", "gopher://", "tcp://", "udp://", "rtmp://", "smb://"
]

# Shell injection metacharacters
SHELL_INJECTION_PATTERN = re.compile(r"[;&|`$><\n\r]")

# Tools that directly execute shell commands and therefore need blanket
# string-argument shell metacharacter validation.  All other tools either:
#   • call Python APIs (no shell risk),
#   • use pathlib/ffprobe through validated wrappers (paths already checked
#     in step 2 via validate_path_confined, which itself calls is_shell_safe), or
#   • store text in session state (append_to_state, exit_loop, etc.).
# Applying the blanket check to tools like append_to_state causes false
# positives because LLM-generated feedback text legitimately contains
# markdown characters (**, *, >, `, $, etc.).
_SHELL_EXECUTING_TOOLS: frozenset[str] = frozenset({
    "execute_ffmpeg_command",
})

# Known sensitive system root paths
DISALLOWED_ROOT_PREFIXES: Sequence[str] = [
    "/etc",
    "/root",
    "/bin",
    "/usr",
    "/var",
    "/sys",
    "/proc",
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\users\\default",
]


def get_default_allowed_directories() -> list[Path]:
    """Retrieve the set of allowed directories for file read/write access.

    Returns:
        List of resolved Path objects covering media, staging, output, and temp dirs.
    """
    allowed = [
        settings.MEDIA_INPUT_DIR.resolve(),
        settings.STAGING_DIR.resolve(),
        settings.OUTPUT_DIR.resolve(),
        Path(tempfile.gettempdir()).resolve(),
        settings.BASE_DIR.resolve(),
    ]
    return allowed


def is_shell_safe(value: str) -> bool:
    """Check if a string is free from dangerous shell metacharacters.

    Args:
        value: The string argument to check.

    Returns:
        True if safe, False if shell metacharacters were detected.
    """
    if not isinstance(value, str):
        return False
    # Check for shell metacharacters
    return not bool(SHELL_INJECTION_PATTERN.search(value))


def validate_path_confined(
    path_str: Union[str, Path],
    allowed_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Path:
    """Ensure a file path does not use path traversal and resides within allowed directories.

    Args:
        path_str: Path string or Path object to validate.
        allowed_dirs: Optional custom allowed directories list.

    Returns:
        The resolved Path object if valid.

    Raises:
        SecuritySandboxingError: If the path is outside allowed boundaries or traverses.
    """
    raw_str = str(path_str).strip()
    if not raw_str:
        raise SecuritySandboxingError("Path cannot be empty.")

    # Check for shell injection characters in paths
    if not is_shell_safe(raw_str):
        raise SecuritySandboxingError(
            f"Shell injection characters detected in path: '{raw_str}'"
        )

    # Check for path traversal characters
    if ".." in raw_str:
        raise SecuritySandboxingError(
            f"Path traversal ('..') detected in path: '{raw_str}'"
        )

    # Check for forbidden protocols in paths
    for proto in FORBIDDEN_PROTOCOLS:
        if proto.lower() in raw_str.lower():
            raise SecuritySandboxingError(
                f"Network protocol '{proto}' is forbidden in path: '{raw_str}'"
            )

    # Check for disallowed system root prefixes
    normalized_lower = raw_str.lower().replace("/", "\\")
    for prefix in DISALLOWED_ROOT_PREFIXES:
        norm_prefix = prefix.lower().replace("/", "\\")
        if normalized_lower.startswith(norm_prefix):
            raise SecuritySandboxingError(
                f"Access to sensitive system path '{prefix}' is blocked: '{raw_str}'"
            )

    # Resolve allowed directories
    if allowed_dirs is None:
        permitted_paths = get_default_allowed_directories()
    else:
        permitted_paths = [Path(d).resolve() for d in allowed_dirs]

    target_path = Path(raw_str).resolve()

    # Verify target path is child of at least one permitted directory
    is_contained = False
    for allowed in permitted_paths:
        try:
            target_path.relative_to(allowed)
            is_contained = True
            break
        except ValueError:
            continue

    if not is_contained:
        allowed_str = ", ".join(str(p) for p in permitted_paths)
        raise SecuritySandboxingError(
            f"Path '{target_path}' is outside designated sandbox directories [{allowed_str}]."
        )

    return target_path


def validate_ffmpeg_command(
    cmd: Sequence[str],
    allowed_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> None:
    """Validate that an FFmpeg or ffprobe CLI command conforms to security sandboxing.

    Ensures:
    1. Executable binary is whitelisted (`ffmpeg` or `ffprobe`).
    2. Flags are whitelisted and no forbidden flags are used.
    3. Arguments do not contain shell injection metacharacters.
    4. Input and output file paths reside within permitted directories.

    Args:
        cmd: Sequence of command line argument strings.
        allowed_dirs: Optional list of permitted directory boundaries.

    Raises:
        SecuritySandboxingError: If any security validation check fails.
    """
    if not cmd:
        raise SecuritySandboxingError("Command argument sequence cannot be empty.")

    # 1. Validate binary
    raw_bin = str(cmd[0]).strip()
    bin_name = Path(raw_bin).name.lower()
    if bin_name not in PERMITTED_BINARIES and not any(
        bin_name.startswith(p) for p in ["ffmpeg", "ffprobe"]
    ):
        raise SecuritySandboxingError(
            f"Binary '{raw_bin}' is not permitted. Only FFmpeg and FFprobe are allowed."
        )

    # 2. Iterate through arguments
    i = 1
    total_args = len(cmd)
    while i < total_args:
        arg = str(cmd[i]).strip()

        # Check shell injection
        if not is_shell_safe(arg):
            raise SecuritySandboxingError(
                f"Shell injection characters detected in command argument: '{arg}'"
            )

        # Check forbidden flags
        if arg.lower() in FORBIDDEN_FLAGS:
            raise SecuritySandboxingError(
                f"Prohibited FFmpeg option flag detected: '{arg}'"
            )

        # Check for forbidden network protocols
        for proto in FORBIDDEN_PROTOCOLS:
            if proto.lower() in arg.lower():
                raise SecuritySandboxingError(
                    f"Network protocol '{proto}' is forbidden in argument: '{arg}'"
                )

        if arg.startswith("-"):
            # It's an option flag
            flag_name = arg.split("=")[0]
            # Some flags like -c:v or -filter_complex:v have suffixes
            base_flag = flag_name.split(":")[0] if ":" in flag_name else flag_name
            if (
                flag_name not in PERMITTED_FFMPEG_FLAGS
                and base_flag not in PERMITTED_FFMPEG_FLAGS
            ):
                raise SecuritySandboxingError(
                    f"Unrecognized or unwhitelisted FFmpeg flag: '{flag_name}'"
                )

            # If it is an input flag, the next arg is an input file path
            if flag_name == "-i" and i + 1 < total_args:
                input_file = cmd[i + 1]
                validate_path_confined(input_file, allowed_dirs)
                i += 1
        else:
            # Non-flag argument: could be input, filter, or output path
            # If it's the last argument, it is typically the output path
            if i == total_args - 1:
                validate_path_confined(arg, allowed_dirs)

        i += 1


def before_tool_sandbox_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any = None,
    allowed_dirs: Optional[Sequence[Union[str, Path]]] = None,
    raise_exception: bool = True,
    **kwargs: Any,
) -> Optional[dict[str, Any]]:
    """ADK before_tool_callback implementation enforcing FFmpeg and tool sandboxing (SPEC-022).

    Inspects tool invocations and parameters:
    - If tool receives `cmd` (e.g., execute_ffmpeg_command): validates the full CLI list.
    - If tool receives path parameters (`clip_path`, `video_path`, `output_path`, `music_path`,
      `directory_path`, `srt_path`): confines them to permitted sandbox directories.
    - If a security check fails:
        - If raise_exception=True, raises SecuritySandboxingError immediately.
        - If raise_exception=False, returns an error dict short-circuiting the tool.
    - If valid, returns None (allowing tool execution to proceed).

    Args:
        tool: The BaseTool or function being invoked.
        args: Dictionary of keyword arguments to the tool.
        tool_context: ADK ToolContext (or None).
        allowed_dirs: Optional custom allowed directories list.
        raise_exception: Whether to raise SecuritySandboxingError or return error dict.

    Returns:
        None if validation succeeds, or error dict if blocked and raise_exception is False.
    """
    tool_name = getattr(tool, "name", str(tool))
    logger.debug("Sandbox validating tool '%s' with args: %s", tool_name, list(args.keys()))

    try:
        # 1. Validate raw cmd list if present
        if "cmd" in args and isinstance(args["cmd"], (list, tuple)):
            validate_ffmpeg_command(args["cmd"], allowed_dirs=allowed_dirs)

        # 2. Validate path-based keyword arguments
        path_keys = [
            "path", "clip_path", "video_path", "output_path", "music_path",
            "directory_path", "srt_path", "file_path", "dir_path",
        ]
        for key in path_keys:
            if key in args and args[key] is not None:
                val = args[key]
                if isinstance(val, (str, Path)):
                    validate_path_confined(val, allowed_dirs=allowed_dirs)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, (str, Path)):
                            validate_path_confined(item, allowed_dirs=allowed_dirs)

        # 3. Check for shell injection in all string arguments — but ONLY for
        #    tools that directly execute shell commands.  For data-only tools
        #    (append_to_state, exit_loop, review_edl_as_critic, etc.) the text
        #    arguments legitimately contain markdown/natural-language characters
        #    like **, >, `, $ which would cause false-positive blocks.
        #    Path arguments are already shell-checked inside validate_path_confined
        #    (step 2), so non-shell tools remain protected against path injection.
        if tool_name in _SHELL_EXECUTING_TOOLS:
            for k, v in args.items():
                if isinstance(v, str) and not is_shell_safe(v):
                    raise SecuritySandboxingError(
                        f"Shell metacharacters detected in argument '{k}': '{v}'"
                    )

    except SecuritySandboxingError as exc:
        logger.warning(
            "[SECURITY TOOL SANDBOX VIOLATION] Blocked tool '%s': %s",
            tool_name,
            exc,
        )
        if raise_exception:
            raise
        return {
            "status": "error",
            "error": str(exc),
            "blocked_by_sandbox": True,
        }

    return None
