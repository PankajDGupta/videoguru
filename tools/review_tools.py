"""Algorithmic Review Tools for VideoGuru (SPEC-010).

Provides functions and custom ADK tools to analyze Edit Decision Lists (EDLs)
against the YouTube recommendation algorithm:
- Pacing metrics (cut frequency, CPM, cadence variance, long cut detection)
- Hook metrics (first 5 seconds retention power, punchiness)
- Retention curve & narrative flow evaluation
- AlgorithmicReviewResult generation with heuristic rules or Gemini 2.0 Flash
- ADK tool review_edl_algorithmically with session state synchronization
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional, Sequence, Union

from google.genai import types

from config import settings
from schemas.edl import EditDecisionList
from schemas.review import (
    AlgorithmicReviewResult,
    HookMetrics,
    PacingMetrics,
    RetentionMetrics,
)

logger = logging.getLogger(__name__)

# Algorithmic benchmarks for YouTube video editing
IDEAL_MIN_AVG_CUT_DURATION = 2.0  # seconds
IDEAL_MAX_AVG_CUT_DURATION = 6.0  # seconds
MAX_CUT_DURATION_THRESHOLD = 10.0  # seconds (cuts longer than this risk drop-off)
MAX_HOOK_DURATION_THRESHOLD = 5.0  # seconds (first cut must establish immediate value)
MIN_PASSING_SCORE = 7.0  # 0.0 - 10.0 scale


def calculate_pacing_metrics(edl: EditDecisionList) -> PacingMetrics:
    """Compute mathematical pacing and cut frequency metrics for an EDL."""
    if not edl:
        raise ValueError("Cannot calculate pacing metrics for an empty EditDecisionList.")

    durations = [entry.duration for entry in edl]
    cut_count = len(durations)
    total_duration = sum(durations)
    avg_cut = total_duration / cut_count if cut_count > 0 else 0.0
    min_cut = min(durations)
    max_cut = max(durations)

    # Cuts per minute (CPM)
    cuts_per_minute = (cut_count / (total_duration / 60.0)) if total_duration > 0 else 0.0

    # Pacing variance (standard deviation of cut durations)
    if cut_count > 1:
        variance = sum((d - avg_cut) ** 2 for d in durations) / cut_count
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    # Cuts exceeding threshold without visual transition
    excessive_cuts = sum(1 for d in durations if d > MAX_CUT_DURATION_THRESHOLD)

    return PacingMetrics(
        avg_cut_duration=round(avg_cut, 3),
        cut_count=cut_count,
        cuts_per_minute=round(cuts_per_minute, 2),
        max_cut_duration=round(max_cut, 3),
        min_cut_duration=round(min_cut, 3),
        pacing_variance=round(std_dev, 3),
        excessive_cuts_count=excessive_cuts,
    )


def calculate_hook_metrics(edl: EditDecisionList) -> HookMetrics:
    """Analyze the opening cut and first 5-second hook retention power."""
    if not edl:
        raise ValueError("Cannot calculate hook metrics for an empty EditDecisionList.")

    first_entry = edl[0]
    hook_duration = first_entry.duration
    first_engagement = first_entry.engagement_score

    # Evaluate duration punchiness
    if hook_duration <= 3.5:
        duration_score = 9.5
    elif hook_duration <= MAX_HOOK_DURATION_THRESHOLD:
        duration_score = 8.5
    elif hook_duration <= 7.0:
        duration_score = 6.0 - (hook_duration - 5.0)
    else:
        duration_score = max(2.0, 4.0 - (hook_duration - 7.0) * 0.5)

    # Blend with engagement score if available
    if first_engagement is not None:
        hook_score = round(0.45 * duration_score + 0.55 * first_engagement, 1)
    else:
        hook_score = round(duration_score, 1)

    hook_score = max(0.0, min(10.0, hook_score))

    # A punchy hook starts in under 5.0s and scores >= 7.0
    is_punchy = (
        hook_duration <= MAX_HOOK_DURATION_THRESHOLD
        and hook_score >= MIN_PASSING_SCORE
        and (first_engagement is None or first_engagement >= 6.5)
    )

    return HookMetrics(
        hook_duration=round(hook_duration, 3),
        first_cut_engagement=round(first_engagement, 1) if first_engagement is not None else None,
        hook_score=hook_score,
        is_punchy=is_punchy,
    )


def calculate_retention_metrics(edl: EditDecisionList, pacing: PacingMetrics) -> RetentionMetrics:
    """Analyze overall timeline retention curve and transition continuity."""
    if not edl:
        raise ValueError("Cannot calculate retention metrics for an empty EditDecisionList.")

    total_duration = edl.total_duration

    # Transition diversity score
    intents = [entry.transition_intent.value for entry in edl]
    unique_intents = set(intents)
    if len(intents) <= 1:
        transition_score = 8.0
    elif len(unique_intents) >= 2:
        transition_score = 9.0
    else:
        transition_score = 8.0

    # Retention curve estimation
    # Base starts at 8.0
    base_retention = 8.5

    # Penalize for excessive cuts that cause retention drop-offs
    if pacing.excessive_cuts_count > 0:
        base_retention -= pacing.excessive_cuts_count * 1.5

    # Penalize if pacing is too slow
    if pacing.avg_cut_duration > IDEAL_MAX_AVG_CUT_DURATION:
        penalty = (pacing.avg_cut_duration - IDEAL_MAX_AVG_CUT_DURATION) * 0.8
        base_retention -= penalty

    # Bonus for good cadence variance (keeps viewer attention refreshed)
    if pacing.cut_count > 2 and 0.5 <= pacing.pacing_variance <= 3.0:
        base_retention += 0.5

    # Consider average engagement across all clips
    scores = [e.engagement_score for e in edl if e.engagement_score is not None]
    if scores:
        avg_engagement = sum(scores) / len(scores)
        retention_score = round(0.5 * base_retention + 0.5 * avg_engagement, 1)
    else:
        retention_score = round(base_retention, 1)

    retention_score = max(0.0, min(10.0, retention_score))

    return RetentionMetrics(
        total_duration=round(total_duration, 3),
        retention_score=retention_score,
        transition_diversity_score=round(transition_score, 1),
    )


def evaluate_edl_heuristically(
    edl: EditDecisionList,
    theme: str = "",
) -> AlgorithmicReviewResult:
    """Evaluate an EDL deterministically using algorithmic retention rules."""
    if not edl:
        raise ValueError("Cannot review an empty EditDecisionList.")

    pacing_metrics = calculate_pacing_metrics(edl)
    hook_metrics = calculate_hook_metrics(edl)
    retention_metrics = calculate_retention_metrics(edl, pacing_metrics)

    # Calculate pacing score
    avg_d = pacing_metrics.avg_cut_duration
    if IDEAL_MIN_AVG_CUT_DURATION <= avg_d <= IDEAL_MAX_AVG_CUT_DURATION:
        pacing_score = 9.0
    elif avg_d < IDEAL_MIN_AVG_CUT_DURATION:
        pacing_score = max(5.0, 9.0 - (IDEAL_MIN_AVG_CUT_DURATION - avg_d) * 2.0)
    else:
        pacing_score = max(3.0, 9.0 - (avg_d - IDEAL_MAX_AVG_CUT_DURATION) * 1.2)

    # Penalize for cuts over threshold
    if pacing_metrics.excessive_cuts_count > 0:
        pacing_score = max(1.0, pacing_score - pacing_metrics.excessive_cuts_count * 1.5)

    pacing_score = round(max(0.0, min(10.0, pacing_score)), 1)

    # Composite algorithmic score (weighted: 35% hook, 35% pacing, 30% retention)
    composite_score = round(
        0.35 * hook_metrics.hook_score
        + 0.35 * pacing_score
        + 0.30 * retention_metrics.retention_score,
        1,
    )
    composite_score = max(0.0, min(10.0, composite_score))

    # Actionable recommendations & diagnostic observations
    recommendations: list[str] = []
    strengths: list[str] = []
    concerns: list[str] = []

    # Hook evaluation
    if hook_metrics.is_punchy:
        strengths.append(
            f"Strong opening hook ({hook_metrics.hook_duration:.1f}s) grabs attention within the crucial 5-second window."
        )
    else:
        if hook_metrics.hook_duration > MAX_HOOK_DURATION_THRESHOLD:
            rec = (
                f"Trim opening cut '{edl[0].file_reference}' from {hook_metrics.hook_duration:.1f}s down to <= 4.0s "
                f"to prevent early viewer abandonment."
            )
            recommendations.append(rec)
            concerns.append(f"Opening cut duration ({hook_metrics.hook_duration:.1f}s) exceeds the 5-second hook threshold.")
        if hook_metrics.first_cut_engagement is not None and hook_metrics.first_cut_engagement < 7.0:
            rec = (
                f"Opening cut '{edl[0].file_reference}' has low engagement ({hook_metrics.first_cut_engagement:.1f}/10). "
                f"Lead with the highest-energy shot to maximize initial retention."
            )
            recommendations.append(rec)
            concerns.append(f"First cut engagement score ({hook_metrics.first_cut_engagement:.1f}/10) is below recommended 7.0 threshold.")

    # Pacing evaluation
    if IDEAL_MIN_AVG_CUT_DURATION <= pacing_metrics.avg_cut_duration <= IDEAL_MAX_AVG_CUT_DURATION:
        strengths.append(
            f"Optimal cut frequency ({pacing_metrics.cuts_per_minute:.1f} CPM, avg {pacing_metrics.avg_cut_duration:.1f}s/cut) aligns with YouTube engagement sweet spots."
        )
    elif pacing_metrics.avg_cut_duration > IDEAL_MAX_AVG_CUT_DURATION:
        concerns.append(
            f"Average cut duration ({pacing_metrics.avg_cut_duration:.1f}s) is sluggish. Viewer attention may flag."
        )
        recommendations.append(
            f"Increase cut frequency by trimming clips to an average of {IDEAL_MIN_AVG_CUT_DURATION:.1f}s - {IDEAL_MAX_AVG_CUT_DURATION:.1f}s."
        )
    else:
        concerns.append(
            f"Average cut duration ({pacing_metrics.avg_cut_duration:.1f}s) is extremely rapid. Ensure cuts have sufficient visual breathing room."
        )

    # Check for long individual cuts
    for idx, entry in enumerate(edl, 1):
        if entry.duration > MAX_CUT_DURATION_THRESHOLD:
            rec = (
                f"Cut #{idx} ('{entry.file_reference}') duration is {entry.duration:.1f}s (> {MAX_CUT_DURATION_THRESHOLD:.1f}s). "
                f"Trim segment or split with b-roll to maintain pacing."
            )
            recommendations.append(rec)
            concerns.append(f"Cut #{idx} is too long ({entry.duration:.1f}s) without a visual transition.")

    # Retention evaluation
    if retention_metrics.retention_score >= 7.5:
        strengths.append(
            f"Timeline exhibits healthy overall momentum with {len(edl)} cut(s) totaling {retention_metrics.total_duration:.1f}s."
        )

    # Passing criteria: score >= 7.0, hook is punchy, no excessive long cuts (> 10s)
    passed = (
        composite_score >= MIN_PASSING_SCORE
        and hook_metrics.is_punchy
        and pacing_metrics.excessive_cuts_count == 0
    )

    # Format feedback markdown string
    status_icon = "[PASS]" if passed else "[REVISE]"
    status_text = "PASSED - Optimized for YouTube Algorithm" if passed else "REVISION REQUIRED - Algorithmic Bottlenecks Detected"

    feedback_lines = [
        f"### Algorithmic Review Report: {status_icon} {status_text}",
        f"**Theme:** {theme or 'General'}",
        f"**Overall Algorithmic Score:** {composite_score:.1f}/10.0 (Hook: {hook_metrics.hook_score:.1f} | Pacing: {pacing_score:.1f} | Retention: {retention_metrics.retention_score:.1f})",
        "",
        "#### Timeline Metrics Breakdown:",
        f"- **Total Cuts:** {pacing_metrics.cut_count}",
        f"- **Total Duration:** {retention_metrics.total_duration:.2f}s",
        f"- **Average Cut Duration:** {pacing_metrics.avg_cut_duration:.2f}s (Cadence Variance: +/-{pacing_metrics.pacing_variance:.2f}s)",
        f"- **Cut Frequency:** {pacing_metrics.cuts_per_minute:.1f} cuts/min",
        f"- **Opening Hook:** {hook_metrics.hook_duration:.2f}s ({'Punchy' if hook_metrics.is_punchy else 'Needs Optimization'})",
        "",
    ]

    if strengths:
        feedback_lines.append("#### Algorithmic Strengths:")
        for s in strengths:
            feedback_lines.append(f"- {s}")
        feedback_lines.append("")

    if concerns:
        feedback_lines.append("#### Retention Risks & Bottlenecks:")
        for c in concerns:
            feedback_lines.append(f"- {c}")
        feedback_lines.append("")

    if recommendations:
        feedback_lines.append("#### Actionable Revision Recommendations:")
        for r in recommendations:
            feedback_lines.append(f"- {r}")
        feedback_lines.append("")
    else:
        feedback_lines.append("#### Recommendations:")
        feedback_lines.append("- Edit decision list satisfies all YouTube algorithm retention and pacing criteria.")
        feedback_lines.append("")

    feedback = "\n".join(feedback_lines)

    return AlgorithmicReviewResult(
        passed=passed,
        score=composite_score,
        hook_score=hook_metrics.hook_score,
        pacing_score=pacing_score,
        retention_score=retention_metrics.retention_score,
        feedback=feedback,
        recommendations=recommendations,
        pacing_metrics=pacing_metrics,
        hook_metrics=hook_metrics,
        retention_metrics=retention_metrics,
    )


def evaluate_edl_with_gemini(
    edl: EditDecisionList,
    theme: str = "",
    client: Optional[Any] = None,
    model: str = settings.GEMINI_MODEL,
) -> AlgorithmicReviewResult:
    """Evaluate an EDL using Gemini 2.0 Flash as a YouTube algorithm expert."""
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key and client is None:
        logger.info("No Gemini API key available; using deterministic heuristic evaluation.")
        return evaluate_edl_heuristically(edl=edl, theme=theme)

    try:
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)

        pacing_metrics = calculate_pacing_metrics(edl)
        hook_metrics = calculate_hook_metrics(edl)
        retention_metrics = calculate_retention_metrics(edl, pacing_metrics)

        cuts_summary = []
        for idx, entry in enumerate(edl, 1):
            score_txt = f", engagement={entry.engagement_score:.1f}" if entry.engagement_score is not None else ""
            cuts_summary.append(
                f"Cut #{idx}: clip='{entry.file_reference}', trim={entry.start_trim:.2f}s->{entry.end_trim:.2f}s "
                f"(duration={entry.duration:.2f}s, transition={entry.transition_intent.value}{score_txt}), "
                f"rationale='{entry.scene_rationale}'"
            )

        prompt = (
            "You are a master YouTube algorithm strategist and viral video analytics specialist.\n"
            "Analyze the following proposed Edit Decision List (EDL) against YouTube recommendation metrics:\n"
            "1. Pacing: Check cut frequency, rhythm, and whether cuts happen often enough to reset attention.\n"
            "2. Hook strength: Scrutinize the first 5 seconds. Does it grab attention immediately?\n"
            "3. Retention curve: Identify any long cuts (>10s) or slow spots that cause viewer drop-off.\n\n"
            f"Creator Theme: {theme or 'General Highlights'}\n"
            f"Pre-computed Pacing: {pacing_metrics.avg_cut_duration}s avg cut, {pacing_metrics.cuts_per_minute} CPM, {pacing_metrics.excessive_cuts_count} cuts >10s\n"
            f"Pre-computed Hook: {hook_metrics.hook_duration}s duration, punchy={hook_metrics.is_punchy}\n\n"
            "Timeline Cuts:\n"
            + "\n".join(cuts_summary)
            + "\n\nProvide your evaluation strictly conforming to the AlgorithmicReviewResult schema."
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AlgorithmicReviewResult,
                temperature=0.2,
            ),
        )

        if response.text:
            return AlgorithmicReviewResult.from_json(response.text)

        logger.warning("Empty response from Gemini; falling back to heuristic review.")
        return evaluate_edl_heuristically(edl=edl, theme=theme)

    except Exception as exc:
        logger.warning("Gemini algorithmic review failed (%s); falling back to heuristic review.", exc)
        return evaluate_edl_heuristically(edl=edl, theme=theme)


def review_edl_algorithmically(
    tool_context: Optional[Any] = None,
    edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]], str]] = None,
    theme: Optional[str] = None,
    offline: bool = False,
) -> AlgorithmicReviewResult:
    """ADK Tool: Analyze an Edit Decision List against YouTube recommendation algorithm metrics.

    Evaluates pacing (cut frequency, CPM), hook strength (first 5 seconds), and viewer retention curve.
    Returns AlgorithmicReviewResult and records results in session state under 'reviewer_passed',
    'reviewer_score', 'reviewer_feedback', and 'algorithmic_review'.

    Args:
        tool_context: ADK ToolContext providing access to session.state.
        edl: Explicit EditDecisionList instance, list of cut dicts, or file path. If None,
            reads 'edl' from tool_context.state.
        theme: Creator's theme. If None, reads 'theme' from tool_context.state.
        offline: If True, forces deterministic heuristic review without LLM calls.

    Returns:
        AlgorithmicReviewResult with scores, metrics, feedback, and pass/fail flag.
    """
    # 1. Resolve theme
    if theme is None and tool_context is not None:
        theme = tool_context.state.get("theme", "")
    theme_str = str(theme or "")

    # 2. Resolve EDL
    resolved_edl: Optional[EditDecisionList] = None
    if edl is not None:
        if isinstance(edl, EditDecisionList):
            resolved_edl = edl
        elif isinstance(edl, str):
            resolved_edl = EditDecisionList.from_file(edl)
        elif isinstance(edl, Sequence):
            resolved_edl = EditDecisionList.from_dict_list(edl)
    elif tool_context is not None:
        state_edl = tool_context.state.get("edl")
        if state_edl:
            if isinstance(state_edl, EditDecisionList):
                resolved_edl = state_edl
            elif isinstance(state_edl, Sequence):
                resolved_edl = EditDecisionList.from_dict_list(state_edl)

    if not resolved_edl:
        raise ValueError(
            "No Edit Decision List provided or found in session state. "
            "Please run Phase II Curation (Curation Agent) before executing algorithmic review."
        )

    # 3. Determine execution mode
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    run_offline = offline or not has_api_key

    if run_offline:
        result = evaluate_edl_heuristically(edl=resolved_edl, theme=theme_str)
    else:
        result = evaluate_edl_with_gemini(edl=resolved_edl, theme=theme_str)

    # 4. Synchronize results to session state
    if tool_context is not None:
        tool_context.state["reviewer_passed"] = result.passed
        tool_context.state["reviewer_score"] = result.score
        tool_context.state["reviewer_feedback"] = result.feedback
        tool_context.state["algorithmic_review"] = result.to_dict()

    return result


def get_review_from_state(state: dict[str, Any]) -> Optional[AlgorithmicReviewResult]:
    """Helper to reconstruct an AlgorithmicReviewResult from session state."""
    raw = state.get("algorithmic_review")
    if not raw:
        return None
    try:
        if isinstance(raw, AlgorithmicReviewResult):
            return raw
        if isinstance(raw, dict):
            return AlgorithmicReviewResult.from_dict(raw)
        if isinstance(raw, str):
            return AlgorithmicReviewResult.from_json(raw)
    except Exception as exc:
        logger.warning("Failed to deserialize AlgorithmicReviewResult from state: %s", exc)
    return None
