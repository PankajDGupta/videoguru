"""Human-Centric Critic Review Tools for VideoGuru (SPEC-011).

Provides functions and custom ADK tools to evaluate Edit Decision Lists (EDLs)
from a human viewer and thumbnail strategist perspective:
- 30-Second Hook & Average View Duration (AVD) evaluation.
- Thumbnail Click-Through Rate (CTR) viability and frame recommendation.
- Emotional engagement, narrative arc, and thematic resonance analysis.
- CriticReviewResult generation with heuristic rules or Gemini 2.0 Flash.
- ADK tool review_edl_as_critic with loop control integration (exit_loop vs append_to_state).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional, Sequence, Union

from google.genai import types

from config import settings
from schemas.critic import (
    CriticReviewResult,
    Hook30sMetrics,
    StorytellingMetrics,
    ThumbnailMetrics,
)
from schemas.edl import EditDecisionList
from tools.loop_tools import append_to_state, exit_loop

logger = logging.getLogger(__name__)

# Human-centric thresholds for YouTube viewing
MAX_30S_WINDOW = 30.0  # seconds
MAX_INTRO_CUT_DURATION = 8.0  # cuts longer than 8s in the first 30s risk viewer drop-off
MIN_CRITIC_PASSING_SCORE = 7.0  # 0.0 to 10.0 scale

# Keywords indicative of visual pop, emotion, and thumbnail potential
THUMBNAIL_POSITIVE_KEYWORDS = {
    "action", "close-up", "face", "expression", "reaction", "smile", "laugh",
    "dramatic", "vibrant", "sunset", "climax", "reveal", "jump", "highlight",
    "energy", "motion", "peak", "dynamic", "stunning", "epic", "focus"
}


def calculate_30s_hook_metrics(edl: EditDecisionList, theme: str = "") -> Hook30sMetrics:
    """Analyze the opening 30-second window for pacing, theme establishment, and dead air."""
    if not edl:
        raise ValueError("Cannot calculate 30s hook metrics for an empty EditDecisionList.")

    total_duration = edl.total_duration
    analyzed_duration = min(MAX_30S_WINDOW, total_duration)

    elapsed = 0.0
    cuts_in_30s = 0
    has_dead_air = False
    early_engagements: list[float] = []
    early_rationales: list[str] = []

    for entry in edl:
        if elapsed >= MAX_30S_WINDOW:
            break
        cuts_in_30s += 1
        cut_duration = entry.duration

        # Flag cuts longer than 8s in intro window as dead air risks
        if cut_duration > MAX_INTRO_CUT_DURATION:
            has_dead_air = True

        if entry.engagement_score is not None:
            early_engagements.append(entry.engagement_score)
        early_rationales.append(entry.scene_rationale.lower())

        elapsed += cut_duration

    # Calculate theme setup score based on early rationales matching creator theme
    theme_lower = theme.lower().strip() if theme else ""
    theme_words = set(re.findall(r"\w+", theme_lower)) if theme_lower else set()

    if theme_words:
        matched_words = 0
        all_early_text = " ".join(early_rationales)
        for w in theme_words:
            if len(w) > 3 and w in all_early_text:
                matched_words += 1
        if matched_words >= 2:
            theme_setup_score = 9.0
        elif matched_words == 1:
            theme_setup_score = 7.5
        else:
            theme_setup_score = 6.0
    else:
        theme_setup_score = 7.5

    # Calculate Hook AVD score
    base_avd = 8.5
    if has_dead_air:
        base_avd -= 2.5

    # Penalty for too few cuts in a long intro (e.g. 30s with only 1 cut)
    if analyzed_duration >= 20.0 and cuts_in_30s < 3:
        base_avd -= 1.5
    elif 3 <= cuts_in_30s <= 8:
        base_avd += 0.5

    # Check first cut punchiness
    first_duration = edl[0].duration
    if first_duration > 5.0:
        base_avd -= 0.8
    elif first_duration <= 3.5:
        base_avd += 0.5

    if early_engagements:
        avg_early = sum(early_engagements) / len(early_engagements)
        hook_avd_score = round(0.5 * base_avd + 0.5 * avg_early, 1)
    else:
        hook_avd_score = round(base_avd, 1)

    hook_avd_score = max(0.0, min(10.0, hook_avd_score))

    is_engaging = (hook_avd_score >= MIN_CRITIC_PASSING_SCORE) and (not has_dead_air)

    return Hook30sMetrics(
        duration_analyzed=round(analyzed_duration, 2),
        cut_count_in_30s=cuts_in_30s,
        hook_avd_score=hook_avd_score,
        theme_setup_score=round(theme_setup_score, 1),
        has_dead_air=has_dead_air,
        is_engaging=is_engaging,
    )


def calculate_thumbnail_metrics(edl: EditDecisionList, theme: str = "") -> ThumbnailMetrics:
    """Identify the most viable candidate cut and frame for a high-CTR YouTube thumbnail."""
    if not edl:
        raise ValueError("Cannot calculate thumbnail metrics for an empty EditDecisionList.")

    best_entry = edl[0]
    best_ctr_score = 0.0
    best_visual_score = 0.0
    best_rationale = ""
    best_timestamp = best_entry.start_trim + (best_entry.duration / 2.0)

    for entry in edl:
        rat_lower = entry.scene_rationale.lower()
        keyword_hits = sum(1 for kw in THUMBNAIL_POSITIVE_KEYWORDS if kw in rat_lower)

        # Baseline visual interest score
        visual_score = 7.0 + min(2.5, keyword_hits * 0.8)

        # Consider engagement score if available
        if entry.engagement_score is not None:
            visual_score = 0.5 * visual_score + 0.5 * entry.engagement_score

        visual_score = max(0.0, min(10.0, visual_score))

        # CTR score favors high visual interest + emotionally resonant cuts
        ctr_score = visual_score
        if keyword_hits > 0:
            ctr_score = min(10.0, ctr_score + 0.4)

        if ctr_score > best_ctr_score:
            best_ctr_score = ctr_score
            best_visual_score = visual_score
            best_entry = entry
            # Pick a representative frame midway through the cut
            best_timestamp = round(entry.start_trim + (entry.duration / 2.0), 2)
            if keyword_hits > 0:
                best_rationale = (
                    f"Selected '{entry.file_reference}' at {best_timestamp:.2f}s due to strong visual keywords "
                    f"and high contrast/energy suited for thumbnail click-through."
                )
            else:
                best_rationale = (
                    f"Selected '{entry.file_reference}' at {best_timestamp:.2f}s as the most engaging moment "
                    f"in the timeline for viewer intrigue."
                )

    is_viable = best_ctr_score >= MIN_CRITIC_PASSING_SCORE

    return ThumbnailMetrics(
        candidate_clip_reference=best_entry.file_reference,
        candidate_timestamp=best_timestamp,
        visual_interest_score=round(best_visual_score, 1),
        ctr_score=round(best_ctr_score, 1),
        thumbnail_rationale=best_rationale,
        is_viable=is_viable,
    )


def calculate_storytelling_metrics(edl: EditDecisionList, theme: str = "") -> StorytellingMetrics:
    """Evaluate the narrative progression, emotional peaks, and thematic alignment."""
    if not edl:
        raise ValueError("Cannot calculate storytelling metrics for an empty EditDecisionList.")

    total_cuts = len(edl)
    total_duration = edl.total_duration

    # 1. Narrative Arc Score
    if total_cuts >= 4:
        narrative_arc = 8.5
    elif total_cuts >= 2:
        narrative_arc = 7.5
    else:
        narrative_arc = 5.5

    # 2. Emotional Peak Score
    scores = [e.engagement_score for e in edl if e.engagement_score is not None]
    if scores:
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        emotional_peak = round(0.6 * max_score + 0.4 * avg_score, 1)
    else:
        emotional_peak = 7.5

    # 3. Theme Alignment Score
    theme_lower = theme.lower().strip() if theme else ""
    theme_words = set(re.findall(r"\w+", theme_lower)) if theme_lower else set()
    all_rationales = " ".join(e.scene_rationale.lower() for e in edl)

    if theme_words:
        matched = sum(1 for w in theme_words if len(w) > 3 and w in all_rationales)
        if matched >= 3:
            theme_align = 9.0
        elif matched >= 1:
            theme_align = 8.0
        else:
            theme_align = 6.5
    else:
        theme_align = 8.0

    # 4. Human Resonance Score
    human_resonance = round((narrative_arc + emotional_peak + theme_align) / 3.0, 1)
    human_resonance = max(0.0, min(10.0, human_resonance))

    return StorytellingMetrics(
        narrative_arc_score=round(narrative_arc, 1),
        emotional_peak_score=round(emotional_peak, 1),
        human_resonance_score=human_resonance,
        theme_alignment_score=round(theme_align, 1),
    )


def evaluate_critic_heuristically(
    edl: EditDecisionList,
    theme: str = "",
) -> CriticReviewResult:
    """Evaluate an EDL deterministically using human viewer and thumbnail rules."""
    if not edl:
        raise ValueError("Cannot review an empty EditDecisionList as critic.")

    hook_metrics = calculate_30s_hook_metrics(edl, theme)
    thumb_metrics = calculate_thumbnail_metrics(edl, theme)
    story_metrics = calculate_storytelling_metrics(edl, theme)

    # Storytelling composite
    story_score = story_metrics.human_resonance_score

    # Composite human-centric score (35% 30s hook, 35% thumbnail CTR, 30% storytelling)
    composite_score = round(
        0.35 * hook_metrics.hook_avd_score
        + 0.35 * thumb_metrics.ctr_score
        + 0.30 * story_score,
        1,
    )
    composite_score = max(0.0, min(10.0, composite_score))

    # Passing determination
    passed = (
        composite_score >= MIN_CRITIC_PASSING_SCORE
        and hook_metrics.is_engaging
        and thumb_metrics.is_viable
    )

    loop_action = "exit_loop" if passed else "append_to_state"

    # Compile strengths, concerns, and actionable recommendations
    strengths: list[str] = []
    concerns: list[str] = []
    recommendations: list[str] = []

    # Hook critique
    if hook_metrics.is_engaging:
        strengths.append(
            f"Engaging opening 30 seconds ({hook_metrics.duration_analyzed:.1f}s analyzed, {hook_metrics.cut_count_in_30s} cuts) "
            f"establishes the theme without dead air."
        )
    else:
        if hook_metrics.has_dead_air:
            concerns.append(
                "Dead air detected in the first 30 seconds: at least one opening cut exceeds 8.0s without a visual change."
            )
            recommendations.append(
                "Trim intro cuts to <= 5.0s and insert high-energy action or b-roll to maintain initial viewer momentum."
            )
        if hook_metrics.hook_avd_score < MIN_CRITIC_PASSING_SCORE:
            concerns.append(
                f"Opening AVD score ({hook_metrics.hook_avd_score:.1f}/10) is below passing benchmark (7.0)."
            )
            recommendations.append(
                "Restructure opening sequence to front-load thematic payoff within the first 15 seconds."
            )

    # Thumbnail critique
    if thumb_metrics.is_viable:
        strengths.append(
            f"Strong thumbnail candidate in '{thumb_metrics.candidate_clip_reference}' at {thumb_metrics.candidate_timestamp:.2f}s "
            f"(CTR Viability: {thumb_metrics.ctr_score:.1f}/10)."
        )
    else:
        concerns.append(
            f"No standout high-CTR thumbnail candidate detected (highest CTR score: {thumb_metrics.ctr_score:.1f}/10)."
        )
        recommendations.append(
            "Include a visually punchy, high-contrast shot with clear facial expression or dynamic action for thumbnail creation."
        )

    # Storytelling critique
    if story_metrics.human_resonance_score >= 7.5:
        strengths.append(
            f"Narrative arc holds emotional resonance with steady progression and theme fidelity ({story_metrics.theme_alignment_score:.1f}/10)."
        )
    elif story_metrics.narrative_arc_score < 7.0:
        concerns.append(
            f"Narrative arc is abrupt ({len(edl)} cuts). Consider adding contextual setup or resolution."
        )
        recommendations.append(
            "Add a concluding cut or reaction shot to provide clear emotional resolution."
        )

    # Format feedback markdown string
    status_badge = "[PASS]" if passed else "[REVISE]"
    action_text = "Advance to Phase IV Human Review (exit_loop invoked)" if passed else "Restart Editing Loop with Feedback (append_to_state invoked)"

    feedback_lines = [
        f"### Critic Agent (Human-Centric) Report: {status_badge}",
        f"**Theme:** {theme or 'General Highlights'}",
        f"**Overall Human-Centric Score:** {composite_score:.1f}/10.0 (30s Hook: {hook_metrics.hook_avd_score:.1f} | Thumbnail CTR: {thumb_metrics.ctr_score:.1f} | Storytelling: {story_score:.1f})",
        f"**Loop Action:** `{loop_action}` — {action_text}",
        "",
        "#### Evaluation Breakdown:",
        f"- **30-Second Hook & AVD:** {hook_metrics.hook_avd_score:.1f}/10.0 (Cuts in 30s: {hook_metrics.cut_count_in_30s}, Dead Air: {'Yes' if hook_metrics.has_dead_air else 'None'})",
        f"- **Theme Setup Score:** {hook_metrics.theme_setup_score:.1f}/10.0",
        f"- **Thumbnail CTR Viability:** {thumb_metrics.ctr_score:.1f}/10.0 (Candidate: `{thumb_metrics.candidate_clip_reference}` @ {thumb_metrics.candidate_timestamp:.2f}s)",
        f"- **Narrative Arc & Payoff:** {story_metrics.narrative_arc_score:.1f}/10.0 (Emotional Peak: {story_metrics.emotional_peak_score:.1f}/10.0)",
        "",
    ]

    if strengths:
        feedback_lines.append("#### Human-Centric Strengths:")
        for s in strengths:
            feedback_lines.append(f"- {s}")
        feedback_lines.append("")

    if concerns:
        feedback_lines.append("#### Viewer Engagement Risks:")
        for c in concerns:
            feedback_lines.append(f"- {c}")
        feedback_lines.append("")

    feedback_lines.append("#### Actionable Revision Recommendations:")
    if recommendations:
        for r in recommendations:
            feedback_lines.append(f"- {r}")
    else:
        feedback_lines.append("- Edit Decision List satisfies all human viewer, thumbnail, and storytelling criteria.")
    feedback_lines.append("")

    feedback = "\n".join(feedback_lines)

    return CriticReviewResult(
        passed=passed,
        score=composite_score,
        hook_30s_score=hook_metrics.hook_avd_score,
        thumbnail_score=thumb_metrics.ctr_score,
        storytelling_score=story_score,
        feedback=feedback,
        recommendations=recommendations,
        candidate_thumbnail=thumb_metrics,
        hook_30s_metrics=hook_metrics,
        storytelling_metrics=story_metrics,
        loop_action=loop_action,
    )


def evaluate_critic_with_gemini(
    edl: EditDecisionList,
    theme: str = "",
    client: Optional[Any] = None,
    model: str = settings.GEMINI_MODEL,
) -> CriticReviewResult:
    """Evaluate an EDL using Gemini 2.0 Flash as a human viewer and thumbnail strategist."""
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key and client is None:
        logger.info("No Gemini API key available; using deterministic heuristic critic evaluation.")
        return evaluate_critic_heuristically(edl=edl, theme=theme)

    try:
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)

        hook_metrics = calculate_30s_hook_metrics(edl, theme)
        thumb_metrics = calculate_thumbnail_metrics(edl, theme)
        story_metrics = calculate_storytelling_metrics(edl, theme)

        cuts_summary = []
        for idx, entry in enumerate(edl, 1):
            score_txt = f", engagement={entry.engagement_score:.1f}" if entry.engagement_score is not None else ""
            cuts_summary.append(
                f"Cut #{idx}: clip='{entry.file_reference}', trim={entry.start_trim:.2f}s->{entry.end_trim:.2f}s "
                f"(duration={entry.duration:.2f}s, transition={entry.transition_intent.value}{score_txt}), "
                f"rationale='{entry.scene_rationale}'"
            )

        prompt = (
            "You are the Critic Agent for VideoGuru, an automated AI video production system.\n"
            "Evaluate the proposed Edit Decision List (EDL) strictly from a human viewer and thumbnail strategist perspective:\n"
            "1. 30-Second Hook & Average View Duration (AVD): Does the opening 30 seconds capture human curiosity and deliver on the theme without dead air?\n"
            "2. Thumbnail CTR Viability: Identify the single most visually compelling, high-contrast frame/clip suitable for a YouTube thumbnail.\n"
            "3. Emotional Storytelling: Evaluate emotional progression, human resonance, and payoff.\n"
            "4. Decision: If passed (score >= 7.0, engaging 30s hook, viable thumbnail), set loop_action to 'exit_loop'. "
            "Otherwise, set loop_action to 'append_to_state' with specific revision recommendations.\n\n"
            f"Creator Theme: {theme or 'General Highlights'}\n"
            f"Pre-computed 30s Hook: {hook_metrics.duration_analyzed}s analyzed, {hook_metrics.cut_count_in_30s} cuts, dead_air={hook_metrics.has_dead_air}\n"
            f"Pre-computed Thumbnail Candidate: '{thumb_metrics.candidate_clip_reference}' at {thumb_metrics.candidate_timestamp}s (CTR: {thumb_metrics.ctr_score})\n\n"
            "Timeline Cuts:\n"
            + "\n".join(cuts_summary)
            + "\n\nProvide your evaluation strictly conforming to the CriticReviewResult schema."
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CriticReviewResult,
                temperature=0.2,
            ),
        )

        if response.text:
            return CriticReviewResult.from_json(response.text)

        logger.warning("Empty response from Gemini; falling back to heuristic critic review.")
        return evaluate_critic_heuristically(edl=edl, theme=theme)

    except Exception as exc:
        logger.warning("Gemini critic review failed (%s); falling back to heuristic critic review.", exc)
        return evaluate_critic_heuristically(edl=edl, theme=theme)


def review_edl_as_critic(
    tool_context: Optional[Any] = None,
    edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]], str]] = None,
    theme: Optional[str] = None,
    offline: bool = False,
) -> CriticReviewResult:
    """ADK Tool: Evaluate an Edit Decision List from a human viewer & thumbnail perspective.

    Analyzes 30-second hook / AVD potential, thumbnail CTR viability, and emotional resonance.
    If the edit passes, automatically triggers exit_loop to advance to Phase IV Human Review.
    If revisions are required, automatically appends feedback to state via append_to_state.

    Records results in session state:
    - 'critic_passed': bool
    - 'critic_score': float
    - 'critic_feedback': str
    - 'critic_review': dict
    - 'candidate_thumbnail': dict

    Args:
        tool_context: ADK ToolContext providing access to session.state and actions.
        edl: Explicit EditDecisionList instance, list of cut dicts, or file path. If None,
            reads 'edl' from tool_context.state.
        theme: Creator's theme. If None, reads 'theme' from tool_context.state.
        offline: If True, forces deterministic heuristic review without LLM calls.

    Returns:
        CriticReviewResult with scores, candidate thumbnail, feedback, and loop decision.
    """
    # 1. Resolve theme
    if theme is None and tool_context is not None and hasattr(tool_context, "state"):
        state_theme = tool_context.state.get("theme") if hasattr(tool_context.state, "get") else None
        if state_theme:
            theme = state_theme
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
    elif tool_context is not None and hasattr(tool_context, "state"):
        state_edl = tool_context.state.get("edl") if hasattr(tool_context.state, "get") else None
        if state_edl:
            if isinstance(state_edl, EditDecisionList):
                resolved_edl = state_edl
            elif isinstance(state_edl, Sequence):
                resolved_edl = EditDecisionList.from_dict_list(state_edl)

    if not resolved_edl:
        raise ValueError(
            "No Edit Decision List provided or found in session state. "
            "Please run Phase II Curation (Curation Agent) before executing critic review."
        )

    # 3. Determine execution mode
    has_api_key = bool(settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
    run_offline = offline or not has_api_key

    if run_offline:
        result = evaluate_critic_heuristically(edl=resolved_edl, theme=theme_str)
    else:
        result = evaluate_critic_with_gemini(edl=resolved_edl, theme=theme_str)

    # 4. Synchronize results to session state
    if tool_context is not None and hasattr(tool_context, "state") and tool_context.state is not None:
        try:
            tool_context.state["critic_passed"] = result.passed
            tool_context.state["critic_score"] = result.score
            tool_context.state["critic_feedback"] = result.feedback
            tool_context.state["critic_review"] = result.to_dict()
            tool_context.state["candidate_thumbnail"] = result.candidate_thumbnail.model_dump()
        except Exception as exc:
            logger.warning("Failed to synchronize critic results to tool_context.state: %s", exc)

        # 5. Trigger autonomous loop action
        if result.passed:
            exit_loop(tool_context)
        else:
            append_to_state(
                key="revision_feedback",
                value=result.feedback,
                tool_context=tool_context,
            )

    return result


def get_critic_review_from_state(state: Any) -> Optional[CriticReviewResult]:
    """Helper to reconstruct a CriticReviewResult from session state."""
    raw = None
    if hasattr(state, "get"):
        raw = state.get("critic_review")
    elif hasattr(state, "__getitem__"):
        try:
            raw = state["critic_review"]
        except (KeyError, TypeError):
            raw = None

    if not raw:
        return None
    try:
        if isinstance(raw, CriticReviewResult):
            return raw
        if isinstance(raw, dict):
            return CriticReviewResult.from_dict(raw)
        if isinstance(raw, str):
            return CriticReviewResult.from_json(raw)
    except Exception as exc:
        logger.warning("Failed to deserialize CriticReviewResult from state: %s", exc)
    return None
