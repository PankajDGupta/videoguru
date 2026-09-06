"""Review Orchestrator Tools for VideoGuru (SPEC-014).

Provides tools for summarizing the Edit Decision List (EDL) and processing user feedback
during Phase IV Human-in-the-Loop Review.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

from schemas.edl import EditDecisionList

logger = logging.getLogger(__name__)


def prepare_review_summary(
    edl: Union[EditDecisionList, Sequence[dict[str, Any]]],
    theme: str,
    otio_path: str,
) -> str:
    """Generate a human-readable markdown summary of the video edit draft.

    Args:
        edl: The Edit Decision List (either EditDecisionList object or list of dicts).
        theme: The creator's intended theme for the video.
        otio_path: Absolute path to the generated OpenTimelineIO (.otio) file.

    Returns:
        Formatted markdown string summarizing the draft edit.
    """
    resolved_edl = edl
    if not isinstance(resolved_edl, EditDecisionList):
        resolved_edl = EditDecisionList.from_dict_list(list(edl))

    cut_count = len(resolved_edl)
    total_duration = resolved_edl.total_duration
    unique_clips = len(resolved_edl.clip_references)

    # Compute transition counts
    transitions = {}
    total_score = 0.0
    score_count = 0

    for entry in resolved_edl:
        trans_name = entry.transition_intent.value
        transitions[trans_name] = transitions.get(trans_name, 0) + 1
        
        if entry.engagement_score is not None:
            total_score += entry.engagement_score
            score_count += 1

    avg_score = total_score / score_count if score_count > 0 else 0.0
    
    trans_list = ", ".join(f"{count} {t}" for t, count in transitions.items())

    summary = (
        "## VideoGuru Edit Draft Ready for Review\n\n"
        f"**Theme:** {theme}\n\n"
        f"### Edit Summary\n"
        f"- **Total Cuts:** {cut_count}\n"
        f"- **Total Duration:** {total_duration:.2f} seconds\n"
        f"- **Unique Clips Used:** {unique_clips}\n"
        f"- **Transitions:** {trans_list or 'None'}\n"
        f"- **Average Engagement Score:** {avg_score:.1f}/10.0\n\n"
        f"### OpenTimelineIO Export\n"
        f"The timeline has been exported and is available at:\n"
        f"`{otio_path}`\n\n"
        "**Please review the edit and reply with:**\n"
        "- 'Approve' to proceed to final rendering.\n"
        "- Specific revision feedback (e.g., 'Make the intro faster', 'Remove the second clip') to iterate on the edit."
    )
    
    return summary


def process_user_review_response(
    user_response: str,
    tool_context: Optional[Any] = None,
) -> str:
    """Process user feedback on the edit draft and update session state.

    Routes 'approve' to advance to rendering, or appends revision feedback
    to state to restart the editing loop.

    Args:
        user_response: The user's text response.
        tool_context: ADK ToolContext providing access to session state.

    Returns:
        Confirmation message acknowledging the feedback.
    """
    response_lower = user_response.lower().strip()
    approval_phrases = ["approve", "approved", "yes", "lgtm", "looks good"]
    
    is_approved = any(phrase in response_lower for phrase in approval_phrases)

    if tool_context is not None and hasattr(tool_context, "state") and tool_context.state is not None:
        state = tool_context.state
        
        if is_approved:
            state["review_approved"] = True
            state["review_status"] = "approved"
            logger.info("User approved the edit draft.")
            return "Edit approved! The draft has been locked and will now proceed to Phase V Rendering."
        else:
            state["review_approved"] = False
            state["review_status"] = "revision_requested"
            
            # Append feedback to revision_feedback list
            current_feedback = list(state.get("revision_feedback", []))
            current_feedback.append(user_response)
            state["revision_feedback"] = current_feedback
            
            logger.info("User requested revisions. Appended feedback to state.")
            return (
                f"Revision requested. Your feedback ('{user_response}') has been recorded. "
                "The Curation Agent will incorporate these changes in the next iteration."
            )
    else:
        logger.warning("process_user_review_response called without tool_context.")
        if is_approved:
            return "Edit approved (no state updated because tool_context is missing)."
        else:
            return "Revision recorded (no state updated because tool_context is missing)."
