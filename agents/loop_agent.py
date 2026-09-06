"""Loop Agent for VideoGuru (SPEC-012).

Responsible for Phase III autonomous editing loop:
- Wraps CurationAgent, ReviewerAgent, and CriticAgent in a sequence.
- Uses ADK LoopAgent to iterate until quality benchmarks are met.
- Max iterations safety circuit-breaker configured via settings.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.agents import LoopAgent

from agents.critic import create_critic_agent
from agents.curation import create_curation_agent
from agents.reviewer import create_reviewer_agent
from config import settings

logger = logging.getLogger(__name__)


def create_loop_agent(
    name: str = "video_editing_loop",
    offline: bool = False,
    max_iterations: Optional[int] = None,
    **kwargs: Any,
) -> LoopAgent:
    """Factory function creating a configured LoopAgent instance.

    Args:
        name: The name of the agent.
        offline: If True, sub-agents run in deterministic offline heuristic mode.
        max_iterations: Maximum loop iterations safety breaker. Defaults to settings.
        **kwargs: Additional kwargs for LoopAgent.

    Returns:
        An initialized LoopAgent wrapping Curation, Reviewer, and Critic.
    """
    curation_agent = create_curation_agent(offline=offline)
    reviewer_agent = create_reviewer_agent(offline=offline)
    critic_agent = create_critic_agent(offline=offline)

    iters = max_iterations if max_iterations is not None else settings.MAX_LOOP_ITERATIONS

    return LoopAgent(
        name=name,
        sub_agents=[curation_agent, reviewer_agent, critic_agent],
        max_iterations=iters,
        **kwargs,
    )


# Default singleton instance
loop_agent = create_loop_agent()
