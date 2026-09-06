"""Unit Tests for register_security_callbacks integration."""

from google.adk.agents import Agent

from callbacks import (
    after_tool_error_recovery_callback,
    before_tool_sandbox_callback,
    input_guardrail_callback,
    on_tool_error_recovery_callback,
    register_security_callbacks,
    schema_validation_callback,
)


def test_register_security_callbacks_attaches_all_hooks():
    agent = Agent(name="test_agent")
    register_security_callbacks(agent)

    assert input_guardrail_callback in agent.before_model_callback
    assert before_tool_sandbox_callback in agent.before_tool_callback
    assert after_tool_error_recovery_callback in agent.after_tool_callback
    assert on_tool_error_recovery_callback in agent.on_tool_error_callback
    assert schema_validation_callback in agent.after_model_callback


def test_register_security_callbacks_selective():
    agent = Agent(name="test_selective_agent")
    register_security_callbacks(
        agent,
        enable_input_guardrail=True,
        enable_tool_sandbox=False,
        enable_error_recovery=False,
        enable_schema_validation=False,
    )

    assert input_guardrail_callback in agent.before_model_callback
    assert agent.before_tool_callback is None
    assert agent.after_tool_callback is None
    assert agent.after_model_callback is None


def test_register_security_callbacks_idempotent():
    agent = Agent(name="test_idempotent_agent")
    register_security_callbacks(agent)
    register_security_callbacks(agent)

    assert agent.before_model_callback.count(input_guardrail_callback) == 1
    assert agent.before_tool_callback.count(before_tool_sandbox_callback) == 1
    assert agent.after_tool_callback.count(after_tool_error_recovery_callback) == 1
    assert agent.on_tool_error_callback.count(on_tool_error_recovery_callback) == 1
    assert agent.after_model_callback.count(schema_validation_callback) == 1
