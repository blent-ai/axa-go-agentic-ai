"""Utilities shared by the agents."""

from __future__ import annotations

from typing import Callable

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

#: Maximum number of model calls for a single agent: a guardrail against loops.
MAX_MODEL_CALLS = 12


def on_tool_error(exception: Exception, request: ToolCallRequest) -> str | None:
    """Send *expected* tool errors back to the model instead of stopping the agent.

    - `ToolException`: business error (unknown claim type, policy section not
      found...). The model receives the message and corrects itself.
    - `NotImplementedError`: on early branches, a tool that is not written yet.
      The agent carries on without it.
    - Any other exception is a bug: it bubbles up.
    """
    name = request.tool_call["name"]
    if isinstance(exception, ToolException):
        return f"Error from tool {name}: {exception}"
    if isinstance(exception, NotImplementedError):
        return f"Tool {name} is not implemented yet ({exception}). Do not call it again."
    return None


def middlewares() -> list:
    """The middlewares common to all ReAct agents of the project."""
    return [
        ToolErrorMiddleware(on_tool_error),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
    ]


def extract_trace(messages: list, tools: list) -> list[dict]:
    """Extract the list of tool calls from an agent's conversation.

    Displaying tool calls and their results is the main debugging tool for an
    agent: you can see whether it picked the right tool, with which arguments,
    and what it got back.

    Parameters
    ----------
    messages : list
        The messages of the agent's conversation.
    tools : list
        The agent's tools. Other calls (the structured-output tool that
        `create_agent` may add) are ignored.

    Returns
    -------
    list[dict]
        ``[{"tool": ..., "arguments": {...}, "result": "...", "error": bool}]``,
        in call order.
    """
    names = {tool.name for tool in tools}
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    trace = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            response = results.get(call["id"])
            if call["name"] not in names or response is None:
                continue
            trace.append(
                {
                    "tool": call["name"],
                    "arguments": call["args"],
                    "result": response.text,
                    "error": response.status == "error",
                }
            )
    return trace


def run_agent(agent, user_message: str, tools: list, on_event: Callable[[dict], None] | None = None) -> dict:
    """Run a ReAct agent while **streaming** its progress.

    Instead of waiting for `agent.invoke` to return, we iterate over the
    successive states of the agent (``stream_mode="values"``) and report every
    new tool call and tool result to ``on_event`` as soon as it happens. The
    last state is the same as what `invoke` would have returned.

    Parameters
    ----------
    agent : CompiledStateGraph
        An agent built with `create_agent`.
    user_message : str
        The user message sent to the agent.
    tools : list
        The agent's tools (calls to other tools, such as the structured-output
        tool, are not reported).
    on_event : Callable[[dict], None], optional
        Receives ``{"type": "tool_call", "tool": ..., "arguments": {...}}`` and
        ``{"type": "tool_result", "tool": ..., "result": "...", "error": bool}``.

    Returns
    -------
    dict
        The final state of the agent (``messages``, ``structured_response``...).
    """
    names = {tool.name for tool in tools}
    call_names: dict[str, str] = {}
    seen, final = 0, {}
    for state in agent.stream({"messages": [{"role": "user", "content": user_message}]}, stream_mode="values"):
        for message in state["messages"][seen:]:
            if on_event is None:
                continue
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    call_names[call["id"]] = call["name"]
                    if call["name"] in names:
                        on_event({"type": "tool_call", "tool": call["name"], "arguments": call["args"]})
            elif isinstance(message, ToolMessage) and call_names.get(message.tool_call_id) in names:
                on_event({
                    "type": "tool_result",
                    "tool": call_names[message.tool_call_id],
                    "result": message.text,
                    "error": message.status == "error",
                })
        seen = len(state["messages"])
        final = state
    return final
