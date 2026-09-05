# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reach a model through cursor-sdk.

Where Helion's `helion/autotuner/llm/transport.py` speaks OpenAI and Anthropic
HTTP directly, with its own retry, mTLS and provider-inference machinery, this
hands all of that to the SDK.

The agent is created with an empty toolset, which makes this a text-in,
text-out channel and nothing more: it cannot run a command, read a file or
reach an MCP server. That is all a search needs -- it hands over a description
of a kernel and gets back a line of JSON -- and the alternative is an agent
holding a shell on the machine being tuned, unattended, for the length of the
run. `text_only_options` therefore refuses to start rather than fall back to
the default toolset.

A round is a subprocess, so nothing can be kept in memory between rounds. The
conversation is held instead by the agent itself: round 0 creates an agent and
records its id in the session file, and every later round resumes that agent,
which is what gives the model the memory of what it already proposed that
Helion gets from its rolling message list.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from .backend import Backend, TransportError, Turn, api_key, parse_model


def text_only_options(sdk: Any, *, api_key: str, model: str, cwd: str) -> Any:
    """Build the agent options: the model `model` names, and no tools at all.

    The empty toolset is checked rather than assumed, because of what is lost
    if it silently is not empty: an agent with a shell on the machine running
    the tuning. An SDK too old to know `tools` would reject the argument, and
    one that took it and ignored it would leave no trace at all, so both are
    turned into a refusal to start.
    """
    model_id, params = parse_model(model)
    # A bare id where there are no parameters, which is the form the SDK
    # documents first and the only one an older one is sure to take.
    selection = model_id
    if params:
        selection = sdk.ModelSelection(
            id=model_id,
            params=[sdk.ModelParameterValue(id=name, value=value) for name, value in params],
        )
    try:
        options = sdk.AgentOptions(
            api_key=api_key,
            model=selection,
            local=sdk.LocalAgentOptions(cwd=cwd),
            tools=[],
        )
    except TypeError as err:
        raise TransportError(
            "this cursor-sdk cannot restrict an agent's toolset, and this "
            "search will not run an agent that can execute commands; upgrade "
            f"it with 'pip install -U cursor-sdk' ({err})",
            started=False,
        ) from err
    if getattr(options, "tools", None) != []:
        raise TransportError(
            "cursor-sdk accepted an empty toolset and did not keep it, so the "
            "agent cannot be held to text; upgrade it with "
            "'pip install -U cursor-sdk'",
            started=False,
        )
    return options


class CursorBackend(Backend):
    """A cursor agent, resumed round to round."""

    name = "cursor"

    def conversation_is_open(self, session: Dict[str, Any]) -> bool:
        return bool(session.get("agentId"))

    def reply(self, turn: Turn) -> str:
        session = turn.session
        transport_started = time.monotonic()
        import_started = transport_started
        try:
            import cursor_sdk as sdk
        except ImportError as err:
            raise TransportError(
                "cursor-sdk is not installed; install it with "
                f"'pip install -r pip_requirements.txt' ({err})",
                started=False,
            ) from err
        import_done = time.monotonic()

        key = api_key(hint="a key can be minted at "
                      "https://cursor.com/dashboard/api?section=user-keys#user-api-keys")

        # Text in, text out, and nothing else. `tools=[]` offers the model no
        # built-in tool at all, so a reply is the only thing it can produce --
        # which is the whole of what this asks for, since a proposal is a line
        # of JSON.
        #
        # Worth being deliberate about rather than trusting a prompt to keep
        # the model in its lane. A local agent runs on the machine doing the
        # tuning, with that machine's shell, filesystem and credentials, and
        # the tuning itself is unattended. Removing the tools removes the
        # question.
        #
        # `local.setting_sources` is left unset, which is the SDK's default and
        # means inline configuration only: no project, user, team or plugin
        # settings, and so no rules, hooks or MCP servers picked up from
        # whatever environment the tuning happens to run in.
        #
        # `cwd` is where the search was started from; with no tool to read it,
        # it serves only to scope the bridge's local conversation store --
        # which is why the first round records it and every later one resumes
        # against the same directory rather than against wherever it happens to
        # be run.
        #
        # The runtime is named explicitly because the SDK picks local silently
        # when neither is given, and a silently-local agent is only correct by
        # accident.
        cwd = session.setdefault("cwd", os.getcwd())
        options_started = time.monotonic()
        options = text_only_options(sdk, api_key=key, model=turn.model, cwd=cwd)
        options_done = time.monotonic()
        agent_id = session.get("agentId")

        # The system prompt rides on the first message rather than being
        # configured separately, so that every backend can carry it. A resumed
        # agent has it in its conversation already.
        text = turn.prompt if agent_id else f"{turn.system_prompt}\n\n{turn.prompt}"

        try:
            # The same options both ways round: a toolset restriction is not
            # kept on the agent, so a resumed round that failed to repeat it
            # would be a round with the full toolset back.
            agent_started = time.monotonic()
            if agent_id:
                manager = sdk.Agent.resume(agent_id, options)
            else:
                manager = sdk.Agent.create(options)
            agent_ready = time.monotonic()
            with manager as agent:
                session["agentId"] = getattr(agent, "agent_id", None) or agent_id
                send_started = time.monotonic()
                run = agent.send(text)
                send_done = time.monotonic()
                first_text = None
                chunks = []
                for chunk in run.iter_text():
                    if first_text is None:
                        first_text = time.monotonic()
                    chunks.append(chunk)
                result = run.wait()
                completed = time.monotonic()
                if getattr(result, "status", None) == "error":
                    raise TransportError(
                        f"the model run failed (run {getattr(result, 'id', '?')})",
                        started=True,
                    )
                reply = "".join(chunks) or run.text() or getattr(result, "result", "") or ""
                session["lastTransportTiming"] = {
                    "sdkImportMs": (import_done - import_started) * 1000.0,
                    "optionsMs": (options_done - options_started) * 1000.0,
                    "agentOpenMs": (agent_ready - agent_started) * 1000.0,
                    "sendMs": (send_done - send_started) * 1000.0,
                    "firstTextMs": ((first_text or completed) - send_done) * 1000.0,
                    "completionMs": (completed - send_done) * 1000.0,
                    "totalMs": (completed - transport_started) * 1000.0,
                    "promptChars": len(text),
                    "responseChars": len(reply),
                    "agentId": session.get("agentId") or "",
                    "runId": getattr(run, "id", ""),
                    "resumed": bool(agent_id),
                }
        except sdk.CursorAgentError as err:
            # The model is named in the message because the spec is the most
            # likely thing to be wrong about it, and a parameter this account
            # does not offer reads as a plain start failure otherwise.
            # `Cursor.models.list()` gives the ids and parameters that are
            # actually available.
            raise TransportError(
                f"could not start a model run for '{turn.model}': {err} "
                f"(retryable={getattr(err, 'is_retryable', 'unknown')})",
                started=False,
            ) from err

        if not reply.strip():
            raise TransportError("the model returned an empty reply", started=True)
        return reply
