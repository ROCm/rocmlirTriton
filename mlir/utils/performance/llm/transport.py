# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Send a prompt to a model and return what it said.

Rewritten. Helion's `helion/autotuner/llm/transport.py` speaks OpenAI and
Anthropic HTTP directly, with its own retry, mTLS and provider-inference
machinery; this goes through cursor-sdk instead, which owns all of that. What
survives from upstream is its 120-second default request timeout and, more
importantly, its insistence on telling two failures apart:

- `CursorAgentError` means the run never started: a bad API key, no network,
  the package not installed. The environment is wrong.
- `result.status == "error"` means the run started and failed. The request is
  wrong, or the model is having a bad day.

Both are fatal to a tuning run, but they are fixed in different places, so
they are reported differently.

A round is a subprocess, so nothing can be kept in memory between rounds. The
conversation is held instead by the agent itself: round 0 creates an agent and
records its id in the session file, and every later round resumes that agent,
which is what gives the model the memory of what it already proposed that
Helion gets from its rolling message list.

Setting $ROCMLIR_LLM_TRANSPORT=stub swaps in a backend that answers from the
config space without talking to anything, which is what lets the tests run
offline.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

# Helion's DEFAULT_REQUEST_TIMEOUT_S. The C++ side enforces its own deadline by
# killing this process, so this is the inner of two and exists to fail with a
# message rather than a signal.
DEFAULT_REQUEST_TIMEOUT_S = 120


class TransportError(Exception):
    """The model could not be reached, or answered with a failure.

    Raised for both of the cases above; `started` says which, since that is
    what decides where to go looking.
    """

    def __init__(self, message: str, *, started: bool):
        super().__init__(message)
        self.started = started


def backend_name() -> str:
    """Which backend to use: `cursor`, or `stub` for offline tests."""
    return os.environ.get("ROCMLIR_LLM_TRANSPORT", "cursor")


def call_model(
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    session: Dict[str, Any],
    space: Dict[str, Sequence[int]],
    default_config: Dict[str, int],
    configs_requested: int,
) -> str:
    """Ask `model` for configs and return its raw reply.

    `session` is read and written in place: the caller persists it, so anything
    stored here survives to the next round.
    """
    name = backend_name()
    if name == "stub":
        return _stub_reply(
            space=space,
            default_config=default_config,
            configs_requested=configs_requested,
            session=session,
        )
    if name != "cursor":
        raise TransportError(
            f"unknown transport backend '{name}'; expected 'cursor' or 'stub'",
            started=False,
        )
    return _cursor_reply(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        session=session,
    )


def _cursor_reply(
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    session: Dict[str, Any],
) -> str:
    try:
        from cursor_sdk import (
            Agent,
            AgentOptions,
            CursorAgentError,
            LocalAgentOptions,
        )
    except ImportError as err:
        raise TransportError(
            "cursor-sdk is not installed; install it with "
            f"'pip install -r pip_requirements.txt' ({err})",
            started=False,
        ) from err

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        raise TransportError(
            "$CURSOR_API_KEY is not set; a key can be minted at "
            "https://cursor.com/dashboard/integrations",
            started=False,
        )

    # Local, against a directory of its own: the agent is being asked to think
    # about numbers, not to read or write this repository. Passed explicitly
    # because the SDK picks local silently when neither runtime is named, and a
    # silently-local agent is only correct by accident.
    local = LocalAgentOptions(cwd=session.get("cwd") or os.getcwd())
    agent_id = session.get("agentId")

    # The system prompt rides on the first message rather than being configured
    # separately, so that every backend can carry it. A resumed agent has it in
    # its conversation already.
    text = prompt if agent_id else f"{system_prompt}\n\n{prompt}"

    try:
        if agent_id:
            manager = Agent.resume(agent_id, AgentOptions(api_key=api_key, model=model,
                                                          local=local))
        else:
            manager = Agent.create(model=model, api_key=api_key, local=local)
        with manager as agent:
            session["agentId"] = getattr(agent, "agent_id", None) or agent_id
            run = agent.send(text)
            result = run.wait()
            if getattr(result, "status", None) == "error":
                raise TransportError(
                    f"the model run failed (run {getattr(result, 'id', '?')})",
                    started=True,
                )
            reply = run.text() or getattr(result, "result", "") or ""
    except CursorAgentError as err:
        raise TransportError(
            f"could not start a model run: {err} "
            f"(retryable={getattr(err, 'is_retryable', 'unknown')})",
            started=False,
        ) from err

    if not reply.strip():
        raise TransportError("the model returned an empty reply", started=True)
    return reply


def _stub_reply(
    *,
    space: Dict[str, Sequence[int]],
    default_config: Dict[str, int],
    configs_requested: int,
    session: Dict[str, Any],
) -> str:
    """Answer from the config space, for tests that must not need a network.

    Walks the parameters in order and, for each, proposes moving it to a value
    it does not currently hold. The result is deterministic, sparse, and made
    of real values, so a test exercises the whole merge-and-check path rather
    than a canned string. Which parameter it starts from advances each round,
    so successive rounds do not repeat themselves and the dedupe path is
    exercised too.
    """
    round_index = int(session.get("stubRound", 0))
    session["stubRound"] = round_index + 1

    movable = [(name, [value
                       for value in values
                       if value != default_config.get(name)])
               for name, values in space.items()]
    movable = [(name, values) for name, values in movable if values]
    if not movable:
        return '{"configs":[]}'

    configs: List[str] = []
    for index in range(configs_requested):
        name, values = movable[(round_index + index) % len(movable)]
        value = values[(round_index + index) // len(movable) % len(values)]
        configs.append(f'{{"{name}":{value}}}')
    print(f"stub transport: answering round {round_index}", file=sys.stderr)
    return '{"configs":[' + ",".join(configs) + "]}"
