# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Send a prompt to a model and return what it said.

Which model, and by what route, is $ROCMLIR_LLM_TRANSPORT's business:

- `openai`, the default: any OpenAI-compatible endpoint, named by
  $ROCMLIR_LLM_BASE_URL -- an on-prem gateway, a local server. See
  llm/openai_backend.py.
- `cursor`: a cursor agent, resumed round to round. See
  llm/cursor_backend.py.
- `stub`: answers out of the config space without a network, which is what
  lets the tests run offline. See llm/stub_backend.py.

Everything the backends share -- the two kinds of failure, the key, the way a
model spec is read -- is in llm/backend.py. Nothing above this module knows
which backend it is talking to: a search hands over a description of a kernel
and gets back a line of JSON.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from .backend import DEFAULT_REQUEST_TIMEOUT_S, KEY_VARIABLE, Backend, TransportError, Turn
from .cursor_backend import CursorBackend
from .openai_backend import OpenAiBackend
from .stub_backend import StubBackend

# Re-exported for the callers that only ever need the failure and the deadline,
# so that reaching a model stays one import.
__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_S",
    "KEY_VARIABLE",
    "TransportError",
    "Turn",
    "backend_name",
    "backend_for",
    "call_model",
    "conversation_is_open",
]

_BACKENDS = {backend.name: backend for backend in (CursorBackend(), OpenAiBackend(), StubBackend())}


def backend_name() -> str:
    """Which backend this run uses."""
    return os.environ.get("ROCMLIR_LLM_TRANSPORT", OpenAiBackend.name)


def backend_for(name: str) -> Backend:
    """The backend `name` selects.

    An unknown name stops the run. Falling back to the default would mean a
    search quietly reaching a service the operator did not choose, on a
    machine that may have been set up precisely to avoid it.
    """
    backend = _BACKENDS.get(name)
    if backend is None:
        expected = ", ".join(f"'{known}'" for known in sorted(_BACKENDS))
        raise TransportError(
            f"unknown transport backend '{name}'; expected one of {expected}",
            started=False,
        )
    return backend


def conversation_is_open(session: Dict[str, Any]) -> bool:
    """Whether an earlier round left a conversation this one continues.

    Which is what decides whether the standing instructions still need
    sending. Each backend holds a conversation its own way, so each answers
    for itself.
    """
    return backend_for(backend_name()).conversation_is_open(session)


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
    a backend stores there survives to the next round.
    """
    return backend_for(backend_name()).reply(
        Turn(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            session=session,
            space=space,
            default_config=default_config,
            configs_requested=configs_requested,
        ))
