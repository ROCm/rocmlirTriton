# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""What every way of reaching a model has in common.

A backend is asked for one thing -- a `Turn` in, the model's words out -- and
is trusted with one thing besides: the session, which is how it remembers a
conversation across rounds. A round is a subprocess, so nothing survives in
memory; whatever a backend needs next time it writes into the session, and the
caller persists it.

The two failures upstream insists on telling apart are the reason `reply` may
raise only `TransportError`, and why that error carries `started`:

- the run never started: a bad key, no network, the package not installed, an
  endpoint that does not serve this model. The environment is wrong.
- the run started and failed. The request is wrong, or the model is having a
  bad day.

Both are fatal to a tuning run, but they are fixed in different places.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

# Helion's DEFAULT_REQUEST_TIMEOUT_S. The C++ side enforces its own deadline by
# killing this process, so this is the inner of two and exists to fail with a
# message rather than a signal.
DEFAULT_REQUEST_TIMEOUT_S = 120

# One name whichever gateway a run talks to, so that pointing a machine at a
# different one is a change of endpoint and not of how it is credentialed.
KEY_VARIABLE = "LLM_GATEWAY_KEY"

# What existing setups export. Read after the standard name, so a machine that
# has both is using the one it was configured with most recently.
_LEGACY_KEY_VARIABLES = ("CURSOR_API_KEY",)


class TransportError(Exception):
    """The model could not be reached, or answered with a failure.

    Raised for both of the cases above; `started` says which, since that is
    what decides where to go looking.
    """

    def __init__(self, message: str, *, started: bool):
        super().__init__(message)
        self.started = started


@dataclass
class Turn:
    """One round's worth of asking, in the terms every backend shares.

    `session` is read and written in place: the caller persists it, so
    anything a backend stores there survives to the next round.

    The last three describe the config space rather than the conversation.
    Only the stub reads them -- it answers out of the space instead of asking
    anyone -- but they travel with the turn so that the question a backend is
    asked is the same question whichever backend it is.
    """

    prompt: str
    system_prompt: str
    model: str
    session: Dict[str, Any]
    space: Dict[str, Sequence[int]]
    default_config: Dict[str, int]
    configs_requested: int


class Backend:
    """One way of reaching a model."""

    #: What $ROCMLIR_LLM_TRANSPORT is set to in order to choose this one.
    name = ""

    def reply(self, turn: Turn) -> str:
        """The model's words, raw. Raises `TransportError` and nothing else."""
        raise NotImplementedError

    def conversation_is_open(self, session: Dict[str, Any]) -> bool:
        """Whether an earlier round left a conversation this one continues.

        Which is what decides whether the standing instructions still need
        sending. A backend that remembers nothing between rounds says no, and
        so is told them every time.
        """
        return False


def api_key(*, hint: str) -> str:
    """The key for whichever gateway this run talks to.

    `hint` says where to get one, which is the only part that differs between
    gateways.
    """
    for name in (KEY_VARIABLE, *_LEGACY_KEY_VARIABLES):
        key = os.environ.get(name)
        if key:
            return key
    raise TransportError(f"${KEY_VARIABLE} is not set; {hint}", started=False)


def parse_model(spec: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split a model spec into a model id and that model's parameters.

    `composer-2.5` is an id by itself. `composer-2.5:fast=true` is the same id
    with its `fast` parameter set, which selects the low-latency variant. That
    is the kind of parameter this search is built to pass, and the default
    spec uses one: a round asks for a batch of configs and reads back a line
    of JSON, and every round after the first is latency the tuning run waits
    through.

    Several parameters go comma-separated. Which parameters a model takes is
    the model's own business -- for cursor `Cursor.models.list()` is the
    authority, and it is account- and team-specific; for an OpenAI-compatible
    endpoint it is whatever that endpoint accepts -- so no name is checked
    here. One the gateway does not offer makes the run fail to start, and that
    is reported rather than retried without it: a search quietly running a
    model nobody asked for is worse than one that stops and says so.
    """
    model_id, _, parameters = spec.partition(":")
    if not model_id:
        raise TransportError(f"the model spec '{spec}' names no model", started=False)
    params: List[Tuple[str, str]] = []
    for item in parameters.split(",") if parameters else []:
        name, separator, value = item.partition("=")
        if not (name and separator and value):
            raise TransportError(
                f"the model spec '{spec}' carries '{item}', which is not a "
                "name=value parameter; write it as 'composer-2.5:fast=true'",
                started=False,
            )
        params.append((name, value))
    return model_id, params
