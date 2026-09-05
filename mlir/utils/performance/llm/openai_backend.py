# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reach a model through an OpenAI-compatible endpoint.

$ROCMLIR_LLM_BASE_URL says which one -- an on-prem gateway, a local server,
api.openai.com -- and the key comes from the same variable every backend
reads. This is the backend for a chip being tuned somewhere that cannot, or
would rather not, call out to a hosted service.

Chat completions carry no state, so unlike the cursor backend there is no
conversation for a later round to resume: what the model remembers is whatever
is resent, and a round is a subprocess, so the messages live in the session
file. What is worth resending is narrower than it looks, which is what
`remembered_messages` is about.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Sequence, Tuple

from .backend import (DEFAULT_REQUEST_TIMEOUT_S, Backend, TransportError, Turn, api_key,
                      parse_model)

BASE_URL_VARIABLE = "ROCMLIR_LLM_BASE_URL"

# What an earlier round's prompt is replaced by once its round is over. The
# turn itself stays so that the roles keep alternating, which some servers
# insist on; its content does not. See `remembered_messages`.
SUPERSEDED_PROMPT = "(an earlier round's request, superseded by the one below)"


def request_arguments(params: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """A model spec's parameters as arguments to the request.

    `temperature=0.2` has to arrive as a number and `stream=false` as a
    boolean, so the values a spec carries as text are read back into what they
    look like. Which arguments an endpoint takes is its own business, the same
    as with cursor: one it does not know fails the request, and that is
    reported rather than quietly dropped.
    """
    arguments: Dict[str, Any] = {}
    for name, text in params:
        value: Any = text
        if text.lower() in ("true", "false"):
            value = text.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(text)
                    break
                except ValueError:
                    continue
        arguments[name] = value
    return arguments


def remembered_messages(session: Dict[str, Any], *, system_prompt: str,
                        prompt: str) -> List[Dict[str, str]]:
    """The conversation to send, with the earlier rounds' prompts stubbed out.

    What is worth remembering is what the model itself proposed, so the
    assistant's replies are kept word for word. The prompts that drew them are
    not: a refinement prompt restates the whole search -- state, anchors,
    results, patterns, refusals -- from scratch every round, so an earlier
    round's copy of all that is not merely redundant, it disagrees. Round 1
    saying "Best so far: 193.36 us" and round 3 saying "120.52 us" are two
    claims about one search, and only the last is true.

    Written into `session`, and returned as the same list.
    """
    messages = session.get("messages")
    if not messages:
        messages = [{"role": "system", "content": system_prompt}]
        session["messages"] = messages
    for message in messages:
        if message.get("role") == "user":
            message["content"] = SUPERSEDED_PROMPT
    messages.append({"role": "user", "content": prompt})
    return messages


class OpenAiBackend(Backend):
    """An OpenAI-compatible chat completion, one per round."""

    name = "openai"

    def conversation_is_open(self, session: Dict[str, Any]) -> bool:
        return bool(session.get("messages"))

    def reply(self, turn: Turn) -> str:
        session = turn.session
        transport_started = time.monotonic()
        try:
            import openai
        except ImportError as err:
            raise TransportError(
                "the openai package is not installed; install it with "
                f"'pip install -r pip_requirements.txt' ({err})",
                started=False,
            ) from err
        import_done = time.monotonic()

        base_url = os.environ.get(BASE_URL_VARIABLE) or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise TransportError(
                f"${BASE_URL_VARIABLE} is not set, so there is no endpoint to "
                "ask; it is the OpenAI-compatible base URL of the gateway, "
                "such as https://llm-api.amd.com/onprem/v1",
                started=False,
            )
        key = api_key(hint=f"it is the key for the gateway at {base_url}")

        model_id, params = parse_model(turn.model)
        client = openai.OpenAI(api_key=key, base_url=base_url, timeout=DEFAULT_REQUEST_TIMEOUT_S)
        client_ready = time.monotonic()

        resumed = self.conversation_is_open(session)
        messages = remembered_messages(session,
                                       system_prompt=turn.system_prompt,
                                       prompt=turn.prompt)

        send_started = time.monotonic()
        try:
            completion = client.chat.completions.create(model=model_id,
                                                        messages=messages,
                                                        **request_arguments(params))
        # Told apart by where the fix is. A gateway that cannot be reached,
        # will not take the key, or does not serve this model is an
        # environment to correct; anything else got as far as the model.
        except (openai.APIConnectionError, openai.AuthenticationError, openai.PermissionDeniedError,
                openai.NotFoundError) as err:
            raise TransportError(
                f"could not reach model '{model_id}' at {base_url}: {err}",
                started=False,
            ) from err
        except openai.OpenAIError as err:
            raise TransportError(f"the model run failed: {err}", started=True) from err
        completed = time.monotonic()

        choices = getattr(completion, "choices", None) or []
        reply = (getattr(choices[0].message, "content", "") or "") if choices else ""
        session["messages"].append({"role": "assistant", "content": reply})
        session["lastTransportTiming"] = {
            "sdkImportMs": (import_done - transport_started) * 1000.0,
            "optionsMs": (client_ready - import_done) * 1000.0,
            "agentOpenMs": 0.0,
            "sendMs": 0.0,
            # Nothing is streamed, so as far as this can tell the first token
            # and the last arrive together.
            "firstTextMs": (completed - send_started) * 1000.0,
            "completionMs": (completed - send_started) * 1000.0,
            "totalMs": (completed - transport_started) * 1000.0,
            "promptChars": sum(len(message.get("content") or "") for message in messages),
            "responseChars": len(reply),
            "responseId": getattr(completion, "id", "") or "",
            "resumed": resumed,
        }
        if not reply.strip():
            raise TransportError("the model returned an empty reply", started=True)
        return reply
