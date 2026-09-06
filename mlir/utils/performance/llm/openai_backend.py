# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reach a model through an OpenAI-compatible endpoint.

$ROCMLIR_LLM_BASE_URL says which one -- an on-prem gateway, a local server --
and the key comes from the same variable every backend reads. This is the
backend for a chip being tuned somewhere that cannot, or would rather not,
call out to a hosted service.

The gateway holds the conversation: a response carries an id, and the next
round names it as `previous_response_id` rather than resending what was
already said. So a round costs its own prompt and nothing more, the same way
resuming an agent does, and the only thing the session file carries between
rounds is that id.
"""

from __future__ import annotations

import getpass
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

from .backend import (DEFAULT_REQUEST_TIMEOUT_S, Backend, TransportError, Turn, api_key,
                      parse_model)

BASE_URL_VARIABLE = "ROCMLIR_LLM_BASE_URL"

# Where the gateway looks for the key. The API-management layer in front of
# the models authenticates on this header of its own and leaves the
# `Authorization` one the SDK always sends alone, which is why the SDK is
# handed a placeholder for it.
KEY_HEADER = "Ocp-Apim-Subscription-Key"
UNUSED_SDK_KEY = "unused"

# What an earlier round's prompt is replaced by once its round is over. See
# `remembered_conversation`.
SUPERSEDED_PROMPT = "(an earlier round's request, superseded by the one below)"


def request_arguments(params: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """A model spec's parameters as arguments to the request.

    `temperature=0.2` has to arrive as a number and `stream=false` as a
    boolean, so the values a spec carries as text are read back into what they
    look like. A dotted name nests: `reasoning.effort=low` is the request's
    `reasoning={"effort": "low"}`, which is how the Responses API asks a model
    to think less, and there is no flat spelling of it to write instead.

    Which arguments an endpoint takes is its own business, so no name is
    checked here. One it does not know fails the request, and that is reported
    rather than quietly dropped.
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
        *outer, last = name.split(".")
        target = arguments
        for step in outer:
            target = target.setdefault(step, {})
        target[last] = value
    return arguments


def remembered_conversation(session: Dict[str, Any], *, system_prompt: str,
                            prompt: str) -> List[Dict[str, str]]:
    """The whole conversation, kept here in case the gateway is not keeping it.

    Naming a `previous_response_id` is the cheaper way to continue, but a
    deployment that does not store responses answers that with a 404, and a
    round that cannot be asked stops the tune. So the conversation is kept
    here too, and is what a round falls back to.

    What is worth remembering is what the model itself proposed, so the
    assistant's replies are kept word for word. The prompts that drew them are
    not: a refinement prompt restates the whole search -- state, anchors,
    results, patterns, refusals -- from scratch every round, so an earlier
    round's copy of all that is not merely redundant, it disagrees. Round 1
    saying "Best so far: 193.36 us" and round 3 saying "120.52 us" are two
    claims about one search, and only the last is true. The turn stays, so
    that the roles keep alternating; its content does not.

    The standing instructions ride in the conversation rather than in the
    request's `instructions`, which `previous_response_id` does not carry
    over. Written into `session`, and returned as the same list.
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


def reasoning_said(response: Any) -> str:
    """What the model worked through before answering, if it says.

    Most of a round's tokens go here -- a measured round spent 2531 of 2653
    output tokens reasoning and 122 answering -- and none of it is in the
    reply, so a transcript without it cannot show why a proposal was made or
    why a round took as long as it did.

    What comes back is up to the endpoint: a summary when the request asked
    for one with `reasoning.summary=auto`, nothing when the deployment keeps
    its reasoning to itself.
    """
    said = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", "") != "reasoning":
            continue
        for part in (getattr(item, "summary", None) or []) + (getattr(item, "content", None) or []):
            text = getattr(part, "text", "") or ""
            if text.strip():
                said.append(text)
    return "\n\n".join(said)


def token_counts(response: Any) -> Dict[str, int]:
    """What the round cost, in the units the model is slow in.

    A round's seconds are a prompt read and a reply written, and the two are
    fixed in different places: a long prompt is this search's to trim, while
    reasoning the model spent before answering is the effort parameter's to
    turn down. Characters cannot tell them apart, and reasoning does not show
    up in the reply at all.

    Empty when the endpoint reports no usage, since a count of zero would read
    as an answer rather than as a silence.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "output_tokens_details", None)
    counts = {
        "inputTokens": getattr(usage, "input_tokens", 0) or 0,
        "outputTokens": getattr(usage, "output_tokens", 0) or 0,
    }
    if details is not None:
        counts["reasoningTokens"] = getattr(details, "reasoning_tokens", 0) or 0
    return counts


class OpenAiBackend(Backend):
    """A response from an OpenAI-compatible endpoint, one per round."""

    name = "openai"

    def conversation_is_open(self, session: Dict[str, Any]) -> bool:
        # Either half of the memory counts: the gateway's, named by a response
        # id, and this backend's own copy, which is what a round falls back to
        # when the gateway turns out not to have kept its half.
        return bool(session.get("messages") or session.get("responseId"))

    def reply(self, turn: Turn) -> str:
        session = turn.session
        transport_started = time.monotonic()
        try:
            import httpx
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
                "ask; it is the base URL of the gateway, such as "
                "https://llm-api.amd.com/OnPrem",
                started=False,
            )
        key = api_key(hint=f"it is the key for the gateway at {base_url}")

        model_id, params = parse_model(turn.model)
        client = openai.OpenAI(
            base_url=base_url,
            api_key=UNUSED_SDK_KEY,
            default_headers={
                KEY_HEADER: key,
                # Who is spending the gateway's budget, which it logs per user
                # rather than per key.
                "user": getpass.getuser(),
            },
            # An intranet gateway serves a certificate signed by a CA the
            # machine tuning a chip has no reason to have been given.
            http_client=httpx.Client(verify=False),
            timeout=DEFAULT_REQUEST_TIMEOUT_S,
        )
        client_ready = time.monotonic()

        conversation = remembered_conversation(session,
                                               system_prompt=turn.system_prompt,
                                               prompt=turn.prompt)
        # An endpoint that answered one `previous_response_id` with a 404 will
        # answer the rest of them the same way, so it is asked once and then
        # taken at its word for the rest of the tune.
        previous = session.get("responseId", "") if session.get("gatewayRemembers", True) else ""
        arguments = request_arguments(params)

        send_started = time.monotonic()
        try:
            sent = conversation
            if previous:
                try:
                    # The cheap way: this round's prompt alone, against a
                    # conversation the gateway is holding for us.
                    sent = [conversation[-1]]
                    response = client.responses.create(model=model_id,
                                                       input=sent,
                                                       previous_response_id=previous,
                                                       **arguments)
                except openai.NotFoundError:
                    # It was not holding it. A deployment that does not store
                    # responses only says so on the round that tries to
                    # continue one, and a round that cannot be asked stops the
                    # whole tune, so say the conversation again instead.
                    session["gatewayRemembers"] = False
                    previous = ""
                    sent = conversation
                    response = client.responses.create(model=model_id, input=sent, **arguments)
            else:
                response = client.responses.create(model=model_id, input=sent, **arguments)
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

        # Whatever the model said in as many words, with any reasoning it did
        # on the way left where it was: the search wants the configs.
        reply = getattr(response, "output_text", "") or ""
        session["responseId"] = getattr(response, "id", "") or ""
        conversation.append({"role": "assistant", "content": reply})
        # For the transcript to print and then forget, rather than to carry
        # one round's thinking through every later round's session file.
        session["lastReasoning"] = reasoning_said(response)
        session["lastTransportTiming"] = {
            **token_counts(response),
            "sdkImportMs": (import_done - transport_started) * 1000.0,
            "optionsMs": (client_ready - import_done) * 1000.0,
            "agentOpenMs": 0.0,
            "sendMs": 0.0,
            # Nothing is streamed, so as far as this can tell the first token
            # and the last arrive together.
            "firstTextMs": (completed - send_started) * 1000.0,
            "completionMs": (completed - send_started) * 1000.0,
            "totalMs": (completed - transport_started) * 1000.0,
            "promptChars": sum(len(message["content"]) for message in sent),
            "responseChars": len(reply),
            "responseId": session["responseId"],
            "resumed": bool(previous),
        }
        if not reply.strip():
            raise TransportError("the model returned an empty reply", started=True)
        return reply
