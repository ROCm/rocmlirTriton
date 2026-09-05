#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Turn one round of tuning-search state into a batch of proposed configs.

This is what `LLMSearch` (mlir/lib/Dialect/Rock/Tuning/LLMSearch.cpp) runs,
once per round:

    proposer.py --request=state.json --response=configs.json [--session=s.json]
                [--transcript=conversation.log]

The request describes the problem, the chip, the config space and everything
measured so far; see the schema in LLMProposer.h. The response is
`{"configs": [...]}`, each config a whole named perf config
(`gemm:mPerBlock=128,...`) built by completing what the model proposed against
the request's `defaultPerfConfig`; see `configs.render_perf_config` for why
the completing happens here rather than being left to the perf-config parser.
A failure that can be explained comes back as `{"error": "..."}`, which the
search reports and stops on -- deliberately, following Helion: falling back to
a search that does not consult the model would make a broken API key look like
a model giving bad advice.

`--transcript` is how a run is read afterwards, or watched while it happens:
the prompt and the reply, in full, appended to a file meant for a person. See
llm/transcript.py.

Run it by hand on a saved request to see a prompt without spending a tuning
run:

    ROCMLIR_LLM_TRANSPORT=stub proposer.py --request=r.json --response=/dev/stdout
    proposer.py --request=r.json --response=/dev/null --print-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

# Importable both as part of the package and as the script the search runs,
# which has no package context at all.
if __package__:
    from . import configs as config_utils
    from . import feedback, prompting, transcript, transport
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm import configs as config_utils
    from llm import feedback, prompting, transcript, transport


def load_session(path: str) -> Dict[str, Any]:
    """The conversation state from earlier rounds, or a fresh one."""
    if not path or not os.path.exists(path):
        return {}
    # Round 0 finds the file already there and empty, since the search makes it
    # rather than leaving a path lying around for anything else to take. That
    # is a fresh session and not a damaged one, so it is not worth a warning.
    if os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r") as handle:
            session = json.load(handle)
        return session if isinstance(session, dict) else {}
    except (OSError, json.JSONDecodeError) as err:
        # A session is an optimization, not a requirement: losing it costs the
        # model its memory of earlier rounds and nothing else.
        print(f"warning: ignoring unreadable session {path}: {err}", file=sys.stderr)
        return {}


def save_session(path: str, session: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "w") as handle:
            json.dump(session, handle, indent=2)
    except OSError as err:
        print(f"warning: could not write session {path}: {err}", file=sys.stderr)


def build_prompt(request: Dict[str, Any]) -> str:
    """The prompt for this round: the initial one, or a refinement."""
    default_config = request.get("defaultConfig", {})
    results = request.get("results", [])
    rejected = request.get("rejected", [])

    # Round 0 is the only one with nothing measured. Keyed on the results
    # rather than on the round number so that `--llm-wait-for-seeds`, which
    # gives round 0 real timings, gets the prompt that can use them.
    if not results:
        return prompting.build_initial_prompt(request)

    return prompting.build_refinement_prompt(
        request,
        search_state=feedback.summarize_search_state_for_llm(results, default_config),
        anchor_configs=feedback.summarize_anchor_configs_for_llm(results, default_config),
        results=feedback.format_results_for_llm(results, default_config),
        top_patterns=feedback.analyze_top_configs(results, default_config),
        failed_patterns=feedback.summarize_failed_configs_for_llm(results, default_config),
        rejected_patterns=feedback.summarize_rejected_configs_for_llm(rejected, default_config),
        unmeasured_count=len(feedback.unmeasured_results(results)),
        total_count=len(results),
        rejected_count=len(rejected),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="the search state, as JSON")
    parser.add_argument("--response", required=True, help="where to write the configs")
    parser.add_argument(
        "--session",
        default="",
        help="where the conversation is kept between rounds, so that a later "
        "round resumes the agent an earlier one created; empty means every "
        "round starts a conversation of its own",
    )
    parser.add_argument(
        "--transcript",
        default="",
        help="where to write down, readably, what was asked and what came "
        "back; appended to, one problem per file",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="write the prompt to stdout and make no request",
    )
    args = parser.parse_args(argv)

    with open(args.request, "r") as handle:
        request = json.load(handle)

    prompt = build_prompt(request)
    if args.print_prompt:
        print(prompt)
        return 0

    space = request.get("space", {})
    default_config = request.get("defaultConfig", {})
    # The exemplar, serialized. Every config this writes back is that string
    # with the fields a proposal named replaced, so a request without one is a
    # request this cannot answer.
    exemplar = request.get("defaultPerfConfig", "")
    if not exemplar:
        with open(args.response, "w") as handle:
            json.dump({"error": "the request carries no 'defaultPerfConfig'"}, handle)
        return 0

    session = load_session(args.session)
    # Which rounds have happened, and nothing they said: what the model was
    # asked and what it answered goes to the transcript, which is written for
    # somebody to read. Keeping it here too would carry every round's prompt
    # through every later round's session file for no reader at all.
    session.setdefault("rounds", []).append({"round": request.get("round")})

    log = transcript.Transcript(args.transcript, request)
    log.begin()
    system_prompt = prompting.build_system_prompt(space)

    response: Dict[str, Any]
    started = time.monotonic()
    # Asking which backend this is can fail -- an unknown one stops the run --
    # and that belongs in the same report as any other way of not reaching a
    # model, rather than as a traceback the driver renders as "the helper
    # failed".
    try:
        # The standing instructions go out with the first message of a
        # conversation and not again, so the transcript is told about them on
        # the same terms the transport sends them on: a round with a
        # conversation to continue is a round that already has them.
        resumed = transport.conversation_is_open(session)
        log.sent(prompt, system_prompt=None if resumed else system_prompt)
        reply = transport.call_model(
            prompt=prompt,
            system_prompt=system_prompt,
            # The C++ side always names one; this default only keeps a
            # hand-written request working, and matches LLMSearchOptions.
            model=request.get("model", "GPT-oss-20B:reasoning.effort=low"),
            session=session,
            space=space,
            default_config=default_config,
            configs_requested=request.get("configsRequested", 15),
        )
        log.received(reply, seconds=time.monotonic() - started)
        log.timing(session.get("lastTransportTiming", {}))
        # A reply with nothing usable in it is an empty round, not a failure:
        # the search treats it as the model having no more to offer and stops,
        # which is what Helion's `_run_refinement_round` does. The trace records
        # it, so a prompt that stops landing is still visible.
        #
        # The model answers sparsely, but what goes back is whole named perf
        # configs: completing them here is the only place the exemplar the
        # prompt was written around is at hand. See `render_perf_config`.
        proposed = config_utils.parse_response_configs(reply, space=space)
        response = {"configs": config_utils.render_perf_configs(exemplar, proposed)}
        log.configs(response["configs"])
    except transport.TransportError as err:
        where = "the model run failed" if err.started else "the environment is wrong"
        response = {"error": f"{err} ({where})"}
        session["rounds"][-1]["error"] = str(err)
        log.failed(response["error"])

    save_session(args.session, session)
    with open(args.response, "w") as handle:
        json.dump(response, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
