# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Answer from the config space, without talking to anything.

Selected with $ROCMLIR_LLM_TRANSPORT=stub, which is what lets the tests run
offline and what makes a prompt inspectable by hand.
"""

from __future__ import annotations

import sys
from typing import List, Sequence

from .backend import Backend, Turn


class StubBackend(Backend):
    """A model-shaped answer built out of the space it was given.

    Walks the parameters in order and, for each, proposes moving it to a value
    it does not currently hold. The result is deterministic, sparse, and made
    of real values, so a test exercises the whole merge-and-check path rather
    than a canned string. Which parameter it starts from advances each round,
    so successive rounds do not repeat themselves and the dedupe path is
    exercised too.
    """

    name = "stub"

    def reply(self, turn: Turn) -> str:
        session = turn.session
        round_index = int(session.get("stubRound", 0))
        session["stubRound"] = round_index + 1

        def alternatives(name: str, values: Sequence[int]) -> List[int]:
            """The values of `name` other than the one the default holds."""
            return [value for value in values if value != turn.default_config.get(name)]

        movable = [(name, alternatives(name, values)) for name, values in turn.space.items()]
        movable = [(name, values) for name, values in movable if values]
        if not movable:
            return '{"configs":[]}'

        configs: List[str] = []
        for index in range(turn.configs_requested):
            name, values = movable[(round_index + index) % len(movable)]
            value = values[(round_index + index) // len(movable) % len(values)]
            configs.append(f'{{"{name}":{value}}}')
        print(f"stub transport: answering round {round_index}", file=sys.stderr)
        return '{"configs":[' + ",".join(configs) + "]}"
