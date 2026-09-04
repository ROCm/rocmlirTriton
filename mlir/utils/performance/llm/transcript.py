# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Write down what was said to the model and what it said back.

The one part of an LLM tuning run that cannot be reconstructed after it: the
timings are in the output file, the counts are in `--search-trace`, and the
prompts and replies are nowhere unless something writes them down. Which is
awkward for the thing most worth reading, since a run that tuned badly is
usually a prompt that read badly or a reply that meant something other than
what was parsed out of it.

So this is prose, not data. `--search-trace` is JSON lines for
analysis/plotSearchTrace.py to plot; this is a file to open, and to `tail -f`
while a run is going. What that costs is that nothing should try to parse it:
if a number wants reading back, it belongs in the trace.

One transcript covers one problem. Round 0 opens it with a banner naming that
problem, so a file that does hold several -- the tuning driver is told where to
write and will append wherever it is pointed -- still reads as one problem
after another rather than as an undifferentiated pile of prompts.

Every section is appended and the file closed again, both because a round is a
subprocess and because a tuning run gets interrupted: what had been said by
the time a run died is the part somebody wants to read.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional, Sequence

# Wide enough for a heading to stand out against wrapped prose, narrow enough
# to read in a terminal beside something else.
_WIDTH = 78


def _rule(char: str) -> str:
    return char * _WIDTH


def _heading(text: str) -> str:
    """A line that can be found by eye in a page of prompt."""
    prefix = f"=== {text} "
    return prefix + "=" * max(_WIDTH - len(prefix), 3)


def _words(text: str) -> str:
    """How much there is to read, in the unit the prompt is budgeted in."""
    count = len(text.split())
    return f"{count} word" if count == 1 else f"{count} words"


def _problem_summary(problem: Dict[str, Any]) -> str:
    """The shape, in one line: what makes this problem not another one.

    Deliberately terser than `workload.describe_problem`, which says all of
    this in prose a few lines further down every transcript. This is the line
    somebody scrolls to, so it holds what tells two problems apart and stops.
    """
    size = problem.get("gemmSize", {})
    dims = " ".join(
        f"{name.upper()}={size[name]}" for name in ("g", "m", "k", "n", "o") if name in size)
    types = " ".join(f"{label}={problem[key]}" for label, key in (
        ("A", "aType"),
        ("B", "bType"),
        ("C", "cType"),
        ("out", "outType"),
    ) if problem.get(key))
    return "  ".join(part for part in (problem.get("kernelType", "unknown"), dims, types) if part)


def _hardware_summary(hardware: Dict[str, Any]) -> str:
    parts = [str(hardware.get("chip", "unknown chip")), f"{hardware.get('numCUs', '?')} CUs"]
    if hardware.get("numChiplets"):
        parts.append(f"{hardware['numChiplets']} chiplets")
    return ", ".join(parts)


class Transcript:
    """One problem's conversation with a model, as a file to read.

    An empty path disables every method, so a caller never has to ask whether
    a transcript was asked for. Nothing a search does depends on this being
    written, so a path that cannot be written is a warning and then silence
    rather than a failed tuning run.
    """

    def __init__(self, path: str, request: Dict[str, Any]):
        self.path = path
        self.request = request
        self.round = request.get("round", 0)
        self.max_rounds = request.get("maxRounds", 0)

    def _append(self, *blocks: str) -> None:
        if not self.path:
            return
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write("\n".join(blocks) + "\n\n")
        except OSError as err:
            print(f"warning: not writing the transcript {self.path}: {err}", file=sys.stderr)
            # Said once. A warning per section would bury the round's own
            # output in a complaint about where it could not be filed.
            self.path = ""

    def _heading(self, what: str) -> str:
        return _heading(f"round {self.round}/{self.max_rounds}  {what}")

    def begin(self) -> None:
        """Name the problem, on the round that opens its transcript.

        Round 0 rather than an empty file, so that a transcript pointed at a
        path some other problem already used says where the second one started
        instead of running on from the first.
        """
        if self.round != 0:
            return
        problem = self.request.get("problem", {})
        hardware = self.request.get("hardware", {})
        model = self.request.get("model", "an unnamed model")
        rounds = self.request.get("maxRounds", "?")
        configs = self.request.get("configsRequested", "?")
        self._append(
            _rule("#"),
            f"# problem   {_problem_summary(problem)}",
            f"# hardware  {_hardware_summary(hardware)}",
            f"# model     {model}, up to {rounds} rounds of {configs} configs",
            f"# started   {time.strftime('%Y-%m-%d %H:%M:%S')}, pid {os.getpid()}",
            _rule("#"),
        )

    def sent(self, prompt: str, system_prompt: Optional[str] = None) -> None:
        """What this round asked for.

        `system_prompt` is the standing instructions, which go out with the
        first message of a conversation; pass None on a round that resumed an
        agent already holding them.

        A round after the first that sent them anyway -- which is what a
        search with no session file does, since it has no agent to resume --
        gets a line saying so rather than the text again. They are built from
        this problem's space and so are the same text every time, and a
        transcript that repeated a thousand words per round to say nothing new
        would be one nobody reads to the end.
        """
        if system_prompt is not None:
            if self.round == 0:
                self._append(self._heading(f"system prompt ({_words(system_prompt)})"),
                             system_prompt)
            else:
                self._append(
                    self._heading("system prompt, again"),
                    "The same standing instructions as round 0, sent again because this "
                    "round\nstarted a conversation rather than resuming one.")
        self._append(self._heading(f"prompt ({_words(prompt)})"), prompt)

    def received(self, reply: str, seconds: float) -> None:
        """What the model said, before anything was made of it.

        Sized in characters where a prompt is sized in words. A reply is one
        line of minified JSON, so every reply there has ever been has logged
        itself as "1 word" however many configs it carried.
        """
        self._append(self._heading(f"reply after {seconds:.1f}s ({len(reply)} chars)"), reply)

    def timing(self, measurements: Dict[str, Any]) -> None:
        """Break a model round into transport and generation phases."""
        if not measurements:
            return

        def milliseconds(name: str) -> str:
            return f"{float(measurements.get(name, 0.0)):.1f} ms"

        self._append(
            self._heading("latency breakdown"),
            f"SDK import: {milliseconds('sdkImportMs')}",
            f"Options: {milliseconds('optionsMs')}",
            f"Agent {'resume' if measurements.get('resumed') else 'create'}: "
            f"{milliseconds('agentOpenMs')}",
            f"Send: {milliseconds('sendMs')}",
            f"First text after send: {milliseconds('firstTextMs')}",
            f"Completion after send: {milliseconds('completionMs')}",
            f"Total transport: {milliseconds('totalMs')}",
            f"Prompt: {measurements.get('promptChars', 0)} chars; "
            f"response: {measurements.get('responseChars', 0)} chars",
            f"Agent: {measurements.get('agentId', '?')}; run: "
            f"{measurements.get('runId', '?')}",
        )

    def configs(self, configs: Sequence[str]) -> None:
        """What was made of it: the configs the search will benchmark.

        Worth having beside the reply it came from, because the step between
        them is where a reply that looked fine turns into nothing usable -- a
        parameter named in prose the parser does not read, a value the space
        does not offer.
        """
        self._append(self._heading(f"{len(configs)} configs read out of that reply"),
                     *(configs or ["(none)"]))

    def failed(self, message: str) -> None:
        self._append(self._heading("failed"), message)
