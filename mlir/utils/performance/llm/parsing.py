# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Extract JSON config payloads from raw LLM responses.

Ported from Helion's `helion/autotuner/llm/parsing.py`. Every function here
exists to handle a specific way a model gets JSON wrong -- Python literals
instead of JSON ones, a fenced code block, a sentence of prose either side of
the object -- and none of it is specific to what is being tuned, so there is
nothing to adapt and every reason to keep it recognizable against upstream.

`close_unfinished_containers` is the one addition, for a way of getting JSON
wrong that upstream does not cover and this tuner met often.
"""

from __future__ import annotations

import json
import re


def fix_python_json(text: str) -> str:
    """Normalize Python literals in LLM output to valid JSON literals."""
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    return re.sub(r"\bFalse\b", "false", text)


def extract_balanced_block(text: str, opener: str, closer: str) -> str | None:
    """Extract the first balanced JSON-like block, respecting quoted strings."""

    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find(opener, start + 1)
    return None


def close_unfinished_containers(text: str) -> str:
    """Finish the objects and arrays an LLM left open, and drop the spare
    closers.

    The failure this exists for is a single character: a reply whose last
    config ends `"bo":0]}` rather than `"bo":0}]}`. The model closed the array
    while an object was still open, and the whole payload -- fifteen configs,
    fourteen of them beyond reproach -- is not JSON. It is not rare enough to
    live with, either: 53 of 938 problems in one tuning run came back with
    nothing at all to benchmark, and 42 of those were this.

    Three repairs, all of them local, and none of which touches text that
    already parses:

      - a closer that arrives while the wrong container is open closes that
        one first, which is the case above;
      - a closer with nothing left open is dropped, since it can only be the
        other half of a mistake already made;
      - whatever is still open at the end is closed, which is what a reply cut
        off mid-array needs.

    A repair is a guess, so this is offered to `parse_jsonish` as one more
    candidate rather than done to the text: a payload that parses as it stands
    is never read from here.
    """
    repaired: list[str] = []
    open_containers: list[str] = []
    closer_of = {"{": "}", "[": "]"}
    opener_of = {"}": "{", "]": "["}
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in closer_of:
            open_containers.append(char)
        elif char in opener_of:
            if opener_of[char] not in open_containers:
                # Nothing this could be closing.
                continue
            while open_containers[-1] != opener_of[char]:
                repaired.append(closer_of[open_containers.pop()])
        repaired.append(char)
        if char in opener_of:
            open_containers.pop()
    if in_string:
        repaired.append('"')
    while open_containers:
        repaired.append(closer_of[open_containers.pop()])
    return "".join(repaired)


def iter_jsonish_candidates(text: str) -> list[str]:
    """Yield likely JSON-ish substrings from raw LLM output."""

    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    for match in re.finditer(r"```(?:json|python)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)
    # Ahead of the balanced blocks, which are the last resort of reading one
    # object out of a payload that cannot be read whole. A repaired payload is
    # the better guess of the two when it parses, since it keeps every config
    # rather than the first.
    if repaired := close_unfinished_containers(text).strip():
        candidates.append(repaired)
    for opener, closer in (("{", "}"), ("[", "]")):
        if candidate := extract_balanced_block(text, opener, closer):
            candidates.append(candidate.strip())
    return list(dict.fromkeys(candidates))


def parse_jsonish(text: str) -> object | None:
    """Parse JSON output with light extraction from wrapped LLM responses."""

    for candidate in iter_jsonish_candidates(fix_python_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
