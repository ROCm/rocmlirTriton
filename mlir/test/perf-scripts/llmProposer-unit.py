# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Unit tests for the llm/ package, the config-proposing helper behind
``--tuning-space=llm``.

Everything here is the half of the LLM search that C++ deliberately does not
do: turning search state into a prompt, and turning a model's prose back into
configs. Both are only exercised end to end by a tuning run that talks to a
model, which is slow, costs money, and gives a different answer every time --
so the parts that can be pinned deterministically are pinned here.

The reply-parsing tests matter most. A model's reply is the one input to this
system that nothing validates upstream, and every one of these cases is a way
a real model has of being not-quite-right: a fenced code block, Python's
``None``, a sentence of explanation either side of the JSON.

Doesn't need a GPU, a network or an API key: the transport's stub backend
answers from the config space.

unittest's exit code is the verdict, so there is nothing to FileCheck; lit
prints the failing test's traceback on a non-zero exit.

# RUN: %python %s
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# The package is installed beside the performance scripts, at
# ${ROCMLIR_BIN_DIR}/llm, which is where rocmlir-tuning-driver looks for it.
# Import it from there rather than from the source tree, so that a file left
# out of the CMake copy fails here instead of during a tuning run.
_script = shutil.which('tuningRunner.py')
if _script is None:
    sys.exit("tuningRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
_bin_dir = Path(_script).parent
sys.path.insert(0, str(_bin_dir))

from llm import configs, feedback, parsing, prompting, transport, workload  # noqa: E402
from llm import proposer, transcript  # noqa: E402

# A space small enough to read, shaped like a real one: a tile ladder that is
# not a power-of-two sequence, a kpack, and a tri-state knob.
SPACE = {
    "mPerBlock": [1, 2, 4, 8, 16, 32, 48, 64],
    "nPerBlock": [1, 2, 4, 8, 16, 32, 48, 64],
    "kpack": [1, 2, 4, 8],
    "useAsyncCopy": [-1, 0, 1],
}

DEFAULT_CONFIG = {"mPerBlock": 32, "nPerBlock": 32, "kpack": 4, "useAsyncCopy": -1}

# The same config as the attribute serializes it, which is what the search
# sends and what every proposal is completed against.
EXEMPLAR = "gemm:mPerBlock=32,nPerBlock=32,kpack=4,useAsyncCopy=-1"


def result(config, time_ns, status="success"):
    return {"config": dict(config), "timeNs": time_ns, "status": status}


class TestResponseParsing(unittest.TestCase):
    """Everything a model does to JSON on the way out."""

    def parse(self, response):
        return configs.parse_response_configs(response, space=SPACE)

    def test_reads_a_plain_reply(self):
        self.assertEqual(self.parse('{"configs": [{"mPerBlock": 64}]}'), [{"mPerBlock": 64}])

    def test_expands_short_response_names(self):
        self.assertEqual(self.parse('{"configs":[{"m":64,"p":8,"ac":0}]}'), [{
            "mPerBlock": 64,
            "kpack": 8,
            "useAsyncCopy": 0,
        }])

    def test_reads_a_bare_list(self):
        # A model told to answer with configs sometimes answers with configs
        # rather than with an object holding them.
        self.assertEqual(self.parse('[{"kpack": 8}]'), [{"kpack": 8}])

    def test_reads_a_fenced_code_block(self):
        response = ('Here are three configs to try:\n\n'
                    '```json\n{"configs": [{"mPerBlock": 64}]}\n```\n\n'
                    'The first widens the M tile.')
        self.assertEqual(self.parse(response), [{"mPerBlock": 64}])

    def test_reads_json_buried_in_prose(self):
        response = 'I would try {"configs": [{"kpack": 2}]} first.'
        self.assertEqual(self.parse(response), [{"kpack": 2}])

    def test_reads_python_literals(self):
        # A model asked for JSON will still write Python when it has been
        # thinking in Python, which every code model has.
        self.assertEqual(self.parse('{"configs": [{"useAsyncCopy": True}]}'), [{"useAsyncCopy": 1}])
        self.assertEqual(self.parse('{"configs": [{"useAsyncCopy": False}]}'), [{
            "useAsyncCopy": 0
        }])

    def test_narrows_a_whole_float(self):
        self.assertEqual(self.parse('{"configs": [{"kpack": 4.0}]}'), [{"kpack": 4}])

    def test_drops_a_fractional_value(self):
        self.assertEqual(self.parse('{"configs": [{"kpack": 4.5}]}'), [])

    def test_drops_a_value_that_is_not_a_number(self):
        self.assertEqual(self.parse('{"configs": [{"kpack": "large"}]}'), [])

    def test_keeps_the_rest_of_a_config_with_one_bad_field(self):
        # A typo in one field is no reason to throw away what the model got
        # right about the same config.
        self.assertEqual(self.parse('{"configs": [{"kpack": "big", "mPerBlock": 64}]}'), [{
            "mPerBlock": 64
        }])

    def test_drops_a_parameter_this_space_does_not_have(self):
        # The space differs by kernel and by chip, and a model that has seen
        # another one will occasionally name a parameter from it.
        self.assertEqual(self.parse('{"configs": [{"numStages": 2, "kpack": 2}]}'), [{"kpack": 2}])

    def test_drops_a_config_of_nothing_but_unknown_parameters(self):
        self.assertEqual(self.parse('{"configs": [{"numStages": 2}]}'), [])

    def test_drops_an_entry_that_is_not_a_config(self):
        self.assertEqual(self.parse('{"configs": ["mPerBlock=64", {"kpack": 2}]}'), [{"kpack": 2}])

    def test_dedupes_within_one_reply(self):
        self.assertEqual(self.parse('{"configs": [{"kpack": 2}, {"kpack": 2}, {"kpack": 8}]}'), [{
            "kpack": 2
        }, {
            "kpack": 8
        }])

    def test_reports_nothing_for_a_reply_with_no_json(self):
        # The caller turns this into "the model had nothing to offer", which
        # ends the search; it must not be confused with a parse crash.
        self.assertEqual(self.parse('I cannot help with that.'), [])

    def test_reports_nothing_for_truncated_json(self):
        # A reply cut off at the token limit is the most common way a large
        # batch fails, and it must not take the round's process down.
        self.assertEqual(self.parse('{"configs": [{"mPerBlock": 64}, {"nPer'), [])

    def test_keeps_configs_sparse(self):
        # The search merges a proposal onto the default, so a field the model
        # did not name has to stay unnamed: inventing a value here would turn
        # "leave this alone" into "set this to what I guessed".
        self.assertEqual(self.parse('{"configs": [{"mPerBlock": 64}]}'), [{"mPerBlock": 64}])


class TestPerfConfigRendering(unittest.TestCase):
    """Completing a sparse proposal into the one form rocmlirTriton accepts.

    The named perf-config parser fills omitted keys from the schema defaults in
    RockAttrDefs.td, which are a serialization fallback and not a config anyone
    would run. What the model means by an unspecified field is the exemplar it
    was shown, so the completing happens here, where the exemplar is.
    """

    def render(self, config):
        return configs.render_perf_config(EXEMPLAR, config)

    def test_replaces_only_what_was_named(self):
        self.assertEqual(self.render({"mPerBlock": 64}),
                         "gemm:mPerBlock=64,nPerBlock=32,kpack=4,useAsyncCopy=-1")

    def test_spells_out_every_field(self):
        # Never left to be reconstructed on parse, for the reason
        # Rock_PerfConfigSchema gives: defaults can change over time.
        rendered = self.render({"mPerBlock": 64})
        for name in ("mPerBlock", "nPerBlock", "kpack", "useAsyncCopy"):
            self.assertIn(f"{name}=", rendered)

    def test_keeps_the_prefix_the_search_sent(self):
        # `gemm:` or `attn:` is the search's business; nothing here decides it.
        rendered = configs.render_perf_config("attn:mPerBlockG0=32,nPerBlockG1=0",
                                              {"nPerBlockG1": 64})
        self.assertEqual(rendered, "attn:mPerBlockG0=32,nPerBlockG1=64")

    def test_renders_the_exemplar_for_a_proposal_that_moves_nothing(self):
        # The search dedupes it against the seed batch, which already ran it.
        self.assertEqual(self.render({}), EXEMPLAR)

    def test_ignores_a_field_the_exemplar_does_not_have(self):
        self.assertEqual(self.render({"numStages": 2}), EXEMPLAR)

    def test_reports_nothing_for_something_that_is_not_a_perf_config(self):
        # Means this and the search disagree about the request schema, which
        # is worth failing on rather than papering over with a guess.
        for bad in ("", "gemm:", "mPerBlock=32", "gemm:mPerBlock", "gemm:=32"):
            with self.subTest(exemplar=bad):
                self.assertIsNone(configs.render_perf_config(bad, {"mPerBlock": 64}))

    def test_dedupes_configs_that_are_the_same_once_completed(self):
        # Two proposals that differ only in a field they both set to what the
        # exemplar already says are one config, and only one benchmark.
        rendered = configs.render_perf_configs(EXEMPLAR, [
            {
                "mPerBlock": 64
            },
            {
                "mPerBlock": 64,
                "kpack": 4
            },
            {
                "mPerBlock": 16
            },
        ])
        self.assertEqual(len(rendered), 2)

    def test_keeps_the_order_the_model_gave(self):
        rendered = configs.render_perf_configs(EXEMPLAR, [{"kpack": 8}, {"kpack": 2}])
        self.assertIn("kpack=8", rendered[0])
        self.assertIn("kpack=2", rendered[1])


class TestJsonSalvage(unittest.TestCase):
    """parsing.py's own corners, which the config tests only reach through."""

    def test_balanced_block_ignores_braces_inside_strings(self):
        text = '{"note": "a } is not a close", "configs": []}'
        self.assertEqual(parsing.extract_balanced_block(text, "{", "}"), text)

    def test_balanced_block_ignores_an_escaped_quote(self):
        text = '{"note": "he said \\"}\\" once", "configs": []}'
        self.assertEqual(parsing.extract_balanced_block(text, "{", "}"), text)

    def test_balanced_block_reports_nothing_when_unclosed(self):
        self.assertIsNone(parsing.extract_balanced_block('{"configs": [', "{", "}"))

    def test_fix_python_json_leaves_words_inside_identifiers(self):
        # The substitution is on word boundaries, so a parameter whose name
        # contains one of the literals survives it.
        self.assertEqual(parsing.fix_python_json('{"NoneOfIt": None}'), '{"NoneOfIt": null}')


class TestSpaceRendering(unittest.TestCase):
    """What the model is told the space is.

    A ladder is enumerated rather than described by a range, so this is the
    only thing standing between the model and proposing values that do not
    exist.
    """

    def test_marks_the_default(self):
        self.assertEqual(configs.render_ladder([1, 2, 4], 2), "[1, 2*, 4]")

    def test_marks_nothing_when_the_default_is_off_the_ladder(self):
        self.assertEqual(configs.render_ladder([1, 2, 4], 3), "[1, 2, 4]")

    def test_states_a_pinned_parameter_as_pinned(self):
        # An axis of one value is a parameter the space has fixed -- e.g.
        # gridGroupSize on an attention kernel -- and offering it as a choice
        # would spend proposals on a field that cannot move.
        self.assertEqual(configs.render_ladder([0], 0), "fixed at 0")

    def test_reports_an_empty_axis_rather_than_an_empty_list(self):
        self.assertEqual(configs.render_ladder([], 0), "(no values)")

    def test_elides_a_long_ladder(self):
        rendered = configs.render_ladder(list(range(200)), 4)
        self.assertIn("more)", rendered)
        # Both ends survive, since they are what says where the ladder stops.
        self.assertTrue(rendered.startswith("[0, 1,"), rendered)
        self.assertTrue(rendered.endswith("199]"), rendered)
        self.assertLess(len(rendered), 400)

    def test_names_the_knobs_by_their_sentinel(self):
        # Read off the values C++ sent, so a knob added to the perf config is
        # described as one without this file being told about it.
        self.assertEqual(configs.knob_names(SPACE), ["useAsyncCopy"])

    def test_leaves_out_a_knob_pinned_to_its_own_sentinel(self):
        # Both callers offer these to the model as knobs it may set. On one
        # convolution three of the eight offered were pinned at -1 by the same
        # space the prompt calls the authority on legal values.
        self.assertEqual(configs.knob_names({**SPACE, "useAsyncCopy": [-1]}), [])

    def test_space_description_names_every_parameter(self):
        rendered = configs.render_space(SPACE, DEFAULT_CONFIG)
        for name in SPACE:
            self.assertIn(name, rendered)
        self.assertIn("default 4", rendered)


class TestFeedback(unittest.TestCase):
    """What a refinement round is told about the last one."""

    def test_ranks_the_measured_results(self):
        results = [
            result({"kpack": 2}, 900.0),
            result({"kpack": 8}, 300.0),
            result({"kpack": 4}, 600.0),
        ]
        ranked = feedback.measured_results(results)
        self.assertEqual([time for _, time in ranked], [300.0, 600.0, 900.0])

    def test_excludes_configs_that_never_ran(self):
        results = [
            result({"kpack": 2}, 900.0),
            result({"kpack": 8}, 0.0, status="notApplicable"),
            result({"kpack": 4}, 0.0, status="failed"),
        ]
        self.assertEqual(len(feedback.measured_results(results)), 1)
        self.assertEqual(len(feedback.unmeasured_results(results)), 2)

    def test_reports_a_config_by_what_it_changed(self):
        # The full config is seventeen fields, most of them the default; the
        # diff is what a model can actually read a pattern out of.
        text = feedback.format_config_diff(DEFAULT_CONFIG, {**DEFAULT_CONFIG, "kpack": 8})
        self.assertIn('"p":8', text)
        self.assertNotIn("mPerBlock", text)

    def test_names_the_default_rather_than_showing_an_empty_diff(self):
        # The diff of the default against itself is empty, and an empty object
        # in a results table reads as a bug rather than as "this row is the
        # heuristic's own config".
        self.assertEqual(feedback.config_diff(DEFAULT_CONFIG, dict(DEFAULT_CONFIG)),
                         {"(default)": True})

    def test_says_so_when_nothing_has_been_measured(self):
        self.assertIn("No successful configs",
                      feedback.summarize_search_state_for_llm([], DEFAULT_CONFIG))

    def test_reports_the_margin_over_the_runner_up(self):
        results = [result({"kpack": 2}, 1000.0), result({"kpack": 8}, 500.0)]
        text = feedback.summarize_search_state_for_llm(results, DEFAULT_CONFIG)
        self.assertIn("Margin vs runner-up: 50.0%", text)

    def test_says_so_when_nothing_was_rejected(self):
        self.assertIn("every config proposed so far was accepted",
                      feedback.summarize_rejected_configs_for_llm([], DEFAULT_CONFIG))

    def test_reports_a_rejection_with_the_check_that_refused_it(self):
        # The space refuses configs on rules its ladders cannot express, and a
        # model not told which rule it hit will keep proposing into it.
        rejected = [{
            "config": {
                **DEFAULT_CONFIG, "mPerBlock": 64,
                "nPerBlock": 64
            },
            "reason": "exceeds LDS capacity",
        }]
        text = feedback.summarize_rejected_configs_for_llm(rejected, DEFAULT_CONFIG)
        self.assertIn("exceeds LDS capacity", text)
        self.assertIn('"m":64', text)

    def test_caps_how_many_rejections_it_lists(self):
        rejected = [{
            "config": {
                **DEFAULT_CONFIG, "mPerBlock": tile
            },
            "reason": f"check {tile}",
        } for tile in range(1, 40)]
        text = feedback.summarize_rejected_configs_for_llm(rejected, DEFAULT_CONFIG)
        self.assertLessEqual(len(text.splitlines()), feedback.MAX_REJECTIONS_IN_PROMPT + 1)

    def test_groups_repeated_rejections(self):
        rejected = [{"config": {"kpack": 8}, "reason": "same check"}] * 5
        text = feedback.summarize_rejected_configs_for_llm(rejected, DEFAULT_CONFIG)
        self.assertEqual(text.count("same check: "), 1)
        self.assertIn("same check=5", text)

    def test_counts_a_pattern_over_the_configs_that_left_the_field_alone(self):
        # A sparse diff that omits mPerBlock is a config running the default
        # mPerBlock, not a config with no opinion. Counting only the diffs that
        # named it reported "always 64" over two configs while the three
        # fastest, listed directly above, all ran the default 32.
        results = [
            result({**DEFAULT_CONFIG, "kpack": 8}, 100.0),
            result({**DEFAULT_CONFIG, "kpack": 2}, 200.0),
            result({**DEFAULT_CONFIG, "kpack": 1}, 300.0),
            result({**DEFAULT_CONFIG, "mPerBlock": 64}, 400.0),
            result({**DEFAULT_CONFIG, "mPerBlock": 64, "kpack": 2}, 500.0),
        ]
        text = feedback.analyze_top_configs(results, DEFAULT_CONFIG)
        self.assertNotIn("mPerBlock: always", text)
        self.assertIn("mostly 32", text)

    def test_counts_the_random_probes_rather_than_calling_them_a_pattern(self):
        # The seed batch is padded with configs drawn at random across every
        # axis at once. Listed as failures they read as patterns to avoid, when
        # all they have in common is having been generated at random.
        probe = {
            "mPerBlock": 64,
            "nPerBlock": 48,
            "kpack": 8,
            "useAsyncCopy": 1,
            "numWaves": 16,
            "numStages": 5,
            "splitKFactor": 3,
            "wavesPerEU": 9,
        }
        results = [result(probe, 0.0, status="notApplicable")]
        text = feedback.summarize_failed_configs_for_llm(results, DEFAULT_CONFIG)
        self.assertIn("1 of the random configs padding the seed batch", text)
        self.assertIn("no pattern in them for you to avoid", text)

    def test_still_shows_a_failure_the_model_could_have_proposed(self):
        # Only the padding is exempt: a sparse config that failed is a real
        # answer to a real proposal and belongs in front of the model.
        results = [result({**DEFAULT_CONFIG, "kpack": 8}, 0.0, status="failed")]
        text = feedback.summarize_failed_configs_for_llm(results, DEFAULT_CONFIG)
        self.assertIn('failed: {"p":8}', text)
        self.assertNotIn("random configs padding", text)


class TestWorkloadDescription(unittest.TestCase):
    """What the model is told the problem is.

    The same M, N and K mean different things depending on which op they came
    from -- a conv's N is a batch times a spatial extent, an attention kernel's
    K is a head dimension -- so a description that only gave the numbers would
    have the model reasoning about a GEMM whatever it was handed.
    """

    GEMM = {
        "kernelType": "Gemm",
        "gemmSize": {
            "g": 1,
            "m": 1024,
            "n": 1024,
            "k": 1024
        },
        "aType": "f16",
        "bType": "f16",
    }
    CONV = {
        "kernelType": "Conv",
        "gemmSize": {
            "g": 1,
            "m": 256,
            "n": 100352,
            "k": 1152
        },
        "aType": "f16",
        "bType": "f16",
    }
    ATTENTION = {
        "kernelType": "Attention",
        "gemmSize": {
            "g": 32,
            "m": 4096,
            "n": 4096,
            "k": 128,
            "o": 128
        },
        "aType": "f16",
        "bType": "f16",
    }
    HARDWARE = {
        "chip": "gfx942",
        "arch": "amdgcn-amd-amdhsa:gfx942",
        "isCDNA": True,
        "numCUs": 304,
        "numChiplets": 8,
        "accelKind": "MFMA",
        "waveSize": 64,
        "ldsSize": 65536,
        "vgprsPerEU": 512,
        "maxWavesPerEU": 8,
        "lastLevelCacheSize": 256 * 1024 * 1024,
        "maxKpack": 8,
        "maxNumCTAs": 1,
        "supportsAsyncCopy": True,
    }

    def test_names_the_kernel_it_is_tuning(self):
        for problem in (self.GEMM, self.CONV, self.ATTENTION):
            with self.subTest(kernel=problem["kernelType"]):
                self.assertIn(problem["kernelType"], workload.describe_problem(problem))

    def test_explains_where_a_convolution_dimension_came_from(self):
        text = workload.describe_problem(self.CONV).lower()
        self.assertIn("channel", text)
        self.assertIn("filter", text)

    def test_explains_an_attention_kernel_as_two_gemms(self):
        text = workload.describe_problem(self.ATTENTION)
        self.assertIn("softmax", text.lower())
        # O is the second GEMM's N, and nothing else in the config names it.
        self.assertIn("128", text)

    def test_says_nothing_about_a_second_gemm_for_a_plain_one(self):
        self.assertNotIn("softmax", workload.describe_problem(self.GEMM).lower())

    def test_compact_problem_summary_keeps_convolution_facts(self):
        text = workload.summarize_problem_for_prompt({
            **self.CONV,
            "strides": [2, 1],
            "kPerBlockAlignment": 15,
        })
        # How K is built, rather than what to do about it: the hints argue for
        # the aligned tiles, and this section says what makes them aligned.
        for expected in ("Conv:", "strides", "K walks the filter's 15 positions for one "
                         "input channel, then the next channel's 15"):
            self.assertIn(expected, text)

    def test_says_which_dimension_each_window_number_moves(self):
        # A bare [2, 1] leaves the reader counting positions, and padding's two
        # numbers per dimension make that worse.
        text = workload.summarize_problem_for_prompt({
            **self.CONV,
            "inputLayout": ["ni", "gi", "ci", "0i", "1i"],
            "strides": [2, 1],
            "dilations": [1, 1],
            "padding": [1, 1, 2, 2],
        })
        self.assertIn("strides H=2 W=1", text)
        self.assertIn("dilations H=1 W=1", text)
        self.assertIn("padding H=1,1 W=2,2 (padding before, after)", text)

    def test_falls_back_to_the_bare_window_lists_without_a_layout(self):
        # Nothing says how many spatial dimensions there are, so tagging them
        # would be a guess.
        text = workload.summarize_problem_for_prompt({
            **{key: value
               for key, value in self.CONV.items() if not key.endswith("Layout")},
            "strides": [2, 1],
        })
        self.assertIn("strides=[2, 1]", text)

    def test_spells_the_layouts_the_way_convolutions_are_written(self):
        # Rock's own gkc01/nigici0i1i spelling says the same thing as GKCYX and
        # NGCHW, to a reader who already knows the suffix convention.
        text = workload.summarize_problem_for_prompt({
            **self.CONV,
            "filterLayout": ["g", "k", "c", "0", "1"],
            "inputLayout": ["ni", "gi", "ci", "0i", "1i"],
            "outputLayout": ["no", "go", "ko", "0o", "1o"],
        })
        self.assertIn("Layouts: filter=GKCYX input=NGCHW output=NGKHW", text)
        self.assertNotIn("nigici", text)
        # N and K are the GEMM's dimensions two lines up, so the letters say
        # whose dimensions they are.
        self.assertIn("not the GEMM's", text)

    def test_keeps_the_order_the_layout_came_in(self):
        # Which dimension is innermost is the whole content of a layout.
        text = workload.summarize_problem_for_prompt({
            **self.CONV,
            "filterLayout": ["g", "0", "1", "c", "k"],
            "inputLayout": ["ni", "0i", "1i", "gi", "ci"],
        })
        self.assertIn("filter=GYXCK input=NHWGC", text)

    def test_names_the_third_spatial_dimension_of_a_3d_convolution(self):
        text = workload.summarize_problem_for_prompt({
            **self.CONV,
            "filterLayout": ["g", "k", "c", "0", "1", "2"],
            "inputLayout": ["ni", "gi", "ci", "0i", "1i", "2i"],
        })
        self.assertIn("filter=GKCZYX input=NGCDHW", text)
        self.assertIn("ZYX the filter window and DHW the image", text)

    def test_compact_hardware_summary_keeps_budgets(self):
        text = workload.summarize_hardware_for_prompt(self.HARDWARE)
        for expected in ("gfx942", "304 CUs", "LDS=", "VGPR/EU=", "max kpack="):
            self.assertIn(expected, text)

    def test_says_the_grid_group_size_default_is_a_heuristic_and_reachable(self):
        # "gridGroupSize=0 selects 7" read as a fact about 0 rather than as one
        # about 7, and a model spent three configs of a run asking for 7.
        text = workload.summarize_hardware_for_prompt({**self.HARDWARE, "defaultGridGroupSize": 7})
        self.assertIn("not ungrouped", text)
        self.assertIn("a heuristic picks 7, and asking for that value changes nothing", text)

    def hints(self, problem):
        return "\n".join(workload.compute_workload_hints(problem, self.HARDWARE))

    def test_hints_at_the_grid_a_plain_gemm_parallelizes_over(self):
        # M x N is where a plain GEMM's parallelism comes from, and on a
        # multi-chiplet part how those tiles are grouped is worth saying.
        hints = self.hints(self.GEMM)
        self.assertIn("workgroups", hints)
        self.assertIn("gridGroupSize", hints)

    def test_does_not_offer_grid_grouping_on_a_gemm_gemm(self):
        # The axes pin gridGroupSize at 0 there, since makeGxNGridLayout takes
        # no group size; suggesting it would be a proposal that cannot land.
        hints = self.hints(self.ATTENTION)
        self.assertNotIn("gridGroupSize", hints)
        # It parallelizes over G x M instead, and N is a loop.
        self.assertIn("mPerBlockG0", hints)

    def test_reads_a_skinny_gemm_as_one(self):
        skinny = {**self.GEMM, "gemmSize": {"g": 1, "m": 64, "n": 64, "k": 65536}}
        self.assertIn("splitKFactor", self.hints(skinny))

    def test_hints_at_the_k_tiles_a_channels_first_conv_wants(self):
        # K is 1152 = 128 channels x a 3x3 filter, so the tiles that keep the
        # window still are the multiples of 9, and 1152 divides by all of them.
        hints = self.hints({**self.CONV, "kPerBlockAlignment": 9})
        self.assertIn("multiple of 9", hints)
        # Named outright, because the model would otherwise have to notice for
        # itself which values in the list are multiples of 9 that divide K.
        # The smallest of them: what a deep kPerBlock costs is the LDS a wide
        # tile needs, and naming the largest sent a model on a real convolution
        # to tiles that measured thirty times slower than the one it had.
        self.assertIn("9, 18, 36, 72", hints)
        self.assertNotIn("288", hints)

    def test_offers_the_aligned_k_tiles_against_the_ordinary_ones(self):
        # Told they were "the ones to reach for", a model spent up to nine
        # proposals of fifteen on them and handed back the config it started
        # from.
        hints = self.hints({**self.CONV, "kPerBlockAlignment": 9})
        self.assertIn("compete with the ordinary tiles", hints)
        self.assertNotIn("ones to reach for", hints)

    def test_hints_that_kpack_is_the_seeds_blind_spot(self):
        # The GEMM sweeps pinned kpack at 1, so the seeds agreeing on it is an
        # absence of evidence rather than evidence.
        hints = " ".join(workload.compute_workload_hints(self.GEMM, self.HARDWARE))
        self.assertIn("kpack", hints)
        self.assertIn("not a measurement", hints)

    def test_says_nothing_of_kpack_where_the_arch_pins_it(self):
        # gfx950 and gfx1250 cap kpack at 1, so the ladder has one value and
        # there is nothing to propose against.
        hardware = {**self.HARDWARE, "maxKpack": 1}
        hints = " ".join(workload.compute_workload_hints(self.GEMM, hardware))
        self.assertNotIn("not a measurement", hints)

    def test_hints_that_a_deep_k_loop_can_pipeline_deeper_than_the_seeds(self):
        deep = {**self.GEMM, "gemmSize": {"g": 1, "m": 1024, "n": 1024, "k": 4096}}
        hints = " ".join(workload.compute_workload_hints(deep, self.HARDWARE))
        self.assertIn("capped", hints)
        self.assertIn("numStages", hints)

    def test_hints_at_no_deeper_pipeline_where_there_is_no_k_to_pipeline(self):
        shallow = {**self.GEMM, "gemmSize": {"g": 1, "m": 1024, "n": 1024, "k": 32}}
        hints = " ".join(workload.compute_workload_hints(shallow, self.HARDWARE))
        self.assertNotIn("capped", hints)

    def test_hints_that_a_chained_dot_wants_four_stages(self):
        # Both the chained-dot pipeline schedule and the pingpong that rides
        # on it test numStages == 4 exactly, and the sweeps behind the seeds
        # stopped at 3, so the value is untried rather than rejected.
        hints = " ".join(workload.compute_workload_hints(self.ATTENTION, self.HARDWARE))
        self.assertIn("exactly 4", hints)

    def test_hints_at_no_stage_depth_for_a_plain_gemm(self):
        # One dot, so there is no chained-dot schedule to unlock.
        hints = " ".join(workload.compute_workload_hints(self.GEMM, self.HARDWARE))
        self.assertNotIn("exactly 4", hints)

    def test_hints_at_no_k_alignment_for_a_plain_gemm(self):
        self.assertNotIn("multiple of", self.hints(self.GEMM))

    def test_reaches_for_no_reduction_where_the_axes_pin_the_factor(self):
        # A shape too small to fill the machine wants splitKFactor, but where
        # the axes pin it the reading has to change its advice rather than
        # send the model after a value it cannot have.
        small = {**self.GEMM, "gemmSize": {"g": 1, "m": 128, "n": 128, "k": 4096}}
        pinned = " ".join(
            workload.compute_workload_hints(small, self.HARDWARE, {"splitKFactor": [1]}))
        self.assertIn("cannot fill the machine", pinned)
        self.assertIn("pins splitKFactor at 1", pinned)

        offered = " ".join(
            workload.compute_workload_hints(small, self.HARDWARE, {"splitKFactor": [1, 2, 4]}))
        self.assertIn("splitKFactor above 1 is the way", offered)

    def test_reads_kpack_off_the_axis_it_was_given(self):
        # The ladder is what the model may choose from, so it is the ladder
        # that decides whether the blind spot is one this run can look into.
        hints = " ".join(workload.compute_workload_hints(self.GEMM, self.HARDWARE, {"kpack": [1]}))
        self.assertNotIn("not a measurement", hints)

    def test_hints_at_no_stage_depth_the_axes_do_not_carry(self):
        # No 4 on the axis, so no config reaches the chained-dot schedule and
        # the paragraph describing it is describing another run's kernel.
        hints = " ".join(
            workload.compute_workload_hints(self.ATTENTION, self.HARDWARE,
                                            {"numStages": [1, 2, 3]}))
        self.assertNotIn("exactly 4", hints)

    def test_describes_the_chip_by_what_it_can_do(self):
        text = workload.describe_hardware(self.HARDWARE)
        self.assertIn("gfx942", text)
        self.assertIn("304", text)
        # Rendered as a size a reader recognizes rather than as a byte count.
        self.assertIn("64 KiB", text)
        # Why a one-value ladder is one value, which the space cannot say.
        self.assertIn("numCTAs is fixed at 1", text)

    def test_says_which_way_each_knob_default_goes(self):
        # The ladders carry -1, 0 and 1 whatever the arch, so nothing in the
        # space says which of the latter two a -1 already means.
        text = workload.describe_hardware({
            **self.HARDWARE, "defaultAsyncCopy": False,
            "defaultBlockPingpong": True,
            "defaultInThreadTranspose": True
        })
        self.assertIn("useAsyncCopy off", text)
        self.assertIn("useBlockPingpong on", text)

    def test_warns_that_pingpong_follows_async_copy_where_it_does(self):
        # gfx950-shaped: pingpong is on only because async copy is, so
        # useAsyncCopy=0 alone silently gives up the pingpong schedule too.
        text = workload.describe_hardware({
            **self.HARDWARE, "defaultAsyncCopy": True,
            "defaultBlockPingpong": True
        })
        self.assertIn("useAsyncCopy=0", text)
        # gfx942-shaped: pingpong is on for its own sake, so there is no
        # coupling to warn about.
        plain = workload.describe_hardware({
            **self.HARDWARE, "defaultAsyncCopy": False,
            "defaultBlockPingpong": True
        })
        self.assertNotIn("useAsyncCopy=0", plain)

    def test_says_what_a_zero_grid_group_size_works_out_to(self):
        # 0 is "let the layout decide", not "no grouping", and the model needs
        # the number it would be competing with.
        text = workload.describe_hardware({**self.HARDWARE, "defaultGridGroupSize": 6})
        self.assertIn("6", text)
        self.assertIn("not ungrouped", text)
        self.assertNotIn("not ungrouped", workload.describe_hardware(self.HARDWARE))

    def test_flags_the_bf16x3_preference_as_a_heuristic(self):
        # An average over shapes that says which of two kernels wins, not a
        # thing the chip can or cannot do, so it is worth measuring both ways
        # and it does not belong among the capabilities.
        text = workload.describe_hardware({**self.HARDWARE, "preferBf16x3ForF32Dot": True})
        self.assertIn("heuristic", text)
        capabilities = next(line for line in text.splitlines() if "Capabilities:" in line)
        self.assertNotIn("bf16", capabilities)

    def test_says_nothing_of_bf16x3_where_the_arch_does_not_prefer_it(self):
        self.assertNotIn("useBf16x3ForF32", workload.describe_hardware(self.HARDWARE))


class TestProblemDetailDescription(unittest.TestCase):
    """The facts beyond the shapes: layout, inter-GEMM fusion and attention's
    masking.

    Two problems with the same M, N and K want different configs when one
    stores A transposed or masks causally, so a description that stopped at the
    dimensions would be asking the model to guess at the rest. Each of these
    checks that a fact C++ put in the request reaches the text, and that a
    question which does not apply to the kernel is not answered anyway.
    """

    def test_names_the_transposed_operands_and_says_what_it_costs(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.GEMM,
            "transposedA": True,
            "transposedB": False,
            "transposedOut": False,
        })
        self.assertIn("A transposed", text)
        self.assertNotIn("B transposed", text)
        # Why it matters, not merely that it is so.
        self.assertIn("coalesce", text)
        self.assertIn("useAsyncCopy", text)

    def test_says_so_when_nothing_is_transposed(self):
        # The plain layout is worth stating rather than leaving to inference:
        # silence would read as "not known" instead of "row-major throughout".
        text = workload.describe_problem({
            **TestWorkloadDescription.GEMM,
            "transposedA": False,
            "transposedB": False,
        })
        self.assertIn("none transposed", text)
        self.assertNotIn("coalesce", text)

    def test_asks_no_layout_question_of_a_convolution(self):
        # A conv states its layout as a dimension order, so "none transposed"
        # would be a claim about it that nothing in the request supports.
        text = workload.describe_problem(TestWorkloadDescription.CONV)
        self.assertNotIn("Operand layout", text)

    def test_gives_the_output_type_as_well_as_the_inputs(self):
        text = workload.describe_problem({**TestWorkloadDescription.GEMM, "cType": "f32"})
        self.assertIn("C=f32", text)

    def test_reports_a_gemm_gemms_second_operand_and_result_apart(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "cType": "f16",
            "outType": "bf16",
        })
        self.assertIn("C=f16", text)
        self.assertIn("out=bf16", text)

    def test_describes_a_convolutions_layout_and_window(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.CONV,
            "filterLayout": ["g", "k", "c", "0", "1"],
            "inputLayout": ["ni", "gi", "ci", "0i", "1i"],
            "strides": [2, 2],
            "dilations": [1, 1],
            "padding": [1, 1, 1, 1],
        })
        self.assertIn("Filter layout: g, k, c, 0, 1", text)
        self.assertIn("strides=[2, 2]", text)
        # A stride above one is why the gather is strided whatever the tile is.
        self.assertIn("the gather is strided", text)

    def test_says_which_multiple_a_channels_first_convs_k_tile_wants(self):
        # C++ sends the factor rather than the filter extents, so this is the
        # only place the aligned kPerBlock can come from.
        text = workload.describe_problem({
            **TestWorkloadDescription.CONV,
            "inputLayout": ["ni", "gi", "ci", "0i", "1i"],
            "kPerBlockAlignment": 9,
        })
        self.assertIn("multiple of 9", text)

    def test_asks_for_no_k_alignment_where_the_layout_earns_none(self):
        # A channels-last conv puts a spatial dim outermost, so no tile size
        # keeps the window still and C++ leaves the field unset.
        text = workload.describe_problem({
            **TestWorkloadDescription.CONV,
            "inputLayout": ["ni", "gi", "0i", "1i", "ci"],
        })
        self.assertNotIn("lands mid-filter", text)

    def test_reports_an_elementwise_fusion_between_the_gemms(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "hasPreSecondGemmFusion": True,
            "numElemwiseInputs": 2,
        })
        self.assertIn("2 extra tensor", text)
        # Its cost is in registers, which is what bounds the tile.
        self.assertIn("live across the first GEMM's accumulator", text)

    def test_says_nothing_of_a_fusion_that_is_not_there(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "hasPreSecondGemmFusion": False,
            "numElemwiseInputs": 0,
        })
        self.assertNotIn("elementwise operation is fused", text)

    def test_warns_that_a_fused_reduction_already_pays_for_atomics(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.GEMM, "hasFusedReduction": True
        })
        self.assertIn("splitKFactor", text)

    def test_reads_unequal_head_counts_as_grouped_query_attention(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "numHeadsQ": 8,
            "numHeadsKV": 2,
        })
        self.assertIn("Grouped-query attention", text)
        # The sharing factor, which is what a K/V tile is reused by.
        self.assertIn("4 query heads", text)

    def test_does_not_call_equal_head_counts_grouped(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "numHeadsQ": 8,
            "numHeadsKV": 8,
        })
        self.assertIn("8 query", text)
        self.assertNotIn("Grouped-query", text)

    def test_explains_causal_masking_as_uneven_work(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "numHeadsQ": 1,
            "causal": True,
        })
        self.assertIn("diagonal", text)

    def test_explains_flash_decoding_as_a_multiplied_grid(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "numHeadsQ": 1,
            "splitKV": 8,
            "hasLse": True,
        })
        self.assertIn("splitKV=8", text)
        self.assertIn("log-sum-exp", text)

    def test_reports_a_sliding_window_and_a_runtime_kv_extent(self):
        text = workload.describe_problem({
            **TestWorkloadDescription.ATTENTION,
            "numHeadsQ": 1,
            "slidingWindowLookBack": 4096,
            "hasLastValidKVIndex": True,
        })
        self.assertIn("look-back of 4096", text)
        self.assertIn("run-time value", text)

    def test_asks_no_attention_question_of_a_gemm(self):
        text = workload.describe_problem(TestWorkloadDescription.GEMM)
        for absent in ("Attention heads", "diagonal", "splitKV", "softmax"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, text)


class TestSystemPromptGating(unittest.TestCase):
    """What the system prompt leaves out. Every parameter it explains is one
    the model then has to consider, so a run whose axes pin a parameter is a
    run whose prompt should not raise it: the advice cannot be acted on, and
    the words spent on it are words not spent on the parameters that can be."""

    # A ladder per parameter the prompt has something to say about, all of them
    # open. Tests below pin the one they are about, so that what changes in the
    # prompt is attributable to that pin.
    OPEN = {
        "mPerBlock": [32, 64],
        "nPerBlock": [32, 64],
        "kPerBlock": [16, 32],
        "kpack": [1, 2],
        "numCTAs": [1, 2],
        "numWaves": [4, 8],
        "matrixInstrNonkdim": [16, 32],
        "splitKFactor": [1, 2],
        "numStages": [1, 2],
        "wavesPerEU": [0, 2],
        "gridGroupSize": [0, 4],
        **{
            knob: [0, 1] for knob in (
                "useAsyncCopy",
                "useBlockPingpong",
                "useInThreadTranspose",
                "useBufferOps",
                "useBufferAtomics",
                "useReductionLayout",
                "useOptimizeEpilogue",
                "useBf16x3ForF32",
            )
        },
    }

    def prompt(self, **pinned):
        """The prompt for a space like `OPEN` but with `pinned` pinned."""
        return prompting.build_system_prompt({**self.OPEN, **pinned})

    def test_explains_a_parameter_the_axes_leave_open(self):
        self.assertIn("splitKFactor: splits the contraction", self.prompt())

    def test_says_nothing_of_a_parameter_the_axes_pin(self):
        # Attention pins splitKFactor, having split that dimension already by
        # another route, so a prompt for one should not suggest the knob.
        self.assertNotIn("splitKFactor: splits the contraction", self.prompt(splitKFactor=[1]))

    def test_says_nothing_of_a_knob_the_target_leaves_no_room_for(self):
        # `addKnobAxes` pins useAsyncCopy where the target has no
        # direct-to-LDS load width, and both of its values then build the one
        # kernel. The hardware section says why; the knob glossary should not
        # be inviting a proposal regardless.
        self.assertNotIn("useAsyncCopy", self.prompt(useAsyncCopy=[-1]))

    def test_drops_the_knob_section_where_no_knob_is_open(self):
        prompt = self.prompt(**{knob: [-1] for knob in prompting._KNOB_BULLETS})
        self.assertNotIn("tri-state", prompt)
        self.assertNotIn("measures nothing", prompt)

    def test_warns_of_a_knob_a_one_stage_pipeline_defeats(self):
        # Neither schedule knob reaches a loop that pipelines one stage deep,
        # so a proposal moving one at numStages=1 re-measures the default.
        self.assertIn("useAsyncCopy=1 builds the kernel", self.prompt())

    def test_says_nothing_of_that_where_the_axes_rule_one_stage_out(self):
        self.assertNotIn("builds the kernel", self.prompt(numStages=[2, 3]))

    def test_keeps_a_duplicate_rule_only_while_its_knob_is_open(self):
        # The rules are about which explicit values re-measure a kernel the
        # default already built, so each one goes with the knob it is about.
        self.assertIn("16 bits wide", self.prompt())
        self.assertNotIn("16 bits wide", self.prompt(useOptimizeEpilogue=[-1]))

    def test_leaves_the_feasibility_arithmetic_to_the_refusals(self):
        # The system prompt no longer restates the checks. What a config has to
        # satisfy beyond the axes is reported per config under Refused Configs,
        # where it is about a config the model actually proposed.
        prompt = self.prompt()
        for check in ("exceedsTritonTensorCap", "wavesPerEURegisterBudget",
                      "compileCostBudget", "ldsBlacklist", "notOnAxis"):
            self.assertNotIn(check, prompt)

    def test_names_the_block_tiles_the_way_this_kernel_spells_them(self):
        gemm = self.prompt()
        self.assertIn("mPerBlock, nPerBlock:", gemm)
        self.assertNotIn("mPerBlockG0", gemm)

        gemm_gemm = prompting.build_system_prompt({**self.OPEN, "nPerBlockG1": [0, 64]})
        self.assertIn("mPerBlockG0, nPerBlockG0:", gemm_gemm)
        self.assertIn("nPerBlockG1: the second GEMM's", gemm_gemm)

    def test_explains_the_chained_dot_schedule_only_to_a_gemm_gemm(self):
        # numStages of exactly 4 buys the chained-dot schedule, which is a fact
        # about two dots in a loop and says nothing to a single GEMM.
        self.assertNotIn("chained-dot", self.prompt())
        self.assertIn("chained-dot",
                      prompting.build_system_prompt({
                          **self.OPEN, "nPerBlockG1": [0, 64]
                      }))

    def test_is_shorter_than_the_prompt_for_a_space_with_every_axis_open(self):
        # The point of the exercise. gfx1201 running a plain GEMM pins the
        # matrix instruction, kpack, the CTA count, async copy, pingpong and
        # (on a non-f32 dot) bf16x3, which is most of a page.
        wmma = self.prompt(matrixInstrNonkdim=[0],
                           kpack=[1],
                           numCTAs=[1],
                           useAsyncCopy=[-1],
                           useBlockPingpong=[-1],
                           useBf16x3ForF32=[-1])
        self.assertLess(len(wmma), len(self.prompt()))

    def test_explains_everything_where_there_is_no_space_to_read(self):
        # The fallback has to be the whole prompt rather than a quietly
        # abridged one, since a caller without a space is a caller that cannot
        # know what it is missing.
        full = prompting.build_system_prompt()
        for name in prompting._PARAM_BULLETS:
            self.assertIn(f"{name}:", full)


class TestPromptConstruction(unittest.TestCase):
    """The prompt is the whole interface to the model, so what has to be in it
    is worth stating: a round that omits the space, or the results it is
    refining, is a round spent asking for guesses."""

    def request(self, **overrides):
        request = {
            "round": 0,
            "maxRounds": 4,
            "configsRequested": 8,
            "model": "composer-2.5:fast=true",
            "problem": TestWorkloadDescription.GEMM,
            "hardware": TestWorkloadDescription.HARDWARE,
            "space": SPACE,
            "defaultConfig": DEFAULT_CONFIG,
            "defaultPerfConfig": EXEMPLAR,
            "seedConfigs": [{
                "mPerBlock": 64,
                "nPerBlock": 64
            }],
            "results": [],
            "rejected": [],
        }
        request.update(overrides)
        return request

    def test_the_first_round_describes_the_space_and_the_chip(self):
        prompt = proposer.build_prompt(self.request())
        self.assertIn("mPerBlock", prompt)
        self.assertIn("gfx942", prompt)

    def test_the_first_round_gives_compact_response_aliases(self):
        prompt = proposer.build_prompt(self.request())
        self.assertIn("## Response Aliases", prompt)
        for alias in ("m=mPerBlock", "n=nPerBlock", "p=kpack", "ac=useAsyncCopy"):
            self.assertIn(alias, prompt)
        self.assertIn('"m":64', prompt)

    def test_the_first_round_offers_the_quick_list(self):
        # The seeds are the heuristic's own answer, which is both a decent
        # starting point and the bar the model is being asked to clear.
        prompt = proposer.build_prompt(self.request())
        self.assertIn("64", prompt)

    def test_the_seeds_say_which_of_their_fields_were_measured(self):
        # The sweeps the quick list is distilled from pinned the knobs,
        # wavesPerEU and gridGroupSize, so every seed agrees on them without
        # having compared them to anything. Unqualified, a column that never
        # varies reads as a consensus.
        prompt = proposer.build_prompt(self.request())
        self.assertIn("unmeasured rather than confirmed", prompt)
        for field in ("wavesPerEU", "gridGroupSize"):
            self.assertIn(field, prompt)

    def test_the_seeds_are_asked_to_be_mutated_and_not_matched(self):
        # `buildSeedBatch` hands every seed to the benchmark before this prompt
        # is sent and `accept` drops a proposal that repeats one, so asking for
        # "configs matching these" spent the round on configs that could not
        # land: eight of fifteen proposals on one convolution.
        prompt = proposer.build_prompt(self.request())
        self.assertIn("already being benchmarked", prompt)
        self.assertIn("propose mutations of them instead", prompt)
        self.assertNotIn("include configs matching these", prompt)

    def test_the_seeds_are_written_as_diffs_like_every_other_config(self):
        # The checked-in list holds whole configs, so printed verbatim one
        # convolution's thirty-four seeds spent 5845 of the prompt's 10234
        # characters repeating nineteen fields to say five.
        prompt = proposer.build_prompt(self.request(seedConfigs=[{
            "mPerBlock": 64,
            "nPerBlock": 48,
            "kpack": DEFAULT_CONFIG["kpack"],
            "useAsyncCopy": DEFAULT_CONFIG["useAsyncCopy"],
        }]))
        seeds = prompt.split("## Heuristic Seed Configs")[1].split("\n## ")[0]
        self.assertIn('  - {"m":64,"n":48}', seeds)
        self.assertNotIn('"ac"', seeds)

    def test_the_seeds_leave_out_what_this_problems_axes_refuse(self):
        # The quick list is checked in for no particular chip, so it can name a
        # tile this problem's space does not carry: on one convolution four
        # seeds asked for mPerBlock=256 where the axis stopped at 160, in a
        # prompt that also calls the Configuration Space the authority. One run
        # copied the 256 and another extrapolated to 192, and both were refused.
        off_axis = max(SPACE["mPerBlock"]) * 2
        prompt = proposer.build_prompt(self.request(seedConfigs=[
            {"mPerBlock": 48, "nPerBlock": 64},
            {"mPerBlock": off_axis, "nPerBlock": 64},
        ]))
        seeds = prompt.split("## Heuristic Seed Configs")[1].split("\n## ")[0]
        self.assertIn('"m":48', seeds)
        self.assertNotIn(str(off_axis), seeds)

    def test_leaves_out_a_chip_limit_the_space_pins_anyway(self):
        # The chip's kpack ceiling and the kpack axis are computed from
        # different things: one convolution advertised "max kpack=2" four lines
        # above "kpack: fixed at 1".
        pinned = dict(SPACE, kpack=[1])
        prompt = proposer.build_prompt(self.request(space=pinned))
        self.assertIn("kpack: fixed at 1", prompt)
        self.assertNotIn("max kpack", prompt)
        self.assertIn("max kpack", proposer.build_prompt(self.request()))

    def test_the_system_prompt_says_which_knob_values_are_duplicates(self):
        # -1 resolves to on unconditionally for the two buffer knobs, and
        # agrees with 1 for the epilogue unless the store is 16-bit, so an
        # explicit 1 there buys a second timing of a kernel already measured.
        system = prompting.build_system_prompt()
        self.assertIn("measures nothing", system)
        self.assertIn("16 bits wide", system)

    def test_the_system_prompt_does_not_cite_the_seeds_for_the_knobs(self):
        # The -1s in the seeds are an artifact of how they were produced, so
        # the knob advice must not lean on them as evidence.
        system = prompting.build_system_prompt()
        self.assertIn("Read nothing into that", system)

    def test_the_aggressive_share_is_pointed_at_the_knobs(self):
        # "A minority, and only where you can say why" read as a reason not to
        # bother: across 2799 proposals the tri-state knobs moved in 2.9% and
        # wavesPerEU in 1.0%, with the aggressive fifth going on bigger tiles
        # instead. A non-default knob is in 29 of 161 winning configs.
        prompt = proposer.build_prompt(self.request())
        self.assertIn("aggressive fifth is where they belong", prompt)
        self.assertIn("least explored part of the space", prompt)

    def test_a_later_round_asks_for_a_knob_experiment(self):
        # The refinement bullets named seven fields to move and no knob among
        # them, so a knob never got tried once the anchors existed.
        prompt = proposer.build_prompt(
            self.request(round=1, results=[result({"kpack": 8}, 500.0)]))
        self.assertIn("flipped from -1 to 0 or 1", prompt)
        self.assertIn("useAsyncCopy", prompt)

    def test_a_later_round_does_not_recommend_a_field_the_space_pins(self):
        # The list used to be fixed, so on a convolution pinning kpack and
        # matrixInstrNonkdim it spent a third of its advice on moves the space
        # refuses.
        prompt = proposer.build_prompt(
            self.request(round=1, space=dict(SPACE, kpack=[4]),
                         results=[result({"mPerBlock": 64}, 500.0)]))
        advice = prompt.split("attributable effects:")[1].split("\n")[0]
        self.assertNotIn("kpack", advice)
        self.assertIn("the block tiles", advice)

    def test_a_later_round_is_told_what_was_measured(self):
        prompt = proposer.build_prompt(self.request(round=1, results=[result({"kpack": 8}, 500.0)]))
        self.assertIn("kpack", prompt)
        # Timings are rendered in microseconds, since nanoseconds of a kernel
        # are six digits of noise.
        self.assertIn("us", prompt)

    def test_a_later_round_is_told_what_was_refused(self):
        prompt = proposer.build_prompt(
            self.request(round=1,
                         results=[result({"kpack": 8}, 500.0)],
                         rejected=[{
                             "config": {
                                 "mPerBlock": 64
                             },
                             "reason": "exceeds LDS capacity"
                         }]))
        self.assertIn("exceeds LDS capacity", prompt)

    def test_the_round_with_results_is_the_refinement_round(self):
        # Keyed on the results and not on the round number, so that
        # --llm-wait-for-seeds -- which gives round 0 real timings -- gets the
        # prompt that can use them.
        initial = proposer.build_prompt(self.request(round=0))
        refinement = proposer.build_prompt(
            self.request(round=0, results=[result({"kpack": 8}, 500.0)]))
        self.assertNotEqual(initial, refinement)
        self.assertIn("Best so far", refinement)
        self.assertNotIn("Best so far", initial)

    def test_wait_for_seeds_keeps_the_problem_context(self):
        # This is the first message in the conversation even though measured
        # seeds put it on the refinement path. The model cannot recover this
        # context from an earlier message because there is no earlier message.
        prompt = proposer.build_prompt(self.request(round=0, results=[result({"kpack": 8}, 500.0)]))
        for section in ("## Problem", "## GPU Hardware", "## Configuration Space",
                        "## Default Configuration", "## Search State"):
            self.assertIn(section, prompt)
        self.assertIn("G=1 M=1024 K=1024 N=1024", prompt)
        self.assertIn("gfx942", prompt)

    def test_a_resumed_refinement_does_not_repeat_the_problem_context(self):
        prompt = proposer.build_prompt(self.request(round=1, results=[result({"kpack": 8}, 500.0)]))
        self.assertIn("## Search State", prompt)
        for section in ("## Problem", "## GPU Hardware", "## Configuration Space",
                        "## Default Configuration"):
            self.assertNotIn(section, prompt)

    def test_the_system_prompt_explains_the_parameters(self):
        # The model has no knowledge of Rock, so every parameter it is allowed
        # to move has to be explained somewhere.
        system = prompting.build_system_prompt()
        for name in ("mPerBlock", "kpack", "splitKFactor", "gridGroupSize"):
            self.assertIn(name, system)

    def test_the_system_prompt_asks_for_sparse_configs(self):
        self.assertIn("default", prompting.build_system_prompt().lower())


class TestTranscript(unittest.TestCase):
    """The record a person reads: what was asked, what came back, and which
    problem it was all about. Nothing parses this, so what is checked is that
    the things somebody scrolls looking for can be found."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "conversation.log"

    def log(self, **overrides):
        request = TestPromptConstruction().request(**overrides)
        return transcript.Transcript(str(self.path), request)

    def text(self):
        return self.path.read_text()

    def test_names_the_problem_before_anything_else(self):
        self.log().begin()
        banner = self.text()
        # The shape, so that two problems in one file are told apart, and the
        # model, since which one answered is the first thing a bad round is
        # blamed on.
        self.assertIn("Gemm", banner)
        self.assertIn("M=1024", banner)
        self.assertIn("K=1024", banner)
        self.assertIn("A=f16", banner)
        self.assertIn("gfx942", banner)
        self.assertIn("304 CUs", banner)
        self.assertIn("composer-2.5:fast=true", banner)

    def test_names_the_problem_once_and_not_again_each_round(self):
        # Round 0 opens a problem's transcript; a later round appends to it.
        self.log(round=0).begin()
        self.log(round=1).begin()
        self.log(round=2).begin()
        self.assertEqual(self.text().count("# problem"), 1)

    def test_keeps_the_prompt_and_the_reply_apart(self):
        log = self.log()
        log.sent("what would you propose?", system_prompt="you are tuning a kernel")
        log.received("I would propose kpack=8", seconds=1.25)
        written = self.text()
        self.assertIn("you are tuning a kernel", written)
        self.assertIn("what would you propose?", written)
        self.assertIn("I would propose kpack=8", written)
        # Each under a heading naming the round, since a file holds several.
        for heading in ("system prompt", "prompt", "reply"):
            self.assertIn(f"round 0/4  {heading}", written)

    def test_says_how_long_the_model_took(self):
        self.log().received("...", seconds=12.5)
        self.assertIn("12.5s", self.text())

    def test_sizes_a_reply_in_characters(self):
        # A reply is one line of minified JSON, so sizing it the way a prompt
        # is sized logged every reply there has ever been as "1 word",
        # whether it carried two configs or twenty.
        self.log().received('{"configs":[{"m":64},{"m":128}]}', seconds=1.0)
        written = self.text()
        self.assertIn("32 chars", written)
        self.assertNotIn("1 word", written)

    def test_records_the_transport_latency_breakdown(self):
        self.log().timing({
            "sdkImportMs": 1.0,
            "optionsMs": 2.0,
            "agentOpenMs": 3000.0,
            "sendMs": 4.0,
            "firstTextMs": 5000.0,
            "completionMs": 9000.0,
            "totalMs": 12007.0,
            "promptChars": 1234,
            "responseChars": 567,
            "agentId": "agent-1",
            "runId": "run-1",
            "resumed": False,
        })
        written = self.text()
        for expected in ("latency breakdown", "Agent create: 3000.0 ms",
                         "First text after send: 5000.0 ms", "Total transport: 12007.0 ms",
                         "1234 chars", "run-1"):
            self.assertIn(expected, written)

    def test_notes_standing_instructions_it_resent_without_repeating_them(self):
        # The same thousand words every round would drown the part of a later
        # round that is new, which is the whole of what a later round says.
        log = self.log(round=1)
        log.sent("refine what you proposed", system_prompt="you are tuning a kernel")
        written = self.text()
        self.assertIn("system prompt, again", written)
        self.assertNotIn("you are tuning a kernel", written)
        self.assertIn("refine what you proposed", written)

    def test_says_nothing_of_a_system_prompt_a_round_did_not_send(self):
        # A resumed conversation already holds the standing instructions, and
        # repeating them here would claim they went out again.
        self.log(round=1).sent("refine what you proposed")
        self.assertNotIn("system prompt", self.text())

    def test_shows_what_was_read_out_of_a_reply(self):
        # Beside the reply it came from: the step between the two is where a
        # good-looking reply turns into nothing to benchmark.
        self.log().configs(["gemm:mPerBlock=64", "gemm:mPerBlock=128"])
        written = self.text()
        self.assertIn("2 configs", written)
        self.assertIn("gemm:mPerBlock=128", written)

        self.log().configs([])
        self.assertIn("0 configs", self.text())

    def test_shows_a_round_that_failed(self):
        self.log(round=2).failed("$CURSOR_API_KEY is not set")
        written = self.text()
        self.assertIn("round 2/4  failed", written)
        self.assertIn("CURSOR_API_KEY", written)

    def test_writes_nowhere_when_asked_for_nowhere(self):
        request = TestPromptConstruction().request()
        log = transcript.Transcript("", request)
        log.begin()
        log.sent("a prompt")
        log.received("a reply", seconds=1.0)
        self.assertFalse(self.path.exists())

    def test_survives_a_path_it_cannot_write(self):
        # A transcript is a record of a tuning run, not a part of one, so a
        # path that turns out to be unwritable must not end the run -- nor
        # complain once per section for the rest of it.
        self.path.write_text("")  # a file where the path wants a directory
        log = transcript.Transcript(str(self.path / "x.log"), TestPromptConstruction().request())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            log.begin()
            log.sent("a prompt")
            log.received("a reply", seconds=1.0)
        self.assertEqual(stderr.getvalue().count("warning"), 1)


class TestProposerCommandLine(unittest.TestCase):
    """The contract with LLMProposer.cpp: a request file in, a response file
    out, and an exit code that says whether to believe it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.request_path = self.tmp / "request.json"
        self.response_path = self.tmp / "response.json"
        self.session_path = self.tmp / "session.json"
        self.transcript_path = self.tmp / "conversation.log"

    def write_request(self, **overrides):
        request = TestPromptConstruction().request(**overrides)
        self.request_path.write_text(json.dumps(request))
        return request

    def run_proposer(self, *extra, transport="stub"):
        """Run proposer.py as the search runs it, in its own process.

        Not by calling main() in-process: the search runs a program, and the
        import path a script has when run by absolute path with no package
        context is a thing that has broken before.
        """
        env = dict(os.environ, ROCMLIR_LLM_TRANSPORT=transport)
        return subprocess.run(
            [
                sys.executable,
                str(_bin_dir / "llm" / "proposer.py"),
                f"--request={self.request_path}",
                f"--response={self.response_path}",
                f"--session={self.session_path}",
                *extra,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    def response(self):
        return json.loads(self.response_path.read_text())

    def test_answers_a_request_with_whole_perf_configs(self):
        self.write_request()
        run = self.run_proposer()
        self.assertEqual(run.returncode, 0, run.stderr)
        proposed = self.response()["configs"]
        self.assertTrue(proposed)
        # Whole named configs, on the ladders, which is all the search asks.
        prefix, _, exemplar_body = EXEMPLAR.partition(":")
        for perf_config in proposed:
            self.assertIsInstance(perf_config, str)
            body = perf_config.removeprefix(f"{prefix}:")
            self.assertNotEqual(body, perf_config, perf_config)
            fields = dict(piece.split("=") for piece in body.split(","))
            # Every field, not just the moved ones.
            self.assertEqual(sorted(fields), sorted(SPACE))
            for name, value in fields.items():
                self.assertIn(int(value), SPACE[name])
        self.assertTrue(exemplar_body)

    def test_reports_a_request_that_carries_no_exemplar(self):
        # Nothing it sends back could be spelled, so this is a schema
        # disagreement with the search rather than a bad round.
        request = TestPromptConstruction().request()
        del request["defaultPerfConfig"]
        self.request_path.write_text(json.dumps(request))
        run = self.run_proposer()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("defaultPerfConfig", self.response()["error"])

    def test_runs_as_a_script_with_no_package_context(self):
        # Covered by every test here, but worth failing on by name: this is how
        # LLMProposer.cpp starts it, and an import written for the package case
        # alone would only break there.
        self.write_request()
        self.assertEqual(self.run_proposer().returncode, 0)

    def test_keeps_the_conversation_between_rounds(self):
        self.write_request()
        self.run_proposer()
        self.assertTrue(self.session_path.exists())
        session = json.loads(self.session_path.read_text())
        self.assertEqual(len(session["rounds"]), 1)

        self.write_request(round=1, results=[result({"kpack": 8}, 500.0)])
        self.run_proposer()
        session = json.loads(self.session_path.read_text())
        self.assertEqual(len(session["rounds"]), 2)
        self.assertEqual(session["rounds"][1]["round"], 1)

    def test_starts_cleanly_from_the_empty_session_the_search_makes(self):
        # The search creates the file and hands over its path, so round 0 finds
        # an empty one. That is a fresh conversation, not a damaged session,
        # and warning about it every run would train people to ignore warnings.
        self.session_path.write_text("")
        self.write_request()
        run = self.run_proposer()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("warning", run.stderr)
        self.assertTrue(self.response()["configs"])

    def test_survives_a_session_it_cannot_read(self):
        # A session is a memory aid, not a requirement, so losing one must not
        # end a tuning run that is otherwise going fine.
        self.session_path.write_text("{ not json")
        self.write_request()
        run = self.run_proposer()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("warning", run.stderr)
        self.assertTrue(self.response()["configs"])

    def test_reports_a_transport_it_cannot_use(self):
        # In band, as an "error" field: the search prints it and stops, which
        # is more use than an exit code the driver renders as "the helper
        # failed".
        self.write_request()
        run = self.run_proposer(transport="not-a-backend")
        self.assertEqual(run.returncode, 0, run.stderr)
        response = self.response()
        self.assertNotIn("configs", response)
        self.assertIn("not-a-backend", response["error"])

    def test_prints_a_prompt_without_asking_a_model(self):
        # How a prompt gets looked at without spending a tuning run.
        self.write_request()
        run = self.run_proposer("--print-prompt", transport="not-a-backend")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("mPerBlock", run.stdout)

    def test_writes_down_the_whole_conversation_when_asked_to(self):
        # What --llm-transcript is for: the run is over, the tuning result was
        # disappointing, and the question is what the model was told and what
        # it said. Two rounds into one file, since that is how a problem's
        # transcript accumulates.
        self.write_request()
        self.assertEqual(self.run_proposer(f"--transcript={self.transcript_path}").returncode, 0)
        self.write_request(round=1, results=[result({"kpack": 8}, 500.0)])
        self.assertEqual(self.run_proposer(f"--transcript={self.transcript_path}").returncode, 0)

        written = self.transcript_path.read_text()
        # The problem, once, and then both rounds in the order they happened.
        self.assertEqual(written.count("# problem"), 1)
        self.assertIn("Gemm", written)
        self.assertLess(written.index("round 0/4"), written.index("round 1/4"))
        for round_index in (0, 1):
            self.assertIn(f"round {round_index}/4  prompt", written)
            self.assertIn(f"round {round_index}/4  reply", written)
        # The prompt as it was sent, not a summary of it, and the configs the
        # reply was turned into.
        self.assertIn("mPerBlock", written)
        self.assertIn(EXEMPLAR.partition(":")[0], written)

    def test_writes_no_transcript_unless_asked(self):
        self.write_request()
        self.assertEqual(self.run_proposer().returncode, 0)
        self.assertFalse(self.transcript_path.exists())


class TestAgentOptions(unittest.TestCase):
    """How the agent asking for configs is set up, which is two decisions.

    It gets no tools: it is asked for a line of JSON and runs on the machine
    being tuned, unattended, so the toolset that could do anything else is
    removed rather than merely unused -- and the removal is checked, since an
    SDK that quietly dropped it would leave nothing to notice.

    And it gets the model the search names, parameters included, since the
    variant a spec asks for may be a parameter of a model rather than a model
    of its own. A spec that cannot be read stops the run instead of being
    approximated."""

    class Options:
        """Stands in for `AgentOptions`, keeping what it was handed."""

        def __init__(self, **fields):
            self.__dict__.update(fields)

    class Local:

        def __init__(self, **fields):
            self.__dict__.update(fields)

    class Selection:
        """Stands in for `ModelSelection`: a model id and its parameters."""

        def __init__(self, *, id, params):  # noqa: A002
            self.id = id
            self.params = list(params)

    class Parameter:
        """Stands in for `ModelParameterValue`: one parameter of a model."""

        def __init__(self, *, id, value):  # noqa: A002
            self.id = id
            self.value = value

    def sdk(self, options_class=None):
        """The handful of cursor-sdk names the options are built out of."""
        return SimpleNamespace(
            AgentOptions=options_class or self.Options,
            LocalAgentOptions=self.Local,
            ModelSelection=self.Selection,
            ModelParameterValue=self.Parameter,
        )

    def build(self, options_class=None, model="composer-2.5"):
        return transport._text_only_options(self.sdk(options_class),
                                            api_key="key",
                                            model=model,
                                            cwd="/tmp")

    def test_asks_for_an_empty_toolset(self):
        options = self.build()
        self.assertEqual(options.tools, [])

    def test_will_not_start_where_the_sdk_has_no_such_option(self):
        # Too old to restrict a toolset. Running anyway would mean an agent
        # with a shell, which is the thing being avoided.
        class Old:

            def __init__(self, *, api_key, model, local):
                pass

        with self.assertRaises(transport.TransportError) as caught:
            self.build(Old)
        self.assertFalse(caught.exception.started)
        self.assertIn("cursor-sdk", str(caught.exception))

    def test_will_not_start_where_the_sdk_takes_it_and_drops_it(self):
        # The quiet failure: options that accept anything and keep none of it.
        class Ignores(TestAgentOptions.Options):

            def __init__(self, **fields):
                fields.pop("tools", None)
                super().__init__(**fields)

        with self.assertRaises(transport.TransportError) as caught:
            self.build(Ignores)
        self.assertFalse(caught.exception.started)

    def test_names_a_model_that_takes_no_parameters_by_id_alone(self):
        options = self.build(model="composer-2.5")
        self.assertEqual(options.model, "composer-2.5")

    def test_asks_composer_for_its_fast_variant(self):
        # The default spec, and the reason parameters are parsed at all: the
        # low-latency variant is a parameter of the model, not a model of its
        # own, so it cannot be passed as a bare id.
        options = self.build(model="composer-2.5:fast=true")
        self.assertEqual(options.model.id, "composer-2.5")
        self.assertEqual([(p.id, p.value) for p in options.model.params], [("fast", "true")])

    def test_passes_on_every_parameter_it_was_given(self):
        options = self.build(model="some-model:fast=true,thinking=high")
        self.assertEqual([(p.id, p.value) for p in options.model.params], [("fast", "true"),
                                                                           ("thinking", "high")])

    def test_will_not_start_on_a_spec_it_cannot_read(self):
        # Reported rather than guessed at. A misread spec would mean a run
        # spent on a model nobody chose, and the bill is per token.
        for spec in ("composer-2.5:fast", "composer-2.5:=true", "composer-2.5:fast=", ":fast=true"):
            with self.subTest(spec=spec):
                with self.assertRaises(transport.TransportError) as caught:
                    self.build(model=spec)
                self.assertFalse(caught.exception.started)
                self.assertIn(spec, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
