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

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

from llm import configs, feedback, parsing, prompting, workload  # noqa: E402
from llm import proposer  # noqa: E402

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
        self.assertIn("kpack", text)
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
        self.assertIn("mPerBlock", text)

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

    def hints(self, problem):
        return "\n".join(workload.compute_workload_hints(problem, self.HARDWARE))

    def test_hints_at_the_grid_a_plain_gemm_parallelizes_over(self):
        # M x N is where a plain GEMM's parallelism comes from, and on a
        # multi-chiplet part how those tiles are grouped is worth saying.
        hints = self.hints(self.GEMM)
        self.assertIn("workgroups", hints)
        self.assertIn("gridGroupSize", hints)

    def test_does_not_offer_grid_grouping_on_a_fused_kernel(self):
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
        self.assertIn("288", hints)
        # Not 576 or 1152, which divide K and are multiples of 9 but are past
        # the largest K tile the space carries.
        self.assertNotIn("576", hints)

    def test_hints_at_no_k_alignment_for_a_plain_gemm(self):
        self.assertNotIn("multiple of", self.hints(self.GEMM))

    def test_describes_the_chip_by_what_it_can_do(self):
        text = workload.describe_hardware(self.HARDWARE)
        self.assertIn("gfx942", text)
        self.assertIn("304", text)
        # Rendered as a size a reader recognizes rather than as a byte count.
        self.assertIn("64 KiB", text)
        # Why a one-value ladder is one value, which the space cannot say.
        self.assertIn("numCTAs is fixed at 1", text)

    def test_flags_the_bf16x3_preference_as_a_heuristic(self):
        # An average over shapes that says which of two kernels wins, not a
        # thing the chip can or cannot do, so it is worth measuring both ways
        # and it does not belong among the capabilities.
        text = workload.describe_hardware({
            **self.HARDWARE, "preferBf16x3ForF32Dot": True
        })
        self.assertIn("heuristic", text)
        capabilities = next(
            line for line in text.splitlines() if "Capabilities:" in line)
        self.assertNotIn("bf16", capabilities)

    def test_says_nothing_of_bf16x3_where_the_arch_does_not_prefer_it(self):
        self.assertNotIn("useBf16x3ForF32",
                         workload.describe_hardware(self.HARDWARE))


class TestProblemDetailDescription(unittest.TestCase):
    """The facts beyond the shapes: layout, fusion and attention's masking.

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
        text = workload.describe_problem({
            **TestWorkloadDescription.GEMM, "cType": "f32"
        })
        self.assertIn("C=f32", text)

    def test_reports_a_fused_ops_second_operand_and_result_apart(self):
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


class TestPromptConstruction(unittest.TestCase):
    """The prompt is the whole interface to the model, so what has to be in it
    is worth stating: a round that omits the space, or the results it is
    refining, is a round spent asking for guesses."""

    def request(self, **overrides):
        request = {
            "round": 0,
            "maxRounds": 4,
            "configsRequested": 8,
            "model": "composer-2.5",
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

    def test_the_first_round_offers_the_quick_list(self):
        # The seeds are the heuristic's own answer, which is both a decent
        # starting point and the bar the model is being asked to clear.
        prompt = proposer.build_prompt(self.request())
        self.assertIn("64", prompt)

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

    def test_the_system_prompt_explains_the_parameters(self):
        # The model has no knowledge of Rock, so every parameter it is allowed
        # to move has to be explained somewhere.
        system = prompting.build_system_prompt()
        for name in ("mPerBlock", "kpack", "splitKFactor", "gridGroupSize"):
            self.assertIn(name, system)

    def test_the_system_prompt_asks_for_sparse_configs(self):
        self.assertIn("default", prompting.build_system_prompt().lower())


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


if __name__ == "__main__":
    unittest.main()
