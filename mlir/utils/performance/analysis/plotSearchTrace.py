#!/usr/bin/env python3
"""Plot how a searched tuning run got where it got.

The searched tuning spaces (--tuning-space=lfbo, llm, llm-lfbo) record one JSON
object per iteration when given --search-trace, which tuningRunner.py does for
every problem under `--debug`. This turns such a trace into a picture: what the
best config was after each iteration, what that iteration cost, how many search
copies were still alive, and what the surrogate model was doing.

A trace holds whichever stages ran, so the panels follow the trace: `lfbo`
plots the surrogate search, `llm` the rounds of proposals, and `llm-lfbo` both,
with the handover marked on the round plot.

Usage:
    $ python3 plotSearchTrace.py <trace.jsonl | dir> [...] [-o OUT.png] [--show]

A directory argument is expanded to the traces inside it, so a whole tuning run
can be plotted with

    $ python3 plotSearchTrace.py results.tsv.search

The search measures kernel times, so that is what is plotted; the TFlops of the
winner are in the tuning output itself.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib

import pandas as pd

# One line per iteration and a running best is a step function, not a curve:
# the best config holds until an iteration replaces it.
STEP = {"where": "post", "linewidth": 2}

BEST = "best time (ms)"


class Trace:
    """One problem's trace, split into the stages that wrote it.

    A hybrid run writes both stages to one file, so which frames are populated
    is what says which stages ran.
    """

    def __init__(self, path: str):
        self.path = path
        self.header: Dict = {}
        self.llm_header: Dict = {}
        self.iterations = pd.DataFrame()
        self.rounds = pd.DataFrame()

    @property
    def has_lfbo(self) -> bool:
        return not self.iterations.empty

    @property
    def has_llm(self) -> bool:
        return not self.rounds.empty


def read_trace(path: str) -> Trace:
    """Reads a trace into its headers and a frame per stage.

    A run that was interrupted leaves a half-written last line, which is worth
    skipping rather than refusing the whole trace over.
    """
    trace = Trace(path)
    iterations: List[Dict] = []
    rounds: List[Dict] = []
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"{path}:{number}: ignoring an incomplete record", file=sys.stderr)
                continue
            kind = record.get("kind")
            if kind == "header":
                trace.header = record
            elif kind == "llm-header":
                trace.llm_header = record
            elif kind == "llm-round":
                rounds.append(record)
            else:
                # Anything else is an LFBO iteration, including the records
                # written before `kind` was added to them.
                iterations.append(record)
    if not iterations and not rounds:
        raise ValueError(f"{path}: no iterations were traced")
    trace.iterations = pd.DataFrame(iterations)
    trace.rounds = pd.DataFrame(rounds)
    return trace


def first_search_win(frame: pd.DataFrame) -> Optional[int]:
    """The iteration at which a config the search found took the lead.

    Until that happens the best config is one of the heuristic's own guesses or
    a random draw used to pad the first batch, and the search has yet to earn
    its keep.
    """
    from_search = frame["bestOrigin"].fillna("").str.startswith("copy")
    if not from_search.any():
        return None
    return int(frame.loc[from_search.idxmax(), "generation"])


def annotate_search_win(axes, frame: pd.DataFrame) -> None:
    won = first_search_win(frame)
    if won is None:
        return
    axes.axvline(won, color="tab:green", linestyle=":", linewidth=1.5)
    axes.annotate("search takes the lead",
                  xy=(won, axes.get_ylim()[0]),
                  xytext=(4, 6),
                  textcoords="offset points",
                  rotation=90,
                  fontsize=8,
                  color="tab:green")


def plot_progress(axes, frame: pd.DataFrame, best: pd.Series) -> None:
    axes.step(frame["generation"], best, color="tab:blue", **STEP)
    axes.set_xlabel("iteration")
    axes.set_ylabel(BEST, color="tab:blue")
    axes.grid(alpha=0.3)

    copies = axes.twinx()
    copies.step(frame["generation"], frame["alive"], color="tab:orange", **STEP)
    copies.set_ylabel("alive copies", color="tab:orange")
    copies.set_ylim(bottom=0)
    annotate_search_win(axes, frame)
    axes.set_title("Progress per iteration")


def plot_cost(axes, frame: pd.DataFrame, best: pd.Series) -> None:
    # An iteration that benchmarks fifty configs costs fifty compiles, so the
    # honest x-axis for "was this worth it" is the budget spent, not the count
    # of iterations.
    axes.step(frame["measured"], best, color="tab:blue", **STEP)
    axes.set_xlabel("configs measured")
    axes.set_ylabel(BEST)
    axes.grid(alpha=0.3)

    minutes = frame["totalMs"] / 60000.0
    clock = axes.twiny()
    clock.step(minutes, best, color="tab:purple", linestyle="--", **STEP)
    clock.set_xlabel("wall clock (minutes)", color="tab:purple")
    axes.set_title("Progress per unit of budget")


def plot_outcomes(axes, frame: pd.DataFrame) -> None:
    # Each iteration reports the batch the previous one proposed, so shift the
    # counts back onto the iteration that chose those configs.
    iterations = frame["generation"]
    succeeded = frame["succeeded"].shift(-1).fillna(0)
    not_applicable = frame["notApplicable"].shift(-1).fillna(0)
    failed = frame["failed"].shift(-1).fillna(0)
    axes.bar(iterations, succeeded, color="tab:green", label="measured")
    axes.bar(iterations,
             not_applicable,
             bottom=succeeded,
             color="tab:orange",
             label="not applicable")
    axes.bar(iterations, failed, bottom=succeeded + not_applicable, color="tab:red", label="failed")
    axes.step(iterations,
              frame["proposed"],
              color="black",
              linestyle=":",
              label="proposed",
              where="post")
    axes.set_xlabel("iteration")
    axes.set_ylabel("configs")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    axes.set_title("What became of each batch")


def plot_surrogate(axes, frame: pd.DataFrame) -> None:
    axes.plot(frame["generation"], frame["trainSize"], color="tab:blue", label="training set")
    axes.plot(frame["generation"], frame["positives"], color="tab:cyan", label="labelled good")
    axes.set_xlabel("iteration")
    axes.set_ylabel("configs")
    axes.legend(fontsize=8, loc="upper left")
    axes.grid(alpha=0.3)

    # How many of the configs the model picked turned out to be in the quantile
    # it was aiming at: a flat line near the quantile itself is a model that has
    # learnt nothing a random draw would not have found.
    precision = axes.twinx()
    precision.plot(frame["generation"],
                   frame["pickPrecision"],
                   color="tab:red",
                   marker=".",
                   linestyle="none")
    precision.set_ylabel("picks that were good", color="tab:red")
    precision.set_ylim(-0.02, 1.02)
    axes.set_title("What the surrogate knew")


def plot_copy_deaths(axes, frame: pd.DataFrame) -> None:
    patience = frame["stoppedOutOfPatience"]
    neighbors = frame["stoppedWithoutNeighbors"]
    selection = frame["stoppedWithoutSelection"]
    axes.bar(frame["generation"], patience, color="tab:blue", label="out of patience")
    axes.bar(frame["generation"],
             neighbors,
             bottom=patience,
             color="tab:orange",
             label="nowhere left to walk")
    axes.bar(frame["generation"],
             selection,
             bottom=patience + neighbors,
             color="tab:red",
             label="nothing selected")
    axes.set_xlabel("iteration")
    axes.set_ylabel("copies stopped")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    axes.set_title("Why copies stopped")


def plot_neighborhood(axes, frame: pd.DataFrame) -> None:
    axes.plot(frame["generation"],
              frame["neighborsGenerated"],
              color="tab:blue",
              label="neighbours generated")
    axes.plot(frame["generation"],
              frame["neighborsSeenBefore"],
              color="tab:orange",
              label="already visited")
    # Moves the space refuses: the LDS blacklist, the compile-cost budget and
    # the rest of the feasibility filters, seen from inside the walk.
    axes.plot(frame["generation"], frame["movesRejected"], color="tab:red", label="moves refused")
    axes.set_xlabel("iteration")
    axes.set_ylabel("candidates")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    axes.set_title("Room to move")


def plot_rounds(axes, frame: pd.DataFrame, best: pd.Series) -> None:
    axes.step(frame["round"], best, color="tab:blue", marker=".", **STEP)
    axes.set_xlabel("round")
    axes.set_ylabel(BEST, color="tab:blue")
    axes.grid(alpha=0.3)
    axes.set_xticks(frame["round"])

    # A round is one request to the model plus the batch it produced, so its
    # cost is mostly latency. Worth seeing next to what it bought.
    minutes = axes.twinx()
    minutes.step(frame["round"], frame["elapsedMs"] / 60000.0, color="tab:purple", **STEP)
    minutes.set_ylabel("round took (minutes)", color="tab:purple")
    minutes.set_ylim(bottom=0)
    axes.set_title("What each round of proposals bought")


def plot_round_outcomes(axes, frame: pd.DataFrame) -> None:
    # What the model asked for, minus what never reached the GPU. A model whose
    # proposals are mostly refused or already-tried is being prompted badly,
    # and that is the failure this panel exists to make obvious.
    rounds = frame["round"]
    proposed = frame["proposed"]
    rejected = frame.get("rejected", pd.Series(0, index=frame.index))
    duplicates = frame.get("duplicates", pd.Series(0, index=frame.index))
    axes.bar(rounds, proposed, color="tab:green", label="benchmarked")
    axes.bar(rounds, rejected, bottom=proposed, color="tab:red", label="refused by the space")
    axes.bar(rounds,
             duplicates,
             bottom=proposed + rejected,
             color="tab:orange",
             label="already tried")
    axes.set_xlabel("round")
    axes.set_ylabel("configs")
    axes.set_xticks(rounds)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    axes.set_title("What became of each proposal")


def summarize(trace: Trace) -> str:
    parts: List[str] = []

    if trace.has_llm:
        rounds = trace.rounds
        last = rounds.iloc[-1]
        best = (rounds["bestNs"] / 1e6).dropna()
        parts.append(f"llm: {len(rounds)} rounds, {int(last['proposals'])} configs proposed, "
                     f"{last['totalMs'] / 60000.0:.1f} min" +
                     (f", best {best.iloc[-1]:.3f} ms" if not best.empty else ""))
        if trace.llm_header:
            parts.append(f"{trace.llm_header.get('model', 'unknown model')} on "
                         f"{trace.llm_header.get('arch', 'unknown arch')}, "
                         f"{trace.llm_header.get('quickSeeds')} heuristic seeds")

    if trace.has_lfbo:
        frame = trace.iterations
        last = frame.iloc[-1]
        best = (frame["bestNs"] / 1e6).dropna()
        outcome = f"best {best.iloc[-1]:.3f} ms, from {last.get('bestOrigin') or 'nowhere'}" \
            if not best.empty else "nothing was measured"
        parts.append(f"lfbo: {len(frame)} iterations, {int(last['measured'])} configs measured, "
                     f"{last['totalMs'] / 60000.0:.1f} min, {outcome}")
        if trace.header:
            parts.append(f"{trace.header.get('arch', 'unknown arch')}, seed "
                         f"{trace.header.get('seed')}, {trace.header.get('copies')} copies")

    return "\n".join(parts)


def plot_trace(path: str, out: Optional[str], show: bool) -> None:
    trace = read_trace(path)

    import matplotlib.pyplot as plt
    rows = (3 if trace.has_lfbo else 0) + (1 if trace.has_llm else 0)
    figure, panels = plt.subplots(rows, 2, figsize=(14, 4.4 * rows), squeeze=False)

    row = 0
    if trace.has_llm:
        rounds = trace.rounds
        best = rounds["bestNs"] / 1e6
        plot_rounds(panels[row][0], rounds, best)
        plot_round_outcomes(panels[row][1], rounds)
        row += 1

    if trace.has_lfbo:
        frame = trace.iterations
        best = frame["bestNs"] / 1e6
        plot_progress(panels[row][0], frame, best)
        plot_cost(panels[row][1], frame, best)
        plot_outcomes(panels[row + 1][0], frame)
        plot_surrogate(panels[row + 1][1], frame)
        plot_copy_deaths(panels[row + 2][0], frame)
        plot_neighborhood(panels[row + 2][1], frame)

    figure.suptitle(summarize(trace), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    target = out or f"{os.path.splitext(path)[0]}.png"
    figure.savefig(target, dpi=140)
    print(f"wrote {target}")
    if show:
        plt.show()
    plt.close(figure)


def collect_traces(inputs: List[str]) -> List[str]:
    traces = []
    for path in inputs:
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "*.jsonl")))
            if not found:
                print(f"{path}: no traces in this directory", file=sys.stderr)
            traces += found
        else:
            traces.append(path)
    return traces


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces",
                        nargs="+",
                        metavar="TRACE",
                        help="trace files written by --search-trace, or directories of them")
    parser.add_argument("-o",
                        "--out",
                        default=None,
                        metavar="PNG",
                        help="where to write the plot (default: beside each trace)")
    parser.add_argument("--show",
                        action="store_true",
                        default=False,
                        help="also open the plot in a window")
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")

    traces = collect_traces(args.traces)
    if not traces:
        parser.error("no traces to plot")
    if args.out and len(traces) > 1:
        parser.error(f"--out: {len(traces)} traces to plot, so each has to keep its own name")

    failures = 0
    for path in traces:
        try:
            plot_trace(path, args.out, args.show)
        except (OSError, ValueError, KeyError) as error:
            print(f"cannot plot {path}: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
