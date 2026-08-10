# Two-stage tuning: top-K + adaptive measurement budget

## Overview

Tuning time is determined by two major, compute-intensive, steps: build and benchmarking steps.
This document describes the two-stage tuning, an approach to reduce benchmarking time while
keeping the same level of accuracy.

### 1. Two-stage idea (measure cheap, then-refine)

The idea that a cheap first pass need not lose the true best comes from [1].
They evaluate every candidate incrementally, use a
confidence bound to discard any candidate whose best possible score is already
worse than the worst possible score of the current best, and "race" the
survivors. Crucially they report: *"In all the cases we
have tested, the [model] chosen by brute force is also contained by the set
returned from Hoeffding Races. Therefore, there is no loss of performance
accuracy."* — i.e. cheap elimination need not cost quality, which is precisely
our "zero precision loss" target for the shortlist.

The structure we actually use comes from [6]: they evaluate many configurations
at a small budget, promote the best fraction to a larger budget, and repeat over
several budget levels. ([1] instead discards candidates adaptively and keeps
measuring the survivors at the same cost, so it backs the "no loss of accuracy"
finding above rather than our two budget levels.) Our two-stage does a similar
thing: a coarse budget over all configs, and then the precise budget over the
top-K.

### 2. Adaptative budget: terminate a measurement early if its result is already reliable

AdaTune [2] is the closest prior work: it accelerates
auto-tuning tensor programs (AutoTVM / TVM) where "the majority of time is spent
on hardware measurement." Their *Adaptive Evaluator* measures a candidate in
micro-batches and, after each, computes the **coefficient of variation of the running performance estimate**;
once it drops below a threshold it **terminates the measurement early**. They
state this "automatically adjusts the hardware measurement costs in a robust and
**model/hardware-independent way**" (the portability property we want).

Reported result: **1.3–3.9× faster** optimization to reach the same quality, or
up to 115% higher GFLOPS under the same time budget, *across different hardware*.

AdaTune also empirically documents *why a fixed budget is wrong*: measurement
variance and its coefficient of variation differ markedly between CPU and GPU and
between models (their Observation 2 / Figure 3). A single hard-coded iteration or
millisecond count cannot be right for all of them, which is our motivation for
an adaptive rule.

Note on the metric: AdaTune's "CV" is computed over the *cumulative* throughput
estimates `Perf_i = i·FLOP / Time(i)`, which converge as `i` grows, so their CV
**shrinks toward zero** with more measurement. That is a convergence signal on
the *estimate*, behaving like the standard error below — not the fixed,
intrinsic sample jitter.

### 3. Measure until the estimate is precise enough (SEM, not CoV)

Two rigorous-benchmarking papers independently arrive at the same adaptive rule,
and they say precisely *which statistic* to stop on.

Georges et al. [3] build a confidence interval (CI) for the mean, and their
JavaStats harness "monitors the variability observed in the measurements to
determine the number of measurements that need to be taken to reach a desired
confidence interval" — it **stops as soon as the CI is within, e.g., 1–2% of the
mean**, or at a preset max (§4.3).

Hoefler & Belli [4] give the same recipe for HPC:
"recompute the (1−α) CI after each `n_i = i·k` measurements and
stop once the required interval is reached" (§4.2.2), with
`n = (t·s / (e·x̄))²` for the normal case. 

CoV vs SEM:

**CoV**: The **coefficient of variation of the raw samples** measures a
kernel's *intrinsic* jitter and **does not shrink** as you measure more — it
converges to a fixed, non-zero value ([4] §3.1.2 describes CoV as a *consistency*
metric of the system), so "measure until CoV is below a small constant" is
impossible for a noisy kernel. 

**SEM**: The **standard error of the mean**
(`SEM = s/√n`, equivalently the CI width) *is* the uncertainty of our estimate and
**does shrink ∝ 1/√n** ([3] §3.2, [4] §3.1.2). AdaTune's shrinking cumulative-CV
is a variant of the same convergence signal.

This is why we use SEM.

### 4. Warmup floor: an adaptive rule still needs a minimum

The SEM only accounts for *random* error, so a few early, still-warming-up
samples can look precise while being systematically wrong. Barrett et al. [5]
use changepoint analysis to show that steady state is not guaranteed and often
never reached, i.e. those first samples are biased, not merely noisy. Small
samples are also a problem for the stopping statistic itself: below n=~30 the
sample variance "can be significantly different from the actual variance"
([3] §3.2.1–3.2.2), so the SEM computed from a handful of samples is unreliable.

We therefore floor the coarse pass: a warmup iteration count with a wall-clock
floor beneath it (`--coarse-warmup-iters`, `--coarse-warmup-floor-ms`), so a fast
GPU still spends enough time for clock ramp to complete, and a minimum number of
measured iterations before the SEM check may fire (`--coarse-min-rep-iters`)

## Design summary

| Component                                         |             References |
|---------------------------------------------------|------------------------|
| Top-K, refine at precise budget                   |                    [6] |
| Shortlist retains the true best                   |                    [1] |
| Cheap coarse pass with early stop on relative SEM |                [2,3,4] |
| Min warmup / min iterations floor                 |  [5]; [3] §3.2.1–3.2.2 |
| Max iteration/time cap                            |               [3] §4.3 |

## References

[1] O. Maron and A. W. Moore. "Hoeffding Races: Accelerating Model Selection Search for Classification and Function Approximation". NeurIPS 1993. 
<https://proceedings.neurips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf>

[2] M. Li, M. Zhang, C. Wang, M. Li. "AdaTune: Adaptive Tensor Program Compilation Made Efficient". NeurIPS 2020. 
<https://proceedings.neurips.cc/paper/2020/file/a9b7ba70783b617e9998dc4dd82eb3c5-Paper.pdf>

[3] A. Georges, D. Buytaert, L. Eeckhout. "Statistically Rigorous Java Performance Evaluation". OOPSLA 2007. 
<https://dri.es/files/oopsla07-georges.pdf>

[4] T. Hoefler and R. Belli. "Scientific Benchmarking of Parallel Computing Systems". SC 2015. 
<https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf>

[5] E. Barrett, C. F. Bolz-Tereick, R. Killick, S. Mount, L. Tratt. "Virtual Machine Warmup Blows Hot and Cold". OOPSLA 2017. 
<https://arxiv.org/abs/1602.00602>

[6] L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization". ICLR 2017 / JMLR 2018. 
<https://arxiv.org/abs/1603.06560>
