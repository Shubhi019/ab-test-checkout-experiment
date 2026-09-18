# A/B Test Analysis — One-Page Checkout

An end-to-end analysis of a conversion experiment: sizing it before it runs, reading it correctly when it finishes, and writing the recommendation that follows.

**Note on the data:** traffic is simulated, with a known effect deliberately built in, so the analysis can be checked against ground truth. This is a methodology exercise, not a report on a live experiment.

---

## The question

An eCommerce team wants to replace a three-step checkout with a single-page checkout. Does it increase the share of sessions that end in a completed order?

## Headline result

| | Sessions | Conversions | Rate |
|---|---|---|---|
| Control | 49,770 | 1,501 | 3.016% |
| Treatment | 49,770 | 1,581 | 3.177% |

Relative lift **+5.33%**, 95% CI **[−0.05%, +0.38%]**, **p = 0.14**.

The interval crosses zero. Read literally: no detectable effect, ship nothing.

**That conclusion is wrong.**

## The actual finding

| Device | Traffic share | Control | Treatment | Lift | p |
|---|---|---|---|---|---|
| Mobile | 65% | 2.229% | 2.112% | **−5.2%** | 0.31 |
| Desktop | 35% | 4.500% | 5.175% | **+15.0%** | 0.003 |

The aggregate was not "no effect". It was two real, opposing effects of similar magnitude cancelling each other out. Because mobile carries roughly two-thirds of traffic, a genuine desktop win was hidden by a mobile regression.

## Recommendation

Ship to **desktop only**. Hold mobile on the existing checkout and instrument the regression before iterating. Re-test a mobile-specific design as its own experiment, sized on the mobile baseline rather than the blended one.

---

## What the analysis covers

**1. Sizing the experiment before it runs.** The minimum detectable effect is set first (+10% relative), and sample size follows from it: 49,778 sessions per variant for 80% power at α = 0.05. Runtime is rounded up to 14 days so the test spans two complete weekly cycles — weekday and weekend shoppers differ, and stopping mid-week biases the result toward whichever days happened to be included.

An underpowered test does not return a wrong answer. It returns no answer, expensively.

**2. The significance test.** Two-sided two-proportion z-test. Pooled standard error for the test statistic (under H₀ both arms share one rate), unpooled for the confidence interval (there the difference is being estimated, not assumed to be zero).

**3. The peeking problem.** Re-running the test daily and stopping at the first p < 0.05 inflates the false positive rate far above the nominal 5% — every additional look is another chance to cross the line on noise. In this dataset:

| Day | Relative lift | p-value |
|---|---|---|
| 5 | +17.5% | 0.007 |
| **6** | **+18.5%** | **0.0018** |
| 10 | +9.1% | 0.038 |
| **14** | **+5.3%** | **0.14** |

An analyst watching daily would have shipped on day 6. The effect evaporated. The fix is to fix the sample size in advance and read the result once — or, if interim looks are genuinely needed, use a sequential design (O'Brien-Fleming boundaries, alpha spending) that budgets error across the looks.

**4. Segmentation, with the caveat stated.** Device was pre-registered as the sole segmentation dimension, on the prior reasoning that a layout change plausibly behaves differently on a small screen. Slicing a null result repeatedly until something reaches significance guarantees finding noise. A post-hoc split is a hypothesis for the next test, not a result of this one.

**5. Novelty check.** Week 1 showed treatment at +14.9%; week 2 at −3.5%. Users engage with a changed interface partly because it changed, and the effect decays. A one-week test would have measured novelty rather than design.

**6. The decision, with its conditions.** The recommendation is stated alongside what would change it, and what would be needed before calling it final: revenue per session rather than conversion rate alone (a lift that shifts mix toward cheaper baskets is not a win), guardrail metrics on refunds, payment failures and support contacts, and a sample ratio mismatch check to confirm assignment was genuinely 50/50.

---

## Running it

```bash
pip install numpy pandas scipy
python ab_test_checkout_experiment.py
```

No data files or network access required — the script generates its own traffic and prints the full analysis.

---

## Why this structure

The interesting part of an experiment is rarely the p-value. It is whether the test was sized to answer the question, whether the result was read once or watched until it said something convenient, and whether the aggregate is concealing something the business needs to know.

Here it was concealing exactly that.

---

**Shubhi Bajpai** · [shubhi019.github.io](https://shubhi019.github.io) · MS Business Analytics & Information Management, University of Delaware
