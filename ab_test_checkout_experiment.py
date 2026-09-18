"""
A/B Test Analysis — One-Page Checkout
=====================================

Scenario
--------
An eCommerce team wants to replace a three-step checkout with a single-page
checkout. The hypothesis is that removing steps reduces abandonment and lifts
the rate at which sessions convert to completed orders.

This script covers the full lifecycle of the decision, in the order the work
actually happens:

  1. Sample sizing and power — decided BEFORE the test runs
  2. Running the test and computing the headline result
  3. Why you don't check the result every day (the peeking problem)
  4. Segmenting the result — where the interesting finding lives
  5. Checking for novelty effects
  6. The ship / don't ship recommendation

Run:  python ab_test_checkout_experiment.py
Needs: numpy, pandas, scipy

Author: Shubhi Bajpai
"""

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# 1. SAMPLE SIZING — done before a single user sees the new checkout
# ----------------------------------------------------------------------------
# The question that decides everything: how small an effect do we care about?
# If we size for a 10% relative lift and the true effect is 3%, the test will
# come back "no significant difference" and we will wrongly conclude the
# feature does nothing. An underpowered test doesn't give a wrong answer --
# it gives no answer, expensively.

BASELINE_CVR = 0.032      # 3.2% of sessions currently convert
MDE_RELATIVE = 0.10       # smallest lift worth shipping for: +10% relative
ALPHA = 0.05              # false positive rate we accept
POWER = 0.80              # chance of detecting the effect if it is real


def required_sample_per_group(p1, relative_mde, alpha=0.05, power=0.80):
    """Sample size per variant for a two-proportion test (two-sided)."""
    p2 = p1 * (1 + relative_mde)
    delta = p2 - p1
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = (2 * p_bar * (1 - p_bar) * (z_alpha + z_beta) ** 2) / (delta ** 2)
    return int(np.ceil(n))


N_PER_GROUP = required_sample_per_group(BASELINE_CVR, MDE_RELATIVE, ALPHA, POWER)
DAILY_SESSIONS = 12_000   # total across both variants

print("=" * 74)
print("1. EXPERIMENT DESIGN")
print("=" * 74)
print(f"Baseline conversion rate        : {BASELINE_CVR:.2%}")
print(f"Minimum detectable effect       : +{MDE_RELATIVE:.0%} relative "
      f"({BASELINE_CVR:.2%} -> {BASELINE_CVR * (1 + MDE_RELATIVE):.2%})")
print(f"Significance level (alpha)      : {ALPHA}")
print(f"Power (1 - beta)                : {POWER:.0%}")
print(f"\nRequired sample per variant     : {N_PER_GROUP:,} sessions")
print(f"Total sample required           : {2 * N_PER_GROUP:,} sessions")
print(f"At {DAILY_SESSIONS:,} sessions/day     : "
      f"{2 * N_PER_GROUP / DAILY_SESSIONS:.1f} days")
print("\nRounded up to 14 days so the test covers two full weekly cycles.")
print("Weekday and weekend shoppers behave differently; stopping mid-week")
print("would bias the result toward whichever days happened to be included.")

RUNTIME_DAYS = 14

# ----------------------------------------------------------------------------
# 2. SIMULATE THE EXPERIMENT
# ----------------------------------------------------------------------------
# Ground truth used to generate the data. In real work you never know these --
# they are here so the analysis below can be checked against reality.
#
# The new checkout genuinely helps on desktop and genuinely hurts on mobile.
# Mobile is the majority of traffic, so the two effects very nearly cancel.

TRUTH = {
    "desktop": {"control": 0.045, "treatment": 0.052},   # real win
    "mobile":  {"control": 0.024, "treatment": 0.021},   # real regression
}
MOBILE_SHARE = 0.65


def simulate(n_per_group, days, mobile_share=MOBILE_SHARE):
    rows = []
    per_day = n_per_group // days
    for day in range(1, days + 1):
        for variant in ("control", "treatment"):
            n_mobile = RNG.binomial(per_day, mobile_share)
            n_desktop = per_day - n_mobile
            for device, n in (("mobile", n_mobile), ("desktop", n_desktop)):
                p = TRUTH[device][variant]
                # Week 1 novelty bump on treatment: users click the new thing
                # simply because it is new. It fades. If you stop the test
                # early you will read this as a real effect.
                if variant == "treatment" and day <= 7:
                    p *= 1.06
                conversions = RNG.binomial(n, p)
                rows.append({
                    "day": day, "variant": variant, "device": device,
                    "sessions": n, "conversions": conversions,
                })
    return pd.DataFrame(rows)


df = simulate(N_PER_GROUP, RUNTIME_DAYS)


# ----------------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------------
def two_proportion_test(conv_a, n_a, conv_b, n_b):
    """Two-sided z-test for a difference in proportions, plus a 95% CI."""
    p_a, p_b = conv_a / n_a, conv_b / n_b

    # Pooled SE for the test statistic (assumes H0: p_a == p_b)
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval on the difference
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    margin = stats.norm.ppf(0.975) * se_unpooled

    return {
        "cvr_control": p_a,
        "cvr_treatment": p_b,
        "abs_diff": p_b - p_a,
        "rel_lift": (p_b - p_a) / p_a,
        "z": z,
        "p_value": p_value,
        "ci_low": (p_b - p_a) - margin,
        "ci_high": (p_b - p_a) + margin,
        "significant": p_value < ALPHA,
    }


def totals(frame):
    g = frame.groupby("variant")[["sessions", "conversions"]].sum()
    return (g.loc["control", "conversions"], g.loc["control", "sessions"],
            g.loc["treatment", "conversions"], g.loc["treatment", "sessions"])


# ----------------------------------------------------------------------------
# 3. HEADLINE RESULT
# ----------------------------------------------------------------------------
print("\n" + "=" * 74)
print("2. HEADLINE RESULT (all traffic, full 14 days)")
print("=" * 74)

overall = two_proportion_test(*totals(df))
c_conv, c_sess, t_conv, t_sess = totals(df)

print(f"Control    : {c_conv:>6,} / {c_sess:>7,} = {overall['cvr_control']:.3%}")
print(f"Treatment  : {t_conv:>6,} / {t_sess:>7,} = {overall['cvr_treatment']:.3%}")
print(f"\nAbsolute difference : {overall['abs_diff']:+.4%}")
print(f"Relative lift       : {overall['rel_lift']:+.2%}")
print(f"95% CI (absolute)   : [{overall['ci_low']:+.4%}, {overall['ci_high']:+.4%}]")
print(f"z = {overall['z']:.3f},  p = {overall['p_value']:.4f}")
print(f"\nSignificant at alpha={ALPHA}? {'YES' if overall['significant'] else 'NO'}")
print("\nRead literally, this says: no detectable effect. Ship nothing, move on.")
print("That conclusion would be wrong, and section 4 shows why.")

# ----------------------------------------------------------------------------
# 4. THE PEEKING PROBLEM
# ----------------------------------------------------------------------------
print("\n" + "=" * 74)
print("3. WHY WE DO NOT CHECK THE RESULT DAILY")
print("=" * 74)
print("Running the same test at the end of every day and stopping the first")
print("time p < 0.05 inflates the false positive rate well above 5% -- each")
print("look is another chance to cross the line by luck alone.\n")
print(f"{'Day':>4} {'Sessions/arm':>13} {'Rel lift':>10} {'p-value':>10}  Would stop?")
print("-" * 74)

crossed = False
for day in range(3, RUNTIME_DAYS + 1):
    upto = df[df["day"] <= day]
    res = two_proportion_test(*totals(upto))
    n_arm = upto[upto["variant"] == "control"]["sessions"].sum()
    flag = "<-- STOP (tempting)" if res["p_value"] < ALPHA else ""
    if res["p_value"] < ALPHA:
        crossed = True
    print(f"{day:>4} {n_arm:>13,} {res['rel_lift']:>9.2%} "
          f"{res['p_value']:>10.4f}  {flag}")

print("-" * 74)
if crossed:
    print("The p-value dipped below 0.05 mid-test and then recovered.")
    print("An analyst watching daily would have shipped on noise.")
else:
    print("It never crossed here -- but that is luck, not method.")
print("\nFix: fix the sample size in advance and read the result once. If you")
print("genuinely need interim looks, use a sequential design (O'Brien-Fleming,")
print("alpha spending) that budgets the error across looks.")

# ----------------------------------------------------------------------------
# 5. SEGMENTATION — the real finding
# ----------------------------------------------------------------------------
print("\n" + "=" * 74)
print("4. SEGMENTED RESULT — WHERE THE FINDING ACTUALLY IS")
print("=" * 74)

segment_rows = []
for device in ("mobile", "desktop"):
    sub = df[df["device"] == device]
    res = two_proportion_test(*totals(sub))
    share = sub["sessions"].sum() / df["sessions"].sum()
    segment_rows.append({
        "Device": device,
        "Traffic share": f"{share:.0%}",
        "Control CVR": f"{res['cvr_control']:.3%}",
        "Treatment CVR": f"{res['cvr_treatment']:.3%}",
        "Rel lift": f"{res['rel_lift']:+.2%}",
        "95% CI": f"[{res['ci_low']:+.3%}, {res['ci_high']:+.3%}]",
        "p-value": f"{res['p_value']:.4f}",
        "Significant": "YES" if res["significant"] else "no",
    })

print(pd.DataFrame(segment_rows).to_string(index=False))
print("\nThe aggregate was not 'no effect'. It was two real and opposite")
print("effects of similar size, cancelling. Desktop users convert materially")
print("better on the new checkout; mobile users convert worse. Because mobile")
print("is ~65% of traffic, the mobile regression swallows the desktop win.")
print("\nCaveat worth stating out loud: device was a pre-registered segment,")
print("chosen before the test because we had a prior reason to expect the")
print("layout to behave differently on a small screen. Slicing an inconclusive")
print("result twenty ways until something is significant is not analysis -- it")
print("is a guarantee of finding noise. Pre-register the cuts, or treat any")
print("post-hoc split as a hypothesis for the NEXT test, not a result of this one.")

# ----------------------------------------------------------------------------
# 6. NOVELTY EFFECT
# ----------------------------------------------------------------------------
print("\n" + "=" * 74)
print("5. NOVELTY CHECK — WEEK 1 VS WEEK 2")
print("=" * 74)

for label, window in (("Week 1", (1, 7)), ("Week 2", (8, 14))):
    sub = df[(df["day"] >= window[0]) & (df["day"] <= window[1])]
    res = two_proportion_test(*totals(sub))
    print(f"{label}: control {res['cvr_control']:.3%} | "
          f"treatment {res['cvr_treatment']:.3%} | "
          f"rel lift {res['rel_lift']:+.2%} | p = {res['p_value']:.4f}")

print("\nTreatment looks better in week 1 than week 2. That gap is a novelty")
print("effect -- people engage with a changed interface because it changed.")
print("It decays. A one-week test would have overstated the treatment and")
print("could have flipped the shipping decision on an artefact.")

# ----------------------------------------------------------------------------
# 7. RECOMMENDATION
# ----------------------------------------------------------------------------
print("\n" + "=" * 74)
print("6. RECOMMENDATION")
print("=" * 74)
print("""
DO NOT ship the one-page checkout to all traffic.

Reasoning
  - The aggregate result is not evidence of 'no effect'. It is evidence of
    two opposing effects that happen to net out.
  - Mobile is ~65% of sessions and shows a real regression. Shipping to all
    traffic knowingly degrades the majority experience.
  - Desktop shows a genuine improvement and should not be thrown away.

Recommended action
  1. Ship the one-page checkout to DESKTOP only. The effect is real,
     measured, and directionally what the hypothesis predicted.
  2. Hold mobile on the existing checkout and investigate the regression
     before iterating. Likely candidates worth instrumenting: form field
     count above the fold, keyboard type on numeric inputs, and whether the
     payment step now falls below the fold on small viewports.
  3. Re-test a mobile-specific design as its own experiment, sized on the
     mobile baseline rather than the blended one.

What would change this recommendation
  - If mobile's regression were inside the CI and not distinguishable from
    zero, a full ship would be defensible on the desktop gain alone.
  - If desktop were under 15% of revenue, the engineering cost of
    maintaining two checkout flows would likely outweigh the gain.

What I would want before calling this final
  - Revenue per session, not just conversion rate. A lift in conversion
    that shifts mix toward lower-value baskets is not a win.
  - Guardrail metrics: refund rate, payment failure rate, support contacts.
  - A sample ratio mismatch check to confirm assignment was actually 50/50
    and no traffic was lost on one arm.
""")

print("=" * 74)
print("Method note: two-sided two-proportion z-test; sample size fixed in")
print("advance for 80% power at alpha=0.05 against a +10% relative MDE;")
print("14-day runtime covering two full weekly cycles; device pre-registered")
print("as the sole segmentation dimension.")
print("=" * 74)
