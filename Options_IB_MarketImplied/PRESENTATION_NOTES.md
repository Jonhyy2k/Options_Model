# Presentation Notes: Market-Implied Carry Inputs

## What Changed

This version of the options engine no longer treats the rate input as a fixed external assumption by default.

Instead, before calibrating BSM / CRR / Merton / Heston, it:

1. Matches calls and puts with the same strike and expiry.
2. Computes the synthetic forward value at each strike:
   `G(K) = C(K) - P(K)`
3. Fits the parity line:
   `G(K) = B(0,T) * (F(0,T) - K) = alpha + beta * K`
4. Recovers:
   `discount factor = B(0,T) = -beta`
   `forward price = F(0,T) = alpha / B(0,T)`
5. Converts those into model inputs:
   `r = -ln(B(0,T)) / T`
   `q = r - ln(F(0,T)/S0) / T`

Those inferred `r` and `q` are then fed into all four pricing models and into the Breeden-Litzenberger density extraction.

## Why This Is Better

The short version to say in the room:

"We do not impose an arbitrary flat rate on the options model. We infer the market-implied forward and discount factor directly from matched call-put pairs in the same expiry we calibrate, then back out the implied financing rate and carry yield. That keeps the calibration internally consistent with the option surface itself."

Why that matters:

- The carry inputs now come from the same market you are calibrating to.
- The risk-neutral density is less exposed to a stale or arbitrary exogenous rate assumption.
- The report now shows diagnostics for the parity fit, so you can defend the quality of the inferred inputs.

## Exact Pitch Language

Use something close to this:

"For the options module, we first infer the market-implied forward price and discount factor from put-call parity across matched strikes. We do that with a weighted regression that emphasizes tighter and more liquid options and downweights noisy deep ITM/OTM strikes. From that, we derive the implied financing rate and dividend/carry yield used in BSM, CRR, Merton, and Heston. So the model is calibrated on inputs that are internally consistent with the same option chain, not on a flat externally imposed rate."

If they ask why this matters:

"Because otherwise you are mixing two worlds: option prices from one market snapshot and a separate rate/carry assumption that may not be synchronized with that snapshot."

## What To Show In The Report

The new report exposes:

- Carry source: `market_implied` or `flat_rate`
- Discount factor `B(0,T)`
- Forward price `F(0,T)`
- Implied `r`
- Implied `q`
- Parity regression `R^2`
- Parity RMSE and max residual
- Number of matched pairs used
- Strike band used in the regression
- Synthetic forward bid/ask bounds
- Standard errors for `B`, `F`, `r`, and `q` when available

These are the key fields to mention verbally:

- "We used X matched call-put pairs."
- "The parity fit had an R-squared of Y."
- "The market-implied forward for this expiry is Z."
- "The implied financing rate is A and implied carry/dividend yield is B."

## Caveats You Should State Explicitly

For US single-stock options, say this clearly:

"For single-name US options, put-call parity is noisier than for European index options because of early exercise features, dividends, and stock-loan effects. So we treat the inferred carry inputs as market-implied estimates, not as exact truths. That is why we report fit diagnostics and synthetic-forward bounds."

That caveat strengthens your credibility; it does not weaken the module.

## When To Prefer Market-Implied Carry

Use `market_implied` when:

- The option chain is liquid.
- You have enough matched strikes.
- The parity fit diagnostics are strong.

Use `flat_rate` as a fallback when:

- The chain is too sparse.
- The parity regression is unstable.
- You want a sensitivity comparison versus a fixed-rate setup.

## Best Slide Framing

Do not say:

"The market thinks there is a 10% real-world chance."

Say:

"The options market prices our target in the right tail, with an options-implied probability mass of roughly X% by this expiry."

If you want to connect it back to the carry inputs:

"The density is extracted after calibrating the surface using market-implied forward and discount inputs from the same option chain."

## Practical Notes

- For the stock pitch, align the expiry horizon with the catalyst horizon as closely as possible.
- Use the multi-DTE output if your thesis spans several dates.
- If the target is about 30% above spot, make sure the strike range still covers the target or at least that the extrapolation boundary is visible on the chart.
- For single stocks around ex-div dates or corporate events, mention that the parity estimate can absorb dividend and borrow effects.
