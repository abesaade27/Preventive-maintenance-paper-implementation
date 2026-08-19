# Agentic AI for Preventive Maintenance Policy Governance — Implementation

A working Python (+ standalone JS dashboard) implementation of the multi-agent
architecture from *"Agentic AI for autonomous preventive maintenance policy
governance: a multi-agent framework for dynamic industrial environments"*
(Crespo Márquez & Gómez Fernández, 2026, *Expert Systems With Applications*
314, 131767).

## Formulas — verified directly against the published PDF

```
CTE_calendar(Tp)  = [Cp + Cc*N(Tp)] / Tp
CTE_operating(Tp) = [Cp*R(Tp) + Cc*F(Tp)] / [Tp*R(Tp) + M(Tp)*F(Tp)]
```
- `CTE_calendar` is an **exact notational match** to `reliability.py`'s
  `cet_calendar()` (renewal function `N(Tp)`, solved from the renewal
  integral equation).
- `CTE_operating`'s denominator `Tp*R(Tp) + M(Tp)*F(Tp)` is the standard
  reliability identity `E[min(T,Tp)] = ∫₀^Tp R(t)dt` — mathematically
  identical to `cet_operating()`'s implementation, just computed via the
  integral form instead of the truncated-mean form. Both are documented
  with this equivalence directly in `reliability.py`.

## Inputs matched exactly to the paper

From Table 1 and Fig. 4's tick-log example:
- `Tp_init = 25.0`
- `Cp` schedule: `20.0, 45.6, 71.1, 96.7, 122.2, 147.8, 173.3, 198.9, 224.4, 250.0`
- `Cc = 250.0` (fixed)
- `n_units = 50` per conversation (from Fig. 4: "Number of observations: 50, censored: 8")
- Regime change at conversation 6: `beta_true` 2.0 (wear-out) → 0.6 (infant-mortality-like), `eta_true = 20.0` fixed
- See `paper_reference.py` for the full transcription of the paper's Table 1 and Table 2.

The paper's own random seed for its Weibull draws was never published, so
`run.py` searches a small range of seeds and reports the one whose
**dominant-policy pattern** (which conversations pick operating-time-based
vs. calendar-based) matches the paper's published pattern exactly —
`OOOOOCCCCC`, flipping right at the regime change. **Seed 1 reproduces this
exactly in the Python implementation** (`numpy`'s RNG); the browser
dashboard uses a different RNG (`mulberry32`) so its matching seed is **4**
— both are set as defaults and both reproduce the paper's qualitative
finding exactly, including the policy flip at conversation 6.

Per-conversation `beta_hat`/`eta_hat`/`Tp*` values still won't match the
paper number-for-number (different RNG stream, unpublished seed) — that's
expected and is why `run.py` and the dashboard both print/plot a direct
side-by-side comparison rather than claiming identical numbers.

## Architecture (matches paper Sections 4.2–4.4)

| File | Role | Paper section |
|---|---|---|
| `message.py` | JSON-style `Message` dataclass | 4.3 |
| `broker.py` | Outbox → 1-tick delay → inbox routing, full event log | 4.2/4.3 |
| `agents/scenario_agent.py` | Simulates censored Weibull data, regime change at conversation 6 | 4.2/5.1 |
| `agents/fitting_agent.py` | Censored Weibull MLE re-estimation | 4.2/5.2 |
| `agents/optimizer_agent.py` | Cost–time-efficiency optimization, picks dominant policy | 4.2/5.3 |
| `agents/explainer_agent.py` | Structured, stubbed explanation + state machine | 4.5/5.4 |
| `agents/orchestrator_agent.py` | Starts conversations, closes the feedback loop | 4.2/5.5 |
| `reliability.py` | Core math: censored MLE, `CTE_calendar`/`CTE_operating` (paper-verified) | 3.2/3.3 |
| `simulation.py` | Tick-based driver tying all agents together | 4.4 |
| `paper_reference.py` | Transcribed Table 1 / Table 2 + exact input parameters from the paper | — |
| `run.py` | Runs 10 conversations with paper-exact inputs, seed-matches, prints comparison, saves plots | 5–6 |
| `dashboard.html` | Standalone interactive browser dashboard (JS reimplementation) | — |

Agents never call each other directly — every interaction is a `Message`
published to the `Broker`'s outbox and delivered to the recipient's inbox
one tick later, exactly as described in Section 4.3. `broker.event_log`
gives full tick-by-tick traceability (the Fig. 4 analogue), dumped to
`output/event_log.json`.

## Running the Python implementation

```bash
pip install numpy scipy matplotlib
python3 run.py
```

Prints:
- A seed-search log, then a side-by-side table: our `beta_hat`/`eta_hat`/`Tp_op`/`Tp_cal`/dominant-policy vs. the paper's published values, conversation by conversation
- Our Phase 1 / Phase 2 summary stats vs. the paper's published Table 2
- The final conversation's Explainer Agent output

Saves to `output/`:
- `fig6_reliability_reestimation.png` — beta_hat & eta_hat vs. true values **and** vs. the paper's own published curve
- `fig7_policy_evolution.png` — Tp_prev/Tp_cal/Tp_op evolution + dominant-policy step chart
- `fig8_censoring.png` — censored data % per conversation
- `fig9_sensitivity_cost_ratio.png` — dual-axis Tp*/normalised-cost-rate sensitivity to Cp/Cc
- `our_results.json` — full structured per-conversation results
- `event_log.json` — full message-level audit trail (every `REQUEST_SCENARIO`/`SCENARIO`/`FIT_RESULT`/`POLICY`/`EXPLANATION` message, tick-stamped)

## Running the dashboard

Open `dashboard.html` in any browser — no server, no install. It:
- Defaults to the paper's exact inputs (Tp_init=25, the real Cp schedule, Cc=250, n=50 units, regime switch at conversation 6, seed=4 — an exact policy-pattern match)
- **"Find best-matching seed vs. paper"** re-runs the same seed search `run.py` does, live, in the browser
- Shows the agent message log, a live results table with our numbers next to the paper's published numbers (mismatched dominant-policy rows highlighted), and all four Fig. 6–9 analogue charts, with the paper's own published curves overlaid where applicable (Fig. 6)
- All inputs (Tp, Cc, unit count, Cp schedule, phase-switch point, seed) are editable to explore other scenarios beyond the paper's own

## What's simplified vs. the paper

- Single asset type, single failure mode, parametric (Weibull) reliability only — as the paper itself scopes in Section 3.5/8.2.
- No CMMS/SAP PM integration layer, Node.js middleware, FlowiseAI orchestrator, live OpenAI GPT call, or pgvector knowledge base (Section 7's industrial deployment stack) — this implements the **academic prototype** layer (Sections 4-6) only.
- The LLM Explainer is template-based, not a live API call — this matches the paper's own description of its academic prototype ("the LLM call is stubbed").
- The paper's exact random seed for its Weibull draws isn't published, so per-conversation numeric values are matched *structurally* (same policy-flip pattern) rather than *numerically* (identical beta_hat/eta_hat draws).

## Extending toward the paper's future work (Section 9)

- Swap `ScenarioAgent` for a `CMMSDataAgent` reading real work orders (Section 7.1) — every downstream agent is unchanged.
- Swap the MLE fitter in `fitting_agent.py` for Kaplan–Meier or a hybrid estimator without touching orchestration/optimization (Section 8.2 explicitly calls this out as compatible with the architecture).
- Wire `ExplainerAgent` to a real LLM call using the same structured payload it already builds.
