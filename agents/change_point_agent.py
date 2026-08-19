"""
change_point_agent.py
PROPOSED ENHANCEMENT (not part of the original paper).

Sits between the ScenarioAgent and the FittingAgent. Implements two things:

1. Adaptive Memory Buffer
   Instead of the FittingAgent re-estimating beta/eta from only the latest
   200-unit batch (as in the original paper), this agent accumulates
   (failure_times, censor_times) across conversations while the operating
   regime looks stable, and forwards the POOLED dataset downstream. This
   directly reduces MLE variance, since more data -> tighter estimates.

2. Change-Point Detection (likelihood-ratio test on RAW survival data)
   Detecting a regime shift by watching beta_hat directly is unreliable,
   because beta_hat is itself a noisy point estimate. Instead, this agent
   compares two hypotheses on the raw pooled data:

       H0 (no change): buffer + new batch share ONE Weibull(beta, eta)
       H1 (change)   : buffer and new batch are TWO DIFFERENT Weibull
                        distributions

   via a likelihood-ratio (LR) test:
       LR = 2 * [ (loglik_buffer_alone + loglik_new_alone) - loglik_pooled ]
   Under H0, LR ~ chi-square with df = 2. A large LR favours H1 -> a
   genuine regime change is declared, the buffer is discarded, and
   estimation restarts fresh from the new batch only.

Integration notes:
  - Forwards a "SCENARIO" message (same schema FittingAgent expects), so
    FittingAgent needs ZERO code changes.
  - weibull_censored_mle() already tolerates heterogeneous per-observation
    censor times, so pooling data across conversations with different Tp
    values is statistically valid -- no special handling required.
  - Only external change needed: one line in scenario_agent.py, routing
    SCENARIO to "ChangePointAgent" instead of "FittingAgent".
"""
import numpy as np
from scipy import stats

from agents.base import Agent
from reliability import weibull_censored_mle


class ChangePointAgent(Agent):
    name = "ChangePointAgent"

    def __init__(self, broker, alpha=0.01, min_buffer_batches=1, max_buffer_size=None):
        super().__init__(broker)
        self.alpha = alpha
        self.min_buffer_batches = min_buffer_batches
        self.max_buffer_size = max_buffer_size
        self.chi2_crit = stats.chi2.ppf(1 - alpha, df=2)

        # --- Adaptive Memory Buffer state ---
        self.buffer_failures = []
        self.buffer_censors = []
        self.buffer_batches = 0

        # --- Audit trail (feeds the "detection delay" metric) ---
        self.change_log = []

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "SCENARIO":
                self._handle_scenario(msg, tick)

    def _handle_scenario(self, msg, tick):
        p = msg.payload
        new_failures = np.array(p["failure_times"], dtype=float)
        new_censors = np.array(p["censor_times"], dtype=float)

        changed, lr_stat = self._detect_change(new_failures, new_censors)

        self.change_log.append({
            "conversation_id": msg.conversation_id,
            "lr_stat": lr_stat,
            "triggered": changed,
            "buffer_batches_before": self.buffer_batches,
        })

        if changed:
            self.buffer_failures = new_failures.tolist()
            self.buffer_censors = new_censors.tolist()
            self.buffer_batches = 1
        else:
            self.buffer_failures.extend(new_failures.tolist())
            self.buffer_censors.extend(new_censors.tolist())
            self.buffer_batches += 1
            self._enforce_cap()

        n_total = len(self.buffer_failures) + len(self.buffer_censors)
        payload = dict(p)
        payload["failure_times"] = list(self.buffer_failures)
        payload["censor_times"] = list(self.buffer_censors)
        payload["censoring_fraction"] = (
            len(self.buffer_censors) / n_total if n_total else 0.0
        )
        payload["buffer_batches"] = self.buffer_batches
        payload["change_detected"] = changed
        payload["lr_stat"] = lr_stat

        self.send("SCENARIO", "FittingAgent", msg.conversation_id, payload, tick)

    def _detect_change(self, new_failures, new_censors):
        if self.buffer_batches < self.min_buffer_batches or len(self.buffer_failures) < 2:
            return False, 0.0
        if len(new_failures) < 2:
            return False, 0.0

        buf_f = np.array(self.buffer_failures)
        buf_c = np.array(self.buffer_censors)

        try:
            _, _, ll_buf = weibull_censored_mle(buf_f, buf_c)
            _, _, ll_new = weibull_censored_mle(new_failures, new_censors)
        except Exception:
            return False, 0.0
        ll_h1 = ll_buf + ll_new

        pooled_f = np.concatenate([buf_f, new_failures])
        pooled_c = np.concatenate([buf_c, new_censors])
        try:
            _, _, ll_h0 = weibull_censored_mle(pooled_f, pooled_c)
        except Exception:
            return False, 0.0

        lr_stat = max(2.0 * (ll_h1 - ll_h0), 0.0)
        changed = lr_stat > self.chi2_crit
        return changed, float(lr_stat)

    def _enforce_cap(self):
        if self.max_buffer_size is None:
            return
        excess = len(self.buffer_failures) + len(self.buffer_censors) - self.max_buffer_size
        if excess <= 0:
            return
        n_drop_f = min(excess, len(self.buffer_failures))
        self.buffer_failures = self.buffer_failures[n_drop_f:]
        remaining = excess - n_drop_f
        if remaining > 0:
            self.buffer_censors = self.buffer_censors[remaining:]