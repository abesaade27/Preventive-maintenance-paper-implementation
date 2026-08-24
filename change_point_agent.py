"""
change_point_agent.py

Proposed research enhancement:
    Change-Point-Gated Adaptive Memory.

For each new scenario batch:
1. Test whether the new batch is statistically different from the current
   memory buffer using a likelihood-ratio test on censored Weibull data.
2. If no change is detected, append the new observations to the buffer.
3. If a change is detected, discard stale pre-change observations and start
   a new regime buffer using the new batch.
4. Forward ONLY the resulting adaptive-memory dataset to FittingAgent.

The implementation keeps failure/censor observations as individual records
so a buffer cap removes the oldest observations without separately truncating
the failure and censor arrays.
"""
import numpy as np
from scipy import stats

from agents.base import Agent
from reliability import weibull_censored_mle


class ChangePointAgent(Agent):
    name = "ChangePointAgent"

    def __init__(
        self,
        broker,
        alpha=0.01,
        min_buffer_batches=1,
        max_buffer_size=None,
        reset_on_change=True,
    ):
        super().__init__(broker)

        self.alpha = float(alpha)
        self.min_buffer_batches = int(min_buffer_batches)
        self.max_buffer_size = (
            None if max_buffer_size is None else int(max_buffer_size)
        )
        self.reset_on_change = bool(reset_on_change)

        # LR has df=2 because H1 estimates two additional Weibull parameters.
        self.chi2_crit = float(stats.chi2.ppf(1.0 - self.alpha, df=2))

        # Each record is {"time": float, "event": 1 failure / 0 censored}.
        self.buffer = []
        self.buffer_batches = 0
        self.regime_id = 0

        # Full audit trail for experiments/dashboard.
        self.change_log = []

    @property
    def buffer_failures(self):
        return [r["time"] for r in self.buffer if r["event"] == 1]

    @property
    def buffer_censors(self):
        return [r["time"] for r in self.buffer if r["event"] == 0]

    def reset(self):
        self.buffer = []
        self.buffer_batches = 0
        self.regime_id = 0
        self.change_log = []

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "SCENARIO":
                self._handle_scenario(msg, tick)

    def _batch_to_records(self, failures, censors):
        records = (
            [{"time": float(t), "event": 1} for t in failures]
            + [{"time": float(t), "event": 0} for t in censors]
        )
        return records

    def _records_to_arrays(self, records):
        failures = np.asarray(
            [r["time"] for r in records if r["event"] == 1],
            dtype=float,
        )
        censors = np.asarray(
            [r["time"] for r in records if r["event"] == 0],
            dtype=float,
        )
        return failures, censors

    def _handle_scenario(self, msg, tick):
        p = msg.payload
        new_failures = np.asarray(p.get("failure_times", []), dtype=float)
        new_censors = np.asarray(p.get("censor_times", []), dtype=float)
        new_records = self._batch_to_records(new_failures, new_censors)

        changed, lr_stat = self._detect_change(new_failures, new_censors)

        old_buffer_size = len(self.buffer)

        if changed and self.reset_on_change:
            self.buffer = list(new_records)
            self.buffer_batches = 1
            self.regime_id += 1
        else:
            self.buffer.extend(new_records)
            self.buffer_batches += 1
            self._enforce_cap()

        if self.regime_id == 0:
            self.regime_id = 1

        failures, censors = self._records_to_arrays(self.buffer)
        n_total = len(failures) + len(censors)

        detection_record = {
            "conversation_id": msg.conversation_id,
            "lr_stat": float(lr_stat),
            "critical_value": self.chi2_crit,
            "triggered": bool(changed),
            "buffer_size_before": old_buffer_size,
            "buffer_size_after": len(self.buffer),
            "buffer_batches": self.buffer_batches,
            "regime_id": self.regime_id,
            "new_failures": len(new_failures),
            "new_censored": len(new_censors),
        }
        self.change_log.append(detection_record)

        payload = dict(p)
        payload["failure_times"] = failures.tolist()
        payload["censor_times"] = censors.tolist()
        payload["censoring_fraction"] = (
            len(censors) / n_total if n_total else 0.0
        )
        payload["buffer_size"] = len(self.buffer)
        payload["buffer_batches"] = self.buffer_batches
        payload["change_detected"] = bool(changed)
        payload["change_point_lr"] = float(lr_stat)
        payload["change_point_critical"] = self.chi2_crit
        payload["regime_id"] = self.regime_id

        # This is the ONLY downstream path in proposed mode.
        self.send(
            "SCENARIO",
            "FittingAgent",
            msg.conversation_id,
            payload,
            tick,
        )

    def _detect_change(self, new_failures, new_censors):
        # No historical regime exists yet.
        if self.buffer_batches < self.min_buffer_batches:
            return False, 0.0

        # A very small sample cannot support a meaningful two-model comparison.
        if len(self.buffer_failures) < 2 or len(new_failures) < 2:
            return False, 0.0

        buf_f, buf_c = self._records_to_arrays(self.buffer)

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
        return bool(changed), float(lr_stat)

    def _enforce_cap(self):
        if self.max_buffer_size is None:
            return

        if self.max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1 or None.")

        excess = len(self.buffer) - self.max_buffer_size
        if excess > 0:
            # Remove oldest observations first.
            del self.buffer[:excess]
