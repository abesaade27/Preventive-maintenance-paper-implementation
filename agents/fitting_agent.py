"""
fitting_agent.py

Weibull maximum-likelihood estimation under right censoring.

The statistical estimator itself is unchanged. The proposed scheme only
changes the dataset supplied to this agent: in baseline mode it receives the
current scenario batch; in proposed mode it receives the ChangePointAgent's
adaptive-memory dataset.
"""
import numpy as np

from agents.base import Agent
from reliability import weibull_censored_mle, mean_time_to_failure


class FittingAgent(Agent):
    name = "FittingAgent"

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "SCENARIO":
                self._handle_scenario(msg, tick)

    def _handle_scenario(self, msg, tick):
        p = msg.payload

        failure_times = np.asarray(p.get("failure_times", []), dtype=float)
        censor_times = np.asarray(p.get("censor_times", []), dtype=float)

        if len(failure_times) < 2:
            beta_hat = 1.5
            eta_hat = max(
                np.mean(censor_times) if len(censor_times) else 20.0,
                1.0,
            )
            loglik = float("nan")
        else:
            beta_hat, eta_hat, loglik = weibull_censored_mle(
                failure_times,
                censor_times,
            )

        mttf_hat = mean_time_to_failure(beta_hat, eta_hat)

        payload = {
            "beta_hat": float(beta_hat),
            "eta_hat": float(eta_hat),
            "mttf_hat": float(mttf_hat),
            "loglik": float(loglik) if np.isfinite(loglik) else loglik,
            "n_failures": int(len(failure_times)),
            "n_censored": int(len(censor_times)),
            "n_observations": int(len(failure_times) + len(censor_times)),
            "censoring_fraction": p.get("censoring_fraction", 0.0),
            "beta_true": p["beta_true"],
            "eta_true": p["eta_true"],
            "Cp": p["Cp"],
            "Tp_used": p["Tp_used"],
            "phase": p["phase"],

            # Experimental provenance for the proposed scheme.
            "buffer_size": p.get("buffer_size", len(failure_times) + len(censor_times)),
            "buffer_batches": p.get("buffer_batches", 1),
            "change_detected": p.get("change_detected", False),
            "change_point_lr": p.get("change_point_lr", 0.0),
            "change_point_critical": p.get("change_point_critical", 0.0),
            "regime_id": p.get("regime_id", 0),
        }

        self.send(
            "FIT_RESULT",
            "OptimizerAgent",
            msg.conversation_id,
            payload,
            tick,
        )
