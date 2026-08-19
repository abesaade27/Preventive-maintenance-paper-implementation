"""
fitting_agent.py
Section 4.2 / 5.2: performs maximum-likelihood estimation of the Weibull
parameters under right-censoring at every conversation, propagating the
updated reliability representation downstream. Purely a statistical
learning task -- no decision logic lives here.
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
        failure_times = np.array(p["failure_times"])
        censor_times = np.array(p["censor_times"])

        if len(failure_times) < 2:
            # degenerate case: not enough failures to fit reliably
            beta_hat, eta_hat = 1.5, max(np.mean(censor_times) if len(censor_times) else 20.0, 1.0)
            loglik = float("nan")
        else:
            beta_hat, eta_hat, loglik = weibull_censored_mle(failure_times, censor_times)

        mttf_hat = mean_time_to_failure(beta_hat, eta_hat)

        payload = {
            "beta_hat": beta_hat,
            "eta_hat": eta_hat,
            "mttf_hat": mttf_hat,
            "loglik": loglik,
            "n_failures": len(failure_times),
            "n_censored": len(censor_times),
            "censoring_fraction": p["censoring_fraction"],
            "beta_true": p["beta_true"],
            "eta_true": p["eta_true"],
            "Cp": p["Cp"],
            "Tp_used": p["Tp_used"],
            "phase": p["phase"],
        }
        self.send("FIT_RESULT", "OptimizerAgent", msg.conversation_id, payload, tick)
