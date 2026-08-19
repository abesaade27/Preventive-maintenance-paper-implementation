"""
scenario_agent.py
Section 4.2 / 5.1: simulates a population of assets under an age-based PM
policy with interval Tp. Units surviving to Tp are right-censored. The true
Weibull shape beta undergoes a regime change halfway through the run:
Phase 1 (wear-out, beta=2.0) -> Phase 2 (infant-mortality-like, beta=0.6).
In an industrial deployment this agent is replaced 1:1 by a CMMSDataAgent
that reconstructs the same structure from real work-order history (Sec. 7.1).
"""
import numpy as np
from agents.base import Agent


class ScenarioAgent(Agent):
    name = "ScenarioAgent"

    def __init__(self, broker, n_units=200, eta_true=20.0,
                 beta_phase1=2.0, beta_phase2=0.6, phase_switch_conv=6, seed=42):
        super().__init__(broker)
        self.n_units = n_units
        self.eta_true = eta_true
        self.beta_phase1 = beta_phase1
        self.beta_phase2 = beta_phase2
        self.phase_switch_conv = phase_switch_conv
        self.rng = np.random.default_rng(seed)

    def true_beta_for(self, conv_id: int) -> float:
        return self.beta_phase1 if conv_id < self.phase_switch_conv else self.beta_phase2

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "REQUEST_SCENARIO":
                self._handle_request(msg, tick)

    def _handle_request(self, msg, tick):
        conv_id = msg.conversation_id
        Tp = msg.payload["Tp"]
        Cp = msg.payload["Cp"]
        beta_true = self.true_beta_for(conv_id)

        raw_times = self.rng.weibull(beta_true, self.n_units) * self.eta_true
        failure_times = raw_times[raw_times <= Tp]
        n_censored = int(np.sum(raw_times > Tp))
        censor_times = np.full(n_censored, Tp)

        payload = {
            "Tp_used": Tp,
            "Cp": Cp,
            "failure_times": failure_times.tolist(),
            "censor_times": censor_times.tolist(),
            "beta_true": beta_true,
            "eta_true": self.eta_true,
            "censoring_fraction": n_censored / self.n_units,
            "phase": "Phase1" if conv_id < self.phase_switch_conv else "Phase 2",
        }

        # before:
        self.send("SCENARIO", "FittingAgent", conv_id, payload, tick)
        # after:
        self.send("SCENARIO", "ChangePointAgent", conv_id, payload, tick)