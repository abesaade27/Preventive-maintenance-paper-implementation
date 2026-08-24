"""
scenario_agent.py

Scenario/data-generation agent.

Baseline mode:
    ScenarioAgent -> FittingAgent

Proposed mode:
    ScenarioAgent -> ChangePointAgent -> FittingAgent

The baseline path is kept unchanged conceptually so that the original
paper implementation remains a clean control. The proposed path routes
the raw scenario only through the ChangePointAgent, whose adaptive buffer
then supplies the fitting dataset.
"""
import numpy as np
from agents.base import Agent


class ScenarioAgent(Agent):
    name = "ScenarioAgent"

    def __init__(
        self,
        broker,
        n_units=200,
        eta_true=20.0,
        beta_phase1=2.0,
        beta_phase2=0.6,
        phase_switch_conv=6,
        seed=42,
        use_change_point=False,
    ):
        super().__init__(broker)
        self.n_units = int(n_units)
        self.eta_true = float(eta_true)
        self.beta_phase1 = float(beta_phase1)
        self.beta_phase2 = float(beta_phase2)
        self.phase_switch_conv = int(phase_switch_conv)
        self.rng = np.random.default_rng(seed)
        self.use_change_point = bool(use_change_point)

    def true_beta_for(self, conv_id: int) -> float:
        return (
            self.beta_phase1
            if conv_id < self.phase_switch_conv
            else self.beta_phase2
        )

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "REQUEST_SCENARIO":
                self._handle_request(msg, tick)

    def _handle_request(self, msg, tick):
        conv_id = msg.conversation_id
        Tp = float(msg.payload["Tp"])
        Cp = float(msg.payload["Cp"])
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
            "phase": "Phase 1" if conv_id < self.phase_switch_conv else "Phase 2",
            "n_units": self.n_units,
            "regime_id": 1 if conv_id < self.phase_switch_conv else 2,
        }

        # IMPORTANT:
        # In proposed mode the raw batch must NOT also go directly to
        # FittingAgent. Otherwise the fitter receives both the raw batch
        # and the buffered batch, defeating the adaptive-memory design.
        recipient = "ChangePointAgent" if self.use_change_point else "FittingAgent"
        self.send("SCENARIO", recipient, conv_id, payload, tick)
