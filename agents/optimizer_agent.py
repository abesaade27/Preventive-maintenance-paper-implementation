"""
optimizer_agent.py
Section 4.2 / 5.3: the economic decision-maker. Evaluates calendar-based
and operating-time-based cost-time efficiency criteria over candidate
preventive intervals, selects the dominant (lower cost-rate) policy, and
reports the decision to both the Orchestrator and the Explainer.
"""
from agents.base import Agent
from reliability import optimize_policy


class OptimizerAgent(Agent):
    name = "OptimizerAgent"

    def __init__(self, broker, Cc=250.0):
        super().__init__(broker)
        self.Cc = Cc  # corrective unit cost, held fixed; Cp evolves across conversations

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "FIT_RESULT":
                self._handle_fit(msg, tick)

    def _handle_fit(self, msg, tick):
        p = msg.payload
        result = optimize_policy(p["beta_hat"], p["eta_hat"], p["Cp"], self.Cc)

        payload = dict(p)  # carry forward reliability + cost context for the Explainer
        payload.update(result)
        payload["Cc"] = self.Cc

        conv_id = msg.conversation_id
        self.send("POLICY", "OrchestratorAgent", conv_id, payload, tick)
        self.send("POLICY", "ExplainerAgent", conv_id, payload, tick)
