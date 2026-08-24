"""
explainer_agent.py
Section 4.5 / 5.4: transforms deterministic optimization outputs into a
traceable, human-readable explanation. No policy evaluation happens here.
In the academic prototype the LLM call is stubbed (a template fills in the
structured facts); in a real deployment this same structured payload would
be handed to an LLM (Section 7, "LLM" component) to produce natural language.
State machine (Fig. 5): IDLE -> RECEIVED -> BUILDING_CONTEXT -> EXPLAINED
"""
from agents.base import Agent


class ExplainerAgent(Agent):
    name = "ExplainerAgent"

    def __init__(self, broker):
        super().__init__(broker)
        self.state = "IDLE"

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "POLICY":
                self._explain(msg, tick)

    def _explain(self, msg, tick):
        self.state = "RECEIVED"
        p = msg.payload
        self.state = "BUILDING_CONTEXT"

        censoring_pct = p["censoring_fraction"] * 100
        explanation = (
            f"[Conversation {msg.conversation_id}, phase {p['phase']}] "
            f"Re-estimated reliability from {p['n_failures']} failures and "
            f"{p['n_censored']} censored units ({censoring_pct:.1f}% censoring): "
            f"beta_hat={p['beta_hat']:.3f}, eta_hat={p['eta_hat']:.3f} "
            f"(true beta={p['beta_true']:.2f}, eta={p['eta_true']:.2f}). "
            f"Under Cp={p['Cp']:.1f}, Cc={p['Cc']:.1f}, the operating-time-based "
            f"policy reaches minimum cost-time-efficiency {p['CTE_op']:.3f} at "
            f"Tp={p['Tp_op']:.2f}, while the calendar-based policy reaches "
            f"{p['CTE_cal']:.3f} at Tp={p['Tp_cal']:.2f}. "
            f"Selected policy: {p['dominant_policy']} with Tp*={p['Tp_star']:.2f} "
            f"(CTE*={p['CTE_star']:.3f})."
        )
        self.state = "EXPLAINED"

        self.send("EXPLANATION", "OrchestratorAgent", msg.conversation_id,
                   {"text": explanation}, tick)
