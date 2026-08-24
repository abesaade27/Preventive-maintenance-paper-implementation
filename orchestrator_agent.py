"""
orchestrator_agent.py
Section 4.2 / 5.5: initiates each conversation (REQUEST_SCENARIO), records
the conversation-level results once POLICY + EXPLANATION arrive, and closes
the adaptive feedback loop by updating the preventive interval Tp used in
the next conversation. This is the only agent with cross-conversation state.
"""
from agents.base import Agent


class OrchestratorAgent(Agent):
    name = "OrchestratorAgent"

    def __init__(self, broker, Tp_init, Cp_schedule, n_conversations):
        super().__init__(broker)
        self.Tp_current = Tp_init
        self.Cp_schedule = Cp_schedule  # list of Cp values, one per conversation
        self.n_conversations = n_conversations
        self.results = []               # one dict per completed conversation
        self._pending_policy = {}       # conv_id -> policy payload, waiting on explanation
        self.conversations_started = 0
        self.conversations_done = 0

    def start_next_conversation(self, tick):
        if self.conversations_started >= self.n_conversations:
            return None
        conv_id = self.conversations_started + 1
        Cp = self.Cp_schedule[self.conversations_started]
        payload = {"Tp": self.Tp_current, "Cp": Cp}
        self.send("REQUEST_SCENARIO", "ScenarioAgent", conv_id, payload, tick)
        self.conversations_started += 1
        return conv_id

    def step(self, tick):
        for msg in self.receive():
            if msg.msg_type == "POLICY":
                self._pending_policy[msg.conversation_id] = msg.payload
            elif msg.msg_type == "EXPLANATION":
                policy = self._pending_policy.pop(msg.conversation_id, None)
                if policy is None:
                    continue
                record = dict(policy)
                record["conversation_id"] = msg.conversation_id
                record["explanation"] = msg.payload["text"]
                self.results.append(record)
                self.conversations_done += 1
                # close the feedback loop: revised interval feeds the next conversation
                self.Tp_current = record["Tp_star"]

    @property
    def all_done(self):
        return self.conversations_done >= self.n_conversations
