"""
simulation.py
Section 4.4: executes the whole system on a discrete tick timeline. Each
tick, every agent reads its inbox, reacts, and (optionally) publishes new
messages; the Broker then advances outbox -> pending -> inbox with a
one-tick delay. A new conversation (full policy-evaluation cycle) is
started as soon as the previous one has finished, until n_conversations
have been completed. This reproduces the "digital maintenance analyst"
closed loop described in Sections 4-5.
"""
from broker import Broker
from agents.scenario_agent import ScenarioAgent
from agents.change_point_agent import ChangePointAgent
from agents.fitting_agent import FittingAgent
from agents.optimizer_agent import OptimizerAgent
from agents.explainer_agent import ExplainerAgent
from agents.orchestrator_agent import OrchestratorAgent


class Simulation:
    def __init__(self, n_conversations=10, Tp_init=25.0, Cp_start=20.0, Cp_step=25.5,
                 Cp_schedule=None, Cc=250.0, n_units=200, phase_switch_conv=6,
                 beta_phase1=2.0, beta_phase2=0.6, eta_true=20.0, seed=42, max_ticks=500,
                 alpha=0.01, min_buffer_batches=1, max_buffer_size=None):
        self.broker = Broker()
        if Cp_schedule is None:
            Cp_schedule = [Cp_start + i * Cp_step for i in range(n_conversations)]
        self.orchestrator = OrchestratorAgent(self.broker, Tp_init, Cp_schedule, n_conversations)
        self.scenario = ScenarioAgent(self.broker, n_units=n_units, eta_true=eta_true,
                                       beta_phase1=beta_phase1, beta_phase2=beta_phase2,
                                       phase_switch_conv=phase_switch_conv, seed=seed)
        self.change_point = ChangePointAgent(self.broker, alpha=alpha,
                                              min_buffer_batches=min_buffer_batches,
                                              max_buffer_size=max_buffer_size)
        self.fitting = FittingAgent(self.broker)
        self.optimizer = OptimizerAgent(self.broker, Cc=Cc)
        self.explainer = ExplainerAgent(self.broker)
        self.agents = [self.orchestrator, self.scenario, self.change_point,
                        self.fitting, self.optimizer, self.explainer]
        self.max_ticks = max_ticks

    def run(self, verbose=False):
        tick = 0
        # kick off the very first conversation
        self.orchestrator.start_next_conversation(tick)

        while not self.orchestrator.all_done and tick < self.max_ticks:
            for agent in self.agents:
                agent.step(tick)
            self.broker.tick(tick)

            # once the previous conversation has fully resolved, start the next one
            if (self.orchestrator.conversations_done == self.orchestrator.conversations_started
                    and self.orchestrator.conversations_started < self.orchestrator.n_conversations):
                self.orchestrator.start_next_conversation(tick + 1)

            if verbose:
                for m in self.broker.event_log:
                    if m.tick_created == tick:
                        print(f"  tick {tick}: {m}")
            tick += 1

        return self.orchestrator.results
