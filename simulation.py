"""
simulation.py

Runs either:

1. Paper baseline:
       ScenarioAgent -> FittingAgent -> OptimizerAgent

2. Proposed scheme:
       ScenarioAgent -> ChangePointAgent -> FittingAgent -> OptimizerAgent

The mathematical Weibull fitting and optimization are shared between both
modes so that experimental differences can be attributed to the adaptive
memory/change-point mechanism.
"""
from broker import Broker
from agents.scenario_agent import ScenarioAgent
from agents.change_point_agent import ChangePointAgent
from agents.fitting_agent import FittingAgent
from agents.optimizer_agent import OptimizerAgent
from agents.explainer_agent import ExplainerAgent
from agents.orchestrator_agent import OrchestratorAgent


class Simulation:
    def __init__(
        self,
        n_conversations=10,
        Tp_init=25.0,
        Cp_start=20.0,
        Cp_step=25.5,
        Cp_schedule=None,
        Cc=250.0,
        n_units=200,
        phase_switch_conv=6,
        beta_phase1=2.0,
        beta_phase2=0.6,
        eta_true=20.0,
        seed=42,
        max_ticks=500,
        use_change_point=False,
        alpha=0.01,
        min_buffer_batches=1,
        max_buffer_size=None,
        reset_on_change=True,
    ):
        self.broker = Broker()
        self.use_change_point = bool(use_change_point)

        if Cp_schedule is None:
            Cp_schedule = [
                Cp_start + i * Cp_step
                for i in range(n_conversations)
            ]

        self.orchestrator = OrchestratorAgent(
            self.broker,
            Tp_init,
            Cp_schedule,
            n_conversations,
        )

        self.scenario = ScenarioAgent(
            self.broker,
            n_units=n_units,
            eta_true=eta_true,
            beta_phase1=beta_phase1,
            beta_phase2=beta_phase2,
            phase_switch_conv=phase_switch_conv,
            seed=seed,
            use_change_point=self.use_change_point,
        )

        # Only instantiate/use the proposed mechanism when requested.
        self.change_point = None
        if self.use_change_point:
            self.change_point = ChangePointAgent(
                self.broker,
                alpha=alpha,
                min_buffer_batches=min_buffer_batches,
                max_buffer_size=max_buffer_size,
                reset_on_change=reset_on_change,
            )

        self.fitting = FittingAgent(self.broker)
        self.optimizer = OptimizerAgent(self.broker, Cc=Cc)
        self.explainer = ExplainerAgent(self.broker)

        self.agents = [
            self.orchestrator,
            self.scenario,
        ]

        if self.change_point is not None:
            self.agents.append(self.change_point)

        self.agents.extend([
            self.fitting,
            self.optimizer,
            self.explainer,
        ])

        self.max_ticks = max_ticks

    def run(self, verbose=False):
        tick = 0

        self.orchestrator.start_next_conversation(tick)

        while (
            not self.orchestrator.all_done
            and tick < self.max_ticks
        ):
            for agent in self.agents:
                agent.step(tick)

            self.broker.tick(tick)

            if (
                self.orchestrator.conversations_done
                == self.orchestrator.conversations_started
                and self.orchestrator.conversations_started
                < self.orchestrator.n_conversations
            ):
                self.orchestrator.start_next_conversation(tick + 1)

            if verbose:
                for m in self.broker.event_log:
                    if m.tick_created == tick:
                        print(f"  tick {tick}: {m}")

            tick += 1

        results = self.orchestrator.results

        # Attach the change-point audit trail without changing the optimizer.
        if self.change_point is not None:
            for result in results:
                conv_id = result["conversation_id"]
                matches = [
                    x for x in self.change_point.change_log
                    if x["conversation_id"] == conv_id
                ]
                if matches:
                    result["change_point"] = matches[-1]

        return results
