"""
broker.py
Section 4.2/4.3: The Broker Agent manages asynchronous communication.
All messages are deposited in a global outbox and delivered to recipients
with a one-tick delay, enforcing loose coupling and full traceability.
"""
from collections import defaultdict, deque
from typing import List
from message import Message


class Broker:
    def __init__(self):
        self.outbox: List[Message] = []       # messages published this tick, awaiting delivery
        self.inboxes = defaultdict(deque)      # agent_name -> deque[Message] ready to be read
        self.pending = []                      # (deliver_at_tick, Message)
        self.event_log: List[Message] = []     # complete tick-by-tick history (Fig. 4)

    def publish(self, msg: Message, tick: int):
        """Agent deposits a message in the outbox."""
        msg.tick_created = tick
        self.outbox.append(msg)
        self.event_log.append(msg)

    def tick(self, current_tick: int):
        """
        Move outbox -> pending (one-tick delay) -> inbox, in that order.
        Called once per simulation tick by the Simulation driver.
        """
        # 1) anything that was pending from last tick becomes deliverable now
        still_pending = []
        for deliver_at, msg in self.pending:
            if current_tick >= deliver_at:
                msg.tick_delivered = current_tick
                self.inboxes[msg.recipient].append(msg)
            else:
                still_pending.append((deliver_at, msg))
        self.pending = still_pending

        # 2) everything published this tick goes into pending for delivery next tick
        for msg in self.outbox:
            self.pending.append((current_tick + 1, msg))
        self.outbox = []

    def receive(self, agent_name: str) -> List[Message]:
        """Agent drains its inbox."""
        msgs = list(self.inboxes[agent_name])
        self.inboxes[agent_name].clear()
        return msgs

    def history_for_conversation(self, conv_id: int) -> List[Message]:
        return [m for m in self.event_log if m.conversation_id == conv_id]
