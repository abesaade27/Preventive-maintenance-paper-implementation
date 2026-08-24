"""base.py -- shared agent scaffolding."""
from message import Message


class Agent:
    name = "BaseAgent"

    def __init__(self, broker):
        self.broker = broker

    def send(self, msg_type, recipient, conversation_id, payload, tick):
        msg = Message(msg_type=msg_type, sender=self.name, recipient=recipient,
                      conversation_id=conversation_id, payload=payload)
        self.broker.publish(msg, tick)
        return msg

    def receive(self):
        return self.broker.receive(self.name)

    def step(self, tick: int):
        """Read inbox and react. Override in subclasses."""
        raise NotImplementedError
