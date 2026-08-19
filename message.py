"""
message.py
Structured JSON-style message used for all inter-agent communication.
Mirrors Section 4.3 of the paper: agents never call each other directly,
they publish Message objects to the Broker's outbox.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict
import itertools

_id_counter = itertools.count(1)


@dataclass
class Message:
    msg_type: str            # e.g. REQUEST_SCENARIO, SCENARIO, FIT_RESULT, POLICY, EXPLANATION
    sender: str
    recipient: str
    conversation_id: int
    payload: Dict[str, Any] = field(default_factory=dict)
    tick_created: int = 0
    tick_delivered: int = None
    msg_id: int = field(default_factory=lambda: next(_id_counter))

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self):
        return (f"<Msg#{self.msg_id} {self.msg_type} conv={self.conversation_id} "
                f"{self.sender}->{self.recipient} tick={self.tick_created}>")
