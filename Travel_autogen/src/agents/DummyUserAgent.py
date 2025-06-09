from autogen_core import BaseAgent, MessageContext
from dataclasses import dataclass

@dataclass
class UserMessage:
    content: str

class DummyUserAgent(BaseAgent):
    def __init__(self):
        super().__init__("DummyUserAgent")

    async def on_message_impl(self, message: UserMessage, ctx: MessageContext):
        print(f"[DummyUserAgent] Received message: {message}")
        return None
