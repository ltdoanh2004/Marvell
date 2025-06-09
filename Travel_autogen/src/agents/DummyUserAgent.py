from autogen_core import MessageContext
from autogen_agentchat.agents import UserProxyAgent
from dataclasses import dataclass
from typing import Any

@dataclass
class UserMessage:
    content: str

# Fake message wrapper nếu cần xài agent chat logic
class DummyUserAgent(UserProxyAgent):
    def __init__(self):
        super().__init__(
            name="user",
            input_func=lambda prompt: input(f"[User Input Required] {prompt}\n> ")
        )

    async def on_message_impl(self, message: UserMessage, ctx: MessageContext) -> Any:
        print(f"[DummyUserAgent] Received message: {message.content}")
        return None
