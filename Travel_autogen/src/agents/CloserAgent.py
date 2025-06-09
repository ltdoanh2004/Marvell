from dataclasses import dataclass
from autogen_core import RoutedAgent, MessageContext, message_handler, default_subscription, type_subscription
from autogen_core import TopicId, DefaultTopicId
from models.travels import TravelQuery, TravelResponse
from typing import Any

@type_subscription(topic_type="closer_reply")
class CloserAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Closer Agent")
        self.responses = {}

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext) -> Any:
        session = ctx.sender.key
        self.responses[session] = {}

        await self.publish_message(message, TopicId("destination_query", session))
        await self.publish_message(message, TopicId("flight_query", session))

    @message_handler
    async def handle_info_response(self, message: TravelResponse, ctx: MessageContext) -> Any:
        session = ctx.topic_id.source
        self.responses.setdefault(session, {})[message.source] = message.content

        if len(self.responses[session]) == 2:
            combined = "\n".join([
                self.responses[session].get("destination", ""),
                self.responses[session].get("flight", "")
            ])
            print(f"[CloserAgent] Final suggestion to user:\n{combined}")

