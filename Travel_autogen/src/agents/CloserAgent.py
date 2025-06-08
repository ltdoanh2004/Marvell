from dataclasses import dataclass
from autogen_core import RoutedAgent, MessageContext, message_handler, default_subscription
from autogen_core.messaging import Topic, DefaultTopicId
from src.models import TravelQuery, TravelResponse

@default_subscription
class CloserAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Closer Agent")
        self.responses = {}

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext):
        self.responses[ctx.source.key] = {}

        # Broadcast to both destination info and flight checker
        await self.publish_message(message, Topic("destination_query", ctx.source.key))
        await self.publish_message(message, Topic("flight_query", ctx.source.key))

    @message_handler
    async def handle_info_response(self, message: TravelResponse, ctx: MessageContext):
        self.responses[ctx.source.key][message.source] = message.content
        
        if len(self.responses[ctx.source.key]) == 2:
            combined = "\n".join([
                self.responses[ctx.source.key].get("destination", ""),
                self.responses[ctx.source.key].get("flight", "")
            ])
            print(f"[CloserAgent] Final suggestion to user:\n{combined}")
