from autogen_core import RoutedAgent, MessageContext, message_handler, default_subscription
from src.models import TravelQuery, TravelResponse
from autogen_core.messaging import Topic

@default_subscription(topic_type="destination_query")
class DestinationInfoAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Destination Info Agent")

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext):
        # Simulate destination info (in real app: use GPT or API)
        content = f"[Destination Info] {message.destination} is sunny, cultural, and has great food."
        await self.publish_message(
            TravelResponse(content=content, source="destination"),
            Topic("closer_reply", ctx.source.key)
        )