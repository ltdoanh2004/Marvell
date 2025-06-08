from autogen_core import RoutedAgent, MessageContext, message_handler, default_subscription
from src.models import TravelQuery, TravelResponse
from autogen_core.messaging import Topic

@default_subscription(topic_type="flight_query")
class FlightCheckerAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Flight Checker Agent")

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext):
        # Simulate flight info
        content = f"[Flight Info] Flights to {message.destination} start from $250 round trip."
        await self.publish_message(
            TravelResponse(content=content, source="flight"),
            Topic("closer_reply", ctx.source.key)
        )