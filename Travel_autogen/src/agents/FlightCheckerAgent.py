from autogen_core import RoutedAgent, MessageContext, message_handler, default_subscription, type_subscription
from models.travels import TravelQuery, TravelResponse
from autogen_core import TopicId
from typing import Any

@type_subscription(topic_type = "flight_query")
class FlightCheckerAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Flight Checker Agent")

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext) -> Any:
        # Simulate flight info
        session = ctx.topic_id
        content = f"[Flight Info] Flights to {message.destination} start from $250 round trip."
        await self.publish_message(
            TravelResponse(content=content, source="flight"),
            TopicId("closer_reply", session)  # Respond back to the closer agent with the session
        )