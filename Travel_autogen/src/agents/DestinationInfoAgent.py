from autogen_core import RoutedAgent, MessageContext, message_handler, type_subscription, TopicId
from models.travels import TravelQuery, TravelResponse
from typing import Any

@type_subscription(topic_type="destination_query")
class DestinationInfoAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Destination Info Agent")

    @message_handler
    async def handle_message(self, message: TravelQuery, ctx: MessageContext) -> Any:
        # Simulate destination info
        content = f"[Destination Info] {message.destination} is sunny, cultural, and has great food."
        
        # Get session key from topic source
        session = ctx.topic_id  # ✅ đúng cách lấy session
        await self.publish_message(
            TravelResponse(content=content, source="destination"),
            TopicId("closer_reply", session)  # Phản hồi lại đúng session
        )
