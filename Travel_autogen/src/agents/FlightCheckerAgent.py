from models.travels import TravelQuery, TravelResponse
from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()

from autogen_core import TopicId
from autogen_core import RoutedAgent, MessageContext, message_handler, type_subscription
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient

# @type_subscription(topic_type = "flight_query")
# class FlightCheckerAgent(RoutedAgent):
#     def __init__(self):
#         super().__init__("Flight Checker Agent")
#         self.model_context = BufferedChatCompletionContext(buffer_size=2)
#         self.model_client = OpenAIChatCompletionClient(
#             model = "gpt-4o-mini",
#             api_key=os.getenv("OPENAI_API_KEY"),
#         )
#         self.agent = AssistantAgent(
#             name="FlightCheckerAgent",
#             model_client=self.model_client,
#             model_context=self.model_context,
#             system_message="You are a flight checker agent. You provide information about flights based on user queries.",
#         )
#         self.cancellation_token = None

#     @message_handler
#     async def handle_message(self, message: TravelQuery, ctx: MessageContext) -> Any:
#         session = ctx.topic_id.source
#         promt = f"Check flights for {message.destination}. just imagine you have data based on destionation. I just want you reply like you have this information about flights. Do not mention that you are an AI or you do not have data. Just reply like you have this information."
#         response = await self.agent.on_messages(
#             [TextMessage(source="user", content=promt)],
#             self.cancellation_token
#         )
#         await self.publish_message(
#             TravelResponse(content=response.chat_message.content , source="flight"),
#             TopicId("closer_reply", session)
#         )
class FlightCheckerAgent(AssistantAgent):
    def __init__(self):
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        model_context = BufferedChatCompletionContext(buffer_size=2)

        super().__init__(
            name="FlightCheckerAgent",
            model_client=model_client,
            model_context=model_context,
            system_message=(
                "You are a flight checker agent. "
                "You help users find flight details based on their destination. "
                "Always respond as if you have the data and never mention you're an AI."
            ),
        )