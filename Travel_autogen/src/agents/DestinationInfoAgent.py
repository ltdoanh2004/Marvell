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

# @type_subscription(topic_type="destination_query")
# class DestinationInfoAgent(RoutedAgent):
#     def __init__(self):
#         super().__init__("Destination Info Agent")
#         self.model_context = BufferedChatCompletionContext(buffer_size=2)
#         self.model_client = OpenAIChatCompletionClient(
#             model = "gpt-4o-mini",
#             api_key=os.getenv("OPENAI_API_KEY"),
#         )
#         self.agent = AssistantAgent(
#             name="DestinationInfoAgent",
#             model_client=self.model_client,
#             model_context=self.model_context,
#             system_message="You are a destination info agent. You provide information about travel destinations based on user queries.",
#         )
#         self.cancellation_token = None
#     @message_handler
#     async def handle_message(self, message: TravelQuery, ctx: MessageContext) -> Any:
#         session  = ctx.topic_id.source
#         promt = f"Provide information about the destination: {message.destination}. Just imagine you have data based on destination. I just want you to reply like you have this information about the destination. Do not mention that you are an AI or you do not have data. Just reply like you have this information."
#         response = await self.agent.on_messages(
#             [TextMessage(source="user", content=promt)],
#             self.cancellation_token
#         )
#         await self.publish_message(
#             TravelResponse(content=response.chat_message.content , source="destination"),
#             TopicId("closer_reply", session)
#         )
class DestinationInfoAgent(AssistantAgent):
    def __init__(self):
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        model_context = BufferedChatCompletionContext(buffer_size=2)

        super().__init__(
            name="DestinationInfoAgent",
            model_client=model_client,
            model_context=model_context,
            system_message=(
                "You are a destination info agent. "
                "You provide travel destination insights like weather, culture, and activities. "
                "Always answer with confidence, do not mention being an AI."
            ),
        )