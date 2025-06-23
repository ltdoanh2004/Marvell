# Example runtime script (e.g., in scripts/run.py)
import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from agents import *
from models.travels import TravelQuery
import os
from dotenv import load_dotenv
# async def main():
#     runtime = SingleThreadedAgentRuntime()

#     await DestinationInfoAgent.register(runtime, "destination_info", lambda: DestinationInfoAgent())
#     await FlightCheckerAgent.register(runtime, "flight_checker", lambda: FlightCheckerAgent())
#     await CloserAgent.register(runtime, "closer", lambda: CloserAgent())
#     await DummyUserAgent.register(runtime, "user", lambda: DummyUserAgent())
#     runtime.start()

#     await runtime.send_message(TravelQuery(destination="Tokyo"), AgentId("closer", "default"), sender=AgentId("user", "session123"))
#     await runtime.stop_when_idle()

async def main():
    destination_agent = DestinationInfoAgent()
    flight_agent = FlightCheckerAgent()
    user_agent = DummyUserAgent() # Người thực gõ vào CLI
    model_client = OpenAIChatCompletionClient(model="gpt-4o")


    groupchat = MagenticOneGroupChat(
    participants=[user_agent, destination_agent, flight_agent],
    model_client=model_client,
    max_turns=8
    )
    await Console(groupchat.run_stream(task="finish confirm the tourist destination with customers"))


if __name__ == "__main__":
    asyncio.run(main())