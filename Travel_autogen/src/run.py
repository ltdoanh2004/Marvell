# Example runtime script (e.g., in scripts/run.py)
import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId

from agents import *
from models.travels import TravelQuery

async def main():
    runtime = SingleThreadedAgentRuntime()

    await DestinationInfoAgent.register(runtime, "destination_info", lambda: DestinationInfoAgent())
    await FlightCheckerAgent.register(runtime, "flight_checker", lambda: FlightCheckerAgent())
    await CloserAgent.register(runtime, "closer", lambda: CloserAgent())
    await DummyUserAgent.register(runtime, "user", lambda: DummyUserAgent())
    runtime.start()

    await runtime.send_message(TravelQuery(destination="Tokyo"), AgentId("closer", "default"), sender=AgentId("user", "session123"))
    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())