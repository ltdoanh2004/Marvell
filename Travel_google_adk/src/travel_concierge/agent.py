# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration of Travel AI Conceirge using Agent Development Kit"""

from google.adk.agents import Agent
from google.adk.memory import BaseMemoryService

from travel_concierge import prompt

from travel_concierge.sub_agents.booking.agent import booking_agent
from travel_concierge.sub_agents.in_trip.agent import in_trip_agent
from travel_concierge.sub_agents.inspiration.agent import inspiration_agent
from travel_concierge.sub_agents.planning.agent import planning_agent
from travel_concierge.sub_agents.post_trip.agent import post_trip_agent
from travel_concierge.sub_agents.pre_trip.agent import pre_trip_agent

from travel_concierge.tools.memory_control import AgenticMemorySystem
from travel_concierge.tools.memory import _load_precreated_itinerary

import click
from google.adk.memory import BaseMemoryService
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService, Session
from typing_extensions import override

from pydantic import BaseModel
from pydantic import Field
from typing import Dict, List
import json
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
from typing import Optional
from google.genai import types
import re
from google.genai.types import Content, Part
from travel_concierge.tools.memory  import TravelMemoryService
from google.adk.runners import Runner
import asyncio


async def main():
    APP_NAME = "Travel Concierge"
    USER_ID = "user123"
    root_agent = Agent(
        model="gemini-2.0-flash-001",
        name="root_agent",
        description="A Travel Concierge using the services of multiple sub-agents",
        instruction=prompt.ROOT_AGENT_INSTR,
        sub_agents=[
            inspiration_agent,
            planning_agent,
            booking_agent,
            pre_trip_agent,
            in_trip_agent,
            post_trip_agent,
        ],
        before_agent_callback=_load_precreated_itinerary
    )
    
    session_service = InMemorySessionService()
    memory_service = TravelMemoryService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )

    session1_id = "test_memory"
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)

    while True:
        query = input('[user]: ')
        if not query or not query.strip():
            continue
        if query == 'exit':
            break

        # Tạo phần nội dung của người dùng
        user_input = types.Content(role='user', parts=[types.Part(text=query)])

        # Gửi yêu cầu và nhận phản hồi từ agent
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_input,
        ):
            if event.content and event.content.parts:
                if text := ''.join(part.text or '' for part in event.content.parts):
                    click.echo(f'[{event.author}]: {text}')
                
                # Lưu thông tin vào bộ nhớ sau mỗi lần trả lời
                await memory_service.add_session_to_memory(session)

    await runner.close()

if __name__ == "__main__":
    asyncio.run(main())