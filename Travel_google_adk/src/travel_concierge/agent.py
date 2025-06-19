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
class MemoryEntry(BaseModel):
  """Represent one memory entry."""

  content: types.Content
  """The main content of the memory."""

  author: Optional[str] = None
  """The author of the memory."""

  timestamp: Optional[str] = None
  """The timestamp when the original content of this memory happened.

  This string will be forwarded to LLM. Preferred format is ISO 8601 format.
  """
class SearchMemoryResponse(BaseModel):
  """Represents the response from a memory search.

  Attributes:
      memories: A list of memory entries that relate to the search query.
  """

  memories: list[MemoryEntry] = Field(default_factory=list)


def _user_key(app_name: str, user_id: str):
  return f'{app_name}/{user_id}'


def _extract_words_lower(text: str) -> set[str]:
  """Extracts words from a string and converts them to lowercase."""
  return set([word.lower() for word in re.findall(r'[A-Za-z]+', text)])

class TravelMemoryService(BaseMemoryService):
    def __init__(self):
        self._session_events: dict[str, dict[str, list[Event]]] = {}
        self.travel_preferences: Dict = {}
        self.itineraries: Dict = {}
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model="gpt-4o-mini",
            api_key=api_key                  # LLM model name
        )
    @override
    async def add_session_to_memory(self, session):
        user_key = _user_key(session.app_name, session.user_id)
        event_list = [event for event in session.events if event.content and event.content.parts]
        event_text = "\n".join(
            [part.text for event in event_list for part in event.content.parts]
        )
        event_content = Content(
            parts=[Part(text=event_text)], role_user=session.user_id, role_assistant=session.app_name
        )
        self.memory_system.add_note(
            content=event_content,
            id=user_key,
            category="session",
            tags=["session", session.app_name, session.user_id],
            timestamp=session.created_at.strftime("%Y%m%d%H%M"),  # YYYYMM
        )

    @override
    async def search_memory(self, *, app_name, user_id, query):
        """Search through memories"""
        response = SearchMemoryResponse()
        results = self.memory_system.search_agentic(query, k=5)
        for result in results:  
            response.memories.append(
              MemoryEntry(
                  content=Content(result["content"]),
              )
          )
            break
        return response

    def store_preferences(self, user_id: str, preferences: Dict):
        """Store user travel preferences"""
        if user_id not in self.travel_preferences:
            self.travel_preferences[user_id] = {}
        self.travel_preferences[user_id].update(preferences)

    def store_itinerary(self, user_id: str, itinerary: Dict):
        """Store user itinerary"""
        if user_id not in self.itineraries:
            self.itineraries[user_id] = {}
        self.itineraries[user_id].update(itinerary)



# root_agent = Agent(
#     model="gemini-2.0-flash-001",
#     name="root_agent",
#     description="A Travel Concierge using the services of multiple sub-agents",
#     instruction=prompt.ROOT_AGENT_INSTR,
#     sub_agents=[
#         inspiration_agent,
#         planning_agent,
#         booking_agent,
#         pre_trip_agent,
#         in_trip_agent,
#         post_trip_agent,
#     ],
#     before_agent_callback=_load_precreated_itinerary,
#     session_service=InMemorySessionService(),
#     memory_service=TravelMemoryService()
# )
