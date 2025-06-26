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

"""Academic_Research: Research advice, related literature finding, research area proposals, web knowledge access."""
import os
import json
import asyncio
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional
from typing_extensions import override
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from pydantic import BaseModel, Field
from . import prompt
from .sub_agents.academic_newresearch import academic_newresearch_agent
from .sub_agents.academic_websearch import academic_websearch_agent
from google.adk.sessions import InMemorySessionService, Session, BaseMemoryService, Event
from google.adk.memory import InMemoryMemoryService # Import MemoryService
from google.adk.runners import Runner

from google.genai import types
from google.genai.types import Content, Part

from mem0 import Memory

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
class InMemoryMemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.

  Uses keyword matching instead of semantic search.
  """

  def __init__(self):
    self._session_events: dict[str, dict[str, list[Event]]] = {}
    self
    """Keys are app_name/user_id, session_id. Values are session event lists."""
  def _user_key(self, app_name: str, user_id: str) -> str:
      return f"{app_name}/{user_id}"

  @override
  async def add_session_to_memory(self, session: Session):
    user_key = self._user_key(session.app_name, session.user_id)
    self._session_events[user_key] = self._session_events.get(
        self._user_key(session.app_name, session.user_id), {}
    )
    self._session_events[user_key][session.id] = [
        event
        for event in session.events
        if event.content and event.content.parts
    ]

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    user_key = self._user_key(app_name, user_id)
    if user_key not in self._session_events:
      return SearchMemoryResponse()

    words_in_query = set(query.lower().split())
    response = SearchMemoryResponse()

    for session_events in self._session_events[user_key].values():
      for event in session_events:
        if not event.content or not event.content.parts:
          continue
        words_in_event = _extract_words_lower(
            ' '.join([part.text for part in event.content.parts if part.text])
        )
        if not words_in_event:
          continue

        if any(query_word in words_in_event for query_word in words_in_query):
          response.memories.append(
              MemoryEntry(
                  content=event.content,
                  author=event.author,
                  timestamp=_utils.format_timestamp(event.timestamp),
              )
          )

    return response
APP_NAME = "memory_example_app"
USER_ID = "mem_user"
MODEL = "gemini-2.0-flash" 

async def main():
    academic_coordinator = LlmAgent(
        name="academic_coordinator",
        model=MODEL,
        description=(
            "analyzing seminal papers provided by the users, "
            "providing research advice, locating current papers "
            "relevant to the seminal paper, generating suggestions "
            "for new research directions, and accessing web resources "
            "to acquire knowledge"
        ),
        instruction=prompt.ACADEMIC_COORDINATOR_PROMPT,
        output_key="seminal_paper",
        tools=[
            AgentTool(agent=academic_websearch_agent),
            AgentTool(agent=academic_newresearch_agent),
        ],
    )


    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService() # Use in-memory for demo

    runner = Runner(
        # Start with the info capture agent
        agent=academic_coordinator,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service # Provide the memory service to the Runner
    )