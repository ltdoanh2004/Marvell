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
from google.adk.tools import load_memory
from pydantic import BaseModel, Field
from . import prompt
from .sub_agents.academic_newresearch import academic_newresearch_agent
from .sub_agents.academic_websearch import academic_websearch_agent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService, BaseMemoryService
from google.adk.runners import Runner

from google.genai import types
from google.genai.types import Content, Part

from mem0 import Memory
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "embedding_model_dims": 1536,
        }
    },
    "version": "v1.1",
}
load_dotenv()
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
class OmemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.
  Uses keyword matching instead of semantic search.
  """
  def __init__(self):
    self.memory = Memory.from_config(config)
    self._session_events = {}
    """Keys are app_name/user_id, session_id. Values are session event lists."""
  def _user_key(self, app_name: str, user_id: str) -> str:
      return f"{app_name}/{user_id}"

  @override
  async def add_session_to_memory(self, session: Session):
    user_key = self._user_key(session.app_name, session.user_id)
    event_list = [event for event in session.events if event.content and event.content.parts]
    event_content = "\n".join(
      [part.text for event in event_list for part in event.content.parts if part.text is not None]
    )
    self._session_events[user_key] = self._session_events.get(user_key, {})
    print("[LOG] Adding session to memory:")
    # print("[LOG]", event_content)
    self.memory.add(
        user_id=user_key,
        messages=event_content,
        infer=False
    )
  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    user_key = self._user_key(app_name, user_id)
    print(f"[LOG] Searching memory for user {user_key} with query: {query}")
    if user_key not in self._session_events:
      return SearchMemoryResponse()
    
    response = SearchMemoryResponse()
    results = self.memory.search(
      query=query,
      user_id=user_key,
    )
    results = "\n".join(
        [f"{i+1}. {m['memory']}" for i, m in  enumerate(results['results'])]
    )
    response.memories.append(
              MemoryEntry(
                  content = Content(parts=[Part(text=results)], role="model")
              ).model_dump())
    return response


async def main():
    APP_NAME = "memory_example_app"
    USER_ID = "mem_user"
    MODEL = "gemini-2.0-flash" 
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    academic_coordinator = LlmAgent(
        name="academic_coordinator",
        model=MODEL,
        description=(
            "analyzing seminal papers provided by the users, "
            "providing research advice, locating current papers "
            "relevant to the seminal paper, generating suggestions "
            "for new research directions, and accessing web resources "
            "to acquire knowledge"
            "Answer the user's question. Use the 'load_memory' tool if the answer might be in past conversations."
        ),
        instruction=prompt.ACADEMIC_COORDINATOR_PROMPT,
        output_key="seminal_paper",
        tools=[
            AgentTool(agent=academic_websearch_agent),
            AgentTool(agent=academic_newresearch_agent),
            load_memory
        ],
    )


    session_service = InMemorySessionService()
    # memory_service = InMemoryMemoryService() # Use in-memory for demo
    memory_service = OmemoryService()
    runner = Runner(
        agent=academic_coordinator,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service # Provide the memory service to the Runner
    )
    # session1_id = "session1"
    # session1 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    # user_input1 = Content(parts=[Part(text="Hello")], role="user")
    
    # final_response_text = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session1_id, new_message=user_input1):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text = event.content.parts[0].text
    # await memory_service.add_session_to_memory(
    #         await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id))
    # print(f"Agent 1 Response: {final_response_text}")
    session_id = "cli_session"
    await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        user_message = Content(parts=[Part(text=user_input)], role="user")
        final_response_text = "(No final response)"
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_message):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
        await memory_service.add_session_to_memory(
            await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        )

        print(f"Agent: {final_response_text}")

if __name__ == "__main__":
    asyncio.run(main())