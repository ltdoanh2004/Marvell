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

"""The 'memorize' tool for several agents to affect session states."""

from datetime import datetime
import json
import os
from typing import Dict, Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions.state import State
from google.adk.tools import ToolContext

from travel_concierge.shared_libraries import constants
from travel_concierge.tools.memory_control import AgenticMemorySystem
SAMPLE_SCENARIO_PATH = os.getenv(
    "TRAVEL_CONCIERGE_SCENARIO", "travel_concierge/profiles/itinerary_empty_default.json"
)


def memorize_list(key: str, value: str, tool_context: ToolContext):
    """
    Memorize pieces of information.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be stored.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    mem_dict = tool_context.state
    if key not in mem_dict:
        mem_dict[key] = []
    if value not in mem_dict[key]:
        mem_dict[key].append(value)
    return {"status": f'Stored "{key}": "{value}"'}


def memorize(key: str, value: str, tool_context: ToolContext):
    """
    Memorize pieces of information, one key-value pair at a time.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be stored.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    mem_dict = tool_context.state
    mem_dict[key] = value
    return {"status": f'Stored "{key}": "{value}"'}
# def memorize(key: str, value: str, tool_context: ToolContext):


def forget(key: str, value: str, tool_context: ToolContext):
    """
    Forget pieces of information.

    Args:
        key: the label indexing the memory to store the value.
        value: the information to be removed.
        tool_context: The ADK tool context.

    Returns:
        A status message.
    """
    if tool_context.state[key] is None:
        tool_context.state[key] = []
    if value in tool_context.state[key]:
        tool_context.state[key].remove(value)
    return {"status": f'Removed "{key}": "{value}"'}


def _set_initial_states(source: Dict[str, Any], target: State | dict[str, Any]):
    """
    Setting the initial session state given a JSON object of states.

    Args:
        source: A JSON object of states.
        target: The session state object to insert into.
    """
    if constants.SYSTEM_TIME not in target:
        target[constants.SYSTEM_TIME] = str(datetime.now())

    if constants.ITIN_INITIALIZED not in target:
        target[constants.ITIN_INITIALIZED] = True

        target.update(source)

        itinerary = source.get(constants.ITIN_KEY, {})
        if itinerary:
            target[constants.ITIN_START_DATE] = itinerary[constants.START_DATE]
            target[constants.ITIN_END_DATE] = itinerary[constants.END_DATE]
            target[constants.ITIN_DATETIME] = itinerary[constants.START_DATE]


def _load_precreated_itinerary(callback_context: CallbackContext):
    """
    Sets up the initial state.
    Set this as a callback as before_agent_call of the root_agent.
    This gets called before the system instruction is contructed.

    Args:
        callback_context: The callback context.
    """    
    data = {}
    with open(SAMPLE_SCENARIO_PATH, "r") as file:
        data = json.load(file)
        print(f"\nLoading Initial State: {data}\n")

    _set_initial_states(data["state"], callback_context.state)




from google.adk.memory import BaseMemoryService
from google.adk.events import Event
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
        print("Initialized TravelMemoryService with AgenticMemorySystem.")
    @override
    async def add_session_to_memory(self, session):
        user_key = _user_key(session.app_name, session.user_id)
        event_list = [event for event in session.events if event.content and event.content.parts]
        event_content = "\n".join(
            [part.text for event in event_list for part in event.content.parts]
        )
        print(f"Adding session content to memory for user {user_key}: {event_content}")
        self.memory_system.add_note(
            content=event_content,
            id=user_key,
            category="session",
            tags=["session", session.app_name, session.user_id],
        )

    @override
    async def search_memory(self, *, app_name, user_id, query):
        """Search through memories"""
        """If user wants to search for memories for called the experience in the past, this function will be called."""
        response = SearchMemoryResponse()
        results = self.memorxy_system.search_agentic(query, k=5)
        print(f"Search results for query '{query}': {results}")
        for result in results:  
            response.memories.append(
              MemoryEntry(
                  content = Content(parts=[Part(text=result["content"])], role="model")
              ).dict()
          )
            break
        return response