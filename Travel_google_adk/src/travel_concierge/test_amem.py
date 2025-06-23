import asyncio
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService # Import MemoryService
from google.adk.runners import Runner
from google.adk.tools import load_memory # Tool to query memory
from google.genai.types import Content, Part
import os
import sys
sys.path.append('/Users/doa_ai/Developer/Marvell/Travel_google_adk/src')  # xAdjust path as needed
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
        response = SearchMemoryResponse()
        results = self.memory_system.search_agentic(query, k=5)
        print(f"Search results for query '{query}': {results}")
        for result in results:  
            response.memories.append(
              MemoryEntry(
                  content = Content(parts=[Part(text=result["content"])], role="model")
              ).dict()
          )
            break
        return response


async def main():
    # --- Constants ---
    APP_NAME = "memory_example_app"
    USER_ID = "mem_user"
    MODEL = "gemini-2.0-flash" # Use a valid model

    # --- Agent Definitions ---
    # Agent 1: Simple agent to capture information
    info_capture_agent = LlmAgent(
        model=MODEL,
        name="InfoCaptureAgent",
        instruction="Acknowledge the user's statement.",
        # output_key="captured_info" # Could optionally save to state too
    )

    # Agent 2: Agent that can use memory
    memory_recall_agent = LlmAgent(
        model=MODEL,
        name="MemoryRecallAgent",
        instruction="Answer the user's question. Use the 'load_memory' tool "
                    "if the answer might be in past conversations.",
        tools=[load_memory] # Give the agent the tool
    )

    # --- Services and Runner ---
    session_service = InMemorySessionService()
    memory_service = TravelMemoryService() # Use in-memory for demo

    runner = Runner(
        # Start with the info capture agent
        agent=info_capture_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service # Provide the memory service to the Runner
    )

# --- Scenario ---
    # Turn 1: Capture some information in a session
    print("--- Turn 1: Capturing Information ---")
    session1_id = "session_info"
    session1 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    user_input1 = Content(parts=[Part(text="My favorite project is Project Alpha.")], role="user")

    # Run the agent
    final_response_text = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session1_id, new_message=user_input1):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text = event.content.parts[0].text
    print(f"Agent 1 Response: {final_response_text}")

    # Get the completed session
    completed_session1 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)

    # Add this session's content to the Memory Service
    print("\n--- Adding Session 1 to Memory ---")
    memory_service = await memory_service.add_session_to_memory(completed_session1)
    print("Session added to memory.")

    # Turn 2: In a *new* (or same) session, ask a question requiring memory
    print("\n--- Turn 2: Recalling Information ---")
    session2_id = "session_recall" # Can be same or different session ID
    session2 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)

    # Switch runner to the recall agent
    runner.agent = memory_recall_agent
    user_input2 = Content(parts=[Part(text="What is my favorite project?")], role="user")

    # Run the recall agent
    print("Running MemoryRecallAgent...")
    final_response_text_2 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session2_id, new_message=user_input2):
        print(f"  Event: {event.author} - Type: {'Text' if event.content and event.content.parts and event.content.parts[0].text else ''}"
            f"{'FuncCall' if event.get_function_calls() else ''}"
            f"{'FuncResp' if event.get_function_responses() else ''}")
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_2 = event.content.parts[0].text
            print(f"Agent 2 Final Response: {final_response_text_2}")
            break # Stop after final response

    # Expected Event Sequence for Turn 2:
    # 1. User sends "What is my favorite project?"
# 2. Agent (LLM) decides to call `load_memory` tool with a query like "favorite project".
# 3. Runner executes the `load_memory` tool, which calls `memory_service.search_memory`.
# 4. `InMemoryMemoryService` finds the relevant text ("My favorite project is Project Alpha.") from session1.
# 5. Tool returns this text in a FunctionResponse event.
# 6. Agent (LLM) receives the function response, processes the retrieved text.
# 7. Agent generates the final answer (e.g., "Your favorite project is Project Alpha.").
if __name__ == "__main__":
    asyncio.run(main())