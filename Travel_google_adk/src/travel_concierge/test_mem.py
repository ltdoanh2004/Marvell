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
import json

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


class MemoryService(BaseMemoryService):
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
            [part.text for event in event_list for part in event.content.parts if part.text is not None]
        )

        self.memory_system.add_note(
            content=event_content,
            category=user_key,
            tags=[session.app_name, session.user_id],
        )
    @override
    async def search_memory(self, *, app_name, user_id, query):
        """Search through memories"""
        response = SearchMemoryResponse()
        results = self.memory_system.search_agentic(query, k=10)
        print(f"Search results for query '{query}': {results}")
        for result in results:  
            response.memories.append(
              MemoryEntry(
                  content = Content(parts=[Part(text=result["content"])], role="model")
              ).model_dump()
          )
        return response

async def main():
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyCa9vwwIoufyZOK2n_Amww8pSdzrqKLDNo'
    APP_NAME = "memory_test_app"
    USER_ID = "test_user"
    MODEL = "gemini-2.0-flash"

    # Agent Definitions
    info_capture_agent = LlmAgent(
        model=MODEL,
        name="InfoCaptureAgent",
        instruction="Acknowledge the user's statement."
    )

    memory_recall_agent = LlmAgent(
        model=MODEL,
        name="MemoryRecallAgent",
        instruction="Answer the user's question. Use the 'load_memory' tool if the answer might be in past conversations.",
        tools=[load_memory]
    )

    parent_agent= LlmAgent(
        model=MODEL,
        name="ParentAgent",
        instruction="You are a parent agent that can delegate tasks to child agents. If the user provides information, delegate to the info_capture_agent. If the user asks a question that requires memory recall, delegate to the memory_recall_agent.",
        sub_agents=[info_capture_agent, memory_recall_agent]
    )
    session_service = InMemorySessionService()
    memory_service = MemoryService()

    runner = Runner(
        agent=parent_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )
    # Đọc các turn từ file JSON
    with open('Travel_google_adk/src/travel_concierge/turns.json', 'r') as f:
        turns = json.load(f)
    results = []
    for turn in turns:
        session_id = turn['session_id']
        print(turn['user_input'])
        user_input = Content(parts=[Part(text=turn['user_input'])], role="user")
        
        # Tạo session mới
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        final_response_text = "(No final response)"
        # Gửi yêu cầu và nhận kết quả từ agent (Memory Recall)
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_input):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            if event.content and event.content.parts and final_response_text != "(No final response)":
                # In ra kết quả trả về từ agent
                print(f"[{session_id}] Agent Response: {final_response_text}")

                # Lưu thông tin session vào memory
                completed_session = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                # Dưới đây, bạn có thể lưu vào memory service nếu cần
                await memory_service.add_session_to_memory(completed_session)
                results.append({
                "session_id": session_id,
                "user_input": turn['user_input'],
                "expected_output": turn['expected_output'],
                "agent_response": final_response_text
                })
    # memory_service = InMemoryMemoryService()  # Use in-memory for demo
    # runner = Runner(
    #     agent=parent_agent,
    #     app_name=APP_NAME,
    #     session_service=session_service,
    #     memory_service=memory_service
    # )
    # # Đọc các turn từ file JSON
    # with open('/Users/doa_ai/Developer/Marvell/Travel_google_adk/src/travel_concierge/turns.json', 'r') as f:
    #     turns = json.load(f)
    # results = []
    # # Vòng lặp qua các turn từ file JSON
    # for turn in turns:
    #     session_id = turn['session_id']
    #     print(turn['user_input'])
    #     user_input = Content(parts=[Part(text=turn['user_input'])], role="user")
        
    #     # Tạo session mới
    #     session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    #     final_response_text = "(No final response)"
    #     # Gửi yêu cầu và nhận kết quả từ agent (Memory Recall)
    #     async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_input):
    #         if event.is_final_response() and event.content and event.content.parts:
    #             final_response_text = event.content.parts[0].text
    #         if event.content and event.content.parts and final_response_text != "(No final response)":
    #             # In ra kết quả trả về từ agent
    #             print(f"[{session_id}] Agent Response: {final_response_text}")

    #             # Lưu thông tin session vào memory
    #             completed_session = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    #             # Dưới đây, bạn có thể lưu vào memory service nếu cần
    #             await memory_service.add_session_to_memory(completed_session)
    #             results.append({
    #             "session_id": session_id,
    #             "user_input": turn['user_input'],
    #             "expected_output": turn['expected_output'],
    #             "agent_response": final_response_text
    #             })
    #     # Sau khi chạy xong tất cả, ghi ra file JSON
    with open("agent_responses_amem.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Uncomment the following lines to run the scenario with multiple turns
    # Uncomment the following lines to run the scenario with multiple turns    
    # # --- Scenario ---
    # # Turn 1: User shares monthly revenue data
    # session1_id = "session1"
    # session1 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    # user_input1 = Content(parts=[Part(text="In January, the revenue was 100,000. In February, the revenue was 120,000.")], role="user")
    
    # final_response_text = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session1_id, new_message=user_input1):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text = event.content.parts[0].text
    # print(f"Agent 1 Response: {final_response_text}")
    
    # # Add session information to memory
    # completed_session1 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    # await memory_service.add_session_to_memory(completed_session1)

    # # Turn 2: User requests total revenue
    # session2_id = "session2"
    # session2 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)
    # user_input2 = Content(parts=[Part(text="What is the total revenue for January and February? If you do not have the data refer to the memory_recall_agnet")], role="user")

    # final_response_text_2 = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session2_id, new_message=user_input2):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text_2 = event.content.parts[0].text
    # print(f"Agent 2 Response: {final_response_text_2}")
    # completed_session2 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)
    # await memory_service.add_session_to_memory(completed_session2)
    # # Turn 3: User updates the revenue for March
    # session3_id = "session3"
    # session3 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session3_id)
    # user_input3 = Content(parts=[Part(text="In March, the revenue was 150,000.")], role="user")
    
    # final_response_text_3 = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session3_id, new_message=user_input3):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text_3 = event.content.parts[0].text
    # print(f"Agent 3 Response: {final_response_text_3}")
    
    # # Update session information with March revenue
    # completed_session3 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session3_id)
    # await memory_service.add_session_to_memory(completed_session3)

    # # Turn 4: User asks for average revenue
    # session4_id = "session4"
    # session4 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session4_id)
    # user_input4 = Content(parts=[Part(text="What is the average revenue for January, February, and March?")], role="user")

    # final_response_text_4 = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session4_id, new_message=user_input4):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text_4 = event.content.parts[0].text
    # print(f"Agent 4 Response: {final_response_text_4}")

    # # Turn 5: User asks for revenue growth between months
    # session5_id = "session5"
    # session5 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session5_id)
    # user_input5 = Content(parts=[Part(text="What is the revenue growth from January to February, and from February to March?")], role="user")

    # final_response_text_5 = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session5_id, new_message=user_input5):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text_5 = event.content.parts[0].text
    # print(f"Agent 5 Response: {final_response_text_5}")

    # # Turn 6: User asks about total revenue after the update
    # session6_id = "session6"
    # session6 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session6_id)
    # user_input6 = Content(parts=[Part(text="What is the total revenue after adding March's revenue?")], role="user")

    # final_response_text_6 = "(No final response)"
    # async for event in runner.run_async(user_id=USER_ID, session_id=session6_id, new_message=user_input6):
    #     if event.is_final_response() and event.content and event.content.parts:
    #         final_response_text_6 = event.content.parts[0].text
    # print(f"Agent 6 Response: {final_response_text_6}")


if __name__ == "__main__":
    asyncio.run(main())