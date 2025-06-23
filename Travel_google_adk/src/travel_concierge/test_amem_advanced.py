import os
import json
import asyncio
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional
from typing_extensions import override
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService, BaseMemoryService
from google.adk.runners import Runner
from google.adk.tools import load_memory
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm

from google.genai import types
from google.genai.types import Content, Part


from travel_concierge.tools.memory_control import LLMController
from travel_concierge.tools.memory_control import AgenticMemorySystem
from travel_concierge.tools.memory_control import calculate_metrics, aggregate_metrics
from travel_concierge.tools.memory import _load_precreated_itinerary

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

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
        backend = "openai"  # or "ollama"
        model = "gpt-4o-mini"  # or "gemini-2
        self.retriever_llm = LLMController(backend=backend, model=model, api_key=api_key)
    async def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        # print("response:{}".format(response))
        return response
    async def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response
    @override
    async def add_session_to_memory(self, session):
        user_key = _user_key(session.app_name, session.user_id)
        event_list = [event for event in session.events if event.content and event.content.parts]
        event_content = "\n".join(
            [part.text for event in event_list for part in event.content.parts if part.text is not None]
        )

        self.memory_system.add_note(
            content=event_content,
            # category=user_key,
            # tags=[session.app_name, session.user_id],
        )
    @override
    async def search_memory(self, *, app_name, user_id, query):
        """Search through memories"""
        response = SearchMemoryResponse()
        keywords = await self.generate_query_llm(query)
        results = self.memory_system.find_related_memories_raw(keywords, k=10)
        response.memories.append(
              MemoryEntry(
                  content = Content(parts=[Part(text=results)], role="model")
              ).model_dump())
        return response

async def main():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # Replace with your OpenAI API key
    APP_NAME = "memory_test_app"
    USER_ID = "test_user"
    MODEL = LiteLlm(model="openai/gpt-4o-mini")
    # MODEL = "gemini-2.0-flash"  # or "gpt-4o-mini"
    logger = setup_logger('agentic_memory_test.log')
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
    with open('travel_concierge/turns.json', 'r') as f:
        turns = json.load(f)
    results = []
    all_metrics = []
    all_categories = []
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
                metrics = calculate_metrics(final_response_text, turn['expected_output']) 
                all_metrics.append(metrics)
                all_categories.append(1)
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    final_results = {
        "aggregate_metrics": aggregate_results,
        "individual_results": results

    }
        # Save results
    output_path = "agent_responses_amem_advanced.json"
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")
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

async def test_memory_service():
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
    # --- Scenario ---
    # Turn 1: User shares monthly revenue data
    session1_id = "session1"
    session1 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    user_input1 = Content(parts=[Part(text="In January, the revenue was 100,000. In February, the revenue was 120,000.")], role="user")
    
    final_response_text = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session1_id, new_message=user_input1):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text = event.content.parts[0].text
    print(f"Agent 1 Response: {final_response_text}")
    
    # Add session information to memory
    completed_session1 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
    await memory_service.add_session_to_memory(completed_session1)

    # Turn 2: User requests total revenue
    session2_id = "session2"
    session2 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)
    user_input2 = Content(parts=[Part(text="What is the total revenue for January and February? If you do not have the data refer to the memory_recall_agnet")], role="user")

    final_response_text_2 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session2_id, new_message=user_input2):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_2 = event.content.parts[0].text
    print(f"Agent 2 Response: {final_response_text_2}")
    completed_session2 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)
    await memory_service.add_session_to_memory(completed_session2)
    # Turn 3: User updates the revenue for March
    session3_id = "session3"
    session3 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session3_id)
    user_input3 = Content(parts=[Part(text="In March, the revenue was 150,000.")], role="user")
    
    final_response_text_3 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session3_id, new_message=user_input3):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_3 = event.content.parts[0].text
    print(f"Agent 3 Response: {final_response_text_3}")
    
    # Update session information with March revenue
    completed_session3 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session3_id)
    await memory_service.add_session_to_memory(completed_session3)

    # Turn 4: User asks for average revenue
    session4_id = "session4"
    session4 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session4_id)
    user_input4 = Content(parts=[Part(text="What is the average revenue for January, February, and March?")], role="user")

    final_response_text_4 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session4_id, new_message=user_input4):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_4 = event.content.parts[0].text
    print(f"Agent 4 Response: {final_response_text_4}")

    # Turn 5: User asks for revenue growth between months
    session5_id = "session5"
    session5 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session5_id)
    user_input5 = Content(parts=[Part(text="What is the revenue growth from January to February, and from February to March?")], role="user")

    final_response_text_5 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session5_id, new_message=user_input5):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_5 = event.content.parts[0].text
    print(f"Agent 5 Response: {final_response_text_5}")

    # Turn 6: User asks about total revenue after the update
    session6_id = "session6"
    session6 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session6_id)
    user_input6 = Content(parts=[Part(text="What is the total revenue after adding March's revenue?")], role="user")

    final_response_text_6 = "(No final response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session6_id, new_message=user_input6):
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text_6 = event.content.parts[0].text
    print(f"Agent 6 Response: {final_response_text_6}")


if __name__ == "__main__":
    asyncio.run(main())