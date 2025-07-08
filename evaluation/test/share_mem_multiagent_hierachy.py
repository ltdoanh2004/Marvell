import os
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from mem0 import Memory
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-001",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0,
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1"
        }
    },
    "vector_store": {  
        "provider": "faiss",  
        "config": {  
            "path": "./faiss_graph_db",  
            "collection_name": "mem0_evaluation"  ,
            "embedding_model_dims": 384
        }  
    },
    "version": "v1.1"
}


# Initialize Mem0 client
mem0 = Memory.from_config(config)

# Define memory function tools
def search_memory(query: str, user_id: str) -> dict:
    """Search through past conversations and memories"""
    memories = mem0.search(query, user_id=user_id)
    if memories:
        memory_context = "\n".join([f"- {mem['memory']}" for mem in memories])
        return {"status": "success", "memories": memory_context}
    return {"status": "no_memories", "message": "No relevant memories found"}

def save_memory(content: str, user_id: str) -> dict:
    """Save important information to memory"""
    try:
        mem0.add([{"role": "user", "content": content}], user_id=user_id)
        return {"status": "success", "message": "Information saved to memory"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save memory: {str(e)}"}
# Travel specialist agent
travel_agent = Agent(
    name="travel_specialist",
    model="gemini-2.0-flash",
    instruction="""You are a travel planning specialist. Use get_user_context to
    understand the user's travel preferences and history before making recommendations.
    After providing advice, use store_interaction to save travel-related information.""",
    description="Specialist in travel planning and recommendations",
    tools=[search_memory, save_memory]
)

# Health advisor agent
health_agent = Agent(
    name="health_advisor",
    model="gemini-2.0-flash",
    instruction="""You are a health and wellness advisor. Use get_user_context to
    understand the user's health goals and dietary preferences.
    After providing advice, use store_interaction to save health-related information.""",
    description="Specialist in health and wellness advice",
    tools=[search_memory, save_memory]
)

# Coordinator agent that delegates to specialists
coordinator_agent = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    instruction="""You are a coordinator that delegates requests to specialist agents.
    For travel-related questions (trips, hotels, flights, destinations), delegate to the travel specialist.
    For health-related questions (fitness, diet, wellness, exercise), delegate to the health advisor.
    Use get_user_context to understand the user before delegation.""",
    description="Coordinates requests between specialist agents",
    tools=[
        AgentTool(agent=travel_agent, skip_summarization=False),
        AgentTool(agent=health_agent, skip_summarization=False)
    ]
)

def chat_with_specialists(user_input: str, user_id: str) -> str:
    """
    Handle user input with specialist agent delegation and memory.

    Args:
        user_input: The user's message
        user_id: Unique identifier for the user

    Returns:
        The specialist agent's response
    """
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name="specialist_system",
        user_id=user_id,
        session_id=f"session_{user_id}"
    )
    runner = Runner(agent=coordinator_agent, app_name="specialist_system", session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=user_input)])
    events = runner.run(user_id=user_id, session_id=session.id, new_message=content)

    for event in events:
        if event.is_final_response():
            response = event.content.parts[0].text

            # Store the conversation in shared memory
            conversation = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ]
            mem0.add(conversation, user_id=user_id)

            return response

    return "No response generated"

# Example usage
response = chat_with_specialists("Plan a healthy meal for my Italy trip", user_id="alice")
print(response)