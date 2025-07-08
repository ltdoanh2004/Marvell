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

# Create agent with memory capabilities
personal_assistant = Agent(
    name="personal_assistant",
    model="gemini-2.0-flash",
    instruction="""You are a helpful personal assistant with memory capabilities.
    Use the search_memory function to recall past conversations and user preferences.
    Use the save_memory function to store important information about the user.
    Always personalize your responses based on available memory.""",
    description="A personal assistant that remembers user preferences and past interactions",
    tools=[search_memory, save_memory]
)

async def chat_with_agent(user_input: str, user_id: str) -> str:
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="memory_assistant",
        user_id=user_id,
        session_id=f"session_{user_id}"
    )
    runner = Runner(agent=personal_assistant, app_name="memory_assistant", session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=user_input)])
    events = runner.run(user_id=user_id, session_id=session.id, new_message=content)

    for event in events:
        if event.is_final_response():
            return event.content.parts[0].text

    return "No response generated"


# Example usage
if __name__ == "__main__":
    response = asyncio.run(chat_with_agent(
        "I love Italian food and I'm planning a trip to Rome next month",
        user_id="alice"
    ))
    print(response)