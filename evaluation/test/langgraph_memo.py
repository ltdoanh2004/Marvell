from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from mem0 import Memory
import os
llm = ChatOpenAI(model="gpt-4")

config = {
    "llm": {
        "provider": "gemini", # follow supported LLMs https://docs.mem0.ai/components/llms/models/openai
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
            "collection_name": "mem0_evaluation",
            "embedding_model_dims": 384
        }  
    },
    "version": "v1.1"
}

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str

def chatbot(state: State):
    
    messages = state["messages"]
    user_id = state["mem0_user_id"]

    # Retrieve relevant memories
    memories = mem0.search(messages[-1].content, user_id=user_id)

    context = "Relevant information from previous conversations:\n"
    for memory in memories:
        context += f"- {memory['memory']}\n"

    system_message = SystemMessage(content=f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
{context}""")

    full_messages = [system_message] + messages
    response = llm.invoke(full_messages)

    # Store the interaction in Mem0
    mem0.add(f"User: {messages[-1].content}\nAssistant: {response.content}", user_id=user_id)
    return {"messages": [response]}



def run_conversation(user_input: str, mem0_user_id: str):
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {"messages": [HumanMessage(content=user_input)], "mem0_user_id": mem0_user_id}

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return
            
if __name__ == "__main__":
    # Configuration
    graph = StateGraph(State)
    
    mem0 = Memory.from_config(config)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", "chatbot")

    compiled_graph = graph.compile()
    print("Welcome to Customer Support! How can I assist you today?")
    mem0_user_id = "customer_123"  # You can generate or retrieve this based on your user management system
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Customer Support: Thank you for contacting us. Have a great day!")
            break
        run_conversation(user_input, mem0_user_id)