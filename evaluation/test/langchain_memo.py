import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
            "path": "./faiss_db",  
            "collection_name": "mem0_evaluation"  ,
            "embedding_model_dims": 384
        }  
    },
    "version": "v1.1"
}


mem0 = Memory.from_config(config)
llm = ChatOpenAI(model="gpt-4o-mini")


prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful travel agent AI. Use the provided context to personalize your responses and remember user preferences and past interactions. 
    Provide travel recommendations, itinerary suggestions, and answer questions about destinations. 
    If you don't have specific information, you can make general suggestions based on common travel knowledge."""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])

def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Retrieve relevant context from Mem0"""
    memories = mem0.search(query, user_id=user_id)
    serialized_memories = ' '.join([mem["memory"] for mem in memories])
    context = [
        {
            "role": "system", 
            "content": f"Relevant information: {serialized_memories}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return context

def generate_response(input: str, context: List[Dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "input": input
    })
    return response.content

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to Mem0"""
    interaction = [
        {
          "role": "user",
          "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    mem0.add(interaction, user_id=user_id)

def chat_turn(user_input: str, user_id: str) -> str:
    # Retrieve context
    context = retrieve_context(user_input, user_id)
    
    # Generate response
    response = generate_response(user_input, context)
    
    # Save interaction
    save_interaction(user_id, user_input, response)
    
    return response

if __name__ == "__main__":
    print("Welcome to your personal Travel Agent Planner! How can I assist you with your travel plans today?")
    user_id = "john"
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Travel Agent: Thank you for using our travel planning service. Have a great trip!")
            break
        
        response = chat_turn(user_input, user_id)
        print(f"Travel Agent: {response}")