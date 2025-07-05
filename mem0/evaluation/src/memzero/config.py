import os
SHARED_OPEN_AI_CONFIG = {  
    "llm": {  
        "provider": "openai",  
        "config": {  
            "model": os.getenv("MODEL", "gpt-4o-mini"),  
            "api_key": os.getenv("OPENAI_API_KEY")  
        }  
    },  
    "embedder": {  
        "provider": "openai",  
        "config": {  
            "model": "text-embedding-3-small"  
        }  
    },  
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        }
    } 
}
SHARED_CONFIG_OPEN_AI_WITH_GRAPH = {  
    "version": "v1.1",  
    "vector_store": {  
        "provider": "faiss",  
        "config": {  
            "path": "./faiss_graph_db",  
            "collection_name": "mem0_evaluation"  
        }  
    },  
    "graph_store": {  
        "provider": "neo4j",  
        "config": {  
            "url": "bolt://localhost:7688",  
            "username": "neo4j",  
            "password": "demodemo"  
        }  
    },  
    "llm": {  
        "provider": "openai",  
        "config": {  
            "api_key": os.getenv("OPENAI_API_KEY"),  
            "model": "gpt-4o",  
            "temperature": 0.2  
        }  
    },  
    "embedder": {  
        "provider": "openai",  
        "config": {  
            "model": "text-embedding-3-small"  
        }  
    },  
    "history_db_path": "./shared_graph_evaluation_history.db"  
}
SHARED_GEMINI_CONFIG = {  
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0,
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    },  
    "embedder": {  
        "provider": "openai",  
        "config": {  
            "model": "text-embedding-3-small"  
        }  
    },  
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        }
    }
}
SHARED_CONFIG_GEMINI_WITH_GRAPH = {  
    "version": "v1.1",  
    "vector_store": {  
        "provider": "faiss",  
        "config": {  
            "path": "./faiss_graph_db",  
            "collection_name": "mem0_evaluation"  ,
            "embedding_model_dims": 384
        }  
    },  
    "graph_store": {  
        "provider": "neo4j",  
        "config": {  
            "url": "bolt://localhost:7688",  
            "username": "neo4j",  
            "password": "demodemo"  
        }  
    },  
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
    "history_db_path": "./shared_graph_evaluation_history.db"  
}
