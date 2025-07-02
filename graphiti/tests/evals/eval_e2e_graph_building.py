"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
from datetime import datetime, timezone
import pandas as pd
import time 
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from mem0 import Memory

SHARED_CONFIG_OPEN_AI_WITH_GRAPH = {  
    "version": "v1.1",  
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
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
            "model": "gpt-4o-mini",  
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
class MemoryADD:
    def __init__(self, config):
        self.config = config
        self.memory = Memory(config=config)


    async def build_subgraph(
        self,
        user_id: str,
        multi_session,
        multi_session_dates,
        session_length: int,
        group_id_suffix: str,
    ): 
        add_episode_results = []
        add_episode_context: list[str] = []
        message_count = 0
        for session_idx, session in enumerate(multi_session):
            for _, msg in enumerate(session):
                if message_count >= session_length:
                    continue
                message_count += 1
                date = multi_session_dates[session_idx] + ' UTC'
                date_format = '%Y/%m/%d (%a) %H:%M UTC'
                date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
                for attempt in range(3):
                    try:
                        episode_body = f'{msg["role"]}: {msg["content"]}'
                        _ = self.memory.add(
                            episode_body, user_id=user_id, metadata={"timestamp": date_string}
                        )
                        semantic_memories, graph_memories, _ = self.memory.get(
                            user_id=user_id, limit=1, metadata={"timestamp": date_string}
                        )
                        results = graph_memories[0]
                        results['content'] = semantic_memories[0]["memory"]
                        results['timestamp'] = semantic_memories[0]["timestamp"]
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1)  # Wait before retrying
                            continue
                        else:
                            raise e
            add_episode_results.append(results)
            add_episode_context.append(msg['content'])
        return user_id, add_episode_results, add_episode_context

    async def build_graph(
        self, group_id_suffix: str, multi_session_count: int, session_length: int
    ):
        # Get longmemeval dataset
        lme_dataset_option = (
            'data/longmemeval_data/longmemeval_oracle.json'  # Can be _oracle, _s, or _m
        )
        lme_dataset_df = pd.read_json(lme_dataset_option)
        add_episode_results = {}
        add_episode_context = {}
        subgraph_results = await semaphore_gather(
            *[
                self.build_subgraph(
                    user_id='lme_oracle_experiment_user_' + str(multi_session_idx),
                    multi_session=lme_dataset_df['haystack_sessions'].iloc[multi_session_idx],
                    multi_session_dates=lme_dataset_df['haystack_dates'].iloc[multi_session_idx],
                    session_length=session_length,
                    group_id_suffix=group_id_suffix,
                )
                for multi_session_idx in range(multi_session_count)
            ]
        )
        for user_id, episode_results, episode_context in subgraph_results:
            add_episode_results[user_id] = episode_results
            add_episode_context[user_id] = episode_context


        return add_episode_results, add_episode_context


    async def eval_graph(multi_session_count: int, session_length: int, llm_client=None) -> float:
        if llm_client is None:
            llm_client = OpenAIClient(config=LLMConfig(model='gpt-4.1-mini'))
        graphiti = Graphiti(NEO4J_URI, NEO4j_USER, NEO4j_PASSWORD, llm_client=llm_client)
        with open('baseline_graph_results.json') as file:
            baseline_results_raw = json.load(file)

            baseline_results: dict[str, list[AddEpisodeResults]] = {
                key: [AddEpisodeResults(**item) for item in value]
                for key, value in baseline_results_raw.items()
            }
        add_episode_results, add_episode_context = await build_graph(
            'candidate', multi_session_count, session_length, graphiti
        )

        filename = 'candidate_graph_results.json'

        candidate_baseline_graph_results = {
            key: [item.model_dump(mode='json') for item in value]
            for key, value in add_episode_results.items()
        }

        with open(filename, 'w') as file:
            json.dump(candidate_baseline_graph_results, file, indent=4, default=str)

        raw_score = 0
        user_count = 0
        for user_id in add_episode_results:
            user_count += 1
            user_raw_score = 0
            for baseline_result, add_episode_result, episodes in zip(
                baseline_results[user_id],
                add_episode_results[user_id],
                add_episode_context[user_id],
                strict=False,
            ):
                context = {
                    'baseline': baseline_result,
                    'candidate': add_episode_result,
                    'message': episodes[0],
                    'previous_messages': episodes[1:],
                }

                llm_response = await llm_client.generate_response(
                    prompt_library.eval.eval_add_episode_results(context),
                    response_model=EvalAddEpisodeResults,
                )

                candidate_is_worse = llm_response.get('candidate_is_worse', False)
                user_raw_score += 0 if candidate_is_worse else 1
                print('llm_response:', llm_response)
            user_score = user_raw_score / len(add_episode_results[user_id])
            raw_score += user_score
        score = raw_score / user_count

        return score
