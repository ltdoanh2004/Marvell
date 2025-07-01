import asyncio
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core import Graphiti  
from graphiti_core.llm_client import OpenAIClient, LLMConfig  
from graphiti_core.nodes import EpisodeType  
from typing import List, Dict, Optional, Union
from datetime import datetime
from utils import calculate_metrics, aggregate_metrics
import random
import os
import pickle
import json
from collections import defaultdict
from dotenv import load_dotenv
import argparse
import logging
from data.load_locomo_dataset import load_locomo_dataset
import dateutil.parser  
import dateutil.tz
from graphiti_core.prompts.models import Message  
from tqdm import tqdm
load_dotenv()
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

class GraphitiAgent:  
    def __init__(self, model, backend, retrieve_k, temperature_c5, neo4j_uri, neo4j_user, neo4j_password):  
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.llm_client = GeminiClient(
        config=LLMConfig(
            api_key=self.api_key,
            model="gemini-2.0-flash"
        )
        )  
        self.embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=self.api_key,
                embedding_model="embedding-001"
            )
        )
        self.graphiti = Graphiti(  
            uri=neo4j_uri,  
            user=neo4j_user,   
            password=neo4j_password,  
            llm_client=self.llm_client,
            embedder=self.embedder,
        )  
        self.retrieve_k = retrieve_k  
        self.temperature_c5 = temperature_c5  
          
    async def initialize(self):  
        """Initialize Graphiti indices and constraints"""  
        await self.graphiti.build_indices_and_constraints()  
    async def add_memory(self, content, time=None, group_id="default"):  
        """Add episode to Graphiti graph"""  
        # Parse string datetime to datetime object if needed  
        content = self.clean_content(content)  # Clean content to avoid JSON parsing issues
        if isinstance(time, str):  
            try:  
                time = dateutil.parser.parse(time)  
            except:  
                # Fallback to current time if parsing fails  
                time = datetime.now(dateutil.tz.UTC)  
        elif time is None:  
            time = datetime.now(dateutil.tz.UTC)  

        try:
            await self.graphiti.add_episode(
                name="",
                episode_body=content,
                reference_time=time,
                source=EpisodeType.message,
                source_description="",
            )
        except Exception as e:
            logger.error(f"Error adding episode: {e}")
            group_id=group_id  
    async def retrieve_memory(self, query, k=10, group_id="default"):  
        """Enhanced retrieval using multiple search strategies"""  
        # Generate optimized queries  
        
        # Use Graphiti's search with different query types  
        search_results = await self.graphiti.search(  
            query=query["semantic_query"],  # Primary semantic search  
            group_ids=[group_id],  
            num_results=k  
        )  
        
        # If needed, can also do additional searches with fulltext keywords  
        if len(search_results) < k // 2:  
            additional_results = await self.graphiti.search(  
                query=query["fulltext_keywords"],  
                group_ids=[group_id],   
                num_results=k - len(search_results)  
            )  
            search_results.extend(additional_results)  
        
        return search_results
    from graphiti_core.prompts.models import Message

    async def generate_search_queries(self, question: str) -> dict:
        """Generate optimized queries for different Graphiti search methods"""  
        prompt = f"""Given the following question, generate search queries optimized for different search methods:  
        
        Question: {question}  
        
        Generate:  
        1. Keywords for fulltext search (BM25) - focus on important nouns and entities  
        2. Semantic query for vector similarity - natural language description  
        3. Entity names that might be relevant for graph traversal  
        
        Format your response as a JSON object with these fields:  
        - "fulltext_keywords": space-separated keywords for BM25 search  
        - "semantic_query": natural language query for vector search    
        - "entity_names": list of potential entity names to search for  
        - "original_query": the original question for fallback  
        
        Example response format:  
        {{  
            "fulltext_keywords": "Caroline LGBTQ support group May 2023",  
            "semantic_query": "When did Caroline attend LGBTQ support group meeting",  
            "entity_names": ["Caroline", "LGBTQ support group"],  
            "original_query": "{question}"  
        }}"""  

        # Thay vì truyền trực tiếp chuỗi, ta cần tạo một đối tượng Message
        messages = [
            Message(role='user', content=prompt)
        ]

        # Gọi hàm generate_response với đối tượng Message
        response = await self.llm_client.generate_response(messages)

        try:
            return json.loads(response)
        except:
            # Fallback to original question
            return {
                "fulltext_keywords": question,
                "semantic_query": question,
                "entity_names": [],
                "original_query": question
            }


    def clean_content(self, content: str) -> str:  
        """Clean content to avoid JSON parsing issues"""  
        # Remove or escape problematic characters  
        content = content.replace('"', "'")  # Replace quotes  
        content = content.replace('\n', ' ')  # Replace newlines  
        content = content.replace('\r', ' ')  # Replace carriage returns  
        return content.strip()   
    async def answer_question(self, question: str, category: int, answer: str, group_id: str = "default"):  
        """Generate answer using Graphiti search results"""  
        # Generate keywords (keep existing logic)  
        keywords = await self.generate_search_queries(question)  
          
        # Use Graphiti search instead of memory retrieval  
        raw_context = await self.retrieve_memory(keywords, k=self.retrieve_k, group_id=group_id)  
        context_text = ""  
        for edge in raw_context:  
            context_text += f"{edge.source_node_name} {edge.fact} {edge.target_node_name}\n"
        assert category in [1,2,3,4,5]
        user_prompt = f"""Context:
                {context_text}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5: 
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context_text}, answer the following question. {question}

                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context_text}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context_text}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:
            user_prompt = f"""Based on the context: {context_text}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
          
        response = await self.llm_client.generate_response(  
            [Message(role='user', content=user_prompt)],  
            temperature=temperature  
        )  
        # Handle different response formats  
        try:  
            if isinstance(response, dict):  
                if 'content' in response:  
                    return response['content'], user_prompt, raw_context  
                elif 'answer' in response:  
                    return response['answer'], user_prompt, raw_context  
            return str(response), user_prompt, raw_context    
            
        except Exception as e:    
            logger.error(f"Error generating response: {e}")    
            return "Error generating response", user_prompt, raw_context
          
          
async def evaluate_dataset(dataset_path: str, model: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, output_path: Optional[str] = None, ratio: float = 1.0, backend: str = "openai", temperature_c5: float = 0.5, retrieve_k: int = 10):
        """Evaluate the agent on the LoComo dataset.
        
        Args:
            dataset_path: Path to the dataset file
            model: Name of the model to use
            output_path: Path to save results
            ratio: Ratio of dataset to evaluate
        """
        # Generate automatic log filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_filename = f"eval_ours_{model}_{backend}_ratio{ratio}_{timestamp}.log"
        log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logger = setup_logger(log_path)
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        samples = load_locomo_dataset(dataset_path)
        logger.info(f"Loaded {len(samples)} samples")
        
        # Select subset of samples based on ratio
        if ratio < 1.0:
            num_samples = max(1, int(len(samples) * ratio))
            samples = samples[:num_samples]
            logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
        
        # Store results
        results = []
        all_metrics = []
        all_categories = []
        total_questions = 0
        category_counts = defaultdict(int)
        
        # Evaluate each sample
        i = 0
        error_num = 0
        memories_dir = os.path.join(os.path.dirname(__file__), "cached_memories_advanced_{}_{}".format(backend, model))
        os.makedirs(memories_dir, exist_ok=True)
        allow_categories = [1,2,3,4,5]
        for sample_idx, sample in enumerate(samples):  
                    # Create Graphiti agent instead of advancedMemAgent  
                    agent = GraphitiAgent(  
                        model=model,   
                        backend=backend,   
                        retrieve_k=retrieve_k,   
                        temperature_c5=temperature_c5,  
                        neo4j_uri=neo4j_uri,  
                        neo4j_user=neo4j_user,  
                        neo4j_password=neo4j_password  
                    )  
                    
                    await agent.initialize()  
                    
                    # Create unique group_id for each sample  
                    group_id = f"locomo_sample_{sample_idx}"  
                    
                    # Add conversations to Graphiti graph instead of memory system  
                    for _, turns in tqdm(sample.conversation.sessions.items()):  
                        for turn in turns.turns:  
                            turn_datetime = turns.date_time  
                            conversation_tmp = f"Speaker {turn.speaker} says: {turn.text}"  
                            await agent.add_memory(conversation_tmp, time=turn_datetime, group_id=group_id)  
                        
                    # Evaluate Q&A - fix indentation và logic  
                    for qa in sample.qa:  
                        if int(qa.category) in allow_categories:  
                            total_questions += 1  
                            category_counts[qa.category] += 1  
                            
                            prediction, user_prompt, raw_context = await agent.answer_question(  
                                qa.question, qa.category, qa.final_answer, group_id=group_id  
                            )  
                         
                            # Handle response parsing  
                            try:  
                                if isinstance(prediction, str) and prediction.startswith('{'):  
                                    prediction = json.loads(prediction)["answer"]  
                            except:  
                                error_num += 1  
                                logger.info(f"Failed to parse prediction as JSON: {prediction}")  
                            
                            # Log results - MOVE INSIDE THE LOOP  
                            logger.info(f"\nQuestion {total_questions}: {qa.question}")  
                            logger.info(f"Prediction: {prediction}")  
                            logger.info(f"Reference: {qa.final_answer}")  
                            logger.info(f"Category: {qa.category}")  
                            
                            # Calculate metrics - MOVE INSIDE THE LOOP  
                            metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {  
                                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,   
                                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,   
                                "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0  
                            }  
                            
                            all_metrics.append(metrics)  
                            all_categories.append(qa.category)  
                            
                            # Store individual result - MOVE INSIDE THE LOOP  
                            result = {  
                                "sample_id": sample_idx,  
                                "question": qa.question,  
                                "prediction": prediction,  
                                "reference": qa.final_answer,  
                                "category": qa.category,  
                                "metrics": metrics  
                            }  
                            results.append(result)  
                            
                            # Log progress  
                            if total_questions % 10 == 0:  
                                logger.info(f"Processed {total_questions} questions so far")
        logger.info(f"Total samples evaluated: {len(samples)}")    
        # Calculate aggregate metrics
        aggregate_results = aggregate_metrics(all_metrics, all_categories)
        
        # Prepare final results
        final_results = {
            "model": model,
            "dataset": dataset_path,
            "total_questions": total_questions,
            "category_distribution": {
                str(cat): count for cat, count in category_counts.items()
            },
            "aggregate_metrics": aggregate_results,
            "individual_results": results
        }
        logger.info(f"Error number: {error_num}")
        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        # Log summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"Total questions evaluated: {total_questions}")
        logger.info("\nCategory Distribution:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
        
        logger.info("\nAggregate Metrics:")
        for split_name, metrics in aggregate_results.items():
            logger.info(f"\n{split_name.replace('_', ' ').title()}:")
            for metric_name, stats in metrics.items():
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
        
        return final_results

async def test_generate_search_queries():
    # Khởi tạo đối tượng GraphitiAgent
    agent = GraphitiAgent(
        model="gpt-4o-mini",  # Model bạn sử dụng
        backend="openai",  # Backend bạn sử dụng
        retrieve_k=10,  # Số lượng kết quả cần lấy
        temperature_c5=0.5,  # Tham số nhiệt độ
        neo4j_uri="bolt://localhost:7687",  # URI Neo4j
        neo4j_user="neo4j",  # Tên người dùng Neo4j
        neo4j_password="demodemo"  # Mật khẩu Neo4j
    )

    # Khởi tạo agent
    await agent.initialize()

    # Câu hỏi mẫu
    question = "When did Caroline attend LGBTQ support group meeting?"

    # Gọi hàm generate_search_queries
    result = await agent.generate_search_queries(question)

    # Kiểm tra kết quả trả về
    if isinstance(result, dict):
        # Kiểm tra các trường dữ liệu
        required_fields = ["fulltext_keywords", "semantic_query", "entity_names", "original_query"]
        for field in required_fields:
            if field not in result:
                print(f"Missing field: {field}")
                return False
        print("Test passed: All fields are present.")
        print(result)  # In kết quả để kiểm tra chi tiết
        return True
    else:
        print("Test failed: Result is not a dictionary.")
        return False



async def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=0.1,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                      help="Backend to use (openai or ollama)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                      help="Retrieve k")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687",  
                      help="Neo4j URI")  
    parser.add_argument("--neo4j-user", type=str, default="neo4j",  
                      help="Neo4j username")    
    parser.add_argument("--neo4j-password", type=str, default = "demodemo" ,
                      help="Neo4j password")  
    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    await evaluate_dataset(  
        dataset_path, args.model, args.neo4j_uri, args.neo4j_user, args.neo4j_password,  
        args.output, args.ratio, args.backend, args.temperature_c5, args.retrieve_k  
    )


if __name__ == "__main__":
    asyncio.run(main())