import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from memzero.config import *
from dotenv import load_dotenv
from openai import OpenAI
import time
from mem0 import Memory
from memzero.utils.metrics import compute_bleu, compute_f1
from tqdm import tqdm
load_dotenv()
client = OpenAI()

CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""
class MemoryEvaluation:
    def __init__(self, data_path=None, batch_size=2, is_graph=False, model ="gpt-4o"):
        if model == "gpt-4o":
            config = SHARED_CONFIG_OPEN_AI_WITH_GRAPH if is_graph else SHARED_OPEN_AI_CONFIG
        elif model == "gemini":
            config = SHARED_CONFIG_GEMINI_WITH_GRAPH if is_graph else SHARED_GEMINI_CONFIG

        config['custon_instructions'] = CUSTOM_INSTRUCTIONS 
        self.mem0_client = Memory.from_config(config)

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data
    
    def process_add_memory(self):
        for entry in tqdm(self.data, desc="add questions to memory"):
            for cur_sess_id, sess_entry, date in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
                metadata = {
                     "session_id": cur_sess_id,
                     "date": date
                }
                for sample in tqdm(sess_entry, desc="Adding memory samples"):
                    self.add_memory(user_id=entry['question_id'], message=sample, metadata=metadata)

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    message, user_id=user_id, metadata=metadata
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def process_add_memory_parallel(self, num_threads=4):
        if not self.data:
            raise ValueError("Data not loaded. Please load data before processing.")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for entry in tqdm(self.data, desc="Processing entries"):
                for cur_sess_id, sess_entry, date in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
                    metadata = {
                        "session_id": cur_sess_id,
                        "date": date
                    }
                    for sample in sess_entry:
                        futures.append(executor.submit(
                            self.add_memory, user_id=entry['question_id'], message=sample, metadata=metadata
                        ))
            for future in tqdm(futures, desc="Waiting for tasks to complete"):
                future.result()
    
    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("Searching with graph")
                    memories = self.mem0_client.search(  
                        query,  
                        user_id=user_id,  
                        limit=self.top_k if hasattr(self, 'top_k') else 5  
                    )
                else:
                    memories = self.mem0_client.search(
                        query, 
                        user_id=user_id, 
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time
    
    def search_and_answer_memory(self):
        results = []
        for entry in tqdm(self.data, desc="Processing memories and evaluate scores"):
            question = entry['question']
            semantic_memories, graph_memories, _ = self.search_memory(
                    user_id=entry['question_id'], 
                    query=question
            )
            prompt = self.build_prompt_question_answer(entry, semantic_memories, graph_memories)
            t1 = time.time()
            llm_answer = self.get_llm_answer(prompt)
            t2 = time.time()
            response_time = t2 - t1
            eval_json = self.get_llm_evaluation(question, answer, llm_answer)
            answer = entry['answer']
            try:
                eval_result = json.loads(eval_json)
            except Exception:
                eval_result = {"score": None, "explanation": eval_json}
            results.append({
                "question": question,
                "reference_answer": answer,
                "llm_answer": llm_answer,
                "evaluation_results": eval_result,
                "response_time": response_time,
                "semantic_memories": semantic_memories,
                "graph_memories": graph_memories,
            })
    
    def build_prompt_question_answer(self, entry, semantic_memories, graph_memories):
        question = entry['question']

        # Xây dựng phần context từ semantic memories
        if semantic_memories:
            memories_text = "\n".join(
                f"- {mem['memory']} (score: {mem['score']}, time: {mem['timestamp']})"
                for mem in semantic_memories
            )
        else:
            memories_text = "No relevant past memories found."

        prompt = (
            f"Context (Past Memories):\n{memories_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        return prompt
        
    def get_llm_answer(prompt, model="gpt-4o"):
        response = client.responses.create(
            model="gpt-4o",
            input=prompt
        )
        return response.output_text

    def get_llm_evaluation(question, reference_answer, llm_answer, model="gpt-4o"):
        eval_prompt = (
            f"Question: {question}\n"
            f"Reference Answer: {reference_answer}\n"
            f"LLM Answer: {llm_answer}\n\n"
            "Evaluate the LLM Answer compared to the Reference Answer. "
            "Give a score from 0 (completely wrong) to 10 (perfectly correct). "
            "Return your result as a JSON object with keys: score (int), explanation (str)."
        )
        response = client.responses.create(
            model="gpt-4o",
            input=eval_prompt
        )
        return response.choices[0].message["content"].strip()        
    
    def evaluate_bleu_f1(self, results):
        bleu_scores = []
        f1_scores = []
        llm_scores = []
        for item in results:
            ref = item["reference_answer"]
            hyp = item["llm_answer"]
            bleu = compute_bleu(ref, hyp)
            f1 = compute_f1(ref, hyp)
            bleu_scores.append(bleu)
            f1_scores.append(f1)
            # Lấy score do LLM chấm nếu có
            score = item.get("evaluation_results", {}).get("score")
            if isinstance(score, (int, float)):
                llm_scores.append(score)
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average LLM Score: {avg_llm_score:.2f}")
        return {"avg_bleu": avg_bleu, "avg_f1": avg_f1, "avg_llm_score": avg_llm_score}
    
    def save_results(self, results, output_path="results.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")

        for i, item in enumerate(results[:3]):
            print(f"\n--- Result {i+1} ---")
            print(json.dumps(item, ensure_ascii=False, indent=2))
        if len(results) > 3:
            print(f"... ({len(results)-3} more results not shown)")