import asyncio  
import json  
from typing import List, Dict, Any  
  
from cognee.eval_framework.eval_config import EvalConfig  
from cognee.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor  
from cognee.eval_framework.corpus_builder.task_getters.TaskGetters import TaskGetters  
from cognee.eval_framework.answer_generation.answer_generation_executor import (  
    AnswerGeneratorExecutor,   
    retriever_options  
)  
from cognee.eval_framework.evaluation.evaluation_executor import EvaluationExecutor  
from cognee.eval_framework.analysis.metrics_calculator import bootstrap_ci 
from cognee.eval_framework.analysis.dashboard_generator import create_dashboard  
def calculate_aggregate_metrics(evaluated_results, metrics):  
    aggregate_metrics = {}  
      
    for metric in metrics:  
        # Extract scores for this metric  
        scores = []  
        for result in evaluated_results:  
            if "metrics" in result and metric in result["metrics"]:  
                scores.append(result["metrics"][metric]["score"])  
          
        if scores:  
            # Use bootstrap_ci to calculate mean and confidence intervals  
            mean, ci_lower, ci_upper = bootstrap_ci(scores)  
              
            aggregate_metrics[metric] = {  
                "scores": scores,  
                "mean": mean,  
                "ci_lower": ci_lower,  
                "ci_upper": ci_upper  
            }  
      
    return aggregate_metrics
async def run_full_evaluation():  
    """  
    Chạy toàn bộ evaluation pipeline từ corpus building đến dashboard generation  
    """  
    # 1. Load configuration  
    config = EvalConfig()  
    # config.questions_path = "evals/hotpot_50_corpus.json"
    config.benchmark = "HotPotQA"
    config.number_of_samples_in_corpus = 50
    config.questions_path = f"questions_{config.benchmark.lower()}_{config.number_of_samples_in_corpus}.json"
    print(f"🔧 Loaded config: {config.benchmark} benchmark with {config.number_of_samples_in_corpus} samples")  
      
    # 2. Build corpus nếu cần  
    if config.building_corpus_from_scratch:  
        print("📚 Building corpus from scratch...")  
          
        # Get task getter  
        task_getter = TaskGetters(config.task_getter_type).getter_func  
          
        # Initialize corpus builder  
        corpus_builder = CorpusBuilderExecutor(  
            benchmark=config.benchmark,  
            task_getter=task_getter,  
        )  
          
        # Build corpus và get questions  
        questions = await corpus_builder.build_corpus(  
            limit=config.number_of_samples_in_corpus,  
            load_golden_context=config.evaluating_contexts,  
            instance_filter=config.instance_filter,  
        )  
          
        # Save questions  
        with open(config.questions_path, "w", encoding="utf-8") as f:  
            json.dump(questions, f, ensure_ascii=False, indent=4)  
          
        print(f"✅ Built corpus with {len(questions)} questions")  
    else:  
        # Load existing questions  
        with open(config.questions_path, "r", encoding="utf-8") as f:  
            questions = json.load(f)  
        print(f"📖 Loaded {len(questions)} existing questions")  
      
    # 3. Generate answers nếu cần  
    if config.answering_questions:  
        print("🤖 Generating answers...")  
          
        # Get retriever  
        retriever_class = retriever_options[config.qa_engine]  
        retriever = retriever_class()  
          
        # Initialize answer generator  
        answer_generator = AnswerGeneratorExecutor()  
          
        # Generate answers  
        answers = await answer_generator.question_answering_non_parallel(  
            questions=questions,  
            retriever=retriever,  
        )  
          
        # Save answers  
        with open(config.answers_path, "w", encoding="utf-8") as f:  
            json.dump(answers, f, ensure_ascii=False, indent=4)  
          
        print(f"✅ Generated {len(answers)} answers")  
    else:  
        # Load existing answers  
        with open(config.answers_path, "r", encoding="utf-8") as f:  
            answers = json.load(f)  
        print(f"📖 Loaded {len(answers)} existing answers")  
      
    # 4. Evaluate answers  
    if config.evaluating_answers:  
        print("📊 Evaluating answers...")  
          
        # Initialize evaluation executor  
        evaluation_executor = EvaluationExecutor(  
            evaluator_engine=config.evaluation_engine,  
            evaluate_contexts=config.evaluating_contexts,  
        )  
          
        # Run evaluation  
        evaluated_results = await evaluation_executor.execute(  
            answers=answers,  
            evaluator_metrics=config.evaluation_metrics,  
        )  
          
        # Save evaluation results  
        with open(config.metrics_path, "w", encoding="utf-8") as f:  
            json.dump(evaluated_results, f, ensure_ascii=False, indent=4)  
          
        print(f"✅ Evaluated {len(evaluated_results)} answers")  
    else:  
        # Load existing evaluation results  
        with open(config.metrics_path, "r", encoding="utf-8") as f:  
            evaluated_results = json.load(f)  
        print(f"📖 Loaded {len(evaluated_results)} existing evaluations")  
      
    # 5. Calculate aggregate metrics  
    if config.calculate_metrics:  
        print("📈 Calculating aggregate metrics...")  
          
        aggregate_metrics = calculate_aggregate_metrics(  
            evaluated_results,   
            config.evaluation_metrics  
        )  
          
        # Save aggregate metrics  
        with open(config.aggregate_metrics_path, "w", encoding="utf-8") as f:  
            json.dump(aggregate_metrics, f, ensure_ascii=False, indent=4)  
          
        print("✅ Calculated aggregate metrics")  
          
        # Print summary  
        for metric, values in aggregate_metrics.items():  
            print(f"  {metric}: {values['mean']:.4f} (CI: {values['ci_lower']:.4f}-{values['ci_upper']:.4f})")  
      
    # 6. Generate dashboard  
    if config.dashboard:    
        print("📊 Generating dashboard...")    
            
        dashboard_file = create_dashboard(    
            metrics_path=config.metrics_path,    
            aggregate_metrics_path=config.aggregate_metrics_path,    
            output_file=config.dashboard_path,    
            benchmark=config.benchmark,  # Sửa từ benchmark_name thành benchmark  
        ) 
          
        print(f"✅ Generated dashboard: {dashboard_file}")  
      
    print("🎉 Evaluation pipeline completed successfully!")  
    return evaluated_results  
  
# Script để chạy với custom config  
async def run_evaluation_with_custom_config(  
    benchmark: str = "HotPotQA",  
    num_samples: int = 10,  
    qa_engine: str = "cognee_completion",  
    evaluation_engine: str = "DeepEval",  
    metrics: List[str] = None,  
):  
    """  
    Chạy evaluation với custom configuration  
    """  
    if metrics is None:  
        metrics = ["correctness", "EM", "f1"]  
      
    # Override config  
    config = EvalConfig()  
    config.benchmark = benchmark  
    config.number_of_samples_in_corpus = num_samples  
    config.qa_engine = qa_engine  
    config.evaluation_engine = evaluation_engine  
    config.evaluation_metrics = metrics  
      
    # Update file paths với benchmark name  
    config.questions_path = f"questions_{benchmark.lower()}_{num_samples}.json"  
    config.answers_path = f"answers_{benchmark.lower()}_{num_samples}.json"  
    config.metrics_path = f"metrics_{benchmark.lower()}_{num_samples}.json"  
    config.aggregate_metrics_path = f"aggregate_metrics_{benchmark.lower()}_{num_samples}.json"  
    config.dashboard_path = f"dashboard_{benchmark.lower()}_{num_samples}.html"  
      
    print(f"🚀 Starting evaluation: {benchmark} with {num_samples} samples")  
    return await run_full_evaluation_with_config(config)  
  
async def run_full_evaluation_with_config(config: EvalConfig):  
    """  
    Chạy evaluation với config được truyền vào  
    """  
    # Tương tự như run_full_evaluation() nhưng sử dụng config được truyền vào  
    # [Implementation tương tự như trên]  
    pass  
  
if __name__ == "__main__":  
    # Chạy với default config  
    asyncio.run(run_full_evaluation())  
      
    # Hoặc chạy với custom config  
    # asyncio.run(run_evaluation_with_custom_config(  
    #     benchmark="HotPotQA",  
    #     num_samples=50,  
    #     qa_engine="cognee_graph_completion",  
    #     evaluation_engine="DeepEval",  
    #     metrics=["correctness", "EM", "f1", "contextual_relevancy"]  
    # ))