import os
import json
import argparse
from memzero.memO_eval import MemoryEvaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate memO's performance on a specific task.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use for evaluation.")
    parser.add_argument("--data_path", type=str, default='data/longmemeval_s.json', help="Path to the evaluation data file.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results.")
    parser.add_argument("--is_graph", action="store_true", help="Use graph memory.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for memory addition.")
    parser.add_argument("--add-mem", action="store_true", help="Add memory before evaluation.")
    parser.add_argument("--percentage_to_process", type=float, default=0.2, help="Percentage of data to add to memory.")
    parser.add_argument("--run_parallel", action="store_true", help="Run the evaluation in parallel mode.")
    return parser.parse_args()

def main(args):
    print(f"Using model: {args.model} on data: {args.data_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"memo_eval_{args.model}.json")

    evaluator = MemoryEvaluation(
        data_path=args.data_path,
        batch_size=args.batch_size,
        is_graph=args.is_graph,
        model=args.model,
        percentage_to_process=args.percentage_to_process,
    )
    if args.add_mem:
        if args.run_parallel:
            evaluator.process_add_memory_parallel(num_threads=8)
        else:
            evaluator.process_add_memory()

    results = evaluator.search_and_answer_memory()
    evaluator.save_results(results, output_path=output_path)
    result_path = os.path.join(args.output_dir, f"memo_eval_results_{args.model}.json")
    evaluator.evaluate_bleu_f1(results, output_path=result_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)