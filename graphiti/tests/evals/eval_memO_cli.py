import argparse  
import asyncio  
import os  
from tests.evals.eval_e2e_memo_graph_building import MemoryADD, SHARED_CONFIG_OPEN_AI_WITH_GRAPH  
  
async def main():  
    parser = argparse.ArgumentParser(  
        description='Run Mem0 vs Graphiti evaluation from the command line.'  
    )  
  
    parser.add_argument(  
        '--multi-session-count',  
        type=int,  
        required=True,  
        help='Integer representing multi-session count',  
    )  
    parser.add_argument(  
        '--session-length',   
        type=int,   
        required=True,   
        help='Length of each session'  
    )  
    parser.add_argument(  
        '--build-baseline',   
        action='store_true',   
        help='If set, builds baseline Mem0 graph results'  
    )  
    parser.add_argument(  
        '--compare-only',  
        action='store_true',  
        help='If set, only runs comparison (requires existing baseline files)'  
    )  
  
    args = parser.parse_args()  
  
    # Initialize MemoryADD with config  
    memory_eval = MemoryADD(config=SHARED_CONFIG_OPEN_AI_WITH_GRAPH)  
    if not args.compare_only:  
        if args.build_baseline:  
            print('Building baseline Mem0 graph...')  
            await memory_eval.build_baseline_graph(  
                multi_session_count=args.multi_session_count,   
                session_length=args.session_length  
            )  
            print('Baseline Mem0 graph built successfully!')  
    # Run comparison  
    # print('Running Mem0 vs Graphiti comparison...')  
    # result = await memory_eval.compare_graph(  
    #     multi_session_count=args.multi_session_count,   
    #     session_length=args.session_length  
    # )  
    # print(f'Comparison result: {result}')  
  
if __name__ == '__main__':  
    asyncio.run(main())