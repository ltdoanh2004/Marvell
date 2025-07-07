.PHONY: eval add-mem eval-all eval-gemini eval-graph clean

eval:
	python src/evaluate.py --model gpt-4o --data_path data/longmemeval_s.json --output_dir results

add-mem-no-parallel:
	python src/evaluate.py --model gpt-4o --data_path data/longmemeval_s.json --output_dir results --add-mem
add-mem-parallel:
	python src/evaluate.py --model gpt-4o --data_path data/longmemeval_s.json --output_dir results --add-mem --run_parallel
eval-all:
	python src/evaluate.py --model gpt-4o --data_path data/longmemeval_s.json --output_dir results --percentage_to_process 1

eval-gemini:
	python src/evaluate.py --model gemini --data_path data/longmemeval_s.json --output_dir results

eval-graph:
	python src/evaluate.py --model gpt-4o --data_path data/longmemeval_s.json --output_dir results --is_graph

clean:
	rm -rf results/*
