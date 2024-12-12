import os
import argparse

from lib.data_loader import DataLoaderFactory
from lib.payload_creator import PayloadCreatorFactory
from lib.api_executor import APIExecutorFactory
from lib.response_evaluator import ResponseEvaluatorFactory
from lib.utils import get_eval_summary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_type", type=str, help="", default="DT", choices=['DT', 'DP'])
    parser.add_argument("--task_type", type=str, help="", default="QG", choices=['QG'])
    
    parser.add_argument("--datasource", type=str, help="", default="huggingface")
    parser.add_argument("--dataset_name", type=str, help="", default="lmqg/qg_squad")
    
    parser.add_argument("--temperature", type=float, help="", default=0.1)
    parser.add_argument("--model", type=str, help="", default="gpt-4o-mini")
    parser.add_argument("--api_type", type=str, help="", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--api_key", type=str, help="", default=os.getenv("OPENAI_API_KEY"))
    
    parser.add_argument("--eval_type", type=str, help="", default="reference_based")
    
    parser.add_argument("--reset", type=bool, help="", default=False)
    
    return parser.parse_args()


def main(args):
    REPO_PATH = os.path.abspath(os.getcwd())
    TEST_PREFIX = f"{args.domain_type}-{args.task_type}"
    
    input_dataset_path = f'{REPO_PATH}/data/raw/{TEST_PREFIX}'
    input_payload_path = f'{REPO_PATH}/data/processed/{TEST_PREFIX}.jsonl'
    system_prompt_path = f'{REPO_PATH}/prompts/{TEST_PREFIX}.txt'
    output_path = f'{REPO_PATH}/results/{TEST_PREFIX}-{args.model}.output.jsonl'
    eval_results_path = f'{REPO_PATH}/results/{TEST_PREFIX}-{args.model}.eval_results.jsonl'
    eval_summary_path = f'{REPO_PATH}/results/{TEST_PREFIX}-{args.model}.eval_summary.json'
    
    # ----------------------------------------------------------------------
    # Load the dataset
    # ----------------------------------------------------------------------
    input_dataset = DataLoaderFactory.get_data_loader(
        source=args.datasource,
        dataset_name=args.dataset_name
    ).load_dataset(
        dataset_path=input_dataset_path,
        split='test',
        reset=args.reset
    )
    
    # ----------------------------------------------------------------------
    # Create the payloads
    # ----------------------------------------------------------------------
    input_payloads = PayloadCreatorFactory.get_payload_creator(
        task_type=args.task_type,
        temperature=args.temperature,
        system_prompt_path=system_prompt_path
    ).create_payload(
        input_dataset=input_dataset,
        payload_path=input_payload_path,
        reset=args.reset
    )
    
    # ----------------------------------------------------------------------
    # Execute the API
    # ----------------------------------------------------------------------
    response_list = APIExecutorFactory.get_api_executor(
        model=args.model,
        api_type=args.api_type,
        api_key=args.api_key
    ).fetch_response(
        input_payloads=input_payloads,
        response_path=output_path,
        reset=args.reset
    )
    
    # ----------------------------------------------------------------------
    # Evaluate the responses
    # ----------------------------------------------------------------------
    eval_results = ResponseEvaluatorFactory.get_evaluator(
        eval_type="reference_based"
    ).evaluate_response(
        input_payloads=input_payloads,
        response_list=response_list,
        results_path=eval_results_path,
        reset=args.reset
    )
    
    # Check the evaluation summary
    eval_summary = get_eval_summary(eval_results, eval_summary_path)
    
    print(f"[[Evaluation Summary]]")
    print(f"- Num of responses: {eval_summary['num_responses']}")
    print(f"- Avg BLEU: {eval_summary['avg_bleu']:.4f}")
    print(f"- Avg ROUGE: {eval_summary['avg_rouge']:.4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)