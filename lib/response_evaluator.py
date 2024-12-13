import os
import jsonlines
from tqdm import tqdm
from evaluate import load


class AbstractResponseEvaluator:
    """
    Abstract class for evaluation handlers.
    """
    def __init__(self, num_responses):
        self.num_responses = num_responses
        self.num_results = 0
    
    def evaluate_response(self):
        """
        Evaluate the submission and return the score.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ReferenceBasedResponseEvaluator(AbstractResponseEvaluator):
    def __init__(self):
        super().__init__(0)
        self.bleu_n = 1
        self.bleu_scorer = load("bleu")
        self.rouge_scorer = load("rouge")
        self.cer_scorer = load("cer")
        
    def evaluate_response(self, **kwargs):
        input_payloads = kwargs['input_payloads']
        response_list = kwargs['response_list']
        self.num_responses = len(response_list)
        
        print(f"[[Evaluating responses]]")
        
        # Step 1: Check to cached evaluation results
        eval_results = []
        if kwargs['reset'] is True:         # TODO: Fix this
            pass
        else:
            eval_results = self.load_cached_results(kwargs['results_path'])
            self.num_results = len(eval_results)
            
            print(f"- Num of responses: {self.num_responses}")
            print(f"- Num of evaluation results: {self.num_results}")
            
            if self.num_results == self.num_responses:
                print(f"Successfully loaded the cached evaluation results!")
                return eval_results
            elif self.num_results > 0:
                print(f"Continuing from {self.num_results} cached evaluation results...")
        
        # Step 2: Evaluate the responses
        for input, output in tqdm(
            zip(kwargs['input_payloads'][self.num_results:], response_list[self.num_results:]),
            desc="Evaluating responses",
        ):
            ground_truth = input["ground_truth"]
            generated_response = output["generated_response"]
            
            bleu_score = self.bleu_scorer.compute(predictions=[generated_response], references=[[ground_truth]], max_order=self.bleu_n)
            cer_score = self.cer_scorer.compute(predictions=[generated_response], references=[ground_truth])
            rouge_scores = self.rouge_scorer.compute(predictions=[generated_response], references=[ground_truth])

            result = {
                "generated_response": generated_response,
                "ground_truth": ground_truth,
                "bleu_4_score": bleu_score['bleu'],
                "cer_score": cer_score,
                "rouge_l_score": rouge_scores['rougeL'],
            }
            self.save_results(result, kwargs['results_path'])
            
            eval_results.append(result)
            
        return eval_results
            
    def save_results(self, eval_result, results_path):
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with jsonlines.open(results_path, mode="a") as writer:
            writer.write(eval_result)
            
    def load_cached_results(self, results_path):
        print(f"Checking for cached evaluation results...")
        
        if os.path.exists(results_path):
            print(f"Evaluation results already exists at '{results_path}'.")
            eval_results = []
            with jsonlines.open(results_path, mode="r") as reader:
                for result in reader:
                    eval_results.append(result)
            return eval_results
        else:
            print(f"No cached evaluation results found.")
            return []


class ResponseEvaluatorFactory:
    """
    A factory class to specify evaluator based on the type of evaluation.
    """
    @staticmethod
    def get_evaluator(eval_type):
        """
        Return an evaluator based on the specified evaluation type.
        """
        if eval_type == "reference_based":
            return ReferenceBasedResponseEvaluator()
        else:
            raise ValueError(f"Unsupported evaluation type: {eval_type}.")