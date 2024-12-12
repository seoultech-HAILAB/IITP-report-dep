import json


def get_eval_summary(eval_results, results_path):
    avg_bleu = sum([result['bleu_4_score'] for result in eval_results]) / len(eval_results)
    avg_rouge = sum([result['rouge_l_score'] for result in eval_results]) / len(eval_results)
    
    eval_summary = {
        "num_responses": len(eval_results),
        "avg_bleu": round(avg_bleu, 4),
        "avg_rouge": round(avg_rouge, 4),
    }
    
    with open(results_path, "w") as f:
        json.dump(eval_summary, f)
    
    return eval_summary
    