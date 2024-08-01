import evaluate
import bert_score
import json

def rouge_score(generated, reference):
    rouge_score = evaluate.load("rouge")
    scores = rouge_score.compute(predictions=generated, references=reference)
    return scores

def bert(generated, reference):
    precision, recall, f1 = bert_score.score(generated, reference, lang="en")
    return precision, recall, f1

if __name__ == '__main__':
    file_name = "./inference_outputs/google_flan-t5-xl_abisee_cnn_dailymail_summaries.json"
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)

    generated_summaries = [sample['generated'] for sample in data]
    reference_summaries = [sample['reference'] for sample in data]
    
    rouge_scores = rouge_score(generated_summaries, reference_summaries)
    print(rouge_scores)
    precision_bert, recall_bert, f1_bert = bert(generated_summaries, reference_summaries)
    print(f"Precision: {precision_bert.mean().item()}")
    print(f"Recall: {recall_bert.mean().item()}")
    print(f"F1: {f1_bert.mean().item()}")