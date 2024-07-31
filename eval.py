import evaluate
import json

if __name__ == '__main__':
    file_name = "google_flan-t5-xl_abisee_cnn_dailymail_summaries.json"
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)

    rouge_score = evaluate.load("rouge")
    generated_summaries = [sample['generated'] for sample in data]
    reference_summaries = [sample['reference'] for sample in data]
    
    scores = rouge_score.compute(predictions=generated_summaries, references=reference_summaries)
    print(scores)