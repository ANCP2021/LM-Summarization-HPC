import evaluate
import json

if __name__ == '__main__':
    file_name = "google_flan-t5-xlabisee_cnn_dailymailsummaries.json"
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)

    print(data)
    quit()
    rouge_score = evaluate.load("rouge")
    scores = rouge_score.compute(
        predictions=[generated_summary], references=[reference_summary]
    )