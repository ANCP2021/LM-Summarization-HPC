from datasets import load_dataset
import json
import os

def load_data(dataset_name, split='train'):
    if dataset_name == "EdinburghNLP/xsum":
        dataset = load_dataset(dataset_name, split=split)
    elif dataset_name == "abisee/cnn_dailymail":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
    else:
        raise ValueError(f"Incorrect dataset: {dataset_name}")
    
    return dataset

def generate_summaries(model, dataset, dataset_name):
    summary_dicts = []
    for data in dataset:
        if dataset_name == "EdinburghNLP/xsum":
            text = data['document']
            reference = data['summary']
        elif dataset_name == "abisee/cnn_dailymail": 
            text = data['article']
            reference = data['highlights']
        else:
            raise ValueError(f"Incorrect dataset: {dataset_name}")
        
        summary = model.summarize(text)
        summary_dicts.append({'generated': summary, 'reference': reference, 'context': text})
    return summary_dicts

def write_to_folder(dir, file_name, summaries):
    file_name = file_name.replace('/', '_')
    file_path = os.path.join(dir, file_name)   
    os.makedirs(dir, exist_ok=True)

    with open(file_path, 'w') as json_file:
        json.dump(summaries, json_file)
