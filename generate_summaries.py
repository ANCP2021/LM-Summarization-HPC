import torch
from datasets import load_dataset
from model import Model
import json

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

if __name__ == '__main__':
    args = {
        # 'model_name': 'openai-community/gpt2',
        # 'model_name': 'openai-community/gpt2-medium',
        # 'model_name': 'openai-community/gpt2-large',
        # 'model_name': 'openai-community/gpt2-xl',
        # 'model_name': 'google/flan-t5-small',
        # 'model_name': 'google/flan-t5-base',
        # 'model_name': 'google/flan-t5-large',
        'model_name': 'google/flan-t5-xl',
        'dataset_name': 'abisee/cnn_dailymail',
        # 'dataset_name': 'EdinburghNLP/xsum',
        'split': 'test[:1]',
    }
    if args['model_name'][:6] == 'google':
        model_family = 't5'
    elif args['model_name'][:6] == 'openai':
        model_family = 'gpt'
    else:
        raise ValueError('Modle family must be of type \'t5\' or \'gpt\'.')

    dataset = load_data(dataset_name=args['dataset_name'], split=args['split'])

    model = Model(model_name=args['model_name'], model_family=model_family)

    summaries = generate_summaries(model, dataset, dataset_name="abisee/cnn_dailymail")

    file_name = args['model_name']+'_'+args['dataset_name']+'_'+'summaries.json'
    file_name = file_name.replace('/', '_')
    with open(file_name, 'w') as json_file:
        json.dump(summaries, json_file)
