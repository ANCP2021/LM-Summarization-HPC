from generate_summaries import (
    load_data, 
    generate_summaries, 
    write_to_folder,
)
from model import Model
import time
import torch

DEBUG = 1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEBUG: print(f'Using device: {device}')
    models = [
        # 'openai-community/gpt2',
        # 'openai-community/gpt2-medium',
        # 'openai-community/gpt2-large',
        'openai-community/gpt2-xl',
        # 'google/flan-t5-small',
        # 'google/flan-t5-base',
        # 'google/flan-t5-large',
        # 'google/flan-t5-xl',
    ]
    datasets = [
        'abisee/cnn_dailymail',
        'EdinburghNLP/xsum'
    ]
    split = 'test[:30]'
    inference_dir = './tester_inference_outputs/'
    execution_dir = './tester_execution_times'


    for dataset in datasets:
        dataset_loaded = load_data(dataset_name=dataset, split=split)
        if DEBUG: print(f'Dataset loaded: {dataset}')
        inference_execution_time = {}
        for model in models:
            if model[:6] == 'google':
                model_family = 't5'
            elif model[:6] == 'openai':
                model_family = 'gpt'
            else:
                raise ValueError('Modle family must be of type \'t5\' or \'gpt\'.')

            model_file_name = model
            model = Model(model_name=model, model_family=model_family)
            if DEBUG: print(f'model loaded {model_file_name} for dataset {dataset}')

            start_time = time.time()
            summaries = generate_summaries(model, dataset_loaded, dataset_name=dataset)
            end_time = time.time()
            if DEBUG: print(f'All summaries are generated for model {model_file_name}')

            execution_time = end_time - start_time
            duration_key = model_file_name+'_'+dataset
            inference_execution_time[duration_key] = execution_time

            file_name = model_file_name+'_'+dataset+'_summaries.json'
            write_to_folder(inference_dir, file_name, summaries)

        execution_file_name = model_file_name+'_'+dataset+'_execution_times.json'
        write_to_folder(execution_dir, execution_file_name, inference_execution_time)

        