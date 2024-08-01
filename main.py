from generate_summaries import (
    load_data, 
    generate_summaries, 
    write_to_folder,
)
from model import Model
import time
import json

if __name__ == '__main__':
    models = [
        'openai-community/gpt2',
        'openai-community/gpt2-medium',
        'openai-community/gpt2-large',
        'openai-community/gpt2-xl',
        'google/flan-t5-small',
        'google/flan-t5-base',
        'google/flan-t5-large',
        'google/flan-t5-xl',
    ]
    datasets = [
        'abisee/cnn_dailymail',
        'EdinburghNLP/xsum'
    ]
    split = 'test[:1]'
    directory = './inference_outputs/'
    inference_execution_time = {}

    for dataset in datasets:
        dataset_loaded = load_data(dataset_name=dataset, split=split)
        for model in models:
            if model[:6] == 'google':
                model_family = 't5'
            elif model[:6] == 'openai':
                model_family = 'gpt'
            else:
                raise ValueError('Modle family must be of type \'t5\' or \'gpt\'.')

            model_file_name = model
            model = Model(model_name=model, model_family=model_family)

            start_time = time.time()
            summaries = generate_summaries(model, dataset_loaded, dataset_name=dataset)
            end_time = time.time()

            execution_time = end_time - start_time
            duration_key = model_file_name+'_'+dataset
            inference_execution_time[duration_key] = execution_time

            file_name = model_file_name+'_'+dataset+'_'+'summaries.json'
            write_to_folder(directory, file_name, summaries)

    execution_file_name = './execution_times.json'
    with open(execution_file_name, 'w') as json_file:
        json.dump(inference_execution_time, json_file)
        