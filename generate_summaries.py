from datasets import load_dataset
from models.gpt import GPT2
from models.t5 import T5

def load_data(dataset_name, split='train'):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def generate_summaries(model, dataset):
    summaries = []
    for data in dataset:
        text = data['document'] 
        summary = model.summarize(text)
        summaries.append(summary)
    return summaries

if __name__ == '__main__':
    xsum_dataset = load_dataset("EdinburghNLP/xsum", split='test[:5]')

    gpt_model = GPT2()

    gpt_summaries = generate_summaries(gpt_model, xsum_dataset)

