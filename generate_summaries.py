from datasets import load_dataset
from model import Model

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

    model = Model()

    summaries = generate_summaries(model, xsum_dataset)
    print(summaries)
