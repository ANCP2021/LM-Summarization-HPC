import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from datasets import load_dataset
from rouge_score import rouge_scorer
import time

DEBUG=1

def check_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# summarize the input text using the defined model and tokenizer
def summarize(model, tokenizer, text, device, model_name):
    summary = ""
    if model_name == "openai-community/gpt2":
        if DEBUG: print("ChatGPT")
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=1024, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elif model_name == "SalmanFaroz/Llama-2-7b-samsum":
        if DEBUG: print("llama")
        prompt = "Summarize the following file. " + text
        inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
        summary = tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=100,)[0],skip_special_tokens=True)

    return summary

# rouge scores between reference and generated summaries
def calculate_rouge_scores(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return scores

if __name__ == "__main__":
    # loading datasets
    # the split will be changed, running small amount for demo
    cnn_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split='test[:5]')
    xsum_dataset = load_dataset("EdinburghNLP/xsum", split='test[:5]')
    if DEBUG: print(f"CNN Dataset: {cnn_dataset}"); print(f"XSum Dataset: {xsum_dataset}")

    # check for device
    device = check_device()
    print(f"Using device: {device}")

    # load model and the respective tokenizer
    model_name = "SalmanFaroz/Llama-2-7b-samsum"
    #model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    datasets = [cnn_dataset, xsum_dataset]
    dataset_names = ["CNN/DailyMail", "XSum"]
    
    # iterate through each of our datasets
    for dataset, name in zip(datasets, dataset_names):
        if DEBUG: print(f"\nProcessing {name} dataset")
        start_time = time.time() # start processing time
        
        summaries = []
        references = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # iterate through examples per dataset
        for i, example in enumerate(dataset):
            if name == "CNN/DailyMail":     # the CNN dataset has context as 'article' and summary as 'highlights'
                text = example['article'] 
                reference = example['highlights']
            else:                           # the XSum dataset has context as 'document' and summary as 'summary'
                text = example['document']
                reference = example['summary']
            if DEBUG: print(f"About to tokenize text")
            tokenized_text = tokenizer.encode(text)
            if len(tokenized_text) < 1024: # check error since some tokenized inputs are greater than what gpt2 can handle
                if DEBUG: print(f"Tokenized text is less than 1024, about to summarize")
                summary = summarize(model, tokenizer, text, device, model_name)
                if DEBUG: print(f"Summarized")
                summaries.append(summary)
                references.append(reference)
            
                # Rouge scores
                scores = calculate_rouge_scores(reference, summary)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            
            if DEBUG: print(f"Processed {i} samples")
                
        end_time = time.time() # end processing time
        elapsed_time = end_time - start_time # total elapsed time
        

        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        
        print(f"Dataset: {name}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Average ROUGE-1: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L: {avg_rougeL:.4f}")
        print(f"Example summary: {summaries[0]}")
