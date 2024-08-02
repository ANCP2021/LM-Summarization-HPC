from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

class Model:
    def __init__(self, model_name: str, model_family: str):
        if model_family == 'gpt': 
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        elif model_family == 't5':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError('Modle family must be of type \'t5\' or \'gpt\'.')

    def summarize(self, text, max_input_length=512, max_output_length=150, num_beams=4):
        prompt = "Summarize the below text:\n\n" + text + "\n\nSummary:"
        inputs = self.tokenizer.encode(prompt, max_length=max_input_length, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs, max_length=max_input_length+max_output_length, num_beams=num_beams, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
