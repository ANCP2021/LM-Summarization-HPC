from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# openai-community/gpt2
# openai-community/gpt2-medium
# openai-community/gpt2-large
# openai-community/gpt2-xl

# google/flan-t5-small
# google/flan-t5-base
# google/flan-t5-large
# google/flan-t5-xl
# google/flan-t5-xxl

class Model:
    def __init__(self, model_name: str = "openai-community/gpt2-xl", model_fam: str = 'gpt'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_fam == 'gpt': 
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_fam == 't5':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError('Modle family must be of type \'t5\' or \'gpt\'.')

    def summarize(self, text: str) -> str:
        prompt = "Summarize the below text:\n" + text
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs, max_length=1024, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
