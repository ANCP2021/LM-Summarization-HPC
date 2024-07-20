from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

class T5:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str) -> str:
        prompt = "Summarize the below text:\n" + text
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs, max_length=1024, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
