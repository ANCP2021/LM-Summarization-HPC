import evaluate
import bert_score

def rouge_score(generated, reference):
    rouge_score = evaluate.load("rouge")
    scores = rouge_score.compute(predictions=generated, references=reference)
    return scores

def bert(generated, reference):
    precision, recall, f1 = bert_score.score(generated, reference, lang="en")
    return precision, recall, f1
