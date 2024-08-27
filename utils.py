#%%
# Define metrics 
from evaluate import load

# exact match
# predictions (list of str): List of predicted texts.
# references (list of str): List of reference texts.
exact_match_metric = load("exact_match")
def exact_match(predictions, references):
    return exact_match_metric.compute(predictions=predictions, references=references)

# match in the string 
# predictions (list of str): List of predicted texts.
# references (list of str): List of reference texts.
def match(predictions, references):
    return sum([1 for p, r in zip(predictions, references) if r in p]) / len(references)

# f1 score
# predictions (list of int): Predicted labels.
# references (list of int): Ground truth labels.
f1_metric = load("f1")
def f1(predictions, references):
    return f1_metric.compute(predictions=predictions, references=references)

# remove ounctuation 
import re
def remove_punct(text):
    return re.sub(r'[^\w\s]', '', text)