#%%
# Define metrics 
from evaluate import load

# exact match
# predictions (list of str): List of predicted texts.
# references (list of str): List of reference texts.
exact_match_metric = load("exact_match")
def exact_match(predictions, references):
    return exact_match_metric.compute(predictions=predictions, references=references)

# remove ounctuation 
import re
def remove_punct(text):
    return re.sub(r'[^\w\s]', '', text)