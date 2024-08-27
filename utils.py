#%%
# Define metrics 
from evaluate import load

# exact match
# predictions (list of str): List of predicted texts.
# references (list of str): List of reference texts.ù
# regexes_to_ignore (list of str): Regex expressions of characters to ignore when calculating the exact matches. Defaults to None. Note: the regex changes are applied before capitalization is normalized.
# ignore_case (bool): If True, turns everything to lowercase so that capitalization differences are ignored. Defaults to False.
# ignore_punctuation (bool): If True, removes punctuation before comparing strings. Defaults to False.
# ignore_numbers (bool): If True, removes all digits before comparing strings. Defaults to False.

exact_match_metric = load("exact_match")
def exact_match(predictions, references):
    return exact_match_metric.compute(predictions=predictions, references=references)

# match in the string 
# predictions (list of str): List of predicted texts.
# references (list of str): List of reference texts.
def match(predictions, references):
    true_l = []
    false_l = []
    matches = 0
    for i, p, r in enumerate(zip(predictions, references)):
        if r in p:
            matches += 1
            true_l.append(i)
        else:
            false_l.append(i)
    return {'match':matches / len(references)}, true_l, false_l

# remove ounctuation 
import re
def remove_punct(text):
    return re.sub(r'[^\w\s\d]', '', text)