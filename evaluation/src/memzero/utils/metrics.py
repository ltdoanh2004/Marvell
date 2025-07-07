import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score

nltk.download('punkt')

def compute_bleu(reference, hypothesis):
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)

def compute_f1(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    # Xây dựng tập từ duy nhất
    all_tokens = list(set(ref_tokens + hyp_tokens))
    ref_vec = [1 if t in ref_tokens else 0 for t in all_tokens]
    hyp_vec = [1 if t in hyp_tokens else 0 for t in all_tokens]
    return f1_score(ref_vec, hyp_vec)