from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def compute_bleu(reference, hypothesis):
    """
    Calcule le score BLEU entre une référence et une hypothèse.

    Args:
        reference (str): Texte de référence.
        hypothesis (str): Texte généré.

    Returns:
        float: Score BLEU.
    """
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu(reference_tokens, hypothesis_tokens)

def compute_rouge(reference, hypothesis):
    """
    Calcule les scores ROUGE entre une référence et une hypothèse.

    Args:
        reference (str): Texte de référence.
        hypothesis (str): Texte généré.

    Returns:
        dict: Scores ROUGE-1 et ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores