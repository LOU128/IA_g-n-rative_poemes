# test_evaluation.py
from src.model.evaluation import compute_bleu, compute_rouge

def test_compute_bleu():
    ref = "Le chat dort paisiblement"
    hyp = "Le chat dort calmement"
    bleu = compute_bleu(ref, hyp)
    assert bleu > 0

def test_compute_rouge():
    ref = "Le chat dort paisiblement"
    hyp = "Le chat dort calmement"
    rouge = compute_rouge(ref, hyp)
    assert rouge["rouge1"].fmeasure > 0