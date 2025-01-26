from src.config import CONFIG
from src.data_processing import load_model_and_tokenizer
from src.model.gpt_generation import generate_text
from src.model.evaluation import compute_bleu, compute_rouge

def main():
    # Charger les modèles et tokenizers
    model_gpt2, tokenizer_gpt2 = load_model_and_tokenizer(CONFIG["models"]["gpt2"])
    model_gpt_neo, tokenizer_gpt_neo = load_model_and_tokenizer(CONFIG["models"]["gpt_neo"])

    # Prompt utilisateur
    prompt = input("Entrez un mot-clé ou une phrase d'inspiration : ")

    # Génération des poèmes
    print("Début de la génération avec GPT-2...")
    poem_gpt2 = generate_text(model_gpt2, tokenizer_gpt2, prompt)
    print("GPT-2 terminé.")

    print("Début de la génération avec GPT-Neo...")
    poem_gpt_neo = generate_text(model_gpt_neo, tokenizer_gpt_neo, prompt)
    print("GPT-Neo terminé.")

    # Afficher les poèmes générés
    print("\n=== Poème GPT-2 ===")
    print(poem_gpt2)
    print("\n=== Poème GPT-Neo ===")
    print(poem_gpt_neo)

    # Évaluation des textes
    bleu_score = compute_bleu(poem_gpt2, poem_gpt_neo)
    rouge_scores = compute_rouge(poem_gpt2, poem_gpt_neo)

    # Afficher les scores
    print("\n=== Scores d'évaluation ===")
    print(f"Score BLEU : {bleu_score:.4f}")
    print(f"ROUGE-1 : {rouge_scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-L : {rouge_scores['rougeL'].fmeasure:.4f}")

if __name__ == "__main__":
    main()
