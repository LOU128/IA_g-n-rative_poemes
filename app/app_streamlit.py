import streamlit as st
from src.data_processing import load_model_and_tokenizer
from src.model.gpt_generation import generate_text
from src.model.evaluation import compute_bleu, compute_rouge
from src.config import CONFIG
from src.model.gpt_generation import format_poem
import sys
import os

# Ajouter le chemin racine du projet au système
current_dir = os.path.dirname(os.path.abspath(__file__))  # Chemin du fichier actuel
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Remonte d'un dossier vers la racine
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)  # Ajoute la racine du projet en priorité

def main():
    # Titre de l'application
    st.title("Générateur de Poèmes avec GPT-2 et GPT-Neo")

    # Message d'accueil
    st.write("Entrez une phrase ou un mot-clé pour générer des poèmes !")

    # Chargement des modèles et tokenizers
    st.write("Chargement des modèles, veuillez patienter...")
    try:
        model_gpt2, tokenizer_gpt2 = load_model_and_tokenizer(CONFIG["models"]["gpt2"])
        model_gpt_neo, tokenizer_gpt_neo = load_model_and_tokenizer(CONFIG["models"]["gpt_neo"])
        st.write("Modèles chargés avec succès !")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return

    # Zone de texte pour entrer un mot-clé ou une phrase d'inspiration
    prompt = st.text_input("Entrez un mot-clé ou une phrase d'inspiration (ex. amour, nature, voyage)")
    prompt = f"Écrivez un poème romantique sur : {prompt}"

    # Bouton pour générer les poèmes
    if st.button("Générer des poèmes"):
        if prompt:
            st.write("Début de la génération...")
            try:
                # Génération des poèmes
                poem_gpt2 = generate_text(model_gpt2, tokenizer_gpt2, prompt)
                poem_gpt_neo = generate_text(model_gpt_neo, tokenizer_gpt_neo, prompt)

                # Reformater les poèmes pour un meilleur affichage
                poem_gpt2 = format_poem(poem_gpt2)
                poem_gpt_neo = format_poem(poem_gpt_neo)

                # Affichage des poèmes générés
                st.subheader("Poèmes Générés")
                st.write("### Poème GPT-2:")
                st.write(poem_gpt2)

                st.write("### Poème GPT-Neo:")
                st.write(poem_gpt_neo)

                # Calcul des scores BLEU et ROUGE
                bleu_score = compute_bleu(poem_gpt2, poem_gpt_neo)
                rouge_scores = compute_rouge(poem_gpt2, poem_gpt_neo)

                # Affichage des scores
                st.subheader("Scores d'évaluation")
                st.write(f"**Score BLEU** : {bleu_score:.4f}")
                st.write("**Scores ROUGE :**")
                st.write(f"ROUGE-1 : {rouge_scores['rouge1'].fmeasure:.4f}")
                st.write(f"ROUGE-L : {rouge_scores['rougeL'].fmeasure:.4f}")
            except Exception as e:
                st.error(f"Erreur lors de la génération des poèmes : {e}")
        else:
            st.warning("Veuillez entrer un mot-clé ou une phrase d'inspiration.")

if __name__ == "__main__":
    main()

