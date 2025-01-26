def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """
    Génère du texte à partir d'un modèle donné avec des paramètres optimisés.

    Arguments :
        model : Modèle pré-entraîné (GPT-2 ou GPT-Neo).
        tokenizer : Tokenizer associé.
        prompt : Texte initial pour la génération.
        max_length : Longueur maximale du texte généré.
        temperature : Contrôle la créativité (plus bas = moins créatif).
        top_p : Utilisé pour le nucleus sampling.
        repetition_penalty : Réduit les répétitions dans le texte généré.

    Retour :
        Texte généré sous forme de chaîne.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2  # Évite les répétitions de phrases
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def format_poem(poem, line_length=50):
    """
    Reformate un texte généré en lignes d'une longueur donnée.

    Arguments :
        poem : Texte du poème à reformater.
        line_length : Longueur maximale de chaque ligne.

    Retour :
        Texte reformatté en lignes courtes.
    """
    lines = [poem[i:i+line_length] for i in range(0, len(poem), line_length)]
    return "\n".join(lines)