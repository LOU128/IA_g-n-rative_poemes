from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name):
    """
    Charge un modèle pré-entraîné et son tokenizer.

    Args:
        model_name (str): Nom du modèle à charger.
    Returns:
        model: Modèle pré-entraîné.
        tokenizer: Tokenizer associé au modèle.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer