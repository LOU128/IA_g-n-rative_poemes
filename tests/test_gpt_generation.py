# test_gpt_generation.py
from src.model.gpt_generation import generate_text
from src.data_processing import load_model_and_tokenizer

def test_generate_text():
    model, tokenizer = load_model_and_tokenizer("gpt2")
    prompt = "Le soleil se lève doucement"
    result = generate_text(model, tokenizer, prompt)
    assert isinstance(result, str)
    assert len(result) > 0