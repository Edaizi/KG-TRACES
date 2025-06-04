from .base_language_model import BaseLanguageModel
from .base_hf_causal_model import HfCausalModel

registed_language_models = {
    'qwen':HfCausalModel,
    'llama': HfCausalModel,
    'others': HfCausalModel
}

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    print("Model is not found in the registed_language_models, return HfCausalModel by default")
    return HfCausalModel