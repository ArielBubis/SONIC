from transformers import AutoModel

def get_encoder(model_name: str) -> AutoModel:
    """
    Get the encoder model from Hugging Face.
    Parameters:
        model_name (str): Model name.
    Returns:
        AutoModel: Encoder model.
    """
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)