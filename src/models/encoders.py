import logging
import os
import shutil
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer


def load_model(model_class, model_name, cache_dir):
    """
    Load a model from local cache or download it if not available.

    This function tries to load a model from a local cache directory first.
    If the model is not available locally or loading fails, it downloads
    the model and saves it to the cache directory.

    Args:
        model_class: The model class to instantiate (SentenceTransformer or CrossEncoder)
        model_name (str): The name or path of the model to load
        cache_dir (str): Directory to cache the downloaded models

    Returns:
        The loaded model instance

    Raises:
        Any exceptions from the model initialization are caught for local loading
        but will be raised if downloading the model fails
    """
    local_model_path = Path(cache_dir) / model_name.replace("/", "_")
    if os.path.exists(local_model_path):
        try:
            logging.info(f"Loading model from local path: {local_model_path}")
            model = model_class(str(local_model_path))
            return model
        except Exception as e:
            logging.warning(
                f"Failed to load model from {local_model_path}, attempting to re-download: {e}"
            )
            shutil.rmtree(local_model_path, ignore_errors=True)

    logging.info(f"Downloading and caching model: {model_name}")
    model = model_class(model_name)
    model.save(str(local_model_path))
    return model


def get_bi_encoder(model_name="all-MiniLM-L6-v2", cache_dir="cached_model"):
    """
    Get a SentenceTransformer bi-encoder model for generating embeddings.

    Bi-encoders encode sentences independently into vectors that can be compared
    using cosine similarity or other distance metrics. They are efficient for
    search and retrieval operations.

    Args:
        model_name (str): The name of the SentenceTransformer model to use
                          Defaults to "all-MiniLM-L6-v2", a lightweight model
                          with good performance for general sentence encoding
        cache_dir (str): Directory to cache the downloaded models

    Returns:
        SentenceTransformer: A loaded bi-encoder model
    """
    return load_model(SentenceTransformer, model_name, cache_dir)


def get_cross_encoder(
    model_name="cross-encoder/ms-marco-MiniLM-L6-v2", cache_dir="cached_model"
):
    """
    Get a CrossEncoder model for scoring sentence pairs.

    Cross-encoders take pairs of sentences as input and output a similarity score.
    They are more accurate than bi-encoders for ranking but less efficient for
    large-scale retrieval since they require comparing each query with each document.

    Args:
        model_name (str): The name of the CrossEncoder model to use
                          Defaults to "cross-encoder/ms-marco-MiniLM-L6-v2",
                          a model trained on the MS MARCO passage ranking dataset
        cache_dir (str): Directory to cache the downloaded models

    Returns:
        CrossEncoder: A loaded cross-encoder model
    """
    return load_model(CrossEncoder, model_name, cache_dir)
