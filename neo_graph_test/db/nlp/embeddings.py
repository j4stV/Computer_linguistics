import os
import threading
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_MODEL: Optional[SentenceTransformer] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_DEVICE: Optional[str] = None
_INIT_LOCK = threading.Lock()


def _get_device() -> str:
    """Return device string. Respects EMBEDDINGS_DEVICE or auto-detects CUDA/MPS."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    env_device = os.environ.get("EMBEDDINGS_DEVICE")
    if env_device:
        _DEVICE = env_device
        return _DEVICE
    # Приоритет: CUDA > MPS > CPU
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _DEVICE = "mps"
    else:
        _DEVICE = "cpu"
    return _DEVICE


def _ensure_tokenizer() -> AutoTokenizer:
    """Lazy-load and cache the HF tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _INIT_LOCK:
        if _TOKENIZER is None:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _TOKENIZER


def _ensure_model() -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model on the chosen device."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _INIT_LOCK:
        if _MODEL is None:
            _MODEL = SentenceTransformer(MODEL_NAME, device=_get_device())
    return _MODEL


def get_chunks(
    texts: Union[str, Sequence[str]],
    chunk_size_tokens: int = 256,
    overlap_tokens: int = 32,
    min_tokens: int = 5,
) -> List[str]:
    """Split text(s) into token-based chunks using a sliding window.

    Arguments
    ---------
    texts: str | list[str]
        One or many texts to split.
    chunk_size_tokens: int
        Target chunk length in tokens.
    overlap_tokens: int
        Overlap between consecutive chunks in tokens.
    min_tokens: int
        Discard chunks shorter than this threshold.

    Returns
    -------
    list[str]
        Decoded text chunks.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    tokenizer = _ensure_tokenizer()
    step = chunk_size_tokens - overlap_tokens

    texts_list: List[str]
    if isinstance(texts, str):
        texts_list = [texts]
    elif isinstance(texts, Iterable):
        texts_list = [t for t in texts if isinstance(t, str)]
    else:
        raise TypeError("texts must be a string or an iterable of strings")

    chunks: List[str] = []
    for text in texts_list:
        if not text:
            continue
        token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        n = len(token_ids)
        if n == 0:
            continue
        i = 0
        while True:
            end = min(i + chunk_size_tokens, n)
            sub_tokens = token_ids[i:end]
            if len(sub_tokens) < min_tokens:
                if end >= n:
                    break
                i += step
                continue
            decoded = tokenizer.decode(sub_tokens, skip_special_tokens=True).strip()
            if decoded:
                chunks.append(decoded)
            if end >= n:
                break
            i += step

    return chunks


def get_embeddings(
    texts: Union[str, Sequence[str]],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Encode text(s) into sentence embeddings.

    If a single string is passed, returns a 1D vector. For multiple strings,
    returns a 2D array of shape (n_texts, embedding_dim).
    """
    model = _ensure_model()

    single_input = isinstance(texts, str)
    if single_input:
        inputs: List[str] = [texts]  # type: ignore[list-item]
    elif isinstance(texts, Iterable):
        inputs = [t for t in texts if isinstance(t, str)]
    else:
        raise TypeError("texts must be a string or an iterable of strings")

    if not inputs:
        return np.array([])

    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    if single_input:
        return vectors[0]
    return vectors


def cos_compare(vec_a: Union[Sequence[float], np.ndarray], vec_b: Union[Sequence[float], np.ndarray]) -> float:
    """Cosine similarity between two 1D vectors. Returns value in [-1, 1]."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both vectors must be 1D")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must have the same length")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


