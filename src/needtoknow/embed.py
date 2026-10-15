# Text embeddings with BAAI/bge-small-en-v1.5, run locally on the CPU through fastembed (ONNX).
# The model is downloaded once into NEEDTOKNOW_MODEL_CACHE and loaded once per process.

from functools import cache

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from needtoknow.config import load_settings

MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
# The bge model card gives this instruction for short queries searched against passages.
# fastembed's query_embed does not add it for this model, so it is added here.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# A batch is padded to its longest passage, so on a CPU one passage at a time is the fastest
# setting measured on Enron. The thread count is NEEDTOKNOW_EMBED_THREADS.
PASSAGE_BATCH = 1

Vector = NDArray[np.float32]


def embed_passages(texts: list[str]) -> list[Vector]:
    vectors = _model().passage_embed(texts, batch_size=PASSAGE_BATCH)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def embed_query(text: str) -> Vector:
    vector = next(iter(_model().query_embed(QUERY_INSTRUCTION + text)))
    return np.asarray(vector, dtype=np.float32)


@cache
def _model() -> TextEmbedding:
    settings = load_settings()
    return TextEmbedding(MODEL, cache_dir=str(settings.model_cache), threads=settings.embed_threads)
