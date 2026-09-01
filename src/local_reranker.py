from model_runtime import configure_quiet_model_loading


configure_quiet_model_loading()

import numpy as np
from sentence_transformers import CrossEncoder


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

RetrievedResult = tuple[float, dict[str, str]]
RerankedResult = tuple[float, float, dict[str, str]]

_model: CrossEncoder | None = None


def get_reranker_model() -> CrossEncoder:
    global _model

    if _model is None:
        _model = CrossEncoder(
            MODEL_ID,
            device="cpu",
            max_length=512,
        )

    return _model


def rerank_results(
    question: str,
    results: list[RetrievedResult],
) -> list[RerankedResult]:
    if not results:
        return []

    pairs = [
        (question, item["text"])
        for _, item in results
    ]

    raw_scores = get_reranker_model().predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    reranker_scores = np.asarray(
        raw_scores,
        dtype=np.float32,
    ).reshape(-1)

    reranked_results = [
        (
            float(reranker_score),
            faiss_score,
            item,
        )
        for reranker_score, (faiss_score, item)
        in zip(reranker_scores, results)
    ]

    reranked_results.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    return reranked_results
