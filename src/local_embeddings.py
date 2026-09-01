from model_runtime import configure_quiet_model_loading


configure_quiet_model_loading()

from sentence_transformers import SentenceTransformer


MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_ID, device="cpu")

    return _model


def create_document_embeddings(
    texts: list[str],
) -> list[list[float]]:
    passages = [f"passage: {text}" for text in texts]

    embeddings = get_embedding_model().encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def create_query_embedding(question: str) -> list[float]:
    embedding = get_embedding_model().encode(
        f"query: {question}",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.tolist()
